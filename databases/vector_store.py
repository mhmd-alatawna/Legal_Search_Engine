"""
Qdrant vector store.

Manages the three embedding databases (paragraphs, sentences, propositions)
as Qdrant collections. Every point carries a payload with `case_id`,
`chunk_id` and the original `text`, enabling downstream score aggregation
per case without extra database round-trips.
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, Iterable, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from databases.chunking import Chunk
from databases.config import QdrantConfig
from databases.embedding import EmbeddingGenerator

logger = logging.getLogger(__name__)

# Logical level -> config attribute holding the collection name.
_LEVEL_TO_COLLECTION_ATTR = {
    "paragraphs": "paragraphs_collection",
    "sentences": "sentences_collection",
    "propositions": "propositions_collection",
}

_UPSERT_BATCH_SIZE = 256


class QdrantVectorStore:
    """Unified access to the paragraph/sentence/proposition vector DBs."""

    def __init__(self, config: QdrantConfig, embedder: EmbeddingGenerator):
        self._config = config
        self._embedder = embedder
        if config.url:
            self._client = QdrantClient(url=config.url, api_key=config.api_key)
            logger.info("Connected to remote Qdrant at %s.", config.url)
        else:
            self._client = QdrantClient(path=config.local_path)
            logger.info("Opened embedded Qdrant storage at %s.", config.local_path)

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _collection_name(self, level: str) -> str:
        if level not in _LEVEL_TO_COLLECTION_ATTR:
            raise ValueError(
                f"Unknown embedding level '{level}'. "
                f"Expected one of: {sorted(_LEVEL_TO_COLLECTION_ATTR)}"
            )
        return getattr(self._config, _LEVEL_TO_COLLECTION_ATTR[level])

    def ensure_collection(self, level: str) -> None:
        """Create the collection for a level when it does not exist yet."""
        name = self._collection_name(level)
        if self._client.collection_exists(name):
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=self._embedder.vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )
        # Payload index on case_id enables fast metadata filtering.
        self._client.create_payload_index(
            collection_name=name,
            field_name="case_id",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
        logger.info("Created Qdrant collection '%s'.", name)

    def recreate_collection(self, level: str) -> None:
        """Drop and re-create the collection for a clean rebuild."""
        name = self._collection_name(level)
        if self._client.collection_exists(name):
            self._client.delete_collection(name)
        self.ensure_collection(level)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    @staticmethod
    def _point_id(level: str, case_id: str, chunk_id: int) -> str:
        """Deterministic point id so re-indexing overwrites in place."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{level}/{case_id}/{chunk_id}"))

    def index_chunks(self, level: str, chunks: Iterable[Chunk]) -> int:
        """Embed and upsert chunks into the collection of the given level.

        Returns the number of points written. Chunks are consumed in
        batches so arbitrarily large corpora can be streamed through.
        """
        self.ensure_collection(level)
        name = self._collection_name(level)

        total = 0
        batch: List[Chunk] = []

        def flush(items: List[Chunk]) -> int:
            if not items:
                return 0
            vectors = self._embedder.encode([c.text for c in items])
            self._client.upsert(
                collection_name=name,
                points=[
                    qmodels.PointStruct(
                        id=self._point_id(level, chunk.case_id, chunk.chunk_id),
                        vector=vector,
                        payload={
                            "case_id": chunk.case_id,
                            "chunk_id": chunk.chunk_id,
                            "text": chunk.text,
                        },
                    )
                    for chunk, vector in zip(items, vectors)
                ],
            )
            return len(items)

        for chunk in chunks:
            batch.append(chunk)
            if len(batch) >= _UPSERT_BATCH_SIZE:
                total += flush(batch)
                batch = []
                if total % (_UPSERT_BATCH_SIZE * 10) == 0:
                    logger.info("Indexed %d points into '%s'...", total, name)
        total += flush(batch)
        logger.info("Finished indexing %d points into '%s'.", total, name)
        return total

    def delete_case(self, level: str, case_id: str) -> None:
        """Remove every point of one case from the given collection."""
        self._client.delete(
            collection_name=self._collection_name(level),
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="case_id",
                            match=qmodels.MatchValue(value=case_id),
                        )
                    ]
                )
            ),
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        level: str,
        query_text: str,
        top_k: int,
        exclude_case_id: Optional[str] = None,
    ) -> List[Tuple[str, int, float]]:
        """Semantic search at the given level.

        Returns a list of (case_id, chunk_id, score) tuples sorted by
        descending cosine similarity. `exclude_case_id` filters out the
        query case itself (a case must not retrieve its own chunks).
        """
        query_vector = self._embedder.encode_one(query_text)
        query_filter = None
        if exclude_case_id is not None:
            query_filter = qmodels.Filter(
                must_not=[
                    qmodels.FieldCondition(
                        key="case_id",
                        match=qmodels.MatchValue(value=exclude_case_id),
                    )
                ]
            )
        response = self._client.query_points(
            collection_name=self._collection_name(level),
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            (point.payload["case_id"], point.payload["chunk_id"], point.score)
            for point in response.points
        ]

    def collection_stats(self) -> Dict[str, int]:
        """Return the point count of every existing collection."""
        stats: Dict[str, int] = {}
        for level in _LEVEL_TO_COLLECTION_ATTR:
            name = self._collection_name(level)
            if self._client.collection_exists(name):
                stats[name] = self._client.count(name, exact=True).count
        return stats
