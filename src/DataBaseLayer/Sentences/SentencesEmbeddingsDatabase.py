import logging
import uuid
from typing import Dict, Iterable, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm

from src.DataBaseLayer.EmbeddingsGenerator import EmbeddingGenerator

logger = logging.getLogger(__name__)

# Massively increased batch size for tiny sentence vectors to minimize disk I/O pauses
_UPSERT_BATCH_SIZE = 16384


class SentencesEmbeddingsDatabase:
    """Simplified local Qdrant collection dedicated exclusively to sentences."""

    def __init__(self, db_path):
        """
        Initializes the local Qdrant connection and stores the embedder.
        db_path: Path to the directory where Qdrant storage should be saved.
        """
        self._embedder = EmbeddingGenerator()
        self.collection_name = "sentences"
        self._client = QdrantClient(path=db_path)
        self.ensure_collection()

    def close(self) -> None:
        """Cleanly close the local database connection."""
        self._client.close()

    def ensure_collection(self) -> None:
        """Creates the sentences collection and its index if it doesn't exist."""
        if self._client.collection_exists(self.collection_name):
            return

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self._embedder.vector_size,
                distance=qmodels.Distance.COSINE,
                # Force Qdrant to hold vectors in temporary memory map instead of immediate disk thrashing
                on_disk=True
            ),
            # CRITICAL: Tell Qdrant NOT to build the HNSW index while we are uploading
            optimizers_config=qmodels.OptimizersConfigDiff(
                indexing_threshold=0
            )
        )

        self._client.create_payload_index(
            collection_name=self.collection_name,
            field_name="case_id",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
        logger.info("Created Qdrant collection '%s' with deferred indexing.", self.collection_name)

    @staticmethod
    def _point_id(case_id: str, sentence_id: str) -> str:
        """Generates a deterministic point id so re-indexing overwrites data in place."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"sentences/{case_id}/{sentence_id}"))

    def index_sentences(self, sentences: Iterable[Tuple[str, str, str]]) -> int:
        """
        Embeds and upserts records into Qdrant with a progress bar.
        Accepts an iterable of tuples: (case_id, sentence_id, text)
        """
        total = 0
        batch = []

        def flush(items: List[Tuple[str, str, str]]) -> int:
            if not items:
                return 0
            import time

            t0 = time.time()
            # Step 1: Encode
            vectors = self._embedder.encode([text for _, _, text in items])

            t1 = time.time()
            # Step 2: Form structures
            points = []
            for (case_id, sent_id, text), vector in zip(items, vectors):
                curr = qmodels.PointStruct(
                    id=self._point_id(case_id, sent_id),
                    vector=vector,
                    payload={"case_id": case_id, "sentence_id": sent_id},
                )
                points.append(curr)

            t2 = time.time()
            # Step 3: Upsert to Qdrant
            self._client.upsert(collection_name=self.collection_name, points=points)
            t3 = time.time()

            # PRINT THE TIMINGS
            progress_bar.write(
                f"TIMING -> Encode: {t1 - t0:.2f}s | Prep: {t2 - t1:.2f}s | Upsert (Qdrant): {t3 - t2:.2f}s")
            return len(items)

        # Wrap the iterable with tqdm for automatic progress tracking
        progress_bar = tqdm(sentences, desc="Indexing Sentences", unit="sent")

        for record in progress_bar:
            batch.append(record)
            if len(batch) >= _UPSERT_BATCH_SIZE:
                total += flush(batch)
                batch = []
                # Adjusted the modulo so it doesn't spam the console too frequently with the larger batch size
                if total % (_UPSERT_BATCH_SIZE * 5) == 0:
                    progress_bar.write(f"Indexed {total} sentences...")

        # Flush the remaining items in the final batch
        if batch:
            total += flush(batch)

        progress_bar.close()
        self.__build_final_index()
        logger.info("Finished indexing %d total sentences.", total)
        return total

    def search(self, query_text, top_k):
        """
        Executes semantic search against stored sentences.
        Returns a list of (case_id, sentence_id, score) tuples.
        """
        query_vector = self._embedder.encode_one(query_text)
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        result = []
        for point in response.points:
            result.append((point.payload["case_id"], point.payload["sentence_id"], point.score))

        return result

    def __build_final_index(self) -> None:
        """Triggers the heavy HNSW indexing process once at the very end."""
        logger.info("Re-enabling Qdrant optimizer and building final HNSW search index...")
        self._client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=qmodels.OptimizersConfigDiff(
                indexing_threshold=20000  # Standard threshold (starts indexing now)
            )
        )