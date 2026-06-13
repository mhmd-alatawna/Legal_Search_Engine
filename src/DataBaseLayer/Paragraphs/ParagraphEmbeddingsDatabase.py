import logging
import uuid
from typing import Dict, Iterable, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm

from src.DataBaseLayer.EmbeddingsGenerator import EmbeddingGenerator

logger = logging.getLogger(__name__)

_UPSERT_BATCH_SIZE = 4096


class ParagraphEmbeddingsDatabase:
    """Simplified local Qdrant collection dedicated exclusively to paragraphs."""

    def __init__(self, db_path):
        """
        Initializes the local Qdrant connection and stores the embedder.
        db_path: Path to the directory where Qdrant storage should be saved.
        embedder: The EmbeddingGenerator instance.
        """
        self._embedder = EmbeddingGenerator()
        self.collection_name = "paragraphs"
        self._client = QdrantClient(path=db_path)
        self.ensure_collection()

    def close(self) -> None:
        """Cleanly close the local database connection."""
        self._client.close()

    def ensure_collection(self) -> None:
        """Creates the paragraphs collection and its index if it doesn't exist."""
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
    def _point_id(case_id: str, paragraph_id: str) -> str:
        """Generates a deterministic point id so re-indexing overwrites data in place."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"paragraphs/{case_id}/{paragraph_id}"))

    def index_paragraphs(self, paragraphs: Iterable[Tuple[str, str, str]]) -> int:
        """
        Embeds and upserts records into Qdrant with a progress bar.
        Accepts an iterable of tuples: (case_id, paragraph_id, text)
        """
        total = 0
        batch = []

        def flush(items: List[Tuple[str, str, str]]) -> int:
            if not items:
                return 0
            import time  # IMPORT TIME HERE

            t0 = time.time()
            # Step 1: Encode
            vectors = self._embedder.encode([text for _, _, text in items])

            t1 = time.time()
            # Step 2: Form structures
            points = []
            for (case_id, para_id, text), vector in zip(items, vectors):
                curr = qmodels.PointStruct(
                    id=self._point_id(case_id, para_id),
                    vector=vector,
                    payload={"case_id": case_id, "paragraph_id": para_id},
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
        progress_bar = tqdm(paragraphs, desc="Indexing Paragraphs", unit="para")

        for record in progress_bar:
            batch.append(record)
            if len(batch) >= _UPSERT_BATCH_SIZE:
                total += flush(batch)
                batch = []
                if total % (_UPSERT_BATCH_SIZE * 10) == 0:
                    # Use progress_bar.write to prevent log messages from breaking the progress bar layout
                    progress_bar.write(f"Indexed {total} paragraphs...")

        # Flush the remaining items in the final batch
        if batch:
            total += flush(batch)

        progress_bar.close()
        self.__build_final_index()
        logger.info("Finished indexing %d total paragraphs.", total)
        return total

    def search(self, query_text, top_k) :
        """
        Executes semantic search against stored paragraphs.
        Returns a list of (case_id, paragraph_id, score) tuples.
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
            result.append((point.payload["case_id"], point.payload["paragraph_id"], point.score))

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