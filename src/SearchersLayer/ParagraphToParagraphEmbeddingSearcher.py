import logging
from typing import Dict, List, Tuple

from pyprojroot import here

from src.DatabaseLayer.DatabasesManagers.tmp_qdrantDB import QdrantDB
from configurations import *
from src.DatabaseLayer.Embedders.TextEmbedder import EmbeddingGenerator

logger = logging.getLogger(__name__)


class ParagraphToParagraphEmbeddingSearcher:
    def __init__(self, db_path, doc_to_segment_count, semantic_thresh = 0.65,macro_B = 0.75):
        """
        db_path: Path to the local Qdrant database directory.
        embedder: The EmbeddingGenerator instance.
        doc_to_segment_count: A dictionary mapping case_id -> total paragraph count.
        semantic_thresh: Hard cutoff to filter out background noise before summing.
        macro_B: Length penalty scaling parameter (0 to 1).
        """
        self.db = QdrantDB(db_path, qdrant_binaries_dir)
        self.doc_to_segment_count = doc_to_segment_count

        # Calculate average document length across the corpus
        self.avg_segment_count = sum(self.doc_to_segment_count.values())
        if len(self.doc_to_segment_count) > 0:
            self.avg_segment_count = self.avg_segment_count / len(self.doc_to_segment_count)
        else:
            self.avg_segment_count = 1

        self.B = macro_B
        self.semantic_thresh = semantic_thresh

    def search(self, query_paragraphs, top_k=5, pool_size_per_query=100):
        """
        Executes an optimized batch semantic search for a full case by querying
        all paragraphs concurrently. Aggregates semantic hits and applies Macro-BM25
        length normalization.
        """
        if not query_paragraphs:
            return []

        aggregated_segment_scores = {}

        # 1. Execute a single optimized batch vector search for all paragraphs at once
        batch_hits = self.db.search_batch(
            queries=query_paragraphs,
            top_k=pool_size_per_query,
            collection_name="paragraphs",
            embedder=EmbeddingGenerator()
        )

        # 2. Iterate through the matrix of results and pool segment scores
        for hits in batch_hits:
            for case_id, para_id, score in hits:
                # CRITICAL: Ignore weak semantic matches to prevent long-document noise accumulation
                if score < self.semantic_thresh:
                    continue

                key = (case_id, para_id)
                # Max-pooling approach per paragraph: if a query paragraph matches a target paragraph
                # multiple times across the loop, we track its strongest match or sum them.
                # Keeping your sum behavior but clean of background noise.
                aggregated_segment_scores[key] = aggregated_segment_scores.get(key, 0.0) + score

        # Reconstruct the tuple format
        combined_segment_results = [
            (case_id, para_id, score)
            for (case_id, para_id), score in aggregated_segment_scores.items()
        ]

        # 3. Roll up to the document level using actual Macro-BM25 length normalization
        scored_documents = self.doc_to_macro_score(combined_segment_results)

        return scored_documents[:top_k]

    def doc_to_macro_score(self, segment_search_results):
        raw_document_scores = {}

        # 1. SUM all the filtered segment scores for each document
        for case_id, _, score in segment_search_results:
            raw_document_scores[case_id] = raw_document_scores.get(case_id, 0.0) + score

        final_documents = []

        # 2. Apply Length Normalization (The Macro-BM25 penalty logic)
        for doc_id, raw_score in raw_document_scores.items():
            doc_length = self.doc_to_segment_count.get(doc_id, 1)

            # Calculate length penalty. If B=0.75, long docs are penalized.
            length_penalty = (1 - self.B) + self.B * (doc_length / self.avg_segment_count)

            # Normalize the aggregated semantic score
            normalized_score = raw_score / length_penalty

            final_documents.append((doc_id, normalized_score))

        # 3. Sort by the new normalized score
        final_documents.sort(key=lambda x: x[1], reverse=True)

        return final_documents