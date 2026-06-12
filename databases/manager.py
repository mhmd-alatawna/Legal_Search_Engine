"""
DatabasesManager: central orchestration component of the data layer.

Initializes the document DB (MongoDB), vector stores (Qdrant) and inverted
indexes (Tantivy), and exposes unified, high-speed read/write access to the
knowledge base. Consumers (the future Searchers module) interact only with
this class and never with the underlying infrastructure.

Typical usage:

    from databases import DatabasesConfig, DatabasesManager

    manager = DatabasesManager(DatabasesConfig())
    manager.build_from_directory("Data/cases")   # one-time ingestion
    text = manager.get_original_text("001234")   # read access
    hits = manager.lexical_search("paragraphs", "duty of care", top_k=50)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from databases.chunking import Chunk, ChunkingEngine
from databases.config import DatabasesConfig
from databases.document_store import MongoDocumentStore
from databases.embedding import EmbeddingGenerator
from databases.inverted_index import TantivyIndexStore
from databases.proposition_generator import GemmaPropositionGenerator
from databases.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

# A four-digit year between 1700 and 2099; the most frequent match in the
# document head is used as a heuristic for the decision year.
_YEAR_PATTERN = re.compile(r"\b(1[7-9]\d{2}|20\d{2})\b")


class DatabasesManager:
    """Facade over all databases of the legal search engine."""

    def __init__(self, config: Optional[DatabasesConfig] = None):
        self.config = config or DatabasesConfig()

        self.documents = MongoDocumentStore(self.config.mongo)
        self.chunker = ChunkingEngine(self.config.chunking)
        self.inverted_indexes = TantivyIndexStore(self.config.inverted_index)

        self.embedder: Optional[EmbeddingGenerator] = None
        self.vectors: Optional[QdrantVectorStore] = None
        if self.config.enable_embeddings:
            self.embedder = EmbeddingGenerator(self.config.embedding)
            self.vectors = QdrantVectorStore(self.config.qdrant, self.embedder)

        self.proposition_generator: Optional[GemmaPropositionGenerator] = None
        if self.config.enable_llm_propositions:
            self.proposition_generator = GemmaPropositionGenerator(self.config.llm)

    def close(self) -> None:
        """Release every underlying connection."""
        self.documents.close()
        if self.vectors is not None:
            self.vectors.close()

    def __enter__(self) -> "DatabasesManager":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # ==================================================================
    # Build pipeline (ingestion)
    # ==================================================================

    def build_from_directory(self, cases_directory: str) -> Dict[str, int]:
        """Build the full knowledge base from a directory of case files.

        Every `*.txt` file is treated as one case; the file name stem is
        the case id. Stages:
          1. Store original texts in MongoDB.
          2. Chunk into paragraphs / sentences / propositions (MongoDB).
          3. Extract basic metadata records (MongoDB).
          4. Build the Tantivy inverted indexes (paragraphs, sentences).
          5. Build the Qdrant embedding collections (if enabled).

        Returns a dictionary of build statistics.
        """
        case_files = self._discover_case_files(cases_directory)
        logger.info(
            "Building knowledge base from %d case files in '%s'.",
            len(case_files),
            cases_directory,
        )

        for case_id, file_path in case_files:
            with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
            self.ingest_case(case_id, text)

        self.build_inverted_indexes()
        if self.config.enable_embeddings:
            self.build_embedding_collections()

        stats = self.stats()
        logger.info("Knowledge base build complete: %s", stats)
        return stats

    @staticmethod
    def _discover_case_files(cases_directory: str) -> List[Tuple[str, str]]:
        if not os.path.isdir(cases_directory):
            raise FileNotFoundError(
                f"Cases directory not found: '{cases_directory}'"
            )
        case_files = [
            (os.path.splitext(name)[0], os.path.join(cases_directory, name))
            for name in sorted(os.listdir(cases_directory))
            if name.lower().endswith(".txt")
        ]
        if not case_files:
            raise ValueError(
                f"No .txt case files found in '{cases_directory}'."
            )
        return case_files

    def ingest_case(self, case_id: str, text: str) -> None:
        """Run the document-level ingestion stages for a single case.

        Index builds (Tantivy / Qdrant) are corpus-level batch operations
        and are NOT triggered here; call `build_inverted_indexes` and
        `build_embedding_collections` after ingesting the corpus.
        """
        self.documents.upsert_original_text(case_id, text)

        paragraphs = self.chunker.chunk_paragraphs(case_id, text)
        sentences = self.chunker.chunk_sentences(case_id, text)
        propositions = self.chunker.extract_proposition_contexts(case_id, text)

        if self.proposition_generator is not None:
            propositions = self._refine_propositions(propositions)

        self.documents.replace_chunks("paragraphs", case_id, paragraphs)
        self.documents.replace_chunks("sentences", case_id, sentences)
        self.documents.replace_chunks("propositions", case_id, propositions)

        self.documents.upsert_metadata(case_id, **self._extract_metadata(text))

    def _refine_propositions(self, raw_contexts: List[Chunk]) -> List[Chunk]:
        """Rewrite raw citation contexts into coherent legal propositions
        via the LLM, falling back to the raw context on failure."""
        refined: List[Chunk] = []
        for chunk in raw_contexts:
            proposition = self.proposition_generator.generate(chunk.text)
            refined.append(
                Chunk(
                    case_id=chunk.case_id,
                    chunk_id=chunk.chunk_id,
                    text=proposition if proposition else chunk.text,
                )
            )
        return refined

    @staticmethod
    def _extract_metadata(text: str) -> Dict[str, Any]:
        """Build the initial metadata record for a case.

        Only the decision year is heuristically extracted from the
        document head at ingestion time; the remaining fields are
        initialized empty and are meant to be enriched later (citation
        graph construction, PageRank computation, category labelling)
        through `update_metadata`.
        """
        head = text[:2000]
        years = _YEAR_PATTERN.findall(head)
        case_year = int(max(set(years), key=years.count)) if years else None
        return {
            "citations": [],
            "judge_name": None,
            "case_year": case_year,
            "court_type": None,
            "main_categories": [],
            "page_rank_score": None,
        }

    def build_inverted_indexes(self) -> Dict[str, int]:
        """(Re)build the Tantivy indexes from the MongoDB chunk data."""
        counts = {}
        for level in ("paragraphs", "sentences"):
            counts[level] = self.inverted_indexes.build_index(
                level, self.documents.iter_chunks(level)
            )
        return counts

    def build_embedding_collections(self) -> Dict[str, int]:
        """(Re)build the Qdrant collections from the MongoDB chunk data."""
        if self.vectors is None:
            raise RuntimeError(
                "Embeddings are disabled (config.enable_embeddings=False)."
            )
        counts = {}
        for level in ("paragraphs", "sentences", "propositions"):
            self.vectors.recreate_collection(level)
            counts[level] = self.vectors.index_chunks(
                level, self.documents.iter_chunks(level)
            )
        return counts

    # ==================================================================
    # Unified read access (consumed by the Searchers module)
    # ==================================================================

    def get_original_text(self, case_id: str) -> Optional[str]:
        return self.documents.get_original_text(case_id)

    def get_paragraphs(self, case_id: str) -> List[Chunk]:
        return self.documents.get_chunks("paragraphs", case_id)

    def get_sentences(self, case_id: str) -> List[Chunk]:
        return self.documents.get_chunks("sentences", case_id)

    def get_propositions(self, case_id: str) -> List[Chunk]:
        return self.documents.get_chunks("propositions", case_id)

    def get_quotes(self, case_id: str) -> List[Chunk]:
        return self.documents.get_chunks("quotes", case_id)

    def get_metadata(self, case_id: str) -> Optional[Dict[str, Any]]:
        return self.documents.get_metadata(case_id)

    def iter_case_ids(self) -> Iterator[str]:
        return self.documents.iter_case_ids()

    # ==================================================================
    # Unified write access (enrichment hooks)
    # ==================================================================

    def update_metadata(self, case_id: str, **fields: Any) -> None:
        """Partially update a metadata record (e.g. with citation lists
        or PageRank scores computed by downstream enrichment jobs)."""
        self.documents.upsert_metadata(case_id, **fields)

    def upsert_quotes(self, case_id: str, quotes: List[str]) -> None:
        """Store the quotes of a case (schema subject to future
        adjustments; quotes are provided externally for now)."""
        chunks = [
            Chunk(case_id=case_id, chunk_id=i, text=quote)
            for i, quote in enumerate(quotes)
        ]
        self.documents.replace_chunks("quotes", case_id, chunks)

    # ==================================================================
    # Search primitives (low-level building blocks, NOT searchers)
    # ==================================================================

    def lexical_search(
        self, level: str, query: str, top_k: int
    ) -> List[Tuple[str, int, float]]:
        """BM25 search over the 'paragraphs' or 'sentences' inverted
        index. Returns (case_id, chunk_id, score) tuples."""
        return self.inverted_indexes.search(level, query, top_k)

    def semantic_search(
        self,
        level: str,
        query: str,
        top_k: int,
        exclude_case_id: Optional[str] = None,
    ) -> List[Tuple[str, int, float]]:
        """Cosine-similarity search over the 'paragraphs', 'sentences' or
        'propositions' embedding collection. Returns (case_id, chunk_id,
        score) tuples."""
        if self.vectors is None:
            raise RuntimeError(
                "Embeddings are disabled (config.enable_embeddings=False)."
            )
        return self.vectors.search(level, query, top_k, exclude_case_id)

    # ==================================================================
    # Diagnostics
    # ==================================================================

    def stats(self) -> Dict[str, int]:
        """Aggregate counts across all databases."""
        stats = {
            "cases": self.documents.count_cases(),
            "paragraphs": self.documents.count_chunks("paragraphs"),
            "sentences": self.documents.count_chunks("sentences"),
            "propositions": self.documents.count_chunks("propositions"),
            "quotes": self.documents.count_chunks("quotes"),
        }
        if self.vectors is not None:
            for name, count in self.vectors.collection_stats().items():
                stats[f"qdrant/{name}"] = count
        return stats
