"""
Tantivy inverted index store.

Builds and queries lightweight full-text indexes for lexical (BM25-style)
retrieval over paragraphs and sentences. Each index stores `case_id`,
`chunk_id` and the chunk text, analyzed by a legal-domain tokenizer
(lowercasing, combined English + legal stopword removal, English stemming).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from typing import Dict, Iterable, List, Tuple

import nltk
import tantivy
from nltk.corpus import stopwords
from tantivy import Filter, SchemaBuilder, TextAnalyzerBuilder, Tokenizer

from databases.chunking import Chunk
from databases.config import InvertedIndexConfig

logger = logging.getLogger(__name__)

nltk.download("stopwords", quiet=True)

_TOKENIZER_NAME = "legal_tokenizer"

_LEGAL_STOPWORDS = [
    "court", "plaintiff", "defendant", "appellant", "appellee", "judgment",
    "ruling", "order", "affirmed", "reversed", "remanded", "petition",
    "motion", "counsel", "testimony", "evidence", "proceedings", "district",
    "circuit", "supreme", "pursuant", "herein", "thereto", "foregoing",
]


class TantivyIndexStore:
    """Unified access to the paragraph and sentence inverted indexes."""

    def __init__(self, config: InvertedIndexConfig):
        self._config = config
        self._indexes: Dict[str, tantivy.Index] = {}

    # ------------------------------------------------------------------
    # Schema / analyzer
    # ------------------------------------------------------------------

    @staticmethod
    def _build_schema() -> tantivy.Schema:
        builder = SchemaBuilder()
        builder.add_text_field("case_id", stored=True, tokenizer_name="raw")
        builder.add_integer_field("chunk_id", stored=True)
        builder.add_text_field("text", stored=True, tokenizer_name=_TOKENIZER_NAME)
        return builder.build()

    @staticmethod
    def _build_tokenizer():
        combined_stopwords = _LEGAL_STOPWORDS + stopwords.words("english")
        return (
            TextAnalyzerBuilder(Tokenizer.regex(r"(?i)([a-z]+)"))
            .filter(Filter.lowercase())
            .filter(Filter.custom_stopword(combined_stopwords))
            .filter(Filter.stemmer("english"))
            .build()
        )

    def _index_path(self, level: str) -> str:
        paths = {
            "paragraphs": self._config.paragraphs_index_path,
            "sentences": self._config.sentences_index_path,
        }
        if level not in paths:
            raise ValueError(
                f"Unknown index level '{level}'. Expected one of: {sorted(paths)}"
            )
        return paths[level]

    def _open_index(self, level: str) -> tantivy.Index:
        if level not in self._indexes:
            path = self._index_path(level)
            os.makedirs(path, exist_ok=True)
            index = tantivy.Index(self._build_schema(), path=path)
            index.register_tokenizer(_TOKENIZER_NAME, self._build_tokenizer())
            self._indexes[level] = index
        return self._indexes[level]

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build_index(self, level: str, chunks: Iterable[Chunk]) -> int:
        """(Re)build the index of one level from a chunk stream.

        The existing index directory is wiped first so the build is
        deterministic. Returns the number of indexed chunks.
        """
        path = self._index_path(level)
        if os.path.isdir(path):
            shutil.rmtree(path)
        self._indexes.pop(level, None)

        index = self._open_index(level)
        writer = index.writer(heap_size=self._config.writer_heap_size)

        count = 0
        for chunk in chunks:
            writer.add_document(
                tantivy.Document(
                    case_id=str(chunk.case_id),
                    chunk_id=int(chunk.chunk_id),
                    text=str(chunk.text),
                )
            )
            count += 1
            if count % 100_000 == 0:
                logger.info("Indexed %d chunks into '%s' index...", count, level)

        writer.commit()
        writer.wait_merging_threads()
        index.reload()
        logger.info("Finished building '%s' index (%d chunks).", level, count)
        return count

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_query(raw_query: str) -> str:
        """Strip punctuation and lowercase so Tantivy never misreads
        uppercase words as boolean operators (AND, OR, NOT)."""
        clean = re.sub(r"[^a-zA-Z0-9\s]", " ", raw_query)
        return clean.lower().strip()

    def search(
        self, level: str, query: str, top_k: int
    ) -> List[Tuple[str, int, float]]:
        """BM25 search over one index.

        Returns a list of (case_id, chunk_id, score) tuples sorted by
        descending BM25 score.
        """
        index = self._open_index(level)
        sanitized = self._sanitize_query(query)
        if not sanitized:
            return []

        searcher = index.searcher()
        parsed = index.parse_query(sanitized, ["text"])
        results = searcher.search(parsed, top_k)

        hits: List[Tuple[str, int, float]] = []
        for score, address in results.hits:
            document = searcher.doc(address)
            hits.append(
                (document["case_id"][0], document["chunk_id"][0], score)
            )
        return hits

    def count_documents(self, level: str) -> int:
        index = self._open_index(level)
        index.reload()
        return index.searcher().num_docs
