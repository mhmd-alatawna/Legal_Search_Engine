"""
Configuration objects for the data layer.

Every infrastructure component (MongoDB, Qdrant, Tantivy, chunking engine,
embedding model, LLM) receives its own dataclass so each component can be
configured, tested and replaced in isolation. `DatabasesConfig` aggregates
them into the single object consumed by `DatabasesManager`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MongoConfig:
    """Connection settings for the MongoDB document store."""

    uri: str = "mongodb://localhost:27017"
    database_name: str = "legal_search_engine"
    # Server selection timeout in milliseconds (fail fast when Mongo is down).
    server_selection_timeout_ms: int = 5000


@dataclass
class QdrantConfig:
    """Connection settings for the Qdrant vector store.

    If `url` is provided, a remote Qdrant server is used; otherwise an
    embedded (local on-disk) instance is created at `local_path`.
    """

    url: Optional[str] = None
    api_key: Optional[str] = None
    local_path: str = "Data/QdrantStorage"
    # Collection names per chunk granularity.
    paragraphs_collection: str = "paragraph_embeddings"
    sentences_collection: str = "sentence_embeddings"
    propositions_collection: str = "proposition_embeddings"


@dataclass
class InvertedIndexConfig:
    """Locations of the Tantivy inverted indexes."""

    paragraphs_index_path: str = "Data/TantivyIndexes/paragraphs"
    sentences_index_path: str = "Data/TantivyIndexes/sentences"
    # Writer heap size in bytes (per index build).
    writer_heap_size: int = 1024 * 1024 * 1024


@dataclass
class ChunkingConfig:
    """Token-window parameters of the pre-processing & chunking strategy."""

    # Paragraphs: natural delimiters first, then sliding window if too long.
    paragraph_max_tokens: int = 200
    paragraph_overlap_tokens: int = 100
    # Sentences: spaCy sentence segmentation, then sliding window if too long.
    sentence_max_tokens: int = 20
    sentence_overlap_tokens: int = 10
    # Propositions: token window captured around each citation tag.
    proposition_window_tokens: int = 200
    citation_tag: str = "<FRAGMENT_SUPPRESSED>"
    # spaCy pipeline used for sentence segmentation.
    spacy_model: str = "en_core_web_sm"


@dataclass
class EmbeddingConfig:
    """Embedding model settings for the Qdrant vector databases."""

    # Legal-domain BERT; alternatively "BAAI/bge-large-en-v1.5".
    model_name: str = "nlpaueb/legal-bert-base-uncased"
    batch_size: int = 32
    max_sequence_length: int = 512
    # "cuda", "cpu" or None for automatic detection.
    device: Optional[str] = None
    normalize: bool = True


@dataclass
class LLMConfig:
    """Settings for the LLM used to generate legal propositions."""

    # Alternatively "google/gemma-7b-it".
    model_name: str = "google/gemma-2-9b-it"
    max_new_tokens: int = 128
    temperature: float = 0.2
    # "cuda", "cpu" or None for automatic detection.
    device: Optional[str] = None


@dataclass
class DatabasesConfig:
    """Aggregated configuration consumed by `DatabasesManager`."""

    mongo: MongoConfig = field(default_factory=MongoConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    inverted_index: InvertedIndexConfig = field(default_factory=InvertedIndexConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    # Build-time switches: heavy stages can be disabled for fast iteration.
    enable_embeddings: bool = True
    enable_llm_propositions: bool = False
