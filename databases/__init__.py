"""
Data layer package for the Legal Search Engine.

This package implements the "Databases" and "DatabasesManager" modules:
  - MongoDB document storage (original texts, chunks, metadata)
  - Qdrant vector stores (paragraph / sentence / proposition embeddings)
  - Tantivy inverted indexes (paragraph / sentence lexical retrieval)
  - Pre-processing & chunking engine (paragraphs, sentences, propositions)
  - LLM-based proposition generation (Gemma, few-shot prompting)

The single public entry point is `DatabasesManager`, which initializes all
underlying infrastructures and exposes unified read/write access to them.

Submodules are imported lazily (PEP 562) so lightweight components (e.g.
the chunking engine or configuration objects) remain usable on machines
where the heavier infrastructure dependencies are not installed.
"""

from databases.config import (
    ChunkingConfig,
    DatabasesConfig,
    EmbeddingConfig,
    InvertedIndexConfig,
    LLMConfig,
    MongoConfig,
    QdrantConfig,
)

# Public name -> defining submodule, resolved on first attribute access.
_LAZY_EXPORTS = {
    "Chunk": "databases.chunking",
    "ChunkingEngine": "databases.chunking",
    "DatabasesManager": "databases.manager",
    "EmbeddingGenerator": "databases.embedding",
    "GemmaPropositionGenerator": "databases.proposition_generator",
    "MongoDocumentStore": "databases.document_store",
    "QdrantVectorStore": "databases.vector_store",
    "TantivyIndexStore": "databases.inverted_index",
}

__all__ = [
    "ChunkingConfig",
    "DatabasesConfig",
    "EmbeddingConfig",
    "InvertedIndexConfig",
    "LLMConfig",
    "MongoConfig",
    "QdrantConfig",
    *_LAZY_EXPORTS,
]


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib

        module = importlib.import_module(_LAZY_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
