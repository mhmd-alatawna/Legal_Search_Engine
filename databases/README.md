# Databases Package — Data Layer

This package implements the **Databases** and **DatabasesManager** modules of the Legal Search Engine: everything related to storing, chunking, indexing and embedding the case-law knowledge base. It deliberately contains **no retrieval logic** — the Searchers, Combiner and Measurer layers are built *on top of* this package and consume it only through the `DatabasesManager` facade.

```
        ┌──────────────────────────────────────────────┐
        │                DatabasesManager              │   <- single entry point
        └──┬──────────┬──────────┬──────────┬──────────┘
           │          │          │          │
     MongoDocument  Tantivy    Qdrant    Chunking      Gemma
        Store      IndexStore VectorStore  Engine   PropositionGen
       (MongoDB)   (lexical)  (semantic) (spaCy)      (LLM)
```

## What's in the folder

| File | Class | Responsibility |
|------|-------|----------------|
| [manager.py](manager.py) | `DatabasesManager` | Central facade: initializes all infrastructures, runs the ingestion/build pipeline, exposes unified read/write/search access. |
| [config.py](config.py) | `DatabasesConfig` (+ per-component configs) | All tunables in one place: connection settings, index paths, chunk window sizes, model names, enable/disable switches. |
| [document_store.py](document_store.py) | `MongoDocumentStore` | The six document databases as MongoDB collections: `original_texts`, `paragraphs`, `sentences`, `propositions`, `quotes`, `metadata`. |
| [chunking.py](chunking.py) | `ChunkingEngine`, `Chunk` | Pre-processing: paragraph windows (200 tokens / 100 overlap), spaCy sentence windows (20 / 10), proposition contexts around `<FRAGMENT_SUPPRESSED>` tags (≤200 tokens, paragraph-bounded). |
| [inverted_index.py](inverted_index.py) | `TantivyIndexStore` | Tantivy BM25 indexes for paragraphs and sentences, with the legal tokenizer (lowercase, English + legal stopwords, stemming). |
| [embedding.py](embedding.py) | `EmbeddingGenerator` | Legal-BERT (`nlpaueb/legal-bert-base-uncased`) text encoder with mean pooling; lazy-loads torch/transformers. |
| [vector_store.py](vector_store.py) | `QdrantVectorStore` | Qdrant collections for paragraph / sentence / proposition embeddings (cosine distance), with per-case payloads. |
| [proposition_generator.py](proposition_generator.py) | `GemmaPropositionGenerator` | Optional LLM stage (`google/gemma-2-9b-it`, few-shot) that rewrites raw citation contexts into coherent legal propositions. |
| [requirements.txt](requirements.txt) | — | Dependencies of this package only. |

### Data model

Every chunk is a `Chunk(case_id, chunk_id, text)` where `chunk_id` is the zero-based ordinal within its case — it maps directly to the `paragraph_id` / `sentence_id` / `proposition_id` columns of the spec. The metadata record per case is:

```
case_id, citations, judge_name, case_year, court_type, main_categories, page_rank_score
```

Only `case_year` is filled heuristically at ingestion; the rest are initialized empty and meant to be enriched later via `manager.update_metadata(...)` (citation-graph construction, PageRank job, category labelling).

## Setup

```bash
pip install -r databases/requirements.txt
python -m spacy download en_core_web_sm   # optional; falls back to a rule-based sentencizer
```

Infrastructure:

- **MongoDB** must be running (default `mongodb://localhost:27017`). Start one locally or via Docker: `docker run -d -p 27017:27017 mongo`.
- **Qdrant** needs no server — by default it runs embedded on disk at `Data/QdrantStorage`. Set `QdrantConfig.url` to use a remote server instead.
- **Tantivy** indexes are plain directories under `Data/TantivyIndexes/`.

## How to use

### One-time build (ingestion)

Point the manager at a directory of `*.txt` case files (the file name stem becomes the `case_id`):

```python
from databases import DatabasesConfig, DatabasesManager

config = DatabasesConfig()
# Fast iteration: skip the heavy stages while developing.
config.enable_embeddings = False        # skip Legal-BERT + Qdrant
config.enable_llm_propositions = False  # keep raw citation contexts (default)

with DatabasesManager(config) as manager:
    stats = manager.build_from_directory("Data/cases")
    print(stats)  # {'cases': ..., 'paragraphs': ..., 'sentences': ..., ...}
```

The build is **idempotent** — re-running it replaces each case's data in place, then rebuilds the Tantivy indexes (and Qdrant collections if enabled) from MongoDB.

### Read access (what the Searchers will call)

```python
with DatabasesManager(config) as manager:
    text       = manager.get_original_text("case_001")
    paragraphs = manager.get_paragraphs("case_001")     # list[Chunk]
    props      = manager.get_propositions("case_001")   # list[Chunk]
    meta       = manager.get_metadata("case_001")       # dict

    # Low-level search primitives (building blocks, not searchers):
    bm25_hits = manager.lexical_search("paragraphs", "duty of care", top_k=50)
    # -> [(case_id, chunk_id, bm25_score), ...]

    cos_hits = manager.semantic_search(
        "sentences", "standard of review", top_k=50,
        exclude_case_id="case_001",   # never retrieve the query case itself
    )
    # -> [(case_id, chunk_id, cosine_score), ...]
```

### Enrichment (filling the metadata over time)

```python
manager.update_metadata("case_001",
                        citations=["case_087", "case_113"],
                        page_rank_score=0.0042)
manager.upsert_quotes("case_001", ["quoted passage one", "quoted passage two"])
```

### Configuration

Everything is a dataclass — override only what you need:

```python
from databases import DatabasesConfig, ChunkingConfig, EmbeddingConfig

config = DatabasesConfig(
    chunking=ChunkingConfig(paragraph_max_tokens=200, paragraph_overlap_tokens=100),
    embedding=EmbeddingConfig(model_name="BAAI/bge-large-en-v1.5", batch_size=64),
)
```

## Next steps (built on top of this package, not inside it)

1. **Searchers module** — each searcher receives a `DatabasesManager` instance and composes the primitives, e.g. `max_max_BM25_proposition_to_paragraphs` = for each of `manager.get_propositions(query_case)`, call `manager.lexical_search("paragraphs", prop.text, k1)`, take the max score per `case_id`, then the max across propositions. The cosine variants do the same with `manager.semantic_search`. `year_scores` and `page_rank` read from `manager.get_metadata`.
2. **Citation-graph enrichment job** — parse citations from the corpus, compute PageRank, write back with `manager.update_metadata`.
3. **Combiner** — `call_searchers` / `thresh_vote_searcher` / decision-tree ensemble over the searchers' ranked lists.
4. **Results Measurer** — recall / precision / F1 over the test queries.
5. **LLM propositions** — once the pipeline works end to end with raw contexts, flip `config.enable_llm_propositions = True` and rebuild to compare retrieval quality with Gemma-refined propositions.

## Notes & limitations

- "Token" in the chunking windows means a whitespace-delimited word — deterministic, model-independent, and safely within BERT's 512-token limit for the chosen window sizes.
- The embedding and LLM models load lazily; constructing a `DatabasesManager` is cheap until you actually embed or generate.
- The `quotes` collection is schema-ready but unpopulated by the build pipeline (spec marks it "subject to future adjustments").
- The existing root-level scripts ([Indexes.py](../Indexes.py), [Searchers.py](../Searchers.py), …) are the standalone v1.0 BM25 engine; this package is the v2 data layer and does not depend on them.