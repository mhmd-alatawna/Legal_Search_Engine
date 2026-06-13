# Ingestion and Processing

The core task of the `DataBaseLayer` is to ingest raw, structurally messy legal cases and transform them into predictable, mathematically uniform chunks (paragraphs) for search indexing. Legal cases are not standard text files; they contain heavily formatted arguments, volatile quantitative data, and varied paragraph markers.

To handle this, the `LegalDocumentProcessor` executes a strict, sequential text normalization and semantic cleaning workflow.

## Step-by-Step Preprocessing Pipeline

To ensure perfect reproducibility, every ingested document is subjected to the following chronological steps before storage:

### 1. Structural Normalization

Artifacts such as `\n`, bullet points (`•`), and standard suppressed tags (e.g., `<FRAGMENT_SUPPRESSED>`) are stripped out and flattened into single spaces. This prevents NLP libraries from misinterpreting arbitrary line breaks as sentence boundaries.

### 2. Paragraph Extraction

A sliding-window regular expression searches for bracketed paragraph markers (e.g., `[1]`). The logic enforces sequential validation—checking if the next parsed number is within a logical rolling count (0 to 6 steps ahead). This filters out false positives and groups the text into verified logical paragraphs.

### 3. Length Chunking

If a logical paragraph is exceedingly long, it is split into overlapping chunks (maximum 10 sentences) using spaCy's robust sentence boundary detection (`senter`). Overlapping ensures contextual continuity between chunks.

### 4. Semantic Masking

Before destructive cleaning occurs, highly volatile quantitative tokens are protected. Custom regular expressions identify and mask money, percentages, and dates (e.g., replacing `$5,000.00` with `MASKEDMONEY`). This normalizes varied numeric formats while protecting legal citations and docket numbers.

### 5. Lemmatization and Cleaning

The text is passed through spaCy to lemmatize purely alphabetic words and strip noisy punctuation. Alphanumeric tokens (like `72(1)`) and our previously injected `MASKED` tags are safely preserved and lowercased.

---

# Concurrency and Storage Design

Processing massive datasets of legal text requires significant compute. The `ParagraphDatabase` relies on Python's `concurrent.futures.ProcessPoolExecutor` to spin up an isolated `LegalDocumentProcessor` on every available CPU core.

To prevent database locking during high-throughput concurrent insertions, the SQLite database is explicitly configured with Write-Ahead Logging (`PRAGMA journal_mode=WAL;`).

This allows simultaneous, unhindered reads and writes across the worker pool.