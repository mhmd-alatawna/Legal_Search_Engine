# 📄 Legal Document Search & Retrieval System

Welcome to the **Legal Document Search & Retrieval System**!

This project is a robust, multiprocessing pipeline designed to ingest, process, and search through large datasets of legal cases. By segmenting documents into logical chunks (**paragraphs, sections, and sentences**) and applying advanced text processing, this system aims to accurately retrieve relevant legal citations and case references.

---

# 🚀 Quick Start Guide

Follow these steps to get the project up and running on your local machine.

## 1. Install Dependencies

First, install all required Python packages:

```bash
pip install -r requirements.txt
```

---

## 2. Generate the Databases

The system relies on local SQLite databases and inverted indexes to function.

Navigate to `DataLayerManager.py`, uncomment the execution code at the bottom of the file, and run it.

> **Note**
>
> This will automatically create a new directory called **`Databases`** in your project root and populate it with the parsed document data.

---

## 3. Run the Evaluators

Once the databases are generated, you can test the system's performance.

Navigate to the result measurer files (for example, `ParagraphToParagraphMeasurer.py`), choose the algorithm you want to test, and execute it to obtain the **Precision**, **Recall**, and **F1-score**.

---

# ⚙️ System Properties & Performance

## Cross-Platform Compatibility

All file paths are calculated dynamically using modern path libraries. There are **zero hardcoded paths**, meaning the project runs seamlessly on **Windows**, **macOS**, and **Linux**.

---

## Dynamic Multiprocessing

The most computationally intensive tasks (such as text processing and database generation) execute in parallel.

The system dynamically detects the number of logical CPU cores available and utilizes all of them, making it highly scalable and suitable for high-performance computing environments.

---

## Efficient Build Time

Generating the complete set of databases from the raw text files takes approximately **1–2 hours** on a standard machine with **16 logical cores**.

---

# 🏗️ Implementation Status

## 1. Data Ingestion & Processing

We have successfully implemented database generators for four distinct text granularities:

* Original Text
* Paragraphs
* Sentences
* Sections

### Smart Cleaning

Before being inserted into the databases, the text undergoes multiple normalization steps to:

* Remove noise
* Mask highly variable numbers (such as dates and monetary values)
* Ensure the resulting text is clean and consistent

### NLP Segmentation

We use **spaCy** to intelligently segment the text.

For example, paragraphs are constructed by securely counting sentences (for instance, capping a paragraph at **10 sentences**). Although computationally expensive, this approach provides highly accurate boundaries.

### Lemmatization

Words are reduced to their base dictionary forms (**lemmas**) using **spaCy**, improving search matching and retrieval quality.

---

## 2. Search Algorithms

We have implemented three specialized searchers:

* Paragraph-to-Paragraph
* Sentence-to-Sentence
* Section-to-Paragraph

### Macro-BM25

All searchers currently rely on a custom **Macro-BM25** implementation.

This method aggregates keyword matches at smaller text segments and rolls them up to score the entire document.

> **Note**
>
> For deeper mathematical details and suggested improvements, refer to:
>
> `Searchers/README.md`

### Top-K Retrieval

The searchers simply return the **Top-K** highest-scoring documents.

They currently do **not** use confidence scores or strict cut-off thresholds.

### Current Limitation (Self-Retrieval)

Because the query case itself is used to search the database, the system currently returns the query case (and exact duplicates) as the **#1 best match**.

These are not true citations and should be filtered out in future updates.

---

## 3. Evaluation & Metrics

We have implemented automated result-measuring scripts for each searcher.

### Fast Testing

To accelerate development, the evaluators currently test against a subset of **200 queries** rather than the entire test set.

### Comparative Insights

All three searchers currently demonstrate similar overall performance.

By analyzing these metrics, we can identify where each searcher performs well and where it struggles, paving the way for a stronger combined approach.

---

# 🗺️ Roadmap & Next Milestones

To push the system toward state-of-the-art performance, the following improvements are planned.

## Semantic Search (Embeddings)

Implement vector databases and dense retrieval searchers using AI embeddings to capture conceptual matches that lack exact keyword overlap.

---

## Proposition Extraction

Build a database and searcher based on atomic **propositions** (factoids or distinct legal claims) rather than raw sentences.

---

## Duplicate Filtering

Implement logic to identify, flag, and remove duplicate cases and self-references from search results to improve retrieval accuracy.

---

## Deep Analytics

Upgrade the result measurers to export **`.csv`** files, enabling granular query-by-query analysis of successes and failures.

---

## The Combinator Module

Build an ensemble module that merges the outputs of:

* Lexical search (BM25)
* Semantic search (Embeddings)
* Structural searchers

into a single, highly accurate final ranking.

---

## LLM Summarization *(Optional)*

Integrate a Large Language Model (LLM) to summarize legal cases into a new database, allowing searches over condensed, high-signal case summaries.

---

## Metadata Integration

Create a database for case metadata (such as dates, courts, and judges) and build searchers capable of filtering or boosting results based on these factual attributes.

---

## PageRank Implementation *(Optional)*

Build a graph database that ranks cases according to how frequently they are cited by other important cases, using the **PageRank** algorithm.
