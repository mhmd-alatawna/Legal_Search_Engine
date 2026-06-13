# Retrieval and Macro-BM25 Aggregation

The `SearchersLayer` is tasked with retrieving relevant documents based on query paragraphs. It leverages a micro-layer (Tantivy) to execute rapid, segment-level searches, and a custom Python layer to aggregate those paragraph-level hits back into document-level scores.

## The Legal Domain Challenge

Legal document retrieval introduces massive structural challenges. Queries are often entire cases themselves, and comparing a 50-paragraph query case to a target database containing both 10-paragraph decisions and 200-page rulings naturally breaks standard search aggregations.

Legal texts are also saturated with boilerplate language (e.g., standard of review, jurisdictional definitions) which can easily hijack search scores.

## The Failures of Standard Aggregation Methods

When determining how to roll micro-scores (paragraph hits) into a final document score, standard aggregation techniques fail when applied to legal text.

| Aggregation Method | How It Works | The Flaw in the Legal Domain |
|-------------------|--------------|------------------------------|
| **Max Segment Score** | Takes the single highest paragraph match as the document's total score. | Ignores evidence density. A massive, irrelevant case sharing one perfect boilerplate paragraph will defeat a highly relevant case containing 15 strong factual matches. |
| **Summing Scores** | Adds up all paragraph match scores across the target document. | Introduces catastrophic **Long Document Bias**. Massive cases accumulate hundreds of low-quality, noisy matches, easily defeating concise, highly relevant cases simply by sheer volume. |

## The Macro-BM25 Solution

To solve the flaws of standard aggregations, the `BM25_searcher` implements a custom Macro-BM25 length normalization approach.

The searcher still sums the segment scores to account for evidence density, but it subjects that raw total to a rigorous mathematical penalty based on the document's total length relative to the corpus.

The length penalty is calculated using the hyperparameter **B** (set to `0.75`):

$$
\text{Penalty} = (1 - B) + B \cdot \frac{|D|}{\text{avgdl}}
$$

Where:

- `$|D|$` is the total number of paragraphs in the target document.
- `$\text{avgdl}$` is the average number of paragraphs per document across the entire index.

### Production Behavior

If a target case is three times longer than the average document, this formula generates a heavy penalty multiplier. The raw summed score is divided by this penalty.

This fundamentally forces the search engine to prioritize **density over volume**. A long document can still rank as the top result, but only if it genuinely contains a proportional amount of high-quality semantic overlap to justify its massive size.

## Open Problems and Future Enhancements

While Macro-BM25 stabilizes document comparison, the retrieval performance is still impacted by two critical open problems currently under active development:

### 1. Query Importance Distribution

Currently, every paragraph in a query case contributes its hits equally to the search pool.

We need to implement a mechanism to calculate the *importance* of each query paragraph (potentially using average IDF scores) and distribute that importance score to the paragraphs in its hit set.

The challenge is distributing a constant score over a standard BM25 hit set without strictly relying on percentages, which inadvertently ignores under-confidence and rewards noise.

### 2. Near-Duplicate Case Deduplication

Legal databases frequently contain duplicate or amended versions of the same case.

Because these duplicate records are not 100% identical at the character level, traditional exact-string deduplication fails.

We need an advanced similarity thresholding mechanism to identify and retain only a single authoritative record per case to prevent index bloat and redundant search results.

### 3. Determining Confidence

currently we simply return the top k results without determining how confident is our searcher is.
by actually accounting for confidence, we might be able to improve the quality of the results and maybe even use it as a feature in the Combinator module later.
