import concurrent.futures

from tqdm import tqdm

from src.DataBaseLayer import DataLayerManager
from src.DataBaseLayer.OriginalText.OriginalTextDatabase import OriginalTextDatabase
from src.DataBaseLayer.Paragraphs.ParagraphDatabase import ParagraphDatabase
# --- NEW IMPORTS ---
from src.SearchersLayer.ParagraphToParagraphEmbeddingSearcher import ParagraphToParagraphEmbeddingSearcher
from src.DataBaseLayer.Paragraphs.ParagraphEmbeddingsDatabase import EmbeddingGenerator


def measure_metrics_at_k(searcher, test_cases, k, pool_size):
    """
    Measures Recall@K, Precision@K, and F1@K for the Semantic Searcher.
    """
    total_relevant_docs = 0
    total_retrieved_relevant_docs = 0
    total_retrieved_docs = 0
    processed_count = 0

    def run_single_query(item):
        local_paragraph_db = ParagraphDatabase(DataLayerManager.paragraphs_DB)
        case_id, ground_truth_citations = item

        # Fetch paragraphs for the query case
        raw_paragraphs = local_paragraph_db.get_paragraphs_by_case(case_id)
        local_paragraph_db.close()

        if not raw_paragraphs:
            return len(ground_truth_citations), 0, 0

        # Ensure we are passing a list of raw text strings to the embedder,
        # just in case your DB returns tuples like (case_id, para_id, text)
        query_paragraphs = [
            p[2] if isinstance(p, tuple) and len(p) == 3 else p
            for p in raw_paragraphs
        ]

        # --- CRITICAL CHANGE ---
        # Pass the case_id down to the searcher so Qdrant excludes the query case from its results
        top_results = searcher.search(
            query_paragraphs=query_paragraphs,
            top_k=k,
            pool_size_per_query=pool_size
        )

        retrieved_case_ids = {res[0] for res in top_results}
        matches = len(set(ground_truth_citations).intersection(retrieved_case_ids))

        return len(ground_truth_citations), matches, len(retrieved_case_ids)

    # Execution Block
    # NOTE: PyTorch handles its own multi-threading for matrix math.
    # If max_workers=16 causes your CPU to freeze or slow down due to thread contention,
    # lower max_workers to 2 or 4.
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(run_single_query, item) for item in test_cases]

        with tqdm(total=len(test_cases), desc="Testing Semantic Search", unit="query") as pbar:
            for future in concurrent.futures.as_completed(futures):
                relevant, matches, retrieved = future.result()
                total_relevant_docs += relevant
                total_retrieved_relevant_docs += matches
                total_retrieved_docs += retrieved
                processed_count += 1

                # --- INTERMEDIATE PRINTING ---
                if processed_count % 50 == 0:
                    current_recall = 0
                    current_precision = 0
                    current_f1 = 0

                    if total_relevant_docs > 0:
                        current_recall = total_retrieved_relevant_docs / total_relevant_docs
                    if total_retrieved_docs > 0:
                        current_precision = total_retrieved_relevant_docs / total_retrieved_docs
                    if (current_precision + current_recall) > 0:
                        current_f1 = 2 * (current_precision * current_recall) / (current_precision + current_recall)

                    pbar.write(
                        f"--> [Update] Queries: {processed_count} | Recall@{k}: {current_recall:.2%} | Precision@{k}: {current_precision:.2%} | F1@{k}: {current_f1:.2%}"
                    )

                pbar.update(1)

    # Final Calculation
    final_recall = 0
    final_precision = 0
    final_f1 = 0

    if total_relevant_docs > 0:
        final_recall = total_retrieved_relevant_docs / total_relevant_docs
    if total_retrieved_docs > 0:
        final_precision = total_retrieved_relevant_docs / total_retrieved_docs
    if (final_precision + final_recall) > 0:
        final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall)

    return final_recall, final_precision, final_f1


# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    raw_query_data = OriginalTextDatabase(DataLayerManager.original_text_DB).get_only_test_case_citations_pairs()
    filtered_query_data = [(case_id, citations) for case_id, citations in raw_query_data if len(citations) > 0]
    test_subset = filtered_query_data[:200]

    paragraph_db = ParagraphDatabase(DataLayerManager.paragraphs_DB)
    case_to_paragraph_count = dict(paragraph_db.get_case_to_paragraph_count())
    paragraph_db.close()

    # Initialize the local embedder (will load from disk if present)
    print("Loading embedding model...")
    embedder = EmbeddingGenerator()

    # Initialize the semantic searcher with Qdrant path
    # Make sure you have added your Qdrant DB path to your DataLayerManager
    searcher = ParagraphToParagraphEmbeddingSearcher(
        db_path=DataLayerManager.paragraphs_embeddings_DB,  # Assuming this exists in your DataLayerManager
        doc_to_segment_count=case_to_paragraph_count,
        semantic_thresh=0.65,
        macro_B=0.75
    )

    # Use dynamic K variable for clean execution and printing
    K_VAL = 5
    recall_score, precision_score, f1_score = measure_metrics_at_k(searcher, test_subset, k=K_VAL, pool_size=30)

    print(f"\n=== Final Semantic Results @ {K_VAL} ===")
    print(f"Recall:    {recall_score:.2%}")
    print(f"Precision: {precision_score:.2%}")
    print(f"F1 Score:  {f1_score:.2%}")