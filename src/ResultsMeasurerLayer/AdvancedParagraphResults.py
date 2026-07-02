import concurrent.futures

from tqdm import tqdm

from src.DatabaseLayer.DatabasesManagers.tmp_sqliteDB import SqliteDB
from src.SearchersLayer.ParagraphToParagraphSearcher import ParagraphToParagraphSearcher


def measure_metrics_at_k(searcher, test_cases, k, pool_size):
    """
    Measures global Recall@K, Precision@K, F1@K, and extended query-level statistics.
    """
    # Keep track of individual query outcomes for deep stats
    # Format: list of tuples -> (relevant_count, match_count, retrieved_count)
    query_results = []

    total_relevant_docs = 0
    total_retrieved_relevant_docs = 0
    total_retrieved_docs = 0
    processed_count = 0

    def run_single_query(item):
        local_paragraph_db = SqliteDB(DataLayerManager.paragraphs_DB)
        case_id, ground_truth_citations = item

        query_paragraphs = local_paragraph_db.get_paragraphs_by_case(case_id)
        local_paragraph_db.close()

        if not query_paragraphs:
            return len(ground_truth_citations), 0, 0

        top_results = searcher.search(query_paragraphs, top_k=k, pool_size_per_query=pool_size)

        retrieved_case_ids = {res[0] for res in top_results}
        matches = len(set(ground_truth_citations).intersection(retrieved_case_ids))

        return len(ground_truth_citations), matches, len(retrieved_case_ids)

    # Execution Block
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(run_single_query, item) for item in test_cases]

        with tqdm(total=len(test_cases), desc="Testing", unit="query") as pbar:
            for future in concurrent.futures.as_completed(futures):
                relevant, matches, retrieved = future.result()

                # Store individual query data
                query_results.append((relevant, matches, retrieved))

                # Add to global totals
                total_relevant_docs += relevant
                total_retrieved_relevant_docs += matches
                total_retrieved_docs += retrieved
                processed_count += 1

                # --- INTERMEDIATE PRINTING ---
                if processed_count % 50 == 0:
                    current_recall = total_retrieved_relevant_docs / total_relevant_docs if total_relevant_docs > 0 else 0
                    current_precision = total_retrieved_relevant_docs / total_retrieved_docs if total_retrieved_docs > 0 else 0
                    current_f1 = 0
                    if (current_precision + current_recall) > 0:
                        current_f1 = 2 * (current_precision * current_recall) / (current_precision + current_recall)

                    pbar.write(
                        f"--> [Update] Queries: {processed_count} | Global Recall@{k}: {current_recall:.2%} | Global Precision@{k}: {current_precision:.2%}"
                    )

                pbar.update(1)

    # --- CALCULATE EXTENDED STATISTICS ---

    # 1. Global Metrics (Micro Average)
    global_recall = total_retrieved_relevant_docs / total_relevant_docs if total_relevant_docs > 0 else 0
    global_precision = total_retrieved_relevant_docs / total_retrieved_docs if total_retrieved_docs > 0 else 0
    global_f1 = 0
    if (global_precision + global_recall) > 0:
        global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall)

    # 2. Split data into "Got Hits" (Matches > 0) and "No Hits" (Matches == 0)
    successful_queries = [res for res in query_results if res[1] > 0]
    failed_queries = [res for res in query_results if res[1] == 0]

    # Optional context: Queries where searcher didn't retrieve ANY documents at all
    zero_retrieval_queries = [res for res in query_results if res[2] == 0]

    # 3. Calculate subset averages
    avg_citations_failed = sum(res[0] for res in failed_queries) / len(failed_queries) if failed_queries else 0
    avg_citations_succ = sum(res[0] for res in successful_queries) / len(
        successful_queries) if successful_queries else 0

    # Macro averages for queries that got hits (Avg of their individual P & R)
    avg_prec_succ = sum(res[1] / res[2] for res in successful_queries) / len(
        successful_queries) if successful_queries else 0
    avg_rec_succ = sum(res[1] / res[0] for res in successful_queries) / len(
        successful_queries) if successful_queries else 0

    # Package all stats into a dictionary for clean handling
    stats = {
        "global_recall": global_recall,
        "global_precision": global_precision,
        "global_f1": global_f1,
        "total_queries": len(query_results),
        "failed_count": len(failed_queries),
        "avg_citations_failed": avg_citations_failed,
        "success_count": len(successful_queries),
        "avg_citations_succ": avg_citations_succ,
        "avg_prec_succ": avg_prec_succ,
        "avg_rec_succ": avg_rec_succ,
        "zero_retrieval_count": len(zero_retrieval_queries)
    }

    return stats


# --- MAIN EXECUTION BLOCK ---

raw_query_data = OriginalTextDatabase(DataLayerManager.original_text_DB).get_only_test_case_citations_pairs()
filtered_query_data = [(case_id, citations) for case_id, citations in raw_query_data if len(citations) > 0]
test_subset = filtered_query_data[:200]

paragraph_db = ParagraphDatabase(DataLayerManager.paragraphs_DB)
case_to_paragraph_count = dict(paragraph_db.get_case_to_paragraph_count())

searcher = ParagraphToParagraphSearcher(DataLayerManager.paragraphs_inverted_index_DB, case_to_paragraph_count)

K_VAL = 5
results = measure_metrics_at_k(searcher, test_subset, k=K_VAL, pool_size=30)

print(f"\n=== Overall Global Results @ {K_VAL} ===")
print(f"Global Recall:    **{results['global_recall']:.2%}**")
print(f"Global Precision: **{results['global_precision']:.2%}**")
print(f"Global F1 Score:  **{results['global_f1']:.2%}**")
print(f"Total Queries Evaluated: {results['total_queries']}")
print(f"Queries w/ Zero Retrieved Docs: {results['zero_retrieval_count']}")

print(f"\n--- Statistics for Failed Queries (0 Correct Hits) ---")
print(f"Count: {results['failed_count']} queries")
print(f"Avg Citation List Size: {results['avg_citations_failed']:.1f} citations")

print(f"\n--- Statistics for Successful Queries (≥1 Correct Hits) ---")
print(f"Count: {results['success_count']} queries")
print(f"Avg Citation List Size: {results['avg_citations_succ']:.1f} citations")
print(f"Avg Query Precision: **{results['avg_prec_succ']:.2%}**")
print(f"Avg Query Recall:    **{results['avg_rec_succ']:.2%}**")