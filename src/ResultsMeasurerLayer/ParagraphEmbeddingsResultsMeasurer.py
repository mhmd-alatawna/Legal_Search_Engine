import concurrent.futures
import os
import shutil
import traceback
from pathlib import Path

import torch
from tqdm import tqdm

from src.DatabaseLayer.DatabasesManagers.tmp_sqliteDB import SqliteDB
from src.DatabaseLayer.Embedders.TextEmbedder import EmbeddingGenerator
from src.SearchersLayer.ParagraphToParagraphEmbeddingSearcher import ParagraphToParagraphEmbeddingSearcher


def measure_metrics_at_k(searcher, test_cases,case_to_paragraphs, k, pool_size):
    """
    Measures Recall@K, Precision@K, and F1@K for the Semantic Searcher.
    Uses a standard loop to prevent PyTorch/Thread deadlocks.
    """
    total_relevant_docs = 0
    total_retrieved_relevant_docs = 0
    total_retrieved_docs = 0
    processed_count = 0
    failed_count = 0
    candidates_dict = {}


    with tqdm(total=len(test_cases), desc="Testing Semantic Search", unit="query") as pbar:
        for item in test_cases:
            case_id, ground_truth_citations = item

            try:

                raw_paragraphs = case_to_paragraphs.get(case_id, [])
                if not raw_paragraphs:
                    pbar.update(1)
                    continue

                query_paragraphs = [
                    p[2] if isinstance(p, tuple) and len(p) == 3 else p
                    for p in raw_paragraphs
                ]

                # 2. Execute Search
                top_results = searcher.search(
                    query_paragraphs=query_paragraphs,
                    top_k=k,
                    pool_size_per_query=pool_size
                )

                # 3. Calculate metrics for this case
                retrieved_case_ids = {str(res[0]) for res in top_results}
                ground_truth_citations = {str(x) for x in ground_truth_citations}
                matches = len(ground_truth_citations.intersection(retrieved_case_ids))
                candidates_dict[case_id] = list(retrieved_case_ids)

                total_relevant_docs += len(ground_truth_citations)
                total_retrieved_relevant_docs += matches
                total_retrieved_docs += len(retrieved_case_ids)
                processed_count += 1

                # Optional: Free GPU memory actively to prevent OOM loops
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                failed_count += 1
                pbar.write(f"\n[!] ERROR processing case {case_id}: {str(e)}")
                # Uncomment the next line if you want to see the full error traceback:
                traceback.print_exc()

            finally:
                pbar.update(1)

                # --- INTERMEDIATE PRINTING ---
                if processed_count > 0 and processed_count % 10 == 0:
                    current_recall = total_retrieved_relevant_docs / total_relevant_docs if total_relevant_docs > 0 else 0
                    current_precision = total_retrieved_relevant_docs / total_retrieved_docs if total_retrieved_docs > 0 else 0
                    current_f1 = 0
                    if (current_precision + current_recall) > 0:
                        current_f1 = 2 * (current_precision * current_recall) / (current_precision + current_recall)

                    pbar.write(
                        f"--> [Update] Queries: {processed_count} | Fails: {failed_count} | Recall@{k}: {current_recall:.2%} | Precision@{k}: {current_precision:.2%} | F1@{k}: {current_f1:.2%}"
                    )

    # Final Calculation
    final_recall = total_retrieved_relevant_docs / total_relevant_docs if total_relevant_docs > 0 else 0
    final_precision = total_retrieved_relevant_docs / total_retrieved_docs if total_retrieved_docs > 0 else 0
    final_f1 = 0
    if (final_precision + final_recall) > 0:
        final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall)

    if failed_count > 0:
        print(f"\nWARNING: {failed_count} cases failed to process. Check the logs above.")

    return final_recall, final_precision, final_f1, candidates_dict

# --- MAIN EXECUTION BLOCK ---
def __to_array(citations_str) :
    citations_str = citations_str.replace("[", '').replace("]", '')
    citations_str = citations_str.replace(",", ' ').replace('"', ' ')
    citations_str = " ".join(citations_str.split())
    result = citations_str.split()
    return result


def run_paragraphs_embeddings_search_measurer():
    raw_query_data = SqliteDB("Databases/SqliteDB").get_all_records("original_text")
    filtered_query_data = [(case_id, __to_array(citations)) for case_id, text, citations, test in raw_query_data if len(__to_array(citations)) > 0]
    test_subset = filtered_query_data[:200]

    # 1. Fetch paragraphs safely
    local_paragraph_db = SqliteDB("Databases/SqliteDB")
    all_paragraphs = local_paragraph_db.get_all_records("paragraphs")
    local_paragraph_db.close()

    case_to_paragraphs = {}
    for case_id, paragraph_id, text in all_paragraphs:
        case_to_paragraphs[case_id] = case_to_paragraphs.get(case_id, []) + [(case_id, paragraph_id, text)]
    case_to_paragraphs_count = {}
    for k in case_to_paragraphs:
        case_to_paragraphs_count[k] = len(case_to_paragraphs[k])

    # Initialize the local embedder (will load from disk if present)
    print("Loading embedding model...")
    embedder = EmbeddingGenerator()
    print("Done")

    # # Initialize the semantic searcher with Qdrant path
    # # Make sure you have added your Qdrant DB path to your DataLayerManager
    # print("Initializing Searcher...")
    # # TODO : this copies from local to tmp dir which is very costly in a normal PC , good for cluster
    # tmp_dir = DataLayerManager.paragraphs_tmp_embeddings_DB
    # if not os.path.exists(DataLayerManager.paragraphs_tmp_embeddings_DB):
    #     Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    #
    # shutil.copytree(DataLayerManager.paragraphs_embeddings_DB, tmp_dir, dirs_exist_ok=True)
    # print("copied database into working directory (tmp dir)")

    searcher = ParagraphToParagraphEmbeddingSearcher(
        db_path="Databases/QdrantStorage",
        doc_to_segment_count=case_to_paragraphs_count,
        semantic_thresh=0.65,
        macro_B=0.75
    )
    print("Done")

    # Use dynamic K variable for clean execution and printing
    K_VAL = 5
    recall_score, precision_score, f1_score, candidates_dict = measure_metrics_at_k(searcher, test_subset, case_to_paragraphs, k=K_VAL, pool_size=30)

    # if os.path.exists(paragraphs_tmp_embeddings_DB):
    #     shutil.rmtree(paragraphs_tmp_embeddings_DB)

    print(f"\n=== Final Semantic Results @ {K_VAL} ===")
    print(f"Recall:    {recall_score:.2%}")
    print(f"Precision: {precision_score:.2%}")
    print(f"F1 Score:  {f1_score:.2%}")
    return recall_score, precision_score, f1_score, test_subset, candidates_dict