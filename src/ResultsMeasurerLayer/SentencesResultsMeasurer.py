import concurrent.futures

from tqdm import tqdm

from src.DatabaseLayer.DatabasesManagers.tmp_sqliteDB import SqliteDB
from src.SearchersLayer.SentenceToSentenceSearcher import SentenceToSentenceSearcher

def measure_metrics_at_k(searcher, test_cases, k, pool_size):
    """
    Measures Recall@K, Precision@K, and F1@K and prints intermediate results safely.
    """
    total_relevant_docs = 0
    total_retrieved_relevant_docs = 0
    total_retrieved_docs = 0  # Tracks how many docs the searcher actually returned
    processed_count = 0
    candidates_dict = {}


    local_sentence_db = SqliteDB("Databases/SqliteDB")
    all_sentences = local_sentence_db.get_all_records("sentences")
    query_sentences_dict = {}
    for case_id, sentence_id, text in all_sentences:
        query_sentences_dict[case_id] = query_sentences_dict.get(case_id, []) + [(case_id, sentence_id, text)]
    local_sentence_db.close()

    with tqdm(total=len(test_cases), desc="Testing", unit="query") as pbar :
        for case_id, ground_truth_citations in test_cases:
            query_sentences = query_sentences_dict[case_id]
            query_sentences = [
                sentence[2] if isinstance(sentence, tuple) and len(sentence) == 3 else sentence
                for sentence in query_sentences
            ]

            if not query_sentences:
                # 0 retrieved, 0 matches
                return len(ground_truth_citations), 0, 0

            # Run the sentence-level searcher
            print("reached1")
            top_results = searcher.search(query_sentences, top_k=k, pool_size_per_query=pool_size)
            print(top_results)

            retrieved_case_ids = {res[0] for res in top_results}
            matches = len(set(ground_truth_citations).intersection(retrieved_case_ids))

            total_relevant_docs += len(ground_truth_citations)
            total_retrieved_relevant_docs += matches
            total_retrieved_docs += len(ground_truth_citations)
            candidates_dict[case_id] = list(retrieved_case_ids)
            processed_count += 1

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

    return final_recall, final_precision, final_f1, candidates_dict


# --- MAIN EXECUTION BLOCK ---
def __to_array(citations_str) :
    citations_str = citations_str.replace("[", '').replace("]", '')
    citations_str = citations_str.replace(",", ' ').replace('"', ' ')
    citations_str = " ".join(citations_str.split())
    result = citations_str.split()
    return result

# --- MAIN EXECUTION BLOCK ---
def run_sentences_lexical_search_measurer() :
    raw_query_data = SqliteDB("Databases/SqliteDB").get_all_records("original_text")
    filtered_query_data = [(case_id, __to_array(citations)) for case_id, text, citations, test in raw_query_data if
                           len(__to_array(citations)) > 0]
    test_subset = filtered_query_data[:200]

    # 1. Fetch paragraphs safely
    local_paragraph_db = SqliteDB("Databases/SqliteDB")
    all_paragraphs = local_paragraph_db.get_all_records("sentences")
    local_paragraph_db.close()

    case_to_sentences = {}
    for case_id, sentence_id, text in all_paragraphs:
        case_to_sentences[case_id] = case_to_sentences.get(case_id, []) + [(case_id, sentence_id, text)]
    case_to_sentences_count = {}
    for k in case_to_sentences:
        case_to_sentences_count[k] = len(case_to_sentences[k])

    # Updated to use SentenceToSentenceSearcher
    searcher = SentenceToSentenceSearcher("Databases/TantivyIndex", case_to_sentences_count)

    # Execute
    K_VAL = 5
    # Note: pool_size=30 is applied here. As discussed earlier, keeping this lower is good for sentences.
    recall_score, precision_score, f1_score, candidates_dict = measure_metrics_at_k(searcher, test_subset, k=K_VAL, pool_size=30)

    print(f"\n=== Final Results @ {K_VAL} ===")
    print(f"Recall:    {recall_score:.2%}")
    print(f"Precision: {precision_score:.2%}")
    print(f"F1 Score:  {f1_score:.2%}")
    return recall_score, precision_score, f1_score, test_subset, candidates_dict
