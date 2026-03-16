from tqdm import tqdm

from DataSetHandler import *
from Searchers import *
import optuna
import logging

import concurrent.futures
from tqdm import tqdm


def multi_thread_measure_recall_k(searcher, query_dict, k):
    max_query_to_run = 1000

    # Safely slice the dictionary items up to our max limit
    queries_to_run = list(query_dict.values())[:max_query_to_run]
    print(f"Total number of queries to run: {len(queries_to_run)}")

    def run_single_query(item):
        query_text, real_doc_id = item
        top_results = searcher.search(query_text, top_k=k)
        if real_doc_id in top_results:
            return 1
        return 0

    success = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_single_query, item) for item in queries_to_run]

        # Use as_completed to update the progress bar the moment any thread finishes a query
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Testing Searcher",
                           unit="query"):
            success += future.result()

    return success / len(queries_to_run)

def measure_recall_k(searcher, query_dict, k) :
    total = 0
    success = 0
    max_query_to_run = 1000
    print(f"total number of queries : {len(query_dict)}")
    with tqdm(total=max_query_to_run, desc="Testing Searcher", unit="chunk") as pbar:
        for query_text, real_doc_id in query_dict.values():
            if total >= max_query_to_run :
                break
            total += 1
            top_results = searcher.search(query_text, top_k=k)
            if real_doc_id in top_results:
                success += 1
            pbar.update(1)
    print(f"recall@{k} : {round(success/total * 100,1)}%")
    return success/total

# InvertedIndexTantivy.generate_index(load_gzipped_json("Data/filtered_data_small.json.gz"))
query_dict = load_gzipped_json("Data/query_dict.json.gz")
print(multi_thread_measure_recall_k(BM25_searcher(), query_dict,10))