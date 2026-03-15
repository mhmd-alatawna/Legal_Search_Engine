from tqdm import tqdm

from DataSetHandler import *
from Searchers import *

def measure_recall_k(searcher, query_dict, k) :
    total = 0
    success = 0
    max_query_to_run = len(query_dict)
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

# InvertedIndex.generate_index(load_gzipped_json("Data/filtered_data_small.json.gz"))
query_dict = load_gzipped_json("Data/query_dict.json.gz")
measure_recall_k(BM25_searcher(), query_dict,10)