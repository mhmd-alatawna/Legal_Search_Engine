import json
from datasets import load_dataset
import gzip


def create_reduced_ids_and_query_dict() :
    # download ground truth dataset , mapping q_id -> doc_id
    print("Downloading ground truth dataset...")
    truth_dataset = load_dataset("AbeHou/CLERC", data_files={"data": f"qrels/qrels.train.doc-level.tsv"})["data"]
    doc_id_column_key_truth = list(truth_dataset[0].keys())[2]
    query_id_column_key_truth = list(truth_dataset[0].keys())[0]

    # extract all doc_ids from the truth dataset , this results in reduction of documents set
    reduced_doc_ids = set()
    for i in range(len(truth_dataset)):
        reduced_doc_ids.add(f"{(truth_dataset[i][doc_id_column_key_truth])}")
    reduced_doc_ids = set(list(reduced_doc_ids)[:5000])

    # extract the query_id -> doc_id mapping
    query_to_doc = {}
    for i in range(len(truth_dataset)):
        query_id = truth_dataset[i][query_id_column_key_truth]
        doc_id = f"{truth_dataset[i][doc_id_column_key_truth]}"
        if doc_id in reduced_doc_ids:
            query_to_doc[query_id] = doc_id



    print("Downloading query dataset...")
    queries_dataset = load_dataset("AbeHou/CLERC", data_files={"data": f"queries/queries.train.tsv"})["data"]

    text_column_key_query = list(queries_dataset[0].keys())[1]
    id_column_key_query = list(queries_dataset[0].keys())[0]

    query_dict = {}
    for i in range(len(queries_dataset)):
        query_id = queries_dataset[i][id_column_key_query]
        query_text = queries_dataset[i][text_column_key_query]
        if query_id in query_to_doc.keys():
            query_dict[query_id] = (query_text,query_to_doc[query_id])

    return reduced_doc_ids,query_dict

def create_reduced_training_set(dataset,reduced_doc_ids):
    def stream_corpus(dataset):
        for i in range(len(dataset)):
            yield dataset[i]

    text_column_key = list(dataset[0].keys())[1]
    id_column_key = list(dataset[0].keys())[0]

    tmp = set()
    x = 0
    for row in stream_corpus(dataset):
        ele = row[text_column_key]
        if ele == "" or not isinstance(ele, str) or ele is None or not ele.strip() or len(
                ele) < 10 or ele in tmp:
            continue
        elif row[id_column_key] in reduced_doc_ids:
            tmp.add((row[id_column_key], ele))
        x += 1
        if x % 50000 == 0:
            print(x)

    documents_dict = {}
    for doc_id, doc_text in tmp:
        documents_dict[doc_id] = doc_text
    return documents_dict

def download_data_and_reduce(reduced_doc_ids) :
    # download documents dataset
    dataset = load_dataset("AbeHou/CLERC", data_files={"data": f"collection/collection.doc.tsv.gz"})["data"]

    # create reduced document set , mapping doc_id -> doc_text
    documents_dict = create_reduced_training_set(dataset,reduced_doc_ids)
    return documents_dict

def flush_data_to_gzipped_json(data, file_path):
    with gzip.open(file_path, 'wt', encoding='utf-8') as gz_file:
        json.dump(data, gz_file, ensure_ascii=False)
    print(f"Successfully compressed and flushed data to {file_path}")

def load_gzipped_json(file_path):
    """Reads a compressed .json.gz file back into a Python dictionary."""
    with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
        return json.load(gz_file)