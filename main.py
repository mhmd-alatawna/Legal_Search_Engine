from DataSetHandler import *

def recreate_all() :
    reduced_training_set = load_gzipped_json("Data/filtered_data.json.gz")
    recreate_query_dict_and_reduced_doc_ids()
    reduced_doc_ids = set(load_gzipped_json("Data/reduced_doc_ids.json.gz"))

    print(len(reduced_training_set))
    print(len(reduced_doc_ids))
    tmp = {}
    for doc_id in reduced_training_set.keys():
        if doc_id in reduced_doc_ids:
            tmp[doc_id] = reduced_training_set[doc_id]
    print(len(tmp))
    flush_data_to_gzipped_json(tmp, "Data/filtered_data_small.json.gz")

def recreate_query_dict_and_reduced_doc_ids():
    reduced_doc_ids, query_dict = create_reduced_ids_and_query_dict()
    flush_data_to_gzipped_json(query_dict, "Data/query_dict.json.gz")
    flush_data_to_gzipped_json(list(reduced_doc_ids), "Data/reduced_doc_ids.json.gz")