from src.DatabaseLayer.DatabasesManagers.tmp_sqliteDB import SqliteDB
from src.DatabaseLayer.Embedders.TextEmbedder import EmbeddingGenerator
from src.DatabaseLayer.Processors.ParagraphProcessor import ParagraphProcessor
from src.DatabaseLayer.Processors.SentencesProcessor import SentencesProcessor
from src.ResultsMeasurerLayer import SentencesResultsMeasurer, ParagraphEmbeddingsResultsMeasurer
from src.ResultsMeasurerLayer.ParagraphEmbeddingsResultsMeasurer import run_paragraphs_embeddings_search_measurer
from src.ResultsMeasurerLayer.SentencesResultsMeasurer import run_sentences_lexical_search_measurer
from src.SearchersLayer.ParagraphToParagraphEmbeddingSearcher import ParagraphToParagraphEmbeddingSearcher
import os
from xmlrpc.server import SimpleXMLRPCServer

from src.SearchersLayer.SentenceToSentenceSearcher import SentenceToSentenceSearcher


def __to_array(citations_str) :
    citations_str = citations_str.replace("[", '').replace("]", '')
    citations_str = citations_str.replace(",", ' ').replace('"', ' ')
    citations_str = " ".join(citations_str.split())
    result = citations_str.split()
    return result

def get_all_cases() :
    raw_query_data = SqliteDB("Databases/SqliteDB").get_all_records("original_text")
    filtered_query_data = [(case_id, __to_array(citations)) for case_id, text, citations, test in raw_query_data]
    print(f"returned {len(filtered_query_data)} cases")
    return filtered_query_data

def get_case_text(to_search_case_id) :
    raw_query_data = SqliteDB("Databases/SqliteDB").get_all_records("original_text")
    full_case_details = [(case_id, text ,__to_array(citations) , test) for case_id, text, citations, test in raw_query_data if
                           to_search_case_id == case_id][0]
    return full_case_details


def require_paragraphs_embeddings_search_measurer():
    recall_score, precision_score, f1_score, test_subset, candidates_dict = run_paragraphs_embeddings_search_measurer()
    return {
        "recall": recall_score,
        "precision": precision_score,
        "f1": f1_score,
        "test_subset": test_subset,
        "candidates_dict": candidates_dict,
    }

def search_paragraph_semantic_by_case(to_search_case_id) :
    raw_query_data = SqliteDB("Databases/SqliteDB").get_all_records("original_text")
    case_tuple = [(case_id, __to_array(citations)) for case_id, text, citations, test in raw_query_data if
                           to_search_case_id == case_id][0]

    # 1. Fetch paragraphs safely
    local_paragraph_db = SqliteDB("Databases/SqliteDB")
    all_paragraphs = local_paragraph_db.get_all_records("paragraphs")
    local_paragraph_db.close()

    case_to_paragraphs = {}
    case_to_paragraphs_count = {}
    for case_id, paragraph_id, text in all_paragraphs:
        if case_id == to_search_case_id :
            case_to_paragraphs[case_id] = case_to_paragraphs.get(case_id, []) + [(case_id, paragraph_id, text)]
        case_to_paragraphs_count[case_id] = case_to_paragraphs_count.get(case_id, 0) + 1

    # Initialize the local embedder (will load from disk if present)
    print("Loading embedding model...")
    embedder = EmbeddingGenerator()
    print("Done")

    searcher = ParagraphToParagraphEmbeddingSearcher(
        db_path="Databases/QdrantStorage",
        doc_to_segment_count=case_to_paragraphs_count,
        semantic_thresh=0.65,
        macro_B=0.75
    )
    print("Done")

    # Use dynamic K variable for clean execution and printing
    K_VAL = 5
    recall_score, precision_score, f1_score, candidates_dict = ParagraphEmbeddingsResultsMeasurer.measure_metrics_at_k(searcher, [case_tuple],
                                                                                    case_to_paragraphs, k=K_VAL,
                                                                                    pool_size=30)

    # TODO : return the data for the result , currently it returns 2 lists , one for the candidates and one for ground truth
    return candidates_dict[to_search_case_id] , case_tuple[1]

def search_paragraph_semantic_by_text(to_search_text, k=5, pool_size=30) :
    processor = ParagraphProcessor()
    query_paragraphs = processor.extract(("0000000",to_search_text))

    # 1. Fetch paragraphs safely
    local_paragraph_db = SqliteDB("Databases/SqliteDB")
    all_paragraphs = local_paragraph_db.get_all_records("paragraphs")
    local_paragraph_db.close()

    case_to_paragraphs_count = {}
    for case_id, paragraph_id, text in all_paragraphs:
        case_to_paragraphs_count[case_id] = case_to_paragraphs_count.get(case_id, 0) + 1


    searcher = ParagraphToParagraphEmbeddingSearcher(
        db_path="Databases/QdrantStorage",
        doc_to_segment_count=case_to_paragraphs_count,
        semantic_thresh=0.65,
        macro_B=0.75
    )
    top_results = searcher.search(
        query_paragraphs=query_paragraphs,
        top_k=k,
        pool_size_per_query=pool_size
    )

    # 3. Calculate metrics for this case
    retrieved_case_ids = {str(res[0]) for res in top_results}
    return list(retrieved_case_ids)




def require_sentences_lexical_search_measure() :
    recall_score, precision_score, f1_score, test_subset, candidates_dict = run_sentences_lexical_search_measurer()
    return {
        "recall": recall_score,
        "precision": precision_score,
        "f1": f1_score,
        "test_subset": test_subset,
        "candidates_dict": candidates_dict,
    }

def search_sentences_lexical_by_case(to_search_case_id) :
    raw_query_data = SqliteDB("Databases/SqliteDB").get_all_records("original_text")
    case_tuple = [(case_id, __to_array(citations)) for case_id, text, citations, test in raw_query_data if
                           to_search_case_id == case_id][0]

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
    recall_score, precision_score, f1_score, candidates_dict = SentencesResultsMeasurer.measure_metrics_at_k(searcher, [case_tuple], k=K_VAL,
                                                                                    pool_size=30)

    # TODO : return the data for the result , currently it returns 2 lists , one for the candidates and one for ground truth
    return candidates_dict[to_search_case_id] , case_tuple[1]

def search_sentence_lexical_by_text(to_search_text, k=5, pool_size=30) :
    processor = SentencesProcessor()
    query_sentences = processor.extract(("0000000",to_search_text))
    query_sentences = [
        sentence[2] if isinstance(sentence, tuple) and len(sentence) == 3 else sentence
        for sentence in query_sentences
    ]

    # 1. Fetch paragraphs safely
    local_paragraph_db = SqliteDB("Databases/SqliteDB")
    all_sentences = local_paragraph_db.get_all_records("sentences")
    local_paragraph_db.close()

    case_to_sentences_count = {}
    for case_id, sentence_id, text in all_sentences:
        case_to_sentences_count[case_id] = case_to_sentences_count.get(case_id, 0) + 1


    searcher = SentenceToSentenceSearcher(
        "Databases/TantivyIndex",
        case_to_sentences_count,
        thresh=0.65,
        macro_B=0.75
    )
    top_results = searcher.search(
        query_sentences=query_sentences,
        top_k=k,
        pool_size_per_query=pool_size
    )

    # 3. Calculate metrics for this case
    retrieved_case_ids = {str(res[0]) for res in top_results}
    return list(retrieved_case_ids)


def serve():
    host = os.getenv("CLUSTER_REMOTE_HOST", "0.0.0.0")
    port = int(os.getenv("CLUSTER_REMOTE_PORT", "9000"))
    server = SimpleXMLRPCServer((host, port), allow_none=True, logRequests=True)

    server.register_function(get_all_cases)
    server.register_function(get_case_text)

    server.register_function(require_paragraphs_embeddings_search_measurer)
    server.register_function(search_paragraph_semantic_by_case)
    server.register_function(search_paragraph_semantic_by_text)

    server.register_function(require_sentences_lexical_search_measure)
    server.register_function(search_sentences_lexical_by_case)
    server.register_function(search_sentence_lexical_by_text)
    print(f"cluster_remote_manager listening on {host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    serve()
