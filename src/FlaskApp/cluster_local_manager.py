import json
import os
import pickle
from pathlib import Path
from xmlrpc.client import Fault, dumps, loads

import paramiko
from paramiko.ssh_exception import ChannelException


CLUSTER_HOST = os.getenv("CLUSTER_HOST", "slurm.bgu.ac.il")
CLUSTER_USERNAME = os.getenv("CLUSTER_USERNAME", "amohamma")
CLUSTER_PASSWORD_FILE = os.getenv(
    "CLUSTER_PASSWORD_FILE",
    r"C:\Users\Owner\OneDrive\Desktop\ClusterSSHPassword.txt",
)
# CLUSTER_REMOTE_HOST = os.getenv("CLUSTER_REMOTE_HOST", "127.0.0.1")
CLUSTER_REMOTE_HOST = "ise-4090-13"
CLUSTER_REMOTE_PORT = int(os.getenv("CLUSTER_REMOTE_PORT", "9000"))

def _cluster_password():
    return Path(CLUSTER_PASSWORD_FILE).read_text(encoding="utf-8").strip()


def _open_ssh_client():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        CLUSTER_HOST,
        username=CLUSTER_USERNAME,
        password=_cluster_password(),
    )
    return client


def _remote_call(function_name, *args):
    client = _open_ssh_client()
    channel = None
    try:
        remote_port = int(CLUSTER_REMOTE_PORT)
        body = dumps(args, methodname=function_name, allow_none=True).encode("utf-8")
        request = (
            "POST /RPC2 HTTP/1.1\r\n"
            f"Host: {CLUSTER_REMOTE_HOST}:{remote_port}\r\n"
            "Content-Type: text/xml\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("ascii") + body

        channel = client.get_transport().open_channel(
            "direct-tcpip",
            (CLUSTER_REMOTE_HOST, remote_port),
            ("127.0.0.1", 0),
        )
        channel.sendall(request)

        chunks = []
        while True:
            chunk = channel.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)

        response = b"".join(chunks)
        _, separator, response_body = response.partition(b"\r\n\r\n")
        if not separator:
            raise RuntimeError("Invalid response from cluster_remote_manager")

        values, _ = loads(response_body)
        return values[0] if values else None
    except ChannelException as exc:
        raise RuntimeError(
            f"Could not connect from {CLUSTER_HOST} to "
            f"{CLUSTER_REMOTE_HOST}:{remote_port}. "
            "Make sure cluster_remote_manager.py is running there. "
            "If it runs inside a Slurm job, set CLUSTER_REMOTE_HOST to the "
            "compute node from SLURM_JOB_NODELIST/job.out, not 127.0.0.1."
        ) from exc
    except Fault as exc:
        raise RuntimeError(f"Remote function {function_name} failed: {exc}") from exc
    finally:
        if channel is not None:
            channel.close()
        client.close()


def _print_response(title, response):
    print(f"\n=== {title} ===")
    print(json.dumps(response, indent=2, ensure_ascii=False))




def get_all_cases() -> list[tuple[str, list[str]]]:
    """
    :return:
    it returns all the cases in the database as pairs of (case_id, citations) where :
    case_id -> str
    citations -> list[str]
    """
    response = _remote_call("get_all_cases")
    _print_response("all cases", response)
    return response


def get_case_text(to_search_case_id):
    """
    :param to_search_case_id:
    :return:
    all details in the database of the case with the given id (all normal data, no embeddings etc ...)
    (case_id, text, citations, test)
    case_id -> str
    text -> str
    citations -> list[str]
    test -> str (either "0" or "1")
    """
    response = _remote_call("get_case_text", to_search_case_id)
    _print_response(f"case {to_search_case_id}", response)
    return response



def require_paragraphs_embeddings_search_measurer():
    """
    :return:
    performs a whole test run of 200 queries on the paragraph to paragraph embeddings searcher
    and returns the following data as a dictionary:
        "recall": recall_score,
        "precision": precision_score,
        "f1": f1_score,
        "test_subset": test_subset,
        "candidates_dict": candidates_dict


    test_subset is a list of tuples of (case_id, citations)
    candidates_dict is a dictionary of {case_id: [list of candidates as str]}
    """
    response = _remote_call("require_paragraphs_embeddings_search_measurer")
    _print_response("paragraphs embeddings search measurer", response)
    return response

def search_paragraph_semantic_by_case(to_search_case_id):
    """
    :param to_search_case_id:
    :return:
    candidates - list[str]
    ground_truth - list[str]
    """
    response = _remote_call("search_paragraph_semantic_by_case", to_search_case_id)
    _print_response(f"semantic paragraph search for {to_search_case_id}", response)
    return response

def search_paragraph_semantic_by_text(to_search_text, k=5, pool_size=30) :
    """
    :param to_search_text:
    :param k:
    :param pool_size:
    :return:
    candidates - list[str]
    """
    response = _remote_call("search_paragraph_semantic_by_text", to_search_text, k, pool_size)
    _print_response(f"semantic paragraph search for the query", response)
    return response



def require_sentences_lexical_search_measure():
    """
    :return:
    performs a whole test run of 200 queries on the sentence to sentence searcher
    and returns the following data as a dictionary:
        "recall": recall_score,
        "precision": precision_score,
        "f1": f1_score,
        "test_subset": test_subset,
        "candidates_dict": candidates_dict


    test_subset is a list of tuples of (case_id, citations)
    candidates_dict is a dictionary of {case_id: [list of candidates as str]}
    """
    response = _remote_call("require_sentences_lexical_search_measure")
    _print_response("sentences search measurer", response)
    return response

def search_sentences_lexical_by_case(to_search_case_id):
    """
    :param to_search_case_id:
    :return:
    candidates - list[str]
    ground_truth - list[str]
    """
    response = _remote_call("search_sentences_lexical_by_case", to_search_case_id)
    _print_response(f"lexical sentences search for {to_search_case_id}", response)
    return response

def search_sentence_lexical_by_text(to_search_text, k=5, pool_size=30) :
    """
    :param to_search_text:
    :param k:
    :param pool_size:
    :return:
    candidates - list[str]
    """
    response = _remote_call("search_sentence_lexical_by_text", to_search_text, k, pool_size)
    _print_response(f"semantic paragraph search for the query", response)
    return response


if __name__ == "__main__":
    results = {}
    all_cases = get_all_cases()
    results["get_all_cases"] = all_cases

    case_id1 = all_cases[0][0]
    my_text1 = get_case_text(case_id1)[1]

    case_id2 = all_cases[1][0]
    my_text2 = get_case_text(case_id2)[1]

    results[f"get_case_text({case_id1})"] = get_case_text(case_id1)
    results[f"get_case_text({case_id2})"] = get_case_text(case_id2)

    results["require_paragraphs_embeddings_search_measurer"] = require_paragraphs_embeddings_search_measurer()
    results[f"search_paragraph_semantic_by_case({case_id1})"] = search_paragraph_semantic_by_case(case_id1)
    results[f"search_paragraph_semantic_by_text({my_text1})"] = search_paragraph_semantic_by_text(my_text1)

    results[f"search_paragraph_semantic_by_case({case_id2})"] = search_paragraph_semantic_by_case(case_id2)
    results[f"search_paragraph_semantic_by_text({my_text2})"] = search_paragraph_semantic_by_text(my_text2)

    # results["require_sentences_lexical_search_measure"] = require_sentences_lexical_search_measure()
    results[f"search_sentences_lexical_by_case({case_id1})"] = search_sentences_lexical_by_case(case_id1)
    results[f"search_sentence_lexical_by_text({my_text1})"] = search_sentence_lexical_by_text(my_text1)

    results[f"search_sentences_lexical_by_case({case_id2})"] = search_sentences_lexical_by_case(case_id2)
    results[f"search_sentence_lexical_by_text({my_text2})"] = search_sentence_lexical_by_text(my_text2)

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

