import os
import pickle
from pathlib import Path

from flask import Flask, jsonify, request


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLASK_APP_DIR = Path(__file__).resolve().parent
MOCK_RESULTS_PATHS = [
    Path(os.getenv("MOCK_RESULTS_PATH", "")) if os.getenv("MOCK_RESULTS_PATH") else None,
    PROJECT_ROOT / "results.pkl",
    FLASK_APP_DIR / "results.pkl",
]

app = Flask(__name__)
_mock_cache = None


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.errorhandler(Exception)
def handle_error(error):
    if isinstance(error, ValueError):
        status = 400
    elif isinstance(error, KeyError):
        status = 404
    elif isinstance(error, FileNotFoundError):
        status = 503
    else:
        status = getattr(error, "code", 500)
    return jsonify({"error": str(error)}), status


def _body():
    return request.get_json(silent=True) or {}


def _int_value(data, key, default):
    try:
        return int(data.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer")


def _mock_path():
    for path in MOCK_RESULTS_PATHS:
        if path and path.exists():
            return path
    checked = [str(path) for path in MOCK_RESULTS_PATHS if path]
    raise FileNotFoundError(
        "results.pkl was not found. Generate it with cluster_local_manager.py "
        f"or set MOCK_RESULTS_PATH. Checked: {checked}"
    )


def _mock_results():
    global _mock_cache
    if _mock_cache is None:
        with _mock_path().open("rb") as file:
            _mock_cache = pickle.load(file)
    return _mock_cache


def _mock_value(key):
    results = _mock_results()
    if key not in results:
        raise KeyError(f"mock result '{key}' was not found")
    return results[key]


def _mock_text_search(function_name, text):
    return _mock_value(f"{function_name}({text})")


@app.get("/health")
def health():
    path = _mock_path()
    return jsonify({"status": "ok", "mock_results_path": str(path)})


@app.get("/mock/keys")
def get_mock_keys():
    return jsonify(sorted(_mock_results().keys()))


@app.post("/mock/reload")
def reload_mock_file():
    global _mock_cache
    _mock_cache = None
    return jsonify({"status": "reloaded", "mock_results_path": str(_mock_path())})


@app.get("/cases")
@app.get("/get_all_cases")
def get_all_cases():
    return jsonify(_mock_value("get_all_cases"))


@app.get("/cases/<case_id>")
@app.get("/get_case_text/<case_id>")
def get_case_text(case_id):
    return jsonify(_mock_value(f"get_case_text({case_id})"))


@app.get("/cases/<case_id>/paragraph-semantic")
@app.get("/search_paragraph_semantic_by_case/<case_id>")
def search_paragraph_semantic_by_case(case_id):
    return jsonify(_mock_value(f"search_paragraph_semantic_by_case({case_id})"))


@app.post("/paragraph-semantic/search")
@app.post("/search_paragraph_semantic_by_text")
def search_paragraph_semantic_by_text():
    data = _body()
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text is required")
    _int_value(data, "k", 5)
    _int_value(data, "pool_size", 30)
    return jsonify(_mock_text_search("search_paragraph_semantic_by_text", text))


@app.get("/measure/paragraphs-embeddings")
@app.get("/require_paragraphs_embeddings_search_measurer")
def require_paragraphs_embeddings_search_measurer():
    return jsonify(_mock_value("require_paragraphs_embeddings_search_measurer"))


@app.get("/measure/sentences-lexical")
@app.get("/require_sentences_lexical_search_measure")
def require_sentences_lexical_search_measure():
    return jsonify(_mock_value("require_sentences_lexical_search_measure"))


@app.get("/cases/<case_id>/sentences-lexical")
@app.get("/search_sentences_lexical_by_case/<case_id>")
def search_sentences_lexical_by_case(case_id):
    return jsonify(_mock_value(f"search_sentences_lexical_by_case({case_id})"))


@app.post("/sentences-lexical/search")
@app.post("/search_sentence_lexical_by_text")
def search_sentence_lexical_by_text():
    data = _body()
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text is required")
    _int_value(data, "k", 5)
    _int_value(data, "pool_size", 30)
    return jsonify(_mock_text_search("search_sentence_lexical_by_text", text))


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
