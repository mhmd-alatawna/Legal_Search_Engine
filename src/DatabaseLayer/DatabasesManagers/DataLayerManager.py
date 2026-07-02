# ---- Database Initialization (Zero configuration leaked) ----
import json
import os
import shutil
import time
from pathlib import Path

from pyprojroot import here
from tqdm import tqdm

from configurations import *
from src.DatabaseLayer.DatabasesManagers.tmp_invertedindex import InvertedIndex
from src.DatabaseLayer.DatabasesManagers.tmp_qdrantDB import QdrantDB
from src.DatabaseLayer.DatabasesManagers.tmp_sqliteDB import SqliteDB
from src.DatabaseLayer.Embedders.TextEmbedder import EmbeddingGenerator
from src.DatabaseLayer.Processors.CopyProcessor import CopyProcessor
from src.DatabaseLayer.Processors.OriginalTextProcessor import OriginalTextProcessor
from src.DatabaseLayer.Processors.ParagraphProcessor import ParagraphProcessor
from src.DatabaseLayer.Processors.SectionsProcessor import SectionsProcessor
from src.DatabaseLayer.Processors.SentencesProcessor import SentencesProcessor

train_cases_dir = f"{root_dir}/RawFiles/task1_train_files_2026"
train_citations_file = f"{root_dir}/RawFiles/task1_train_labels_2026.json"
test_cases_dir = f"{root_dir}/RawFiles//task1_test_files_2026"
test_citations_file = f"{root_dir}/RawFiles//task1_test_labels_2026.json"


tmp_databases_dir = str(Path(tmp_dir) / "Databases")
real_databases_dir = str(Path(root_dir) / "Databases")

def safe_remove(path: Path) -> None:
    """
    Defensively removes a file or directory on Windows, handling directories,
    and utilizing a retry backoff to bypass transient IDE indexing locks.
    """
    if not path.exists():
        return

    if path.is_dir():
        shutil.rmtree(path)
        return

    # Retry loop to beat transient PyCharm indexing or file system locks
    for attempt in range(5):
        try:
            path.unlink()
            return
        except PermissionError:
            time.sleep(0.3)  # Brief pause to let external locks clear

    # Fallback explanation if a process is aggressively holding the file open
    raise PermissionError(
        f"[WinError 5] Access is denied to '{path}'. This file is locked. "
        "Please check your Windows Task Manager and terminate any dangling "
        "background Python processes from previous runs."
    )

def extract_records(cases_dir, is_test, citations_file):
    """
    Parses document files from a directory, attaches sanitized citation mappings,
    and returns a collected list of unified data tuples.
    """
    if os.path.exists(citations_file):
        with open(citations_file, 'r', encoding='utf-8') as f:
            citations_map = json.load(f)
    else:
        citations_map = {}
        print(f"Warning: {citations_file} not found.")

    path_list = list(Path(cases_dir).glob("*.txt"))
    records = []

    print(f"Reading files from {cases_dir}...")

    with tqdm(total=len(path_list), desc="Reading Cases", unit="file") as pbar:
        for file_path in path_list:
            case_id = file_path.stem
            filename = file_path.name
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                raw_citations = citations_map.get(filename, [])
                clean_citations = [c.replace(".txt", "") for c in raw_citations]

                # Store data record directly into the list memory space
                records.append((case_id, text, json.dumps(clean_citations), is_test))
            except Exception as e:
                pbar.write(f"Error processing {filename}: {e}")

            pbar.update(1)

    print(f"Successfully processed {len(records)} cases.")
    return records


def create_sqlite_database():
    scratch_path = Path(tmp_databases_dir) / "SqliteDB"
    real_path = Path(real_databases_dir) / "SqliteDB"

    safe_remove(scratch_path)

    original_tuples_list = extract_records(train_cases_dir, False, train_citations_file)
    original_tuples_list.extend(extract_records(test_cases_dir, True, test_citations_file))

    sqlite_db = SqliteDB(str(scratch_path))
    sqlite_db.index_documents(original_tuples_list, "original_text", OriginalTextProcessor())
    sqlite_db.index_documents(original_tuples_list, "paragraphs", ParagraphProcessor())
    sqlite_db.index_documents(original_tuples_list, "sentences", SentencesProcessor())
    sqlite_db.index_documents(original_tuples_list, "sections", SectionsProcessor())
    sqlite_db.close()

    real_path.mkdir(parents=True, exist_ok=True)
    safe_remove(real_path)
    shutil.move(str(scratch_path), str(real_path))


def create_inverted_index_database() :
    scratch_path = Path(tmp_databases_dir) / "TantivyIndex"
    real_path = Path(real_databases_dir) / "TantivyIndex"

    safe_remove(scratch_path)

    sqlite_db = SqliteDB(str(Path(real_databases_dir) / "SqliteDB"))
    inverted_index_db = InvertedIndex(str(scratch_path))

    CopyProcessor.schema = SqliteDB.get_schema("paragraphs")
    records = sqlite_db.get_all_records("paragraphs")
    inverted_index_db.index_documents(records, "paragraphs", CopyProcessor())

    CopyProcessor.schema = SqliteDB.get_schema("sentences")
    records = sqlite_db.get_all_records("sentences")
    inverted_index_db.index_documents(records, "sentences", CopyProcessor())

    inverted_index_db.close()
    sqlite_db.close()

    safe_remove(real_path)
    shutil.move(str(scratch_path), str(real_path))


def create_qdrant_database() :
    scratch_path = Path(tmp_databases_dir) / "QdrantStorage"
    real_path = Path(real_databases_dir) / "QdrantStorage"

    safe_remove(scratch_path)

    sqlite_db = SqliteDB(str(Path(real_databases_dir) / "SqliteDB"))
    qdrant_db = QdrantDB(str(scratch_path), qdrant_binaries_dir,)

    records = sqlite_db.get_all_records("paragraphs")
    qdrant_db.index_documents(records, "paragraphs", EmbeddingGenerator())
    records = sqlite_db.get_all_records("sentences")
    qdrant_db.index_documents(records, "sentences", EmbeddingGenerator())

    qdrant_db.close()
    sqlite_db.close()
    safe_remove(real_path)
    shutil.move(str(scratch_path), str(real_path))

if __name__ == "__main__":
    create_sqlite_database()
    create_inverted_index_database()
    create_qdrant_database()