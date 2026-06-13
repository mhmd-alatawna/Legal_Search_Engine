import json
import os
import sqlite3
from pathlib import Path

from tqdm import tqdm


class OriginalTextDatabase:
    def __init__(self, db_name):
        Path(db_name).parent.mkdir(parents=True, exist_ok=True)
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    original_text TEXT,
                    citations TEXT,
                    is_test BOOLEAN
                )
            ''')
        # Performance trick: WAL mode allows faster concurrent writes/reads
        self.cursor.execute("PRAGMA journal_mode=WAL;")
        self.conn.commit()

    def add_many_records(self, records):
        """Batch insert records for maximum efficiency."""
        if not records:
            return
        query = "INSERT OR IGNORE INTO cases (case_id, original_text, citations, is_test) VALUES (?, ?, ?, ?)"
        self.cursor.executemany(query, records)
        self.conn.commit()

    def add_record(self, record):
        query = "INSERT OR IGNORE INTO cases (case_id, original_text, citations, is_test) VALUES (?, ?, ?, ?)"
        self.cursor.execute(query, record)
        self.conn.commit()

    def extract_records(self, cases_dir, is_test, citations_file, batch_size=100):
        """Parses files and populates the database using batching."""
        if os.path.exists(citations_file):
            with open(citations_file, 'r', encoding='utf-8') as f:
                citations_map = json.load(f)
        else:
            citations_map = {}
            print(f"Warning: {citations_file} not found.")

        path_list = list(Path(cases_dir).glob("*.txt"))
        counter = 0
        curr_batch = []

        print(f"Reading files from {cases_dir}...")

        with tqdm(total=len(path_list), desc="Indexing Cases", unit="file") as pbar:
            for file_path in path_list:
                case_id = file_path.stem
                filename = file_path.name
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()

                    raw_citations = citations_map.get(filename, [])
                    clean_citations = [c.replace(".txt", "") for c in raw_citations]

                    curr_batch.append((case_id, text, json.dumps(clean_citations), is_test))
                    if len(curr_batch) >= batch_size:
                        self.add_many_records(curr_batch)
                        curr_batch = []

                    counter += 1
                except Exception as e:
                    # Use pbar.write to avoid breaking the progress bar layout
                    pbar.write(f"Error processing {filename}: {e}")

                # Update progress bar by 1 for every file
                pbar.update(1)

        # --- Handle the remaining records ---
        if curr_batch:
            self.add_many_records(curr_batch)

        print(f"Successfully indexed {counter} cases into {self.db_name}.")

    def get_all_case_citations_pairs(self):
        self.cursor.execute("SELECT case_id, citations FROM cases")
        result = self.cursor.fetchall()
        return [(case_id, json.loads(citations)) for (case_id, citations) in result]

    def get_only_test_case_citations_pairs(self):
        self.cursor.execute("SELECT case_id, citations FROM cases WHERE is_test = TRUE")
        result = self.cursor.fetchall()
        return [(case_id, json.loads(citations)) for (case_id, citations) in result]

    def close(self):
        """Cleanly close the connection."""
        self.conn.close()