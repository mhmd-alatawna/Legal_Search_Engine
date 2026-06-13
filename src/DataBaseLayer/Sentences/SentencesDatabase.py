import concurrent.futures
import sqlite3
from pathlib import Path

from tqdm import tqdm

from src.DataBaseLayer.LegalDocumentProcessor import LegalDocumentProcessor

worker_processor = None


def init_worker():
    """
    Runs once per CPU core when the process pool starts.
    Initializes the LegalDocumentProcessor so it isn't reloaded per file.
    """
    global worker_processor
    worker_processor = LegalDocumentProcessor()


def process_single_file(file_path):
    """
    Reads and processes a single file using the core's local processor.
    Returns a tuple of (Success_Boolean, Filename, Data_or_Error).
    """
    case_id = file_path.stem
    filename = file_path.name
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Switch to extract_sentences
        sentence_dict = worker_processor.extract_sentences(text)

        # Flatten the dictionary into a list of tuples for SQLite
        records = []
        for sentence_id, sentence_text in sentence_dict.items():
            records.append((case_id, str(sentence_id), sentence_text))

        return True, filename, records
    except Exception as e:
        return False, filename, str(e)


class SentenceDatabase:
    def __init__(self, db_name):
        Path(db_name).parent.mkdir(parents=True, exist_ok=True)
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

        # Updated table name and primary keys to reflect sentences
        self.cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sentences (
                        case_id TEXT,
                        sentence_id TEXT,
                        text TEXT,
                        PRIMARY KEY (case_id, sentence_id)
                    )
                ''')
        # Performance trick: WAL mode allows faster concurrent writes/reads
        self.cursor.execute("PRAGMA journal_mode=WAL;")
        # Extra safety for bulk inserts
        self.cursor.execute("PRAGMA synchronous=NORMAL;")
        self.conn.commit()

    def __add_many_records(self, records):
        """Batch insert records for maximum efficiency."""
        if not records:
            return
        query = "INSERT OR IGNORE INTO sentences (case_id, sentence_id, text) VALUES (?, ?, ?)"
        self.cursor.executemany(query, records)
        self.conn.commit()

    def extract_records(self, cases_dir, batch_size=3000):
        """Parses files across multiple CPU cores and populates the database."""
        path_list = list(Path(cases_dir).glob("*.txt"))
        counter = 0
        curr_batch = []

        print(f"Reading files from {cases_dir} using Multiprocessing...")

        # Create a pool of worker processes.
        with concurrent.futures.ProcessPoolExecutor(initializer=init_worker) as executor:

            # Map the worker function to the files, wrapping in tqdm for a progress bar
            results = list(
                tqdm(executor.map(process_single_file, path_list), total=len(path_list), desc="Indexing Cases",
                     unit="file"))

            for success, filename, result_data in results:
                if success:
                    curr_batch.extend(result_data)
                    counter += 1

                    # Write to the DB from the main thread only
                    if len(curr_batch) >= batch_size:
                        self.__add_many_records(curr_batch)
                        curr_batch = []
                else:
                    print(f"\nError processing {filename}: {result_data}")

        # --- Handle the remaining records ---
        if curr_batch:
            self.__add_many_records(curr_batch)

        print(f"Successfully indexed {counter} cases into {self.db_name}.")

    def close(self):
        """Cleanly close the connection."""
        self.conn.close()

    def get_all_records(self):
        self.cursor.execute("SELECT * FROM sentences")
        return self.cursor.fetchall()

    def get_all_sentence_ids(self):
        self.cursor.execute("SELECT case_id, sentence_id FROM sentences")
        return self.cursor.fetchall()

    def get_sentences_by_case(self, case_id):
        """Fetches all sentences for a specific case, ordered by ID."""
        self.cursor.execute(
            "SELECT text FROM sentences WHERE case_id = ? ORDER BY CAST(sentence_id AS INTEGER) ASC",
            (case_id,)
        )
        return [row[0] for row in self.cursor.fetchall()]

    def get_case_to_sentence_count(self):
        """Much faster than loading all records: uses SQL aggregation."""
        self.cursor.execute("SELECT case_id, COUNT(*) FROM sentences GROUP BY case_id")
        return self.cursor.fetchall()