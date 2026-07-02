import json
import os
import pickle
import sqlite3
from pathlib import Path

# TODO : the current schemas for original_text table do not handle the citations as a list
#  changing the code blindly also is not an option because this affects many places !
class SqliteDB:
    _instance = None
    metadata_directory = {}
    metadata_directory_file = "metadata_directory.pkl"
    database_file = "sqlite.db"

    # SINGLETON PATTERN: Ensures only one instance exists.
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SqliteDB, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).mkdir(parents=True, exist_ok=True)
        Path(f"{db_path}/{SqliteDB.database_file}").parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(f"{db_path}/{SqliteDB.database_file}", check_same_thread=False)
        self.cursor = self.conn.cursor()

        # Core database speed tunings from your original implementation
        self.cursor.execute("PRAGMA journal_mode=WAL;")
        self.cursor.execute("PRAGMA synchronous=NORMAL;")
        self.conn.commit()

        self.__load_metadata_directory()
        if len(SqliteDB.metadata_directory) > 0:
            for collection_name, schema in SqliteDB.metadata_directory.items():
                print(f"Loaded {collection_name} collections from metadata directory.")
        else :
            print("created a new empty database")
        self.initialized = True

    def index_documents(self, documents, collection_name, processor):
        # a tuple of tuples of size 2 , tuple(tuple(field_name1,field_type1),...)
        fields = processor.schema.fields
        # a tuple of field_names , tuple(field_name1,...)
        primary_keys = processor.schema.primary_keys
        columns = ", ".join([f"{field} {field_type}" for field,field_type in fields])
        primary_keys = ", ".join(primary_keys)

        self.cursor.execute(f"PRAGMA table_info({collection_name})")
        existing_table = self.cursor.fetchall()
        if existing_table:
            # PRAGMA returns (cid, name, type, notnull, dflt_value, pk). We compare (name, type).
            existing_schema = {(row[1].lower(), row[2].upper()) for row in existing_table}
            target_schema = {(f[0].lower(), f[1].upper()) for f in fields}

            if existing_schema != target_schema:
                print(f"Warning: Table '{collection_name}' exists with a different schema. Dropping original table.")
                self.cursor.execute(f"DROP TABLE {collection_name}")
            else:
                print(f"Using already existing schema for table '{collection_name}'.")

        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {collection_name} ({columns}, PRIMARY KEY ({primary_keys}))")
        self.conn.commit()
        self.__update_metadata_directory(collection_name, processor.schema)

        final_records = processor.extract_many(documents)
        # Batch Insert optimized writes
        if final_records:
            placeholders = ", ".join(["?"] * len(fields))
            query = f"INSERT OR IGNORE INTO {collection_name} VALUES ({placeholders})"

            # Flush in batches
            batch_size = 2000
            for i in range(0, len(final_records), batch_size):
                self.cursor.executemany(query, final_records[i:i + batch_size])
                self.conn.commit()
            print(f"Successfully indexed {len(final_records)} records into table '{collection_name}'.")

    def __load_metadata_directory(self):
        file_name = f"{self.db_path}/{SqliteDB.metadata_directory_file}"
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                SqliteDB.metadata_directory = pickle.load(f)
        else :
            SqliteDB.metadata_directory = {}

    def __update_metadata_directory(self, collection_name, collection_definition):
        file_name = f"{self.db_path}/{SqliteDB.metadata_directory_file}"
        SqliteDB.metadata_directory[collection_name] = collection_definition
        with open(file_name, "wb") as f:
            pickle.dump(SqliteDB.metadata_directory, f)



    def search(self, query: str, top_k: int, collection_name: str) -> list:
        # Relational metadata fallbacks or direct filtering
        self.cursor.execute(f"SELECT * FROM {collection_name} LIMIT ?", (top_k,))
        return self.cursor.fetchall()

    def get_all_records(self, collection_name):
        self.cursor.execute(f"SELECT * FROM {collection_name}")
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()

    @staticmethod
    def get_schema(collection_name):
        return SqliteDB.metadata_directory.get(collection_name, None)