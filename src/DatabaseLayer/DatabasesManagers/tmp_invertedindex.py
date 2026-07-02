import os
import pickle
import re
import shutil
from pathlib import Path

import tantivy
from tantivy import SchemaBuilder, Filter

from src.DatabaseLayer.Processors.Schema import Schema


class InvertedIndex:
    _instance = None
    metadata_directory = {}
    metadata_directory_file = "metadata_directory.pkl"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(InvertedIndex, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, db_path: str):
        if self.initialized:
            return

        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.__load_metadata_directory()

        if InvertedIndex.metadata_directory:
            for collection_name in InvertedIndex.metadata_directory:
                print(f"Loaded {collection_name} collection from metadata directory.")
        else:
            print("created a new empty inverted index database")

        self.initialized = True

    def index_documents(self, documents, collection_name, processor):
        schema = processor.schema
        collection_path = self.db_path / collection_name
        tantivy_schema = self.__build_tantivy_schema(schema)
        if collection_path.exists():
            print("collection already exists , loading collection")
            index = tantivy.Index.open(str(collection_path))
        else :
            print("Creating a new collection...")
            Path(collection_path).resolve().mkdir(parents=True, exist_ok=True)
            index = tantivy.Index(tantivy_schema, path=str(collection_path))
        collection_path.mkdir(parents=True, exist_ok=True)
        self.__update_metadata_directory(collection_name, schema)

        self.__register_tokenizer(index)

        writer = index.writer(heap_size=512 * 1024 * 1024)
        count = 0
        extracted = processor.extract_many(documents)
        for record in extracted:
            tantivy_doc = self.__to_tantivy_document(record, schema)
            writer.add_document(tantivy_doc)
            count += 1

        writer.commit()
        writer.wait_merging_threads()
        index.reload()
        print(f"Successfully indexed {count} records into collection '{collection_name}'.")

    def search(self, query, top_k, collection_name):
        collection_path = self.db_path / collection_name
        if not collection_path.exists():
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        schema = InvertedIndex.metadata_directory[collection_name]
        index = tantivy.Index.open(str(collection_path))
        self.__register_tokenizer(index)

        searcher = index.searcher()
        parsed_query = self.__parse_query(index, query, schema.main_field)
        if parsed_query is None:
            return []
        results = searcher.search(parsed_query, top_k)

        rows = []
        for score, address in results.hits:
            stored_doc = searcher.doc(address)
            stored_values = stored_doc.to_dict()
            row = [stored_values.get(field_name)[0] for field_name,field_type in schema.fields if schema.main_field != field_name]
            row.append(score)
            rows.append(tuple(row))
        return rows

    def close(self):
        pass

    def __build_tantivy_schema(self, schema):
        builder = SchemaBuilder()
        for field_name, field_type in schema.fields:
            field_type = field_type.upper()
            if field_type == "TEXT":
                if field_name == schema.main_field:
                    builder.add_text_field(field_name,stored=False,tokenizer_name="legal_tokenizer")
                else:
                    builder.add_text_field(field_name, stored=True)
            elif field_type in {"INTEGER", "INT", "NUMERIC"}:
                builder.add_integer_field(field_name, stored=True)
            elif field_type in {"BOOLEAN", "BOOL"}:
                builder.add_boolean_field(field_name, stored=True)
            else:
                raise ValueError(f"Unsupported Tantivy field type: {field_type}")

        return builder.build()

    def __to_tantivy_document(self, record, schema: Schema):
        values = {}
        if len(record) != len(schema.fields):
            raise ValueError(f"Record does not match schema. Expected {len(schema.fields)} values, got {len(record)}.")
        for value, (field_name, field_type) in zip(record, schema.fields):
            field_type = field_type.upper()

            if field_type == "TEXT":
                values[field_name] = str(value)
            elif field_type in {"INTEGER", "INT", "NUMERIC"}:
                values[field_name] = int(value)
            elif field_type in {"BOOLEAN", "BOOL"}:
                values[field_name] = bool(value)

        return tantivy.Document(**values)

    def __register_tokenizer(self, index):
        tokenizer = (
            tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.regex(r"(\S+)"))
            .filter(Filter.lowercase())
            .build()
        )

        index.register_tokenizer("legal_tokenizer", tokenizer)

    def __parse_query(self, index, query, main_field):
        query = str(query)
        try:
            return index.parse_query(query, [main_field])
        except ValueError:
            safe_query = re.sub(r"[^A-Za-z0-9_]+", " ", query)
            safe_query = " ".join(safe_query.split())
            if not safe_query:
                return None
            return index.parse_query(safe_query, [main_field])

    def __load_metadata_directory(self):
        file_path = self.db_path / InvertedIndex.metadata_directory_file
        if file_path.exists():
            with open(file_path, "rb") as f:
                InvertedIndex.metadata_directory = pickle.load(f)
        else:
            InvertedIndex.metadata_directory = {}

    def __update_metadata_directory(self, collection_name, schema):
        InvertedIndex.metadata_directory[collection_name] = schema
        file_path = self.db_path / InvertedIndex.metadata_directory_file
        with open(file_path, "wb") as f:
            pickle.dump(InvertedIndex.metadata_directory, f)

    @staticmethod
    def get_schema(collection_name):
        return InvertedIndex.metadata_directory.get(collection_name, None)
