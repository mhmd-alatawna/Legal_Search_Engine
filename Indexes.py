import os
import json
import re
import shutil
import subprocess
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import tantivy
from tantivy import SchemaBuilder, TextAnalyzerBuilder, Tokenizer, Filter
from Chunkers import BasicChunker
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)



class InvertedIndexPyserini:
    def __init__(self, index_path="Data/InvertedIndex"):
        self.index_path = index_path
        self.searcher = None
        self.last_b = -1
        self.last_k1 = -1

    @staticmethod
    def generate_index(documents_dict, index_path="Data/InvertedIndex", temp_jsonl_dir="Data/TempCollection",segment_size=10,overlap_size=5):
        shutil.rmtree("Data/InvertedIndex")
        os.makedirs(index_path, exist_ok=True)
        os.makedirs(temp_jsonl_dir, exist_ok=True)

        chunker = BasicChunker(segment_size, overlap_size)

        # ---------------------------------------------------------
        # STEP 1: PREPARE DATA IN JSONL FORMAT (Pyserini Requirement)
        # ---------------------------------------------------------
        jsonl_file_path = os.path.join(temp_jsonl_dir, "chunks.jsonl")
        print(f"Chunking and pre-processing {len(documents_dict)} documents...")

        with open(jsonl_file_path, 'w', encoding='utf-8') as f:
            for doc_id, text in tqdm(documents_dict.items(), desc="Writing JSONL", unit="doc"):
                segments = chunker.chunk_document(text, doc_id)

                for segment_id, segment in segments:
                    cleaned_segment = InvertedIndexPyserini.sanitize_and_preprocess(segment)

                    # Pyserini STRICTLY requires an 'id' and 'contents' key
                    doc_dict = {
                        "id": str(segment_id),
                        "contents": cleaned_segment
                    }
                    f.write(json.dumps(doc_dict) + '\n')

        chunker.flush_datastructures() # Keep this if your chunker still uses it

        # ---------------------------------------------------------
        # STEP 2: RUN PYSERINI INDEXER
        # ---------------------------------------------------------
        print("Starting Lucene Indexing via Pyserini...")
        # We call Pyserini's highly optimized indexer as a subprocess module
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", temp_jsonl_dir,
            "--index", index_path,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "4",  # Adjust based on your CPU cores
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw"
        ]

        subprocess.run(cmd, check=True)
        print("Lexical Indexing complete!")

    @staticmethod
    def sanitize_and_preprocess(text):
        """
        Replaces Tantivy's TextAnalyzerBuilder.
        Handles regex tokenization, lowercasing, and custom stopword removal in Python.
        (Stemming will be handled natively by Pyserini's default EnglishAnalyzer).
        """
        LEGAL_STOPWORDS = {
            "court", "plaintiff", "defendant", "appellant", "appellee", "judgment",
            "ruling", "order", "affirmed", "reversed", "remanded", "petition",
            "motion", "counsel", "testimony", "evidence", "proceedings", "district",
            "circuit", "supreme", "pursuant", "herein", "thereto", "foregoing"
        }
        STANDARD_ENGLISH = set(stopwords.words('english'))
        COMBINED_STOPWORDS = LEGAL_STOPWORDS | STANDARD_ENGLISH

        # 1. Lowercase
        text = text.lower()
        # 2. Keep only alphabetic characters
        words = re.findall(r"(?i)([a-z]+)", text)
        # 3. Filter stopwords
        filtered_words = [w for w in words if w not in COMBINED_STOPWORDS]

        return " ".join(filtered_words)

    def query_parser(self, query):
        """
        Pre-processes the query to match the tokens stored in the index.
        Pyserini's `searcher.search()` takes this raw string directly.
        """
        return InvertedIndexPyserini.sanitize_and_preprocess(query)

    def search(self, query, top_k, k1=1, b=0.8):
        if self.searcher is None or self.last_b != b or self.last_k1 != k1:
            from pyserini.search.lucene import LuceneSearcher
            self.searcher = LuceneSearcher(self.index_path)
            self.searcher.set_bm25(k1, b)
            self.last_b = b
            self.last_k1 = k1


        query = self.query_parser(query)
        results = self.searcher.search(query, top_k)
        tmp = []
        for hit in results:
            segment_id = hit.docid
            score = hit.score
            tmp.append((segment_id, score))
        return tmp


# TODO : adjust implementation so it can create the index without the need to use the full documents_dict
class InvertedIndexTantivy:
    def __init__(self, index_path="Data/InvertedIndex"):
        schema = InvertedIndexTantivy.get_schema()
        tokenizer = InvertedIndexTantivy.get_tokenizer()
        self.index = tantivy.Index(schema, path=index_path)
        self.index.register_tokenizer("legal_tokenizer", tokenizer)

    @staticmethod
    def generate_index(documents_dict, index_path="Data/InvertedIndex",segment_size=10,overlap_size=5):
        shutil.rmtree("Data/InvertedIndex")
        os.makedirs(index_path, exist_ok=True)
        schema = InvertedIndexTantivy.get_schema()
        tokenizer = InvertedIndexTantivy.get_tokenizer()

        index = tantivy.Index(schema, path=index_path)
        index.register_tokenizer("legal_tokenizer", tokenizer)

        chunker = BasicChunker(segment_size,overlap_size)


        # 5. SETUP WRITER AND LOOP
        # 2GB RAM budget (Ample space for 46k documents)
        writer = index.writer(heap_size=2 * 1024 * 1024 * 1024)
        print(f"Starting Indexing for {len(documents_dict)} documents...")
        for doc_id, text in documents_dict.items():
            # Pass kwargs directly to tantivy.Document as you successfully did before
            segments = chunker.chunk_document(text, doc_id)
            for segment_id, segment in segments:
                writer.add_document(tantivy.Document(
                    segment_id=str(segment_id),
                    text=str(segment)
                ))

        # 6. FINALIZE
        print("Flushing index to disk and merging segments...")
        writer.commit()
        writer.wait_merging_threads()
        chunker.flush_datastructures()
        print("Lexical Indexing complete!")

    def query_parser(self,query):
        safe_query_string = InvertedIndexTantivy.sanitize_query(query)
        # Parse the query. Now Tantivy knows how to process the query string!
        return self.index.parse_query(safe_query_string, ["text"])

    @staticmethod
    def get_schema():
        schema_builder = SchemaBuilder()
        schema_builder.add_text_field("segment_id", stored=True)
        schema_builder.add_text_field("text", stored=True, tokenizer_name="legal_tokenizer")
        schema = schema_builder.build()

        return schema

    @staticmethod
    def get_tokenizer():
        LEGAL_STOPWORDS = [
            "court", "plaintiff", "defendant", "appellant", "appellee", "judgment",
            "ruling", "order", "affirmed", "reversed", "remanded", "petition",
            "motion", "counsel", "testimony", "evidence", "proceedings", "district",
            "circuit", "supreme", "pursuant", "herein", "thereto", "foregoing"
        ]
        STANDARD_ENGLISH = stopwords.words('english')
        COMBINED_STOPWORDS = LEGAL_STOPWORDS + STANDARD_ENGLISH

        tokenizer = (
            TextAnalyzerBuilder(Tokenizer.regex(r"(?i)([a-z]+)"))
            .filter(Filter.lowercase())
            .filter(Filter.custom_stopword(COMBINED_STOPWORDS))
            .filter(Filter.stemmer("english"))
            .build()
        )

        return tokenizer

    @staticmethod
    def sanitize_query(raw_query):
        """
        Strips out all punctuation and lowercases the text to prevent
        Tantivy from misinterpreting uppercase words as boolean operators (AND, OR, NOT).
        """
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', raw_query)
        return clean_text.lower()

    def search(self, query, top_k, k1=1, b=0.8):
        searcher = self.index.searcher()

        query = self.query_parser(query)
        results = searcher.search(query, top_k)
        tmp = []
        for score, segment_address in results.hits:
            doc = searcher.doc(segment_address)
            segment_id = doc["segment_id"][0]
            tmp.append((segment_id, score))
        return tmp
