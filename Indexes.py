import json
from datasets import load_dataset
import gzip
import os
import tantivy
from tantivy import SchemaBuilder, TextAnalyzerBuilder, Tokenizer, Filter
import re
import nltk

from Chunkers import BasicChunker
from DataSetHandler import flush_data_to_gzipped_json

nltk.download('stopwords')
from nltk.corpus import stopwords

def get_schema():
    schema_builder = SchemaBuilder()
    schema_builder.add_text_field("segment_id", stored=True)
    schema_builder.add_text_field("text", stored=True, tokenizer_name="legal_tokenizer")
    schema = schema_builder.build()

    return schema


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


def sanitize_query(raw_query):
    """
    Strips out all punctuation and lowercases the text to prevent
    Tantivy from misinterpreting uppercase words as boolean operators (AND, OR, NOT).
    """
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', raw_query)
    return clean_text.lower()
# TODO : adjust implementation so it can create the index without the need to use the full documents_dict
class InvertedIndex:
    def __init__(self, index_path="Data/InvertedIndex"):
        schema = get_schema()
        tokenizer = get_tokenizer()
        self.index = tantivy.Index(schema, path=index_path)
        self.index.register_tokenizer("legal_tokenizer", tokenizer)

    @staticmethod
    def generate_index(documents_dict, index_path="Data/InvertedIndex"):
        os.makedirs(index_path, exist_ok=True)
        schema = get_schema()
        tokenizer = get_tokenizer()

        index = tantivy.Index(schema, path=index_path)
        index.register_tokenizer("legal_tokenizer", tokenizer)

        chunker = BasicChunker(10,5)


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

    def get_searcher(self):
        return self.index.searcher()

    def query_parser(self,query):
        safe_query_string = sanitize_query(query)
        # Parse the query. Now Tantivy knows how to process the query string!
        return self.index.parse_query(safe_query_string, ["text"])
