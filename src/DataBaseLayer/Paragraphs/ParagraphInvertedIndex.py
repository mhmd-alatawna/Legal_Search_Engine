import os
import shutil
from pathlib import Path

import tantivy
from nltk.corpus import stopwords
from tantivy import SchemaBuilder, Filter

from src.DataBaseLayer.LegalDocumentProcessor import LegalDocumentProcessor


class ParagraphInvertedIndex:
    def __init__(self, index_path):
        schema = ParagraphInvertedIndex.get_schema()
        tokenizer = ParagraphInvertedIndex.get_tokenizer()
        Path(index_path).mkdir(parents=True, exist_ok=True)
        self.index = tantivy.Index(schema, path=index_path)
        self.index.register_tokenizer("legal_tokenizer", tokenizer)
        self.doc_processor = LegalDocumentProcessor()


    @staticmethod
    def generate_index(all_cases, index_path):
        shutil.rmtree(index_path, ignore_errors=True)
        os.makedirs(index_path, exist_ok=True)

        schema = ParagraphInvertedIndex.get_schema()
        tokenizer = ParagraphInvertedIndex.get_tokenizer()

        index = tantivy.Index(schema, path=index_path)
        index.register_tokenizer("legal_tokenizer", tokenizer)

        # 2GB RAM budget (Ample space for 46k documents)
        writer = index.writer(heap_size=2 * 1024 * 1024 * 1024)
        print(f"Starting Indexing for {len(all_cases)} documents...")

        for case_id, paragraph_id, text in all_cases:
            # Assuming 'text' here is already passed through spaCy's lemmatize_and_clean!
            writer.add_document(tantivy.Document(
                case_id=str(case_id),
                paragraph_id=str(paragraph_id),
                text=str(text)
            ))

        print("Flushing index to disk and merging segments...")
        writer.commit()
        writer.wait_merging_threads()
        print("Lexical Indexing complete!")

    def query_parser(self, query):
        """
        Instead of a raw regex, we use the spaCy processor to ensure the query
        looks EXACTLY like the indexed documents (masked money, lemmatized, etc.)
        """
        safe_query_string = self.doc_processor.lemmatize_and_clean(query)

        if not safe_query_string.strip():
            return None
        try:
            return self.index.parse_query(safe_query_string, ["text"])
        except ValueError:
            return None

    @staticmethod
    def get_schema():
        schema_builder = SchemaBuilder()
        # 'stored=True' means Tantivy saves the original string to return in results.
        schema_builder.add_text_field("case_id", stored=True)
        schema_builder.add_text_field("paragraph_id", stored=True)
        schema_builder.add_text_field("text", stored=False, tokenizer_name="legal_tokenizer")
        # NOTE: I changed text stored=False to save massive disk space, assuming you
        # pull the actual display text from your database using the returned case_id.

        return schema_builder.build()

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
            # Use \S+ to split ONLY by whitespace. This preserves MASKEDMONEY and 72(1)
            tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.regex(r"(\S+)"))
            .filter(Filter.lowercase())  # Safe fallback, though spaCy did this.
            .filter(Filter.custom_stopword(COMBINED_STOPWORDS))
            # STEMMER REMOVED: Trust the spaCy lemmas!
            .build()
        )

        return tokenizer

    def search(self, query, top_k):
        searcher = self.index.searcher()

        parsed_query = self.query_parser(query)
        if parsed_query is None:
            return []

        results = searcher.search(parsed_query, top_k)
        tmp = []
        for score, segment_address in results.hits:
            doc = searcher.doc(segment_address)
            case_id = doc["case_id"][0]
            paragraph_id = doc["paragraph_id"][0]
            tmp.append((case_id, paragraph_id, score))

        return tmp