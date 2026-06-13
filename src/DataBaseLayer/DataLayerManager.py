import os
import shutil
from pathlib import Path

from pyprojroot import here

from src.DataBaseLayer.OriginalText.OriginalTextDatabase import OriginalTextDatabase
from src.DataBaseLayer.Paragraphs.ParagraphDatabase import ParagraphDatabase
from src.DataBaseLayer.Paragraphs.ParagraphEmbeddingsDatabase import ParagraphEmbeddingsDatabase
from src.DataBaseLayer.Paragraphs.ParagraphInvertedIndex import ParagraphInvertedIndex
from src.DataBaseLayer.Sections.SectionsDatabase import SectionsDatabase
from src.DataBaseLayer.Sections.SectionsInvertedIndex import SectionsInvertedIndex
from src.DataBaseLayer.Sentences.SentenceInvertedIndex import SentenceInvertedIndex
from src.DataBaseLayer.Sentences.SentencesDatabase import SentenceDatabase
from src.DataBaseLayer.Sentences.SentencesEmbeddingsDatabase import SentencesEmbeddingsDatabase

dir_path = str(here() / "Databases")


sections_DB = f"{dir_path}/Sections.db"
sections_inverted_index_DB = f"{dir_path}/SectionsInvertedIndex"

sentences_DB = f"{dir_path}/Sentences.db"
sentences_inverted_index_DB = f"{dir_path}/SentencesInvertedIndex"
sentences_embeddings_DB = f"{dir_path}/SentencesEmbeddings"
sentences_tmp_embeddings_DB = f"/tmp/SentencesEmbeddings"

paragraphs_DB = f"{dir_path}/Paragraphs.db"
paragraphs_inverted_index_DB = f"{dir_path}/ParagraphsInvertedIndex"
paragraphs_embeddings_DB = f"{dir_path}/ParagraphsEmbeddings"
paragraphs_tmp_embeddings_DB = f"/tmp/ParagraphsEmbeddings"

original_text_DB = f"{dir_path}/LegalCases.db"

train_cases_dir = str(here() / "RawFiles/task1_train_files_2026")
train_citations_file = str(here() / "RawFiles/task1_train_labels_2026.json")
test_cases_dir = str(here() / "RawFiles/task1_test_files_2026")
test_citations_file = str(here() / "RawFiles/task1_test_labels_2026.json")

def configure_original_test_database():
    database = OriginalTextDatabase(original_text_DB)
    try:
        database.extract_records(train_cases_dir,False, train_citations_file)
        database.extract_records(test_cases_dir,True, test_citations_file)
    finally:
        database.close()

def configure_paragraphs_database():
    database = ParagraphDatabase(paragraphs_DB)
    try:
        database.extract_records(train_cases_dir)
        database.extract_records(test_cases_dir)
    finally:
        database.close()

def configure_paragraphs_inverted_index():
    my_db = ParagraphDatabase(paragraphs_DB)
    all_records = my_db.get_all_records()
    inverted_index = ParagraphInvertedIndex(paragraphs_inverted_index_DB)
    ParagraphInvertedIndex.generate_index(all_records, paragraphs_inverted_index_DB)

def configure_paragraphs_embeddings():
    my_db = ParagraphDatabase(paragraphs_DB)
    all_records = my_db.get_all_records()

    if os.path.exists(paragraphs_tmp_embeddings_DB):
        shutil.rmtree(paragraphs_tmp_embeddings_DB)
    embeddings_db = ParagraphEmbeddingsDatabase(paragraphs_tmp_embeddings_DB)
    embeddings_db.index_paragraphs(all_records)
    embeddings_db.close()
    if paragraphs_tmp_embeddings_DB is not None:
        shutil.copytree(paragraphs_tmp_embeddings_DB, paragraphs_embeddings_DB, dirs_exist_ok=True)
        if os.path.exists(paragraphs_tmp_embeddings_DB):
            shutil.rmtree(paragraphs_tmp_embeddings_DB)


def configure_sentences_database():
    database = SentenceDatabase(sentences_DB)
    try:
        database.extract_records(train_cases_dir)
        database.extract_records(test_cases_dir)
    finally:
        database.close()

def configure_sentences_inverted_index():
    my_db = SentenceDatabase(sentences_DB)
    all_records = my_db.get_all_records()
    inverted_index = SentenceInvertedIndex(sentences_inverted_index_DB)
    SentenceInvertedIndex.generate_index(all_records, sentences_inverted_index_DB)

def configure_sentences_embeddings():
    my_db = SentenceDatabase(sentences_DB)
    all_records = my_db.get_all_records()

    if os.path.exists(sentences_tmp_embeddings_DB):
        shutil.rmtree(sentences_tmp_embeddings_DB)
    embeddings_db = SentencesEmbeddingsDatabase(paragraphs_tmp_embeddings_DB)
    embeddings_db.index_sentences(all_records)
    embeddings_db.close()
    if sentences_tmp_embeddings_DB is not None:
        shutil.copytree(sentences_tmp_embeddings_DB, sentences_embeddings_DB, dirs_exist_ok=True)
        if os.path.exists(sentences_tmp_embeddings_DB):
            shutil.rmtree(sentences_tmp_embeddings_DB)

# TODO : probably it is useless to run the sections database on the training set because
#  it shouldn't contain any sections at all , maybe only run it on the test set ?
def configure_sections_database():
    database = SectionsDatabase(sections_DB)
    try:
        database.extract_records(train_cases_dir)
        database.extract_records(test_cases_dir)
    finally:
        database.close()

def configure_sections_inverted_index():
    my_db = SectionsDatabase(sections_DB)
    all_records = my_db.get_all_records()
    inverted_index = SectionsInvertedIndex(sections_inverted_index_DB)
    SectionsInvertedIndex.generate_index(all_records, sections_inverted_index_DB)

if __name__ == "__main__":
#     dir_path = Path("Databases")
#     if dir_path.is_dir():
#         shutil.rmtree(dir_path)
#     dir_path.mkdir(parents=True, exist_ok=True)
#
#
#     configure_original_test_database()
#     configure_paragraphs_database()
#     configure_paragraphs_inverted_index()
#     configure_sentences_database()
#     configure_sentences_inverted_index()
#     configure_sections_database()
#     configure_sections_inverted_index()
    configure_paragraphs_embeddings()
    configure_sentences_embeddings()
