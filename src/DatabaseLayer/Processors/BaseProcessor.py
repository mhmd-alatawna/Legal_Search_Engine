import abc
import concurrent
import os
import re
from abc import ABC

import spacy
from tqdm import tqdm

_worker_processor = None


def _init_sqlite_worker(processor_class):
    global _worker_processor
    # Dynamically instantiate the processor class inside the worker process
    _worker_processor = processor_class()

def _process_chunk_batch(data_tuple):
    try:
        # returns a list of all records as tuples
        extracted = _worker_processor.extract(data_tuple)
        return True, extracted
    except Exception as e:
        return False, str(e)


# TODO : for some reason it is slower to run the multiprocessing here than before , check why and fix it .
class BaseProcessor(ABC):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner"])
        self.nlp.enable_pipe("senter")

    def _mask_quantities(self, text):
        """
        Safely masks highly variable quantitative numbers (money, dates, percentages)
        while leaving informative legal citations, section numbers, and dockets completely untouched.
        """
        # 1. Mask Money (Matches $500, $5,000.00, etc.)
        text = re.sub(r'\$\s*\d+(?:,\d{3})*(?:\.\d+)?', ' MASKEDMONEY ', text)

        # 2. Mask Percentages (Matches 10%, 99.9%, etc.)
        text = re.sub(r'\b\d+(?:\.\d+)?\s*%', ' MASKEDPERCENT ', text)

        # 3. Mask Numeric Dates (Matches 12/31/2022, 12-31-2022)
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', ' MASKEDDATE ', text)

        # 4. Mask Written Dates (Matches January 1st, 2022, Oct. 14, 2021, etc.)
        date_pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b'
        text = re.sub(date_pattern, ' MASKEDDATE ', text, flags=re.IGNORECASE)

        return text

    def lemmatize_and_clean(self, text):
        """
        Lemmatizes text for similarity comparison while strictly preserving numbers
        and stripping out noisy punctuation.
        """
        # --- NEW STEP: Apply the safe masking before spaCy reads it ---
        text = self._mask_quantities(text)

        doc = self.nlp(text)
        cleaned_tokens = []

        for token in doc:
            # 1. Preserve numbers (matches "123", "72(1)", "1st", or any token containing a digit)
            # Since our masked tags (e.g., MASKEDMONEY) have no digits, they bypass this step.
            if token.like_num or any(char.isdigit() for char in token.text):
                cleaned_tokens.append(token.text.lower())

            # 2. Lemmatize purely alphabetic words
            elif token.is_alpha:
                # The masked tags will be caught here, lowercased to 'maskedmoney', and kept safe!
                cleaned_tokens.append(token.lemma_.lower())

        return ' '.join(cleaned_tokens)

    def _split_long_text(self, text, max_sentences, overlap):
        """
        Splits a long string into chunks based on a maximum number of sentences
        using spaCy for robust sentence boundary detection.
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if len(sentences) <= max_sentences:
            return [text]

        chunks = []
        step = max(1, max_sentences - overlap)

        for i in range(0, len(sentences), step):
            chunk_sentences = sentences[i: i + max_sentences]
            chunks.append(' '.join(chunk_sentences))

            if i + max_sentences >= len(sentences):
                break

        return chunks

    @abc.abstractmethod
    def extract(self,data_tuple):
        pass

    def extract_many(self,documents):
        final_records = []
        try:
            available_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cores = os.cpu_count() or 4

        ideal_chunksize = max(1, len(documents) // (available_cores * 4))
        calculated_chunksize = min(25, ideal_chunksize)
        print(f"Processing and chunking {len(documents)} items across {available_cores} processes...")

        # Switch to ThreadPoolExecutor to easily share the memory space
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=available_cores,
                initializer=_init_sqlite_worker,
                initargs=(self.__class__,)
        ) as executor:
            results = list(tqdm(
                executor.map(_process_chunk_batch, documents, chunksize=calculated_chunksize),
                total=len(documents),
                desc=f"Chunking -> documents on ({available_cores} cores)"
            ))
            for success, result_data in results:
                if success:
                    final_records.extend(result_data)
                else:
                    print(f"Error processing document: {result_data}")
        return final_records