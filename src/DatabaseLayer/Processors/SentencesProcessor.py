import re

from src.DatabaseLayer.Processors.BaseProcessor import BaseProcessor
from src.DatabaseLayer.Processors.Schema import Schema


class SentencesProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        fields = (("case_id","TEXT"), ("sentence_id","TEXT"), ("text","TEXT"))
        primary_keys = ("case_id", "sentence_id")
        self.schema = Schema(fields, primary_keys, "text")

    def __organize(self, paragraphs_dict, case_id):
        # we assume structure (case_id, sentence_id, text)
        final_records = []
        for sentence_id, sentence_text in paragraphs_dict.items():
            final_records.append((case_id,sentence_id, sentence_text))
        return final_records

    def extract(self, data_tuple, window_length=3):
        """
        Extracts individual sentences from the text, applies structural normalization
        to protect boundary detection, strips paragraph markers, and lemmatizes each.
        """
        case_id = data_tuple[0]
        text = data_tuple[1]

        # 1. Normalize text to protect spaCy's sentence boundary detection
        text = text.replace('\n', ' ').replace('•', '').replace('<FRAGMENT_SUPPRESSED>', ' ').replace(
            '<REFERENCE_SUPPRESSED>', ' ').replace('<CITATION_SUPPRESSED>', ' ')

        # Strip out the paragraph markers (e.g., [1], [42]) so they don't become
        # isolated noisy sentences or attach awkwardly to the start of valid sentences.
        text = re.sub(r'\[(\d{1,3})\]', ' ', text)

        # Flatten whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # 2. Parse the entire cleaned text through spaCy to detect sentence boundaries
        doc = self.nlp(text)
        raw_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        final_sentences = {}
        new_id = 1

        # 3. Clean, Lemmatize, and assign final sequential IDs
        for sent in raw_sentences:
            processed_sent = self.lemmatize_and_clean(sent)

            # Failsafe: only add the sentence if it isn't empty after processing
            if processed_sent.strip():
                final_sentences[str(new_id)] = processed_sent
                new_id += 1

        return self.__organize(final_sentences, case_id)

