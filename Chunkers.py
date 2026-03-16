import math

import nltk
from nltk.tokenize import sent_tokenize

from DataSetHandler import flush_data_to_gzipped_json


class BasicChunker:
    def __init__(self, segment_size, overlap_size):
        # Ensure the unsupervised Punkt model is downloaded for sentence splitting
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

        self.segment_size = segment_size
        self.overlap_size = overlap_size
        self.segment_to_doc = {}
        self.doc_to_segment_count = {}

    def chunk_document(self, document, doc_id):
        # NLTK handles the sentence boundary detection cleanly natively
        sentences = sent_tokenize(document)

        segments = []
        for i in range(0, len(sentences), self.segment_size - self.overlap_size):
            segment = ' '.join(sentences[i:i + self.segment_size])
            segments.append(segment)
            if i + self.segment_size >= len(sentences):
                break

        self.doc_to_segment_count[doc_id] = len(segments)

        tmp = []
        for count, segment in enumerate(segments):
            seg_id = f"{doc_id}_{count + 1}"
            self.segment_to_doc[seg_id] = doc_id
            tmp.append((seg_id, segment))

        segments = tmp
        return segments

    def flush_datastructures(self):
        flush_data_to_gzipped_json(self.segment_to_doc , "Data/segment_to_doc.json.gz")
        flush_data_to_gzipped_json(self.doc_to_segment_count , "Data/doc_to_segment_count.json.gz")
