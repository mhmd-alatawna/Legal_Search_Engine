import re

from src.DatabaseLayer.Processors.BaseProcessor import BaseProcessor
from src.DatabaseLayer.Processors.Schema import Schema


class ParagraphProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        fields = (("case_id","TEXT"), ("paragraph_id","TEXT"), ("text","TEXT"))
        primary_keys = ("case_id", "paragraph_id")
        self.schema = Schema(fields, primary_keys, "text")

    def __organize(self, paragraphs_dict, case_id):
        # we assume structure (case_id, paragraph_id, text)
        final_records = []
        for paragraph_id, paragraph_text in paragraphs_dict.items():
            final_records.append((case_id,paragraph_id, paragraph_text))
        return final_records

    def extract(self, data_tuple, max_sentences=10):
        """
        Extracts paragraphs using sequential logic to ignore false positives,
        splits them if they contain too many sentences, and assigns fresh sequential IDs.
        """
        case_id = data_tuple[0]
        text = data_tuple[1]

        # 1. Normalize text
        text = text.replace('\n', ' ').replace('•', '').replace('<FRAGMENT_SUPPRESSED>', ' ').replace(
            '<REFERENCE_SUPPRESSED>', ' ').replace('<CITATION_SUPPRESSED>', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = ' '.join(text.split())

        pattern = r'\[(\d{1,3})\]'
        start_indices = [match.start() for match in re.finditer(pattern, text)]

        raw_paragraphs = []

        # 2. Extract logical paragraphs
        if start_indices:
            rolling_count = 1

            for i in range(len(start_indices)):
                start = start_indices[i]
                target_text = text[start:start + 6]

                match = re.search(pattern, target_text)
                if not match:
                    continue

                number = int(match.group(1))
                end = start_indices[i + 1] if i + 1 < len(start_indices) else len(text)

                check_no = number - rolling_count

                if 0 <= check_no <= 6:
                    marker_len = len(match.group(0))
                    content = text[start + marker_len:end].strip()
                    raw_paragraphs.append(content)
                    rolling_count = number + 1
                else:
                    if len(raw_paragraphs) > 0:
                        raw_paragraphs[-1] += " " + text[start:end].strip()
                    else:
                        raw_paragraphs.append(text[start:end].strip())
        else:
            raw_paragraphs.append(text)

        # 3. Chunk long paragraphs, Lemmatize, and assign final sequential IDs
        final_paragraphs = {}
        new_id = 1

        for para in raw_paragraphs:
            if not para.strip():
                continue

            chunks = self._split_long_text(para, max_sentences, max_sentences // 2)

            for chunk in chunks:
                # Apply the advanced text processing before saving the final chunk
                processed_chunk = self.lemmatize_and_clean(chunk)

                # Failsafe: only add the chunk if it isn't empty after processing
                if processed_chunk.strip():
                    final_paragraphs[str(new_id)] = processed_chunk
                    new_id += 1

        return self.__organize(final_paragraphs, case_id)
