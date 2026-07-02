import re

from src.DatabaseLayer.Processors.BaseProcessor import BaseProcessor
from src.DatabaseLayer.Processors.Schema import Schema


class SectionsProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        fields = (("case_id","TEXT"), ("section_id","TEXT"), ("text","TEXT"))
        primary_keys = ("case_id", "section_id")
        self.schema = Schema(fields, primary_keys, "text")

    def __organize(self, paragraphs_dict, case_id):
        # we assume structure (case_id, section_id, text)
        final_records = []
        for section_id, section_text in paragraphs_dict.items():
            final_records.append((case_id,section_id, section_text))
        return final_records

    def extract(self, data_tuple, window_length=3):
        """
        Extracts context windows surrounding any citation tag
        (<FRAGMENT_SUPPRESSED>, <CITATION_SUPPRESSED>, or <REFERENCE_SUPPRESSED>).
        A section is the sentence containing the tag, plus up to 'window_length'
        sentences before and after it, strictly bounded by the paragraph limits.
        The tags are then stripped from the final text before lemmatization.
        """
        case_id = data_tuple[0]
        text = data_tuple[1]

        # 1. Normalize text BUT PRESERVE ALL THREE TARGET TAGS for boundary detection
        text = text.replace('\n', ' ').replace('•', '')

        # Erase paragraph markers here if you do not want them in the final text
        text = re.sub(r'\[\d{1,3}\]', ' ', text)

        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Define our target tags
        target_tags = ("<FRAGMENT_SUPPRESSED>", "<CITATION_SUPPRESSED>", "<REFERENCE_SUPPRESSED>")

        # 2. Extract logical paragraphs using your existing sliding window logic
        pattern = r'\[(\d{1,3})\]'
        start_indices = [match.start() for match in re.finditer(pattern, text)]
        raw_paragraphs = []

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

        # 3. Find target sections using a dynamic sliding sentence window
        final_sections = {}
        section_id = 1

        for para in raw_paragraphs:
            # Fast skip if the paragraph doesn't contain ANY of our targets
            if not any(tag in para for tag in target_tags):
                continue

                # Parse sentences using spaCy
            doc = self.nlp(para)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

            # Iterate to find the targets (handles multiple citations in one paragraph)
            for idx, sent in enumerate(sentences):
                if any(tag in sent for tag in target_tags):
                    # Calculate window boundaries, clamped to the paragraph length
                    start_idx = max(0, idx - window_length)
                    end_idx = min(len(sentences), idx + window_length + 1)

                    # Extract and join the window
                    window_sentences = sentences[start_idx:end_idx]
                    section_text = ' '.join(window_sentences)

                    # --- NEW STEP: Erase the target tags before lemmatizing ---
                    section_text = re.sub(r'<(?:FRAGMENT|CITATION|REFERENCE)_SUPPRESSED>', ' ', section_text)

                    # Clean and lemmatize the final section (tags are now gone)
                    processed_section = self.lemmatize_and_clean(section_text)

                    if processed_section.strip():
                        final_sections[str(section_id)] = processed_section
                        section_id += 1

        return self.__organize(final_sections, case_id)
