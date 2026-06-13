import re
import spacy


class LegalDocumentProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner"])
        self.nlp.enable_pipe("senter")

    def __mask_quantities(self, text):
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
        text = self.__mask_quantities(text)

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

    def __split_long_text(self, text, max_sentences, overlap):
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

    def extract_paragraphs(self, text, max_sentences=10):
        """
        Extracts paragraphs using sequential logic to ignore false positives,
        splits them if they contain too many sentences, and assigns fresh sequential IDs.
        """
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

            chunks = self.__split_long_text(para, max_sentences, max_sentences // 2)

            for chunk in chunks:
                # Apply the advanced text processing before saving the final chunk
                processed_chunk = self.lemmatize_and_clean(chunk)

                # Failsafe: only add the chunk if it isn't empty after processing
                if processed_chunk.strip():
                    final_paragraphs[str(new_id)] = processed_chunk
                    new_id += 1

        return final_paragraphs

    def extract_sentences(self, text):
        """
        Extracts individual sentences from the text, applies structural normalization
        to protect boundary detection, strips paragraph markers, and lemmatizes each.
        """
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

        return final_sentences

    def extract_sections(self, text, window_length=3):
        """
        Extracts context windows surrounding any citation tag
        (<FRAGMENT_SUPPRESSED>, <CITATION_SUPPRESSED>, or <REFERENCE_SUPPRESSED>).
        A section is the sentence containing the tag, plus up to 'window_length'
        sentences before and after it, strictly bounded by the paragraph limits.
        The tags are then stripped from the final text before lemmatization.
        """
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

        return final_sections
