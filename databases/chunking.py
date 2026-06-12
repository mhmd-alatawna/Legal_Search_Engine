"""
Pre-processing & chunking engine.

Implements the three chunking strategies of the data layer:
  - Paragraphs: natural delimiters, sliding window for long paragraphs
    (max 200 tokens, 100-token overlap).
  - Sentences: spaCy segmentation, sliding window for long sentences
    (max 20 tokens, 10-token overlap).
  - Propositions: token window of up to `proposition_window_tokens`
    captured around each hidden citation tag, strictly respecting
    paragraph boundaries.

"Token" here means a whitespace-delimited word. This keeps the window
arithmetic deterministic and model-independent, while staying well within
the context limits of BERT-class models for the chosen window sizes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

from databases.config import ChunkingConfig

logger = logging.getLogger(__name__)

# Two or more newlines (possibly with intermediate whitespace) delimit
# natural paragraphs in the raw case texts.
_PARAGRAPH_DELIMITER = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    """A single text chunk produced by the engine.

    `chunk_id` is the zero-based ordinal of the chunk within its case,
    matching the `paragraph_id` / `sentence_id` / `proposition_id`
    columns of the corresponding MongoDB collections.
    """

    case_id: str
    chunk_id: int
    text: str


class ChunkingEngine:
    """Stateless engine converting a raw case text into chunk lists."""

    def __init__(self, config: ChunkingConfig):
        self._config = config
        # Loaded lazily: spaCy is only required for sentence chunking.
        self._nlp = None

    def _get_nlp(self):
        if self._nlp is None:
            self._nlp = self._load_spacy(self._config.spacy_model)
        return self._nlp

    @staticmethod
    def _load_spacy(model_name: str):
        """Load the spaCy pipeline, falling back to a blank English
        pipeline with a rule-based sentencizer when the packaged model
        is not installed. Heavy components are disabled since only
        sentence boundaries are needed."""
        import spacy

        try:
            nlp = spacy.load(model_name, exclude=["ner", "lemmatizer", "tagger"])
            logger.info("Loaded spaCy model '%s'.", model_name)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found; falling back to blank 'en' "
                "pipeline with a rule-based sentencizer.",
                model_name,
            )
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
        # Legal documents can be very long; raise the processing limit.
        nlp.max_length = 5_000_000
        return nlp

    # ------------------------------------------------------------------
    # Paragraphs
    # ------------------------------------------------------------------

    def chunk_paragraphs(self, case_id: str, text: str) -> List[Chunk]:
        """Split raw text on natural paragraph delimiters, then apply a
        sliding window to any paragraph exceeding the token budget."""
        max_tokens = self._config.paragraph_max_tokens
        overlap = self._config.paragraph_overlap_tokens

        chunks: List[Chunk] = []
        for paragraph in self.split_natural_paragraphs(text):
            tokens = paragraph.split()
            if len(tokens) <= max_tokens:
                chunks.append(Chunk(case_id, len(chunks), paragraph))
            else:
                for window in self._sliding_windows(tokens, max_tokens, overlap):
                    chunks.append(Chunk(case_id, len(chunks), window))
        return chunks

    @staticmethod
    def split_natural_paragraphs(text: str) -> List[str]:
        """Split on blank-line delimiters and drop empty fragments."""
        paragraphs = _PARAGRAPH_DELIMITER.split(text)
        return [p.strip() for p in paragraphs if p and p.strip()]

    # ------------------------------------------------------------------
    # Sentences
    # ------------------------------------------------------------------

    def chunk_sentences(self, case_id: str, text: str) -> List[Chunk]:
        """Tokenize the text into sentences with spaCy, then apply a
        sliding window to any sentence exceeding the token budget."""
        max_tokens = self._config.sentence_max_tokens
        overlap = self._config.sentence_overlap_tokens

        chunks: List[Chunk] = []
        # Process paragraph by paragraph so spaCy never sees an oversized
        # document and sentence boundaries cannot cross paragraphs.
        nlp = self._get_nlp()
        for paragraph in self.split_natural_paragraphs(text):
            doc = nlp(paragraph)
            for sent in doc.sents:
                sentence = sent.text.strip()
                if not sentence:
                    continue
                tokens = sentence.split()
                if len(tokens) <= max_tokens:
                    chunks.append(Chunk(case_id, len(chunks), sentence))
                else:
                    for window in self._sliding_windows(tokens, max_tokens, overlap):
                        chunks.append(Chunk(case_id, len(chunks), window))
        return chunks

    # ------------------------------------------------------------------
    # Propositions
    # ------------------------------------------------------------------

    def extract_proposition_contexts(self, case_id: str, text: str) -> List[Chunk]:
        """Locate every hidden citation tag and capture a surrounding
        window of up to `proposition_window_tokens` tokens, strictly
        clamped to the boundaries of the containing paragraph.

        The window is centered on the tag (half the budget on each side);
        when the tag sits near a paragraph edge the window simply gets
        truncated rather than borrowing tokens from neighbouring
        paragraphs. The tag itself is removed from the captured context.
        """
        tag = self._config.citation_tag
        window_budget = self._config.proposition_window_tokens
        half_window = window_budget // 2

        chunks: List[Chunk] = []
        for paragraph in self.split_natural_paragraphs(text):
            if tag not in paragraph:
                continue
            tokens = paragraph.split()
            tag_positions = [i for i, tok in enumerate(tokens) if tag in tok]
            for position in tag_positions:
                start = max(0, position - half_window)
                end = min(len(tokens), position + half_window + 1)
                window_tokens = [
                    tok for tok in tokens[start:end] if tag not in tok
                ]
                context = " ".join(window_tokens).strip()
                if context:
                    chunks.append(Chunk(case_id, len(chunks), context))
        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sliding_windows(
        tokens: List[str], window_size: int, overlap: int
    ) -> List[str]:
        """Yield text windows of `window_size` tokens advancing by
        `window_size - overlap` tokens per step."""
        if overlap >= window_size:
            raise ValueError("Overlap must be smaller than the window size.")
        stride = window_size - overlap
        windows: List[str] = []
        for start in range(0, len(tokens), stride):
            window = tokens[start : start + window_size]
            windows.append(" ".join(window))
            if start + window_size >= len(tokens):
                break
        return windows
