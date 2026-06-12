"""
Embedding generator.

Wraps a HuggingFace transformer encoder (Legal-BERT by default, BGE as an
alternative) behind a minimal batched-encoding API. Heavy dependencies
(torch / transformers) are imported lazily so the rest of the data layer
can be used on machines without a deep-learning stack installed.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

from databases.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Batched text-to-vector encoder with mean pooling."""

    def __init__(self, config: EmbeddingConfig):
        self._config = config
        self._tokenizer = None
        self._model = None
        self._device = None
        self._vector_size: int = 0

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        device = self._config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(
            "Loading embedding model '%s' on device '%s'...",
            self._config.model_name,
            device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._config.model_name)
        self._model = AutoModel.from_pretrained(self._config.model_name)
        self._model.to(device)
        self._model.eval()
        self._device = device
        self._vector_size = int(self._model.config.hidden_size)
        logger.info("Embedding model ready (vector size: %d).", self._vector_size)

    @property
    def vector_size(self) -> int:
        """Dimensionality of the produced embeddings."""
        self._ensure_loaded()
        return self._vector_size

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode a sequence of texts into embedding vectors.

        Uses attention-masked mean pooling over the last hidden state,
        optionally followed by L2 normalization (so cosine similarity
        reduces to a dot product in the vector store).
        """
        if not texts:
            return []
        self._ensure_loaded()

        import torch

        vectors: List[List[float]] = []
        batch_size = self._config.batch_size
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._config.max_sequence_length,
                    return_tensors="pt",
                ).to(self._device)

                output = self._model(**encoded)
                hidden = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).type_as(hidden)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts

                if self._config.normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(pooled.cpu().tolist())
        return vectors

    def encode_one(self, text: str) -> List[float]:
        """Convenience wrapper for single-text encoding (query time)."""
        return self.encode([text])[0]
