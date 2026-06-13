import logging
import os
from typing import List, Sequence

from pyprojroot import here

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Batched text-to-vector encoder with mean pooling (Raw PyTorch implementation)."""

    def __init__(self):
        """
        Initializes the generator settings without loading heavy dependencies.
        """
        # Define the local fallback path
        self.local_model_path = str(here() / "Models/local_legal_bert")

        # Check if local files exist; if so, default to them immediately
        if os.path.exists(self.local_model_path):
            self.model_name = self.local_model_path
            logger.info("Local model found. Using offline path: %s", self.model_name)
        else:
            self.model_name = "nlpaueb/legal-bert-base-uncased"
            logger.info("Local model not found. Will download from Hugging Face on first run.")

        self.batch_size = 256
        self.max_sequence_length = 512  # LEGAL-BERT absolute max is 512
        self.normalize = True
        self._explicit_device = None

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

        device = self._explicit_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # If we are pointing to Hugging Face, download it and save it locally first
        if self.model_name == "nlpaueb/legal-bert-base-uncased" and not os.path.exists(self.local_model_path):
            logger.info("Downloading '%s' from Hugging Face Hub...", self.model_name)

            # Temporary download to save files locally
            temp_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            temp_model = AutoModel.from_pretrained(self.model_name)

            logger.info("Saving model weights offline to '%s' for cluster use...", self.local_model_path)
            os.makedirs(self.local_model_path, exist_ok=True)
            temp_tokenizer.save_pretrained(self.local_model_path)
            temp_model.save_pretrained(self.local_model_path)

            # Switch the target pointer to our newly created local directory
            self.model_name = self.local_model_path

        logger.info(
            "Loading embedding model from '%s' on device '%s'...",
            self.model_name,
            device,
        )

        # This will now always load from the local directory if it was just downloaded or existed prior
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(device)
        self._model.eval()
        self._device = device
        self._vector_size = int(self._model.config.hidden_size)
        logger.info("Embedding model ready (vector size: %d, max length: %d).",
                    self._vector_size, self.max_sequence_length)

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
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start: start + self.batch_size])

                # Truncation=True ensures overly long texts do not crash the model
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_sequence_length,
                    return_tensors="pt",
                ).to(self._device)

                output = self._model(**encoded)
                hidden = output.last_hidden_state

                # Mean Pooling Math
                mask = encoded["attention_mask"].unsqueeze(-1).type_as(hidden)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts

                if self.normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(pooled.cpu().tolist())
        return vectors

    def encode_one(self, text: str) -> List[float]:
        """Convenience wrapper for single-text encoding (query time)."""
        return self.encode([text])[0]