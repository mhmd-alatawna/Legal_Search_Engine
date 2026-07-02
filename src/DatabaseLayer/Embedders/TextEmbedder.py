import logging
import os
from typing import List, Sequence

from pyprojroot import here

from src.DatabaseLayer.Processors.Schema import Schema

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Optimized batched text-to-vector encoder with mixed precision and adaptive safety."""
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingGenerator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.local_model_path = str(here() / "Models/local_legal_bert")

        if os.path.exists(self.local_model_path):
            self.model_name = self.local_model_path
            logger.info("Local model found. Using offline path: %s", self.model_name)
        else:
            self.model_name = "nlpaueb/legal-bert-base-uncased"
            logger.info("Local model not found. Will download from Hugging Face on first run.")

        self.batch_size = 128
        self.max_sequence_length = 512
        self.normalize = True
        self._explicit_device = None

        self._tokenizer = None
        self._model = None
        self._device = None
        self._vector_size: int = 0


        fields = (("case_id","TEXT"), ("paragraph_id","TEXT"), ("text","TEXT"))
        primary_keys = ("case_id", "paragraph_id")
        self.schema = Schema(fields, primary_keys, "text", self.vector_size)
        self._initialized = True

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        device = self._explicit_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if self.model_name == "nlpaueb/legal-bert-base-uncased" and not os.path.exists(self.local_model_path):
            logger.info("Downloading '%s' from Hugging Face Hub...", self.model_name)
            temp_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            temp_model = AutoModel.from_pretrained(self.model_name)

            logger.info("Saving model weights offline to '%s'...", self.local_model_path)
            os.makedirs(self.local_model_path, exist_ok=True)
            temp_tokenizer.save_pretrained(self.local_model_path)
            temp_model.save_pretrained(self.local_model_path)
            self.model_name = self.local_model_path

        logger.info("Loading embedding model from '%s' on device '%s'...", self.model_name, device)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(device)
        self._model.eval()
        self._device = device
        self._vector_size = int(self._model.config.hidden_size)
        logger.info("Embedding model ready (vector size: %d).", self._vector_size)

    @property
    def vector_size(self) -> int:
        self._ensure_loaded()
        return self._vector_size

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        self._ensure_loaded()

        import torch

        # OPTIMIZATION: "Smart Batching"
        # Tracking original indices lets us sort by length to minimize padding bloat,
        # then re-sort back to the user's original order at the end.
        indexed_texts = sorted(enumerate(texts), key=lambda x: len(x[1]))
        sorted_vectors: List[List[float]] = []

        vectors_dict = {}

        # OPTIMIZATION: Setup mixed precision context if running on GPU
        device_type = "cuda" if "cuda" in str(self._device) else "cpu"

        with torch.no_grad():
            for start in range(0, len(indexed_texts), self.batch_size):
                batch_tuples = indexed_texts[start: start + self.batch_size]
                batch_indices = [t[0] for t in batch_tuples]
                batch_texts = [t[1] for t in batch_tuples]

                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,  # Now padding is hyper-efficient because lengths are similar!
                    truncation=True,
                    max_length=self.max_sequence_length,
                    return_tensors="pt",
                ).to(self._device)

                # OPTIMIZATION: Activate 16-bit execution context (Safe 2x-3x speedup on GPU)
                with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                    output = self._model(**encoded)
                    hidden = output.last_hidden_state

                    # Mean Pooling Math
                    mask = encoded["attention_mask"].unsqueeze(-1).type_as(hidden)
                    summed = (hidden * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1e-9)
                    pooled = summed / counts

                    if self.normalize:
                        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

                # Convert to floats and match vectors back to original indices
                batch_vectors = pooled.cpu().tolist()
                for idx, vec in zip(batch_indices, batch_vectors):
                    vectors_dict[idx] = vec

        # Reconstruct the original list order requested by the caller
        return [vectors_dict[i] for i in range(len(texts))]

    def encode_one(self, text: str) -> List[float]:
        return self.encode([text])[0]