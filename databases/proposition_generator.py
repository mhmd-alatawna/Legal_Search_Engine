"""
LLM-based proposition generator.

Converts the raw token windows extracted around `<FRAGMENT_SUPPRESSED>`
citation tags into coherent, self-contained legal propositions using a
locally hosted Gemma instruction-tuned model with few-shot prompting.

The model is loaded lazily: constructing the class is free, and the data
layer can be built end-to-end with LLM refinement disabled.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from databases.config import LLMConfig

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a legal text analyst. You receive an excerpt from a court "
    "decision in which a citation to a precedent case has been removed. "
    "Rewrite the excerpt as a single, coherent, self-contained legal "
    "proposition: the legal rule, standard or holding that the removed "
    "citation supported. Requirements:\n"
    "- Output exactly one proposition and nothing else.\n"
    "- Do not mention that a citation was removed.\n"
    "- Do not invent case names, parties or facts not present in the excerpt.\n"
    "- Keep the proposition under 60 words.\n"
)

# Few-shot examples (in-context learning) steering the model toward short,
# rule-like outputs rather than summaries of the excerpt.
_FEW_SHOT_EXAMPLES = [
    (
        "The respondent argues that the application is moot. As held in "
        "the decision to which we are referred, a matter is moot when a "
        "decision will not have the effect of resolving a live controversy "
        "affecting the rights of the parties.",
        "A matter is moot when a court decision will not resolve a live "
        "controversy that affects the rights of the parties.",
    ),
    (
        "Counsel submits that the standard of review on questions of "
        "procedural fairness is correctness, and that no deference is owed "
        "to the tribunal on such questions.",
        "Questions of procedural fairness are reviewed on a standard of "
        "correctness, with no deference owed to the tribunal.",
    ),
    (
        "It is well established that the onus rests on the applicant to "
        "establish a serious issue to be tried, irreparable harm, and that "
        "the balance of convenience favours granting the injunction.",
        "An applicant for an injunction must establish a serious issue to "
        "be tried, irreparable harm, and a balance of convenience in favour "
        "of granting the injunction.",
    ),
]


class GemmaPropositionGenerator:
    """Few-shot proposition generation with a Gemma instruction model."""

    def __init__(self, config: LLMConfig):
        self._config = config
        self._pipeline = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return

        import torch
        from transformers import pipeline

        device = self._config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(
            "Loading LLM '%s' on device '%s'...", self._config.model_name, device
        )
        self._pipeline = pipeline(
            "text-generation",
            model=self._config.model_name,
            device=device,
            torch_dtype="auto",
        )
        logger.info("LLM ready.")

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(context: str) -> List[dict]:
        """Build a chat-format prompt with the system instructions and
        few-shot examples.

        Gemma has no dedicated system role, so the system prompt is
        prepended to the first user turn.
        """
        messages: List[dict] = []
        first_excerpt, first_proposition = _FEW_SHOT_EXAMPLES[0]
        messages.append(
            {
                "role": "user",
                "content": f"{_SYSTEM_PROMPT}\nExcerpt:\n{first_excerpt}",
            }
        )
        messages.append({"role": "assistant", "content": first_proposition})
        for excerpt, proposition in _FEW_SHOT_EXAMPLES[1:]:
            messages.append({"role": "user", "content": f"Excerpt:\n{excerpt}"})
            messages.append({"role": "assistant", "content": proposition})
        messages.append({"role": "user", "content": f"Excerpt:\n{context}"})
        return messages

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, context: str) -> Optional[str]:
        """Generate one legal proposition from a citation context.

        Returns None when the context is empty or generation fails, so a
        batch build can fall back to the raw context and keep going.
        """
        context = (context or "").strip()
        if not context:
            return None
        self._ensure_loaded()

        try:
            output = self._pipeline(
                self._build_messages(context),
                max_new_tokens=self._config.max_new_tokens,
                do_sample=self._config.temperature > 0,
                temperature=max(self._config.temperature, 1e-5),
                return_full_text=False,
            )
            proposition = output[0]["generated_text"].strip()
            return proposition or None
        except Exception:
            logger.exception("Proposition generation failed; returning None.")
            return None

    def generate_batch(self, contexts: List[str]) -> List[Optional[str]]:
        """Generate propositions for a list of contexts, preserving order."""
        return [self.generate(context) for context in contexts]
