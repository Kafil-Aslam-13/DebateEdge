"""Direct, lightweight replacements for Guardrails Hub validators.
No hub downloads, no Presidio's default large models — uses the
small spaCy model (en_core_web_sm) and a regex/wordlist profanity
filter instead of ML toxicity classifiers, to keep memory low.
"""

from src.core.logger import get_logger

logger = get_logger(__name__)


class DirectPIIGuard:
    """PII detection using Presidio directly with the small spaCy model."""

    def __init__(self, entities: list[str]):
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        # Force the small model instead of Presidio's default (often larger)
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        nlp_engine = provider.create_engine()

        self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        self._entities = entities

    def validate(self, text: str) -> None:
        results = self._analyzer.analyze(
            text=text, entities=self._entities, language="en"
        )
        if results:
            found = ", ".join(sorted({r.entity_type for r in results}))
            raise ValueError(f"PII detected: {found}")


class DirectToxicGuard:
    """Lightweight toxicity/profanity check — no ML model, near-zero memory."""

    def __init__(self, threshold: float = 0.5):
        from better_profanity import profanity
        profanity.load_censor_words()
        self._profanity = profanity

    def validate(self, text: str) -> None:
        if self._profanity.contains_profanity(text):
            raise ValueError("Toxic/profane language detected")


class DirectGibberishGuard:
    """Heuristic gibberish detection — character/word ratio based, no model."""

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold

    def validate(self, text: str) -> None:
        words = text.strip().split()
        if not words:
            raise ValueError("Empty text treated as gibberish")

        # Heuristic: ratio of alphabetic characters to total, and
        # ratio of words that contain at least one vowel
        alpha_chars = sum(c.isalpha() for c in text)
        total_chars = max(len(text), 1)
        alpha_ratio = alpha_chars / total_chars

        vowels = set("aeiouAEIOU")
        words_with_vowel = sum(1 for w in words if any(c in vowels for c in w))
        vowel_ratio = words_with_vowel / len(words)

        # Low alpha ratio or very few vowel-containing words → likely gibberish
        if alpha_ratio < 0.5 or vowel_ratio < 0.4:
            raise ValueError("Text appears to be gibberish")