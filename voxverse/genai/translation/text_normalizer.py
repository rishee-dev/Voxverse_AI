"""Text normalization, preprocessing, and script transliteration for IndicTrans2.

Pure Python implementation decoupling inference from fragile Cython extensions.

Owner: P2 — Speech Understanding Engineer (Translation & NLP)
"""

from __future__ import annotations

import re
from typing import ClassVar

from indicnlp.tokenize import indic_detokenize
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer

from voxverse.config.logging import get_logger

logger = get_logger("translation.normalizer")


class IndicTransTextNormalizer:
    """Preprocesses English text and postprocesses translated Telugu text for IndicTrans2.

    Handles:
    - Normalization of whitespace and punctuation.
    - Protection of entities (URLs, emails, numerals) via placeholder tags.
    - Sentence segmentation for multi-sentence inputs.
    - Transliteration from unified Devanagari representation to Telugu script.
    - Post-generation detokenization and placeholder restoration.
    """

    COMMON_ABBREVIATIONS: ClassVar[list[str]] = [
        "Dr.",
        "Mr.",
        "Mrs.",
        "Ms.",
        "Prof.",
        "e.g.",
        "i.e.",
        "etc.",
        "vs.",
        "approx.",
        "Dept.",
        "Govt.",
        "Gen.",
        "Col.",
        "St.",
        "U.S.",
        "U.K.",
        "No.",
    ]

    # Precompiled regex patterns
    URL_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"\b(?<![\w/.])(?:(?:https?|ftp)://)?(?:(?:[\w-]+\.)+(?!\.))(?:[\w/\-?#&=%.]+)+(?!\.\w+)\b"
    )
    EMAIL_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}"
    )
    MULTISPACE_PATTERN: ClassVar[re.Pattern] = re.compile(r"\s+")

    def __init__(self) -> None:
        self.en_normalizer = MosesPunctNormalizer()
        self.en_tokenizer = MosesTokenizer(lang="en")
        self.en_detokenizer = MosesDetokenizer(lang="en")
        self.transliterator = UnicodeIndicTransliterator()

    def clean_whitespace(self, text: str) -> str:
        """Strip leading/trailing whitespace and collapse repeated whitespace."""
        return self.MULTISPACE_PATTERN.sub(" ", text).strip()

    def split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences while protecting abbreviations and decimals.

        Ensures long multi-sentence paragraphs are split on legitimate boundaries
        rather than truncated.
        """
        cleaned = self.clean_whitespace(text)
        if not cleaned:
            return []

        # 1. Mask abbreviations
        masked = cleaned
        for i, abbr in enumerate(self.COMMON_ABBREVIATIONS):
            masked = masked.replace(abbr, f"__ABBR_{i}__")

        # 2. Mask decimal numbers (e.g., 3.14)
        masked = re.sub(r"(\d+)\.(\d+)", r"\1__DECIMAL_DOT__\2", masked)

        # 3. Split on sentence-final punctuation followed by whitespace
        raw_parts = re.split(r"(?<=[.!?])\s+", masked)

        sentences: list[str] = []
        for part in raw_parts:
            s = part.strip()
            if not s:
                continue

            # Restore decimal dots and abbreviations
            s = s.replace("__DECIMAL_DOT__", ".")
            for i, abbr in enumerate(self.COMMON_ABBREVIATIONS):
                s = s.replace(f"__ABBR_{i}__", abbr)

            sentences.append(s)

        return sentences

    def wrap_placeholders(self, text: str) -> tuple[str, dict[str, str]]:
        """Protect URLs, emails, and technical identifiers with placeholder tags.

        Returns
        -------
        tuple[str, dict[str, str]]
            (wrapped_text, placeholder_to_entity_map)
        """
        mapping: dict[str, str] = {}
        serial_no = 1
        wrapped = text

        # 1. Protect emails
        for match in set(self.EMAIL_PATTERN.findall(wrapped)):
            tag = f"<ID{serial_no}>"
            mapping[tag] = match
            wrapped = wrapped.replace(match, tag)
            serial_no += 1

        # 2. Protect URLs
        for match in set(self.URL_PATTERN.findall(wrapped)):
            if len(match.replace(".", "")) >= 4:
                tag = f"<ID{serial_no}>"
                mapping[tag] = match
                wrapped = wrapped.replace(match, tag)
                serial_no += 1

        return wrapped, mapping

    def preprocess_sentence(
        self,
        sentence: str,
        source_language: str = "eng_Latn",
        target_language: str = "tel_Telu",
    ) -> tuple[str, dict[str, str]]:
        """Preprocess an English sentence for IndicTrans2 input.

        Punctuation normalization -> Entity placeholder wrapping -> Moses tokenization
        -> Prepend language tags.
        """
        cleaned = self.clean_whitespace(sentence)

        # Punctuation normalization
        normed = self.en_normalizer.normalize(cleaned)

        # Entity protection
        wrapped, placeholder_map = self.wrap_placeholders(normed)

        # Moses tokenization
        tokens = self.en_tokenizer.tokenize(wrapped, escape=False)
        tokenized_sentence = " ".join(tokens)

        # Prepend FLORES language tags as required by IndicTrans2
        formatted = f"{source_language} {target_language} {tokenized_sentence}"
        return formatted, placeholder_map

    def postprocess_sentence(
        self,
        raw_output: str,
        target_language: str = "tel_Telu",
        placeholder_map: dict[str, str] | None = None,
    ) -> str:
        """Convert model output (Devanagari intermediate representation) to Telugu script.

        Transliterates Devanagari to Telugu Unicode script, applies Telugu detokenization,
        and restores protected entity placeholders.
        """
        # If output is for Telugu, convert Devanagari script to Telugu script
        if target_language == "tel_Telu":
            # Transliterate from Devanagari ("hi") to Telugu ("te")
            telugu_text = self.transliterator.transliterate(raw_output, "hi", "te")
            # Clean spacing around Telugu punctuation
            detokenized = indic_detokenize.trivial_detokenize(telugu_text, "te")
        else:
            detokenized = raw_output

        # Restore entity placeholders (URLs, emails, numerals)
        if placeholder_map:
            for tag, original_val in placeholder_map.items():
                detokenized = detokenized.replace(tag, original_val)
                # Also handle variants where whitespace might have been inserted into the tag
                detokenized = detokenized.replace(f"< {tag[1:-1]} >", original_val)
                detokenized = detokenized.replace(f"[{tag[1:-1]}]", original_val)

        return self.clean_whitespace(detokenized)
