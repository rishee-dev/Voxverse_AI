"""Unit tests for translation text normalization, entity protection, and script transliteration.

Owner: P2 — Speech Understanding Engineer
"""

from __future__ import annotations

from voxverse.genai.translation.text_normalizer import IndicTransTextNormalizer


def test_clean_whitespace() -> None:
    normalizer = IndicTransTextNormalizer()
    raw = "   Hello,   world!   \n This is a    test.   "
    cleaned = normalizer.clean_whitespace(raw)
    assert cleaned == "Hello, world! This is a test."


def test_split_into_sentences_simple() -> None:
    normalizer = IndicTransTextNormalizer()
    text = "First sentence. Second sentence! Is this the third sentence?"
    sents = normalizer.split_into_sentences(text)
    assert len(sents) == 3
    assert sents[0] == "First sentence."
    assert sents[1] == "Second sentence!"
    assert sents[2] == "Is this the third sentence?"


def test_split_into_sentences_with_abbreviations_and_decimals() -> None:
    normalizer = IndicTransTextNormalizer()
    text = (
        "Dr. Rao met Mr. Smith at the hospital. "
        "The sensor recorded 3.14 units of output! "
        "Did Prof. Sharma confirm the results? Yes, he did."
    )
    sents = normalizer.split_into_sentences(text)
    assert len(sents) == 4
    assert sents[0] == "Dr. Rao met Mr. Smith at the hospital."
    assert sents[1] == "The sensor recorded 3.14 units of output!"
    assert sents[2] == "Did Prof. Sharma confirm the results?"
    assert sents[3] == "Yes, he did."


def test_split_into_sentences_empty() -> None:
    normalizer = IndicTransTextNormalizer()
    assert normalizer.split_into_sentences("") == []
    assert normalizer.split_into_sentences("   \t\n  ") == []


def test_wrap_placeholders_and_restore() -> None:
    normalizer = IndicTransTextNormalizer()
    text = "Contact support@voxverse.ai or visit https://voxverse.ai/docs for assistance."
    wrapped, mapping = normalizer.wrap_placeholders(text)

    assert "support@voxverse.ai" not in wrapped
    assert "https://voxverse.ai/docs" not in wrapped
    assert "<ID" in wrapped

    # Restore placeholders
    restored = normalizer.postprocess_sentence(wrapped, target_language="tel_Telu", placeholder_map=mapping)
    assert "support@voxverse.ai" in restored
    assert "https://voxverse.ai/docs" in restored


def test_preprocess_sentence_structure() -> None:
    normalizer = IndicTransTextNormalizer()
    formatted, p_map = normalizer.preprocess_sentence(
        "Hello, world!",
        source_language="eng_Latn",
        target_language="tel_Telu",
    )
    assert formatted.startswith("eng_Latn tel_Telu")
    assert "Hello" in formatted
    assert isinstance(p_map, dict)


def test_postprocess_sentence_transliteration_to_telugu() -> None:
    normalizer = IndicTransTextNormalizer()
    # Devanagari representation produced by IndicTrans2
    devanagari_output = "हलो मीरु ऎला उन्नारु ?"
    telugu_output = normalizer.postprocess_sentence(
        devanagari_output,
        target_language="tel_Telu",
    )
    # Check that output is in Telugu script (Unicode range \u0C00 - \u0C7F)
    assert any("\u0c00" <= char <= "\u0c7f" for char in telugu_output)
    assert "హలో" in telugu_output
    assert "మీరు" in telugu_output
