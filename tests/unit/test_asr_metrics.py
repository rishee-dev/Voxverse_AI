"""Unit tests for ASR evaluation metrics (WER, CER, normalization).

Owner: P2 — Speech Understanding Engineer
"""

from __future__ import annotations

import pytest

from voxverse.genai.asr.metrics import (
    calculate_cer,
    calculate_wer,
    normalize_transcript,
)


def test_normalize_transcript_basic():
    text = "Hello, World! This is a TEST."
    expected = "hello world this is a test"
    assert normalize_transcript(text) == expected


def test_normalize_transcript_contractions():
    text = "We didn't know it wasn't working."
    expected = "we didn't know it wasn't working"
    assert normalize_transcript(text) == expected


def test_normalize_transcript_empty_and_whitespace():
    assert normalize_transcript("") == ""
    assert normalize_transcript("   \n\t  ") == ""


def test_calculate_wer_identical():
    ref = "The quick brown fox jumps over the lazy dog"
    hyp = "The quick brown fox jumps over the lazy dog"
    wer, s, d, i, n = calculate_wer(ref, hyp)
    assert wer == 0.0
    assert s == 0 and d == 0 and i == 0
    assert n == 9


def test_calculate_wer_with_punctuation_difference():
    ref = "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS"
    hyp = "Concord returned to its place amidst the tents."
    wer, s, d, i, n = calculate_wer(ref, hyp, normalize=True)
    assert wer == 0.0
    assert n == 8


def test_calculate_wer_substitution():
    ref = "red green blue"
    hyp = "red yellow blue"
    wer, s, d, i, n = calculate_wer(ref, hyp)
    assert wer == pytest.approx(1 / 3)
    assert s == 1 and d == 0 and i == 0
    assert n == 3


def test_calculate_wer_deletion():
    ref = "one two three four"
    hyp = "one three four"
    wer, s, d, i, n = calculate_wer(ref, hyp)
    assert wer == pytest.approx(1 / 4)
    assert s == 0 and d == 1 and i == 0
    assert n == 4


def test_calculate_wer_insertion():
    ref = "one two three"
    hyp = "one extra two three"
    wer, s, d, i, n = calculate_wer(ref, hyp)
    assert wer == pytest.approx(1 / 3)
    assert s == 0 and d == 0 and i == 1
    assert n == 3


def test_calculate_wer_empty_reference():
    wer, s, d, i, n = calculate_wer("", "")
    assert wer == 0.0

    wer, s, d, i, n = calculate_wer("", "some hypothesis")
    assert wer == 1.0


def test_calculate_cer_basic():
    ref = "cat"
    hyp = "bat"
    cer, s, d, i, n = calculate_cer(ref, hyp)
    assert cer == pytest.approx(1 / 3)
    assert s == 1 and d == 0 and i == 0
    assert n == 3


def test_calculate_cer_identical():
    ref = "hello world"
    hyp = "hello world"
    cer, s, d, i, n = calculate_cer(ref, hyp)
    assert cer == 0.0
