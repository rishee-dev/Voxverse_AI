"""Evaluation metrics for speech recognition: Word Error Rate (WER) and Character Error Rate (CER).

Owner: P2 — Speech Understanding Engineer

Standard formula:
    WER = (Substitutions + Deletions + Insertions) / Total Reference Words
    CER = (Substitutions + Deletions + Insertions) / Total Reference Characters
"""

from __future__ import annotations

import re
import string


def normalize_transcript(text: str) -> str:
    """Normalize transcript text for objective speech recognition evaluation.

    Normalization steps:
    1. Lowercase all text.
    2. Remove standard punctuation (period, comma, exclamation, question mark, etc.).
    3. Retain alphanumeric characters, hyphens within words, and apostrophes.
    4. Normalize all contiguous whitespace sequences to a single space.
    5. Strip leading and trailing whitespace.

    Invariants:
    - Never silently removes numbers or vocabulary words.
    - Deterministic across all platforms.
    """
    if not text:
        return ""

    lowered = text.lower()
    # Replace punctuation characters with whitespace, except apostrophes inside words
    # e.g. "don't" -> "don't", but "hello, world!" -> "hello world"
    cleaned = re.sub(r"[^\w\s\']", " ", lowered)
    # Remove isolated apostrophes
    cleaned = re.sub(r"(?<!\w)\'|\'(?!\w)", " ", cleaned)
    # Collapse multiple whitespace
    normalized = " ".join(cleaned.split())
    return normalized


def _levenshtein_alignment[T](ref: list[T], hyp: list[T]) -> tuple[int, int, int]:
    """Compute (Substitutions, Deletions, Insertions) using dynamic programming.

    Parameters
    ----------
    ref:
        Reference token sequence.
    hyp:
        Hypothesis token sequence.

    Returns
    -------
    tuple[int, int, int]
        (substitutions, deletions, insertions)
    """
    n = len(ref)
    m = len(hyp)

    # dp[i][j] stores (cost, S, D, I)
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)  # i deletions

    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)  # j insertions

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Option 1: Substitution
                sub_cost = dp[i - 1][j - 1][0] + 1
                sub_tuple = (sub_cost, dp[i - 1][j - 1][1] + 1, dp[i - 1][j - 1][2], dp[i - 1][j - 1][3])

                # Option 2: Deletion (token in ref missing in hyp)
                del_cost = dp[i - 1][j][0] + 1
                del_tuple = (del_cost, dp[i - 1][j][1], dp[i - 1][j][2] + 1, dp[i - 1][j][3])

                # Option 3: Insertion (token in hyp extra)
                ins_cost = dp[i][j - 1][0] + 1
                ins_tuple = (ins_cost, dp[i][j - 1][1], dp[i][j - 1][2], dp[i][j - 1][3] + 1)

                best = min((sub_tuple, del_tuple, ins_tuple), key=lambda x: x[0])
                dp[i][j] = best

    _, s, d, ins = dp[n][m]
    return s, d, ins


def calculate_wer(
    reference: str,
    hypothesis: str,
    *,
    normalize: bool = True,
) -> tuple[float, int, int, int, int]:
    """Calculate Word Error Rate (WER) between reference and hypothesis.

    Parameters
    ----------
    reference:
        Ground truth reference text.
    hypothesis:
        Model transcription output.
    normalize:
        If True, applies normalize_transcript before tokenizing.

    Returns
    -------
    tuple[float, int, int, int, int]
        (wer, substitutions, deletions, insertions, reference_word_count)
    """
    ref_text = normalize_transcript(reference) if normalize else reference.strip()
    hyp_text = normalize_transcript(hypothesis) if normalize else hypothesis.strip()

    ref_words = ref_text.split() if ref_text else []
    hyp_words = hyp_text.split() if hyp_text else []

    n_ref = len(ref_words)
    if n_ref == 0:
        if len(hyp_words) == 0:
            return 0.0, 0, 0, 0, 0
        return 1.0, 0, 0, len(hyp_words), 0

    s, d, i = _levenshtein_alignment(ref_words, hyp_words)
    wer = (s + d + i) / float(n_ref)
    return wer, s, d, i, n_ref


def calculate_cer(
    reference: str,
    hypothesis: str,
    *,
    normalize: bool = True,
) -> tuple[float, int, int, int, int]:
    """Calculate Character Error Rate (CER) between reference and hypothesis.

    Parameters
    ----------
    reference:
        Ground truth reference text.
    hypothesis:
        Model transcription output.
    normalize:
        If True, applies normalize_transcript before character sequence extraction.

    Returns
    -------
    tuple[float, int, int, int, int]
        (cer, substitutions, deletions, insertions, reference_char_count)
    """
    ref_text = normalize_transcript(reference) if normalize else reference.strip()
    hyp_text = normalize_transcript(hypothesis) if normalize else hypothesis.strip()

    ref_chars = list(ref_text)
    hyp_chars = list(hyp_text)

    n_ref = len(ref_chars)
    if n_ref == 0:
        if len(hyp_chars) == 0:
            return 0.0, 0, 0, 0, 0
        return 1.0, 0, 0, len(hyp_chars), 0

    s, d, i = _levenshtein_alignment(ref_chars, hyp_chars)
    cer = (s + d + i) / float(n_ref)
    return cer, s, d, i, n_ref
