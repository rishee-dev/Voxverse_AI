"""Evaluation metrics for machine translation quality assessment.

Uses SacreBLEU to compute standard corpus-level BLEU, chrF, and chrF++ scores.

Owner: P2 — Speech Understanding Engineer (Translation & NLP)
"""

from __future__ import annotations

import sacrebleu

from voxverse.genai.translation.schemas import EvaluationMetricsResult


def compute_translation_metrics(
    hypotheses: list[str],
    references: list[str],
    dataset_name: str = "FLORES-200",
    split: str = "devtest",
    duration_seconds: float = 0.0,
) -> EvaluationMetricsResult:
    """Compute SacreBLEU, chrF, and chrF++ scores for hypothesis translations.

    Parameters
    ----------
    hypotheses:
        List of generated translation strings.
    references:
        List of ground-truth reference translation strings.
    dataset_name:
        Name of evaluation benchmark dataset.
    split:
        Dataset partition split (e.g., 'devtest').
    duration_seconds:
        Total evaluation runtime in seconds.

    Returns
    -------
    EvaluationMetricsResult
        Calculated metric scores.

    Raises
    ------
    ValueError
        If hypotheses and references have mismatched lengths or are empty.
    """
    if len(hypotheses) != len(references):
        msg = f"Length mismatch: {len(hypotheses)} hypotheses vs {len(references)} references."
        raise ValueError(msg)

    if not hypotheses:
        msg = "Cannot compute metrics on empty lists."
        raise ValueError(msg)

    # SacreBLEU expects references as a list of reference sets: [ [ref_0, ref_1, ...] ]
    bleu_score = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf_score = sacrebleu.corpus_chrf(hypotheses, [references])
    chrf_pp_score = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)

    return EvaluationMetricsResult(
        dataset_name=dataset_name,
        split=split,
        sample_count=len(hypotheses),
        bleu=round(float(bleu_score.score), 2),
        chrf=round(float(chrf_score.score), 2),
        chrf_plus_plus=round(float(chrf_pp_score.score), 2),
        duration_seconds=round(duration_seconds, 2),
    )
