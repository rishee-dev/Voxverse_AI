"""VoxVerse translation package for English-to-Indic neural machine translation.

Owner: P2 — Speech Understanding Engineer (Translation & NLP)
"""

from __future__ import annotations

from voxverse.genai.translation.base import BaseTranslationEngine
from voxverse.genai.translation.exceptions import (
    EmptyInputError,
    InferenceError,
    ModelLoadError,
    PreprocessingError,
    TextLengthExceededError,
    TranslationError,
    UnsupportedLanguageError,
)
from voxverse.genai.translation.indictrans2_engine import IndicTrans2Engine
from voxverse.genai.translation.metrics import compute_translation_metrics
from voxverse.genai.translation.schemas import (
    EvaluationMetricsResult,
    SUPPORTED_SOURCE_LANGUAGES,
    SUPPORTED_TARGET_LANGUAGES,
    TranslationModelInfo,
    TranslationRequest,
    TranslationResult,
    TranslationSegment,
)
from voxverse.genai.translation.service import TranslationService
from voxverse.genai.translation.text_normalizer import IndicTransTextNormalizer

__all__ = [
    "BaseTranslationEngine",
    "EmptyInputError",
    "EvaluationMetricsResult",
    "IndicTrans2Engine",
    "IndicTransTextNormalizer",
    "InferenceError",
    "ModelLoadError",
    "PreprocessingError",
    "SUPPORTED_SOURCE_LANGUAGES",
    "SUPPORTED_TARGET_LANGUAGES",
    "TextLengthExceededError",
    "TranslationError",
    "TranslationModelInfo",
    "TranslationRequest",
    "TranslationResult",
    "TranslationSegment",
    "TranslationService",
    "UnsupportedLanguageError",
    "compute_translation_metrics",
]
