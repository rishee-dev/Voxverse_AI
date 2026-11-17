"""API routes for English-to-Indic text translation.

Owner: P4 — Pipeline Engineer with P2 — Speech Understanding Engineer
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from voxverse.config.logging import get_logger
from voxverse.genai.translation.exceptions import (
    EmptyInputError,
    InferenceError,
    TranslationError,
    UnsupportedLanguageError,
)
from voxverse.genai.translation.schemas import (
    TranslationRequest,
    TranslationResult,
)
from voxverse.genai.translation.service import TranslationService

logger = get_logger("api.routes.translation")

router = APIRouter(prefix="/api/v1/translate", tags=["Translation"])

# Singleton translation service instance for the API process
_translation_service: TranslationService | None = None


def get_translation_service() -> TranslationService:
    """Return the application-wide TranslationService singleton."""
    global _translation_service  # noqa: PLW0603
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service


@router.post(
    "",
    response_model=TranslationResult,
    status_code=status.HTTP_200_OK,
    summary="Translate text from English to Indic languages (default Telugu)",
)
async def translate_text_endpoint(
    request: TranslationRequest,
    service: TranslationService = Depends(get_translation_service),
) -> TranslationResult:
    """Translate English text into Telugu using IndicTrans2.

    Parameters
    ----------
    request:
        TranslationRequest containing source text and optional source/target language codes.
    service:
        Injected TranslationService singleton.

    Returns
    -------
    TranslationResult
        Translated Telugu text and performance telemetry.
    """
    try:
        return service.translate_text(
            request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            beam_size=request.beam_size,
        )
    except EmptyInputError as e:
        logger.warning("Empty text submitted for translation: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except UnsupportedLanguageError as e:
        logger.warning("Unsupported language requested: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except InferenceError as e:
        logger.error("Inference failure during translation: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation inference failed: {e}",
        ) from e
    except TranslationError as e:
        logger.error("Unexpected translation error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {e}",
        ) from e
