"""Domain-specific exceptions for Machine Translation.

Owner: P2 — Speech Understanding Engineer (Translation & NLP)
"""

from __future__ import annotations


class TranslationError(Exception):
    """Base exception for all machine translation errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ModelLoadError(TranslationError):
    """Raised when the translation model or tokenizer fails to load into memory."""


class UnsupportedLanguageError(TranslationError):
    """Raised when an unsupported language pair or tag is requested."""

    def __init__(self, language: str, supported: list[str]) -> None:
        message = f"Language '{language}' is not supported. Supported languages: {supported}"
        super().__init__(message, details={"language": language, "supported": supported})
        self.language = language
        self.supported = supported


class EmptyInputError(TranslationError):
    """Raised when the input text is empty or contains only whitespace."""

    def __init__(self, message: str = "Input text cannot be empty or purely whitespace.") -> None:
        super().__init__(message)


class TextLengthExceededError(TranslationError):
    """Raised when the input text exceeds the maximum permitted characters or tokens."""

    def __init__(self, length: int, max_length: int) -> None:
        message = f"Input length of {length} characters exceeds maximum allowed of {max_length}."
        super().__init__(message, details={"length": length, "max_length": max_length})
        self.length = length
        self.max_length = max_length


class InferenceError(TranslationError):
    """Raised when translation model inference fails during execution."""


class PreprocessingError(TranslationError):
    """Raised when text normalization or script transliteration fails."""
