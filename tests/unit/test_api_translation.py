"""Unit tests for POST /api/v1/translate endpoint.

Owner: P4 — Pipeline Engineer with P2 — Speech Understanding Engineer
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from voxverse.api.app import app
from voxverse.api.routes.translation import get_translation_service
from voxverse.genai.translation.base import BaseTranslationEngine
from voxverse.genai.translation.exceptions import InferenceError
from voxverse.genai.translation.schemas import TranslationModelInfo
from voxverse.genai.translation.service import TranslationService


class MockAPITranslationEngine(BaseTranslationEngine):
    """Predictable mock engine for API route unit testing."""

    def __init__(self, should_fail: bool = False) -> None:
        super().__init__(model_name="mock-api-engine", device="cpu")
        self.should_fail = should_fail

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def is_loaded(self) -> bool:
        return True

    def get_model_info(self) -> TranslationModelInfo:
        return TranslationModelInfo(
            model_name="mock-api-engine",
            backend="mock",
            device="cpu",
            compute_type="float32",
            parameter_count=1000,
            max_sequence_length=256,
        )

    def translate(
        self,
        text: str,
        *,
        source_language: str = "eng_Latn",
        target_language: str = "tel_Telu",
        beam_size: int = 4,
        max_length: int = 256,
    ) -> str:
        return self.translate_batch([text])[0]

    def translate_batch(
        self,
        texts: list[str],
        *,
        source_language: str = "eng_Latn",
        target_language: str = "tel_Telu",
        beam_size: int = 4,
        max_length: int = 256,
    ) -> list[str]:
        if self.should_fail:
            raise InferenceError("Engine exploded")
        return [f"[MOCK_TE: {t}]" for t in texts]


@pytest.fixture
def client_with_mock_translation() -> TestClient:
    engine = MockAPITranslationEngine()
    service = TranslationService(engine=engine)
    app.dependency_overrides[get_translation_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def client_with_failing_translation() -> TestClient:
    engine = MockAPITranslationEngine(should_fail=True)
    service = TranslationService(engine=engine)
    app.dependency_overrides[get_translation_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_api_translate_success(client_with_mock_translation: TestClient) -> None:
    response = client_with_mock_translation.post(
        "/api/v1/translate",
        json={
            "text": "Good morning, friend!",
            "source_language": "eng_Latn",
            "target_language": "tel_Telu",
            "beam_size": 4,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source_text"] == "Good morning, friend!"
    assert data["translated_text"] == "[MOCK_TE: Good morning, friend!]"
    assert data["source_language"] == "eng_Latn"
    assert data["target_language"] == "tel_Telu"
    assert len(data["segments"]) == 1
    assert data["model_info"]["backend"] == "mock"


def test_api_translate_empty_text_rejected(client_with_mock_translation: TestClient) -> None:
    response = client_with_mock_translation.post(
        "/api/v1/translate",
        json={
            "text": "",
            "source_language": "eng_Latn",
            "target_language": "tel_Telu",
        },
    )
    # Pydantic validation rejects min_length=1 with 422
    assert response.status_code == 422


def test_api_translate_whitespace_only_rejected(client_with_mock_translation: TestClient) -> None:
    response = client_with_mock_translation.post(
        "/api/v1/translate",
        json={
            "text": "   \n\t  ",
            "source_language": "eng_Latn",
            "target_language": "tel_Telu",
        },
    )
    # Service validation raises EmptyInputError which maps to 400
    assert response.status_code == 400


def test_api_translate_unsupported_language(client_with_mock_translation: TestClient) -> None:
    response = client_with_mock_translation.post(
        "/api/v1/translate",
        json={
            "text": "Hello",
            "source_language": "invalid_lang",
            "target_language": "tel_Telu",
        },
    )
    assert response.status_code == 422


def test_api_translate_internal_failure(client_with_failing_translation: TestClient) -> None:
    response = client_with_failing_translation.post(
        "/api/v1/translate",
        json={
            "text": "Hello world",
            "source_language": "eng_Latn",
            "target_language": "tel_Telu",
        },
    )
    assert response.status_code == 500
    assert "Translation inference failed" in response.json()["detail"]
