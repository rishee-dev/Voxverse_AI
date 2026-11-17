"""Unit tests for Text-to-Speech (TTS) data models, exceptions, and base interface.

Owner: P3 — Voice Generation Engineer
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voxverse.genai.tts.base import BaseTTSEngine
from voxverse.genai.tts.exceptions import (
    MissingConsentError,
    MissingReferenceTranscriptError,
    ModelLoadError,
    ReferenceAudioError,
    SynthesisError,
    TeluguScriptValidationError,
    TTSError,
    UnsupportedEngineError,
    VoiceProfileNotFoundError,
)
from voxverse.genai.tts.schemas import (
    SpeakerConditioningData,
    TTSModelInfo,
    TTSRequest,
    TTSResult,
)


def test_speaker_conditioning_data_valid() -> None:
    data = SpeakerConditioningData(
        voice_id="voice-test-123",
        owner_id="user-456",
        speaker_name="Kalyan",
        reference_audio_path="/data/voices/ref.wav",
        reference_transcript="Hello, this is my authorized reference voice.",
        consent_hash="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    )
    assert data.voice_id == "voice-test-123"
    assert data.speaker_name == "Kalyan"
    assert data.sample_rate == 16000
    assert "authorized" in data.reference_transcript


def test_speaker_conditioning_data_missing_required() -> None:
    with pytest.raises(ValidationError):
        SpeakerConditioningData(
            voice_id="voice-test-123",
            owner_id="user-456",
            # Missing speaker_name, reference_audio_path, reference_transcript, consent_hash
        )


def test_tts_model_info_valid() -> None:
    info = TTSModelInfo(
        model_name="ai4bharat/IndicF5",
        backend="indicf5-flow-matching",
        device="cpu",
        output_sample_rate=24000,
        parameter_count=300000000,
        supports_cross_lingual_cloning=True,
    )
    assert info.model_name == "ai4bharat/IndicF5"
    assert info.backend == "indicf5-flow-matching"
    assert info.supports_cross_lingual_cloning is True
    assert info.output_sample_rate == 24000


def test_tts_request_valid() -> None:
    req = TTSRequest(
        text="నమస్కారం, ఇది వోక్స్ వర్స్ తెలుగు స్వర అనువాదం.",
        language="tel_Telu",
        voice_id="voice-abc-123",
        nfe_steps=32,
        speed=1.0,
    )
    assert req.text.startswith("నమస్కారం")
    assert req.voice_id == "voice-abc-123"
    assert req.nfe_steps == 32
    assert req.speed == 1.0


def test_tts_request_defaults() -> None:
    req = TTSRequest(text="హలో ప్రపంచం")
    assert req.language == "tel_Telu"
    assert req.voice_id is None
    assert req.nfe_steps == 32
    assert req.speed == 1.0


def test_tts_request_empty_text() -> None:
    with pytest.raises(ValidationError):
        TTSRequest(text="")


def test_tts_request_bounds() -> None:
    # nfe_steps must be between 8 and 64
    with pytest.raises(ValidationError):
        TTSRequest(text="నమస్కారం", nfe_steps=4)
    with pytest.raises(ValidationError):
        TTSRequest(text="నమస్కారం", nfe_steps=128)

    # speed must be between 0.5 and 2.0
    with pytest.raises(ValidationError):
        TTSRequest(text="నమస్కారం", speed=0.1)
    with pytest.raises(ValidationError):
        TTSRequest(text="నమస్కారం", speed=3.5)


def test_tts_result_valid() -> None:
    info = TTSModelInfo(
        model_name="ai4bharat/IndicF5",
        backend="indicf5-flow-matching",
        device="cpu",
        output_sample_rate=24000,
        parameter_count=300000000,
    )
    res = TTSResult(
        audio_path="/data/output/synth_1.wav",
        audio_duration_seconds=4.5,
        sample_rate=24000,
        channels=1,
        inference_duration_seconds=9.0,
        real_time_factor=2.0,
        model_info=info,
        speaker_similarity_score=0.82,
        peak_memory_mb=3150.0,
    )
    assert res.audio_duration_seconds == 4.5
    assert res.inference_duration_seconds == 9.0
    assert res.real_time_factor == 2.0
    assert res.speaker_similarity_score == 0.82
    assert res.peak_memory_mb == 3150.0


def test_tts_exceptions_hierarchy() -> None:
    base = TTSError("General TTS failure", details={"step": "vocoder"})
    assert isinstance(base, Exception)
    assert base.message == "General TTS failure"
    assert base.details["step"] == "vocoder"

    load_err = ModelLoadError("Weight file corrupt")
    assert isinstance(load_err, TTSError)

    not_found = VoiceProfileNotFoundError("v-999")
    assert isinstance(not_found, TTSError)
    assert not_found.voice_id == "v-999"
    assert "v-999" in str(not_found)

    consent_err = MissingConsentError("v-999", "Speaker revoked permission.")
    assert isinstance(consent_err, TTSError)
    assert consent_err.voice_id == "v-999"
    assert "revoked" in str(consent_err)

    ref_err = ReferenceAudioError("Audio duration too short (<3s)", path="/audio/short.wav")
    assert isinstance(ref_err, TTSError)
    assert ref_err.path == "/audio/short.wav"

    transcript_err = MissingReferenceTranscriptError("v-999")
    assert isinstance(transcript_err, TTSError)
    assert "reference transcript" in str(transcript_err)

    synth_err = SynthesisError("Euler ODE step divergence")
    assert isinstance(synth_err, TTSError)

    script_err = TeluguScriptValidationError("ABC 123", "Unrecognized non-Indic characters")
    assert isinstance(script_err, TTSError)
    assert script_err.reason == "Unrecognized non-Indic characters"

    engine_err = UnsupportedEngineError("unknown-tts", ["indicf5-flow-matching", "vits-mms"])
    assert isinstance(engine_err, TTSError)
    assert engine_err.engine_name == "unknown-tts"
    assert "vits-mms" in engine_err.supported


def test_base_tts_engine_contract() -> None:
    # Cannot instantiate abstract class directly
    with pytest.raises(TypeError):
        BaseTTSEngine("ai4bharat/IndicF5")  # type: ignore[abstract]

    class MockTTSEngine(BaseTTSEngine):
        def __init__(self) -> None:
            super().__init__(model_name="mock-indicf5", device="cpu")
            self._loaded = False

        def load(self) -> None:
            self._loaded = True

        def unload(self) -> None:
            self._loaded = False

        def is_loaded(self) -> bool:
            return self._loaded

        def get_model_info(self) -> TTSModelInfo:
            return TTSModelInfo(
                model_name=self.model_name,
                backend="mock",
                device=self.device,
                output_sample_rate=24000,
                parameter_count=300000000,
            )

        def synthesize(
            self,
            text: str,
            *,
            conditioning_data: SpeakerConditioningData | None = None,
            nfe_steps: int = 32,
            speed: float = 1.0,
            output_path: str | None = None,
        ) -> TTSResult:
            return TTSResult(
                audio_path=output_path or "/tmp/test.wav",
                audio_duration_seconds=3.0,
                sample_rate=24000,
                channels=1,
                inference_duration_seconds=6.0,
                real_time_factor=2.0,
                model_info=self.get_model_info(),
            )

    engine = MockTTSEngine()
    assert not engine.is_loaded()
    engine.load()
    assert engine.is_loaded()
    info = engine.get_model_info()
    assert info.backend == "mock"
    result = engine.synthesize("నమస్కారం")
    assert result.audio_duration_seconds == 3.0
    assert result.real_time_factor == 2.0
    engine.unload()
    assert not engine.is_loaded()
