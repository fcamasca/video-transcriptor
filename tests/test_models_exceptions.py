# tests/test_models_exceptions.py — Tests unitarios para models.py y exceptions.py

import pytest
from models import (
    ProcessingStatus, DuplicatePolicy, WhisperModel,
    TranscriptionSegment, AppConfig, VideoResult,
)
from exceptions import (
    AudioExtractionError, NoAudioStreamError,
    TranscriptionError, AudioTooShortError,
    DiarizationError, MissingHuggingFaceTokenError,
    OutputWriteError, FfmpegNotFoundError,
)


# --- Enums ---

class TestProcessingStatus:
    def test_valid_values(self):
        assert ProcessingStatus("OK") == ProcessingStatus.OK
        assert ProcessingStatus("SIN_AUDIO") == ProcessingStatus.SIN_AUDIO

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ProcessingStatus("INVALID")

    def test_is_str(self):
        assert isinstance(ProcessingStatus.OK, str)


class TestDuplicatePolicy:
    def test_valid_values(self):
        assert DuplicatePolicy("skip") == DuplicatePolicy.SKIP
        assert DuplicatePolicy("overwrite") == DuplicatePolicy.OVERWRITE

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            DuplicatePolicy("delete")

    def test_is_str(self):
        assert isinstance(DuplicatePolicy.SKIP, str)


class TestWhisperModel:
    def test_valid_values(self):
        for v in ("tiny", "base", "small", "medium", "large"):
            assert WhisperModel(v).value == v

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            WhisperModel("xlarge")

    def test_is_str(self):
        assert isinstance(WhisperModel.BASE, str)


# --- Dataclasses ---

class TestAppConfig:
    def test_defaults(self):
        cfg = AppConfig()
        assert cfg.input_dir == "Videos/"
        assert cfg.output_dir == "Transcripcion/"
        assert cfg.language == "es"
        assert cfg.diarization is True
        assert cfg.duplicate_policy == DuplicatePolicy.SKIP
        assert cfg.log_file is None
        assert cfg.whisper_model == WhisperModel.BASE

    def test_custom_values(self):
        cfg = AppConfig(
            input_dir="in/",
            output_dir="out/",
            language="en",
            diarization=False,
            duplicate_policy=DuplicatePolicy.OVERWRITE,
            log_file="app.log",
            whisper_model=WhisperModel.SMALL,
        )
        assert cfg.language == "en"
        assert cfg.diarization is False
        assert cfg.duplicate_policy == DuplicatePolicy.OVERWRITE
        assert cfg.whisper_model == WhisperModel.SMALL


class TestTranscriptionSegment:
    def test_required_fields(self):
        seg = TranscriptionSegment(start=0.0, end=1.5, text="Hola")
        assert seg.start == 0.0
        assert seg.end == 1.5
        assert seg.text == "Hola"
        assert seg.speaker is None

    def test_with_speaker(self):
        seg = TranscriptionSegment(start=1.0, end=3.0, text="Texto", speaker="Speaker 1")
        assert seg.speaker == "Speaker 1"


class TestVideoResult:
    def test_defaults(self):
        result = VideoResult(video_name="video.mp4", status=ProcessingStatus.OK)
        assert result.segments == []
        assert result.error_message is None

    def test_with_error(self):
        result = VideoResult(
            video_name="video.mp4",
            status=ProcessingStatus.ERROR_EXTRACCION_AUDIO,
            error_message="ffmpeg falló",
        )
        assert result.error_message == "ffmpeg falló"


# --- Jerarquía de excepciones ---

class TestExceptionHierarchy:
    def test_no_audio_stream_is_audio_extraction(self):
        assert issubclass(NoAudioStreamError, AudioExtractionError)
        assert issubclass(NoAudioStreamError, Exception)

    def test_audio_too_short_is_transcription(self):
        assert issubclass(AudioTooShortError, TranscriptionError)
        assert issubclass(AudioTooShortError, Exception)

    def test_missing_token_is_diarization(self):
        assert issubclass(MissingHuggingFaceTokenError, DiarizationError)
        assert issubclass(MissingHuggingFaceTokenError, Exception)

    def test_output_write_error_is_exception(self):
        assert issubclass(OutputWriteError, Exception)

    def test_ffmpeg_not_found_is_exception(self):
        assert issubclass(FfmpegNotFoundError, Exception)

    def test_exceptions_are_catchable_by_parent(self):
        with pytest.raises(AudioExtractionError):
            raise NoAudioStreamError("sin audio")

        with pytest.raises(TranscriptionError):
            raise AudioTooShortError("muy corto")

        with pytest.raises(DiarizationError):
            raise MissingHuggingFaceTokenError("token ausente")
