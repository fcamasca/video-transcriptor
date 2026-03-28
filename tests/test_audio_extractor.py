# Feature: video-transcriptor, Propiedad 3: Limpieza de archivos temporales
# Tests unitarios y de propiedad para AudioExtractor

import os
import subprocess
import tempfile
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_extractor import AudioExtractor
from exceptions import AudioExtractionError, FfmpegNotFoundError, NoAudioStreamError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_completed_process(returncode: int, stderr: bytes = b""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stderr = stderr
    proc.stdout = b""
    return proc


# ---------------------------------------------------------------------------
# Tests unitarios — 5.3
# ---------------------------------------------------------------------------

class TestAudioExtractorUnit:

    def test_ffmpeg_not_found_raises_ffmpeg_error(self):
        extractor = AudioExtractor()
        with patch("audio_extractor.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(FfmpegNotFoundError):
                with extractor.extract("video.mp4"):
                    pass

    def test_no_audio_stream_raises_no_audio_error(self):
        extractor = AudioExtractor()
        stderr = b"no audio stream found"
        proc = _make_completed_process(returncode=1, stderr=stderr)
        with patch("audio_extractor.subprocess.run", return_value=proc):
            with pytest.raises(NoAudioStreamError):
                with extractor.extract("video.mp4"):
                    pass

    def test_ffmpeg_generic_failure_raises_extraction_error(self):
        extractor = AudioExtractor()
        stderr = b"some unexpected ffmpeg error"
        proc = _make_completed_process(returncode=1, stderr=stderr)
        with patch("audio_extractor.subprocess.run", return_value=proc):
            with pytest.raises(AudioExtractionError):
                with extractor.extract("video.mp4"):
                    pass

    def test_temp_file_deleted_after_success(self):
        extractor = AudioExtractor()
        proc = _make_completed_process(returncode=0)
        captured_path = []

        with patch("audio_extractor.subprocess.run", return_value=proc):
            with extractor.extract("video.mp4") as wav_path:
                captured_path.append(wav_path)
                assert os.path.exists(wav_path)

        assert not os.path.exists(captured_path[0])

    def test_temp_file_deleted_after_error(self):
        extractor = AudioExtractor()
        proc = _make_completed_process(returncode=0)
        captured_path = []

        with patch("audio_extractor.subprocess.run", return_value=proc):
            with pytest.raises(RuntimeError):
                with extractor.extract("video.mp4") as wav_path:
                    captured_path.append(wav_path)
                    assert os.path.exists(wav_path)
                    raise RuntimeError("error dentro del bloque")

        assert not os.path.exists(captured_path[0])

    def test_temp_file_deleted_after_ffmpeg_failure(self):
        extractor = AudioExtractor()
        stderr = b"some unexpected ffmpeg error"
        proc = _make_completed_process(returncode=1, stderr=stderr)
        captured_path = []

        original_mkstemp = tempfile.mkstemp

        def patched_mkstemp(**kwargs):
            fd, path = original_mkstemp(**kwargs)
            captured_path.append(path)
            return fd, path

        with patch("audio_extractor.subprocess.run", return_value=proc):
            with patch("audio_extractor.tempfile.mkstemp", side_effect=patched_mkstemp):
                with pytest.raises(AudioExtractionError):
                    with extractor.extract("video.mp4"):
                        pass

        if captured_path:
            assert not os.path.exists(captured_path[0])


# ---------------------------------------------------------------------------
# Test de propiedad P3 — 5.2
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    succeed=st.booleans(),
    error_stage=st.sampled_from(["ffmpeg_fail", "inner_error", "no_audio"]),
)
def test_p3_temp_file_always_cleaned_up(succeed, error_stage):
    """
    Propiedad 3: el archivo temporal no existe en el sistema de archivos
    al finalizar el procesamiento del video, independientemente del resultado.
    """
    extractor = AudioExtractor()
    captured_path = []

    original_mkstemp = tempfile.mkstemp

    def patched_mkstemp(*args, **kwargs):
        fd, path = original_mkstemp(*args, **kwargs)
        captured_path.append(path)
        return fd, path

    if succeed:
        proc = _make_completed_process(returncode=0)
        with patch("audio_extractor.subprocess.run", return_value=proc):
            with patch("audio_extractor.tempfile.mkstemp", side_effect=patched_mkstemp):
                with extractor.extract("video.mp4"):
                    pass
    else:
        if error_stage == "ffmpeg_fail":
            proc = _make_completed_process(returncode=1, stderr=b"generic error")
            with patch("audio_extractor.subprocess.run", return_value=proc):
                with patch("audio_extractor.tempfile.mkstemp", side_effect=patched_mkstemp):
                    with pytest.raises(AudioExtractionError):
                        with extractor.extract("video.mp4"):
                            pass
        elif error_stage == "no_audio":
            proc = _make_completed_process(returncode=1, stderr=b"no audio stream")
            with patch("audio_extractor.subprocess.run", return_value=proc):
                with patch("audio_extractor.tempfile.mkstemp", side_effect=patched_mkstemp):
                    with pytest.raises(NoAudioStreamError):
                        with extractor.extract("video.mp4"):
                            pass
        else:  # inner_error
            proc = _make_completed_process(returncode=0)
            with patch("audio_extractor.subprocess.run", return_value=proc):
                with patch("audio_extractor.tempfile.mkstemp", side_effect=patched_mkstemp):
                    with pytest.raises(RuntimeError):
                        with extractor.extract("video.mp4"):
                            raise RuntimeError("error interno")

    if captured_path:
        assert not os.path.exists(captured_path[0]), (
            f"El archivo temporal {captured_path[0]} no fue eliminado"
        )
