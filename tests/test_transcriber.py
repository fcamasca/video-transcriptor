# Feature: video-transcriptor, Propiedad 4: Invariante de segmentos temporales
# Tests unitarios y de propiedad para Transcriber

import os
import struct
import tempfile
import wave
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exceptions import AudioTooShortError, TranscriptionError
from models import TranscriptionSegment
from transcriber import Transcriber


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(duration_seconds: float, sample_rate: int = 16000) -> str:
    """Crea un archivo WAV temporal con la duración indicada."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    n_frames = int(duration_seconds * sample_rate)
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return tmp_path


def _make_model(segments: list) -> MagicMock:
    """Crea un mock de modelo Whisper que retorna los segmentos dados."""
    model = MagicMock()
    model.transcribe.return_value = {"segments": segments}
    return model


def _whisper_seg(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text}


# ---------------------------------------------------------------------------
# Tests unitarios — 6.3
# ---------------------------------------------------------------------------

class TestTranscriberUnit:

    def test_audio_too_short_raises_error(self):
        wav = _make_wav(0.5)
        try:
            transcriber = Transcriber()
            with pytest.raises(AudioTooShortError):
                transcriber.transcribe(wav, model=MagicMock())
        finally:
            os.remove(wav)

    def test_audio_exactly_one_second_does_not_raise(self):
        wav = _make_wav(1.0)
        try:
            model = _make_model([_whisper_seg(0.0, 1.0, "hola")])
            transcriber = Transcriber()
            segments = transcriber.transcribe(wav, model)
            assert len(segments) == 1
        finally:
            os.remove(wav)

    def test_whisper_error_raises_transcription_error(self):
        wav = _make_wav(2.0)
        try:
            model = MagicMock()
            model.transcribe.side_effect = RuntimeError("whisper crash")
            transcriber = Transcriber()
            with pytest.raises(TranscriptionError):
                transcriber.transcribe(wav, model)
        finally:
            os.remove(wav)

    def test_segments_have_correct_start_end_text(self):
        wav = _make_wav(5.0)
        try:
            raw_segs = [
                _whisper_seg(0.0, 2.0, " hola mundo "),
                _whisper_seg(2.5, 4.0, " adiós "),
            ]
            model = _make_model(raw_segs)
            transcriber = Transcriber()
            segments = transcriber.transcribe(wav, model)

            assert segments[0].start == 0.0
            assert segments[0].end == 2.0
            assert segments[0].text == "hola mundo"

            assert segments[1].start == 2.5
            assert segments[1].end == 4.0
            assert segments[1].text == "adiós"
        finally:
            os.remove(wav)

    def test_empty_segments_returns_empty_list(self):
        wav = _make_wav(3.0)
        try:
            model = _make_model([])
            transcriber = Transcriber()
            segments = transcriber.transcribe(wav, model)
            assert segments == []
        finally:
            os.remove(wav)

    def test_speaker_is_none_by_default(self):
        wav = _make_wav(2.0)
        try:
            model = _make_model([_whisper_seg(0.0, 1.5, "texto")])
            transcriber = Transcriber()
            segments = transcriber.transcribe(wav, model)
            assert segments[0].speaker is None
        finally:
            os.remove(wav)


# ---------------------------------------------------------------------------
# Test de propiedad P4 — 6.2
# ---------------------------------------------------------------------------

# Estrategia: genera segmentos con start >= 0, end > start, text no vacío
_seg_strategy = st.builds(
    dict,
    start=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    end=st.floats(min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False),
    text=st.text(min_size=1, max_size=200),
).filter(lambda s: s["end"] > s["start"])


@settings(max_examples=100, deadline=None)
@given(raw_segments=st.lists(_seg_strategy, min_size=0, max_size=20))
def test_p4_segment_temporal_invariant(raw_segments):
    """
    Propiedad 4: todos los segmentos generados por el Transcriptor cumplen
    start >= 0, end > start y text no vacío.
    """
    wav = _make_wav(max(5.0, sum(s["end"] for s in raw_segments) + 1 if raw_segments else 5.0))
    try:
        model = _make_model(raw_segments)
        transcriber = Transcriber()
        segments = transcriber.transcribe(wav, model)

        for seg in segments:
            assert seg.start >= 0, f"start negativo: {seg.start}"
            assert seg.end > seg.start, f"end ({seg.end}) no mayor que start ({seg.start})"
            assert seg.text, f"text vacío en segmento {seg}"
    finally:
        os.remove(wav)
