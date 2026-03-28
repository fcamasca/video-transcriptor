# Feature: video-transcriptor, Propiedad 5: Etiquetas de hablante válidas
# Tests unitarios y de propiedad para Diarizer

import os
import re
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diarizer import Diarizer
from exceptions import DiarizationError, MissingHuggingFaceTokenError
from models import TranscriptionSegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(start: float, end: float, text: str = "texto") -> TranscriptionSegment:
    return TranscriptionSegment(start=start, end=end, text=text)


def _diar_turn(start: float, end: float, label: str):
    """Crea un mock de (turn, _, label) como retorna pyannote itertracks."""
    turn = MagicMock()
    turn.start = start
    turn.end = end
    return (turn, None, label)


def _make_pipeline_mock(diar_turns: list):
    """Crea un mock de pyannote Pipeline que retorna los turns dados."""
    diarization = MagicMock()
    diarization.itertracks.return_value = diar_turns

    pipeline_instance = MagicMock()
    pipeline_instance.return_value = diarization

    pipeline_cls = MagicMock()
    pipeline_cls.from_pretrained.return_value = pipeline_instance
    return pipeline_cls


# ---------------------------------------------------------------------------
# Tests unitarios — 8.3
# ---------------------------------------------------------------------------

class TestDiarizerUnit:

    def test_missing_token_raises_error(self, monkeypatch):
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
        diarizer = Diarizer()
        with pytest.raises(MissingHuggingFaceTokenError):
            diarizer.diarize("audio.wav", [_seg(0.0, 2.0)])

    def test_pyannote_not_installed_raises_diarization_error(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "fake-token")
        diarizer = Diarizer()
        with patch("diarizer.Pipeline", None):
            with pytest.raises(DiarizationError, match="pyannote.audio no está instalado"):
                diarizer.diarize("audio.wav", [_seg(0.0, 2.0)])

    def test_pyannote_failure_raises_diarization_error(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "fake-token")
        diarizer = Diarizer()
        with patch("diarizer.Pipeline") as mock_pipeline_cls:
            mock_pipeline_cls.from_pretrained.side_effect = RuntimeError("pyannote crash")
            with pytest.raises(DiarizationError):
                diarizer.diarize("audio.wav", [_seg(0.0, 2.0)])

    def test_segment_with_no_overlap_gets_desconocido(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "fake-token")
        diarizer = Diarizer()
        # Segmento de transcripción [0, 1], diarización en [5, 6] — sin overlap
        turns = [_diar_turn(5.0, 6.0, "SPEAKER_00")]
        pipeline_cls = _make_pipeline_mock(turns)
        with patch("diarizer.Pipeline", pipeline_cls):
            segments = diarizer.diarize("audio.wav", [_seg(0.0, 1.0)])
        assert segments[0].speaker == "Speaker Desconocido"

    def test_segment_with_overlap_gets_correct_label(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "fake-token")
        diarizer = Diarizer()
        turns = [_diar_turn(0.0, 3.0, "SPEAKER_00")]
        pipeline_cls = _make_pipeline_mock(turns)
        with patch("diarizer.Pipeline", pipeline_cls):
            segments = diarizer.diarize("audio.wav", [_seg(0.5, 2.5)])
        assert segments[0].speaker == "Speaker 1"

    def test_labels_normalized_to_speaker_n(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "fake-token")
        diarizer = Diarizer()
        turns = [
            _diar_turn(0.0, 2.0, "SPEAKER_00"),
            _diar_turn(2.0, 4.0, "SPEAKER_01"),
        ]
        pipeline_cls = _make_pipeline_mock(turns)
        segs = [_seg(0.0, 2.0), _seg(2.0, 4.0)]
        with patch("diarizer.Pipeline", pipeline_cls):
            result = diarizer.diarize("audio.wav", segs)
        assert result[0].speaker == "Speaker 1"
        assert result[1].speaker == "Speaker 2"

    def test_best_overlap_wins(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "fake-token")
        diarizer = Diarizer()
        # Segmento [0, 4]: overlap con SPEAKER_00 = 1s, con SPEAKER_01 = 3s
        turns = [
            _diar_turn(0.0, 1.0, "SPEAKER_00"),
            _diar_turn(1.0, 4.0, "SPEAKER_01"),
        ]
        pipeline_cls = _make_pipeline_mock(turns)
        with patch("diarizer.Pipeline", pipeline_cls):
            result = diarizer.diarize("audio.wav", [_seg(0.0, 4.0)])
        assert result[0].speaker == "Speaker 2"  # SPEAKER_01 tiene mayor overlap

    def test_exact_tie_overlap_first_speaker_wins(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "fake-token")
        diarizer = Diarizer()
        # Segmento [1, 3]: overlap exacto de 1s con SPEAKER_00 [0,2] y SPEAKER_01 [2,4]
        turns = [
            _diar_turn(0.0, 2.0, "SPEAKER_00"),  # overlap = min(3,2)-max(1,0) = 1s
            _diar_turn(2.0, 4.0, "SPEAKER_01"),  # overlap = min(3,4)-max(1,2) = 1s
        ]
        pipeline_cls = _make_pipeline_mock(turns)
        with patch("diarizer.Pipeline", pipeline_cls):
            result = diarizer.diarize("audio.wav", [_seg(1.0, 3.0)])
        # Gana el primero encontrado (SPEAKER_00 → Speaker 1)
        assert result[0].speaker == "Speaker 1"

    def test_empty_segments_returns_empty(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "fake-token")
        diarizer = Diarizer()
        pipeline_cls = _make_pipeline_mock([])
        with patch("diarizer.Pipeline", pipeline_cls):
            result = diarizer.diarize("audio.wav", [])
        assert result == []


# ---------------------------------------------------------------------------
# Test de propiedad P5 — 8.2
# ---------------------------------------------------------------------------

_SPEAKER_PATTERN = re.compile(r"^Speaker \d+$")

# Estrategia: genera turns de diarización aleatorios
_turn_strategy = st.builds(
    lambda start, duration, label: (start, start + duration, label),
    start=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    duration=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    label=st.sampled_from(["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"]),
)

_seg_strategy = st.builds(
    lambda start, duration: (start, start + duration),
    start=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    duration=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)


@settings(max_examples=100)
@given(
    raw_turns=st.lists(_turn_strategy, min_size=0, max_size=10),
    raw_segs=st.lists(_seg_strategy, min_size=1, max_size=10),
)
def test_p5_speaker_labels_always_valid(raw_turns, raw_segs):
    """
    Propiedad 5: todas las etiquetas de hablante asignadas siguen el patrón
    "Speaker N" (N entero positivo) o "Speaker Desconocido".
    Ningún segmento tiene etiqueta None o vacía.
    """
    diarizer = Diarizer()

    diar_turns = [_diar_turn(s, e, lbl) for s, e, lbl in raw_turns]
    segments = [_seg(s, e) for s, e in raw_segs]

    result = diarizer._assign_speakers(segments, diar_turns)

    for seg in result:
        assert seg.speaker is not None, "speaker no debe ser None"
        assert seg.speaker != "", "speaker no debe ser vacío"
        assert (
            _SPEAKER_PATTERN.match(seg.speaker) or seg.speaker == "Speaker Desconocido"
        ), f"etiqueta inválida: '{seg.speaker}'"
