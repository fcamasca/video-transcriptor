# Feature: video-transcriptor, Propiedad 6: Formato del archivo de salida
# Tests unitarios y de propiedad para OutputWriter

import os
import re
import tempfile
import shutil
from typing import List
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exceptions import OutputWriteError
from models import ProcessingStatus, TranscriptionSegment
from output_writer import OutputWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(start: float, end: float, text: str, speaker: str = None) -> TranscriptionSegment:
    return TranscriptionSegment(start=start, end=end, text=text, speaker=speaker)


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _lines(path: str) -> List[str]:
    return _read(path).splitlines()


# ---------------------------------------------------------------------------
# Tests unitarios — 9.3
# ---------------------------------------------------------------------------

class TestOutputWriterUnit:

    def test_creates_output_dir_if_not_exists(self, tmp_path):
        out_dir = str(tmp_path / "nueva_carpeta")
        writer = OutputWriter()
        path = writer.write("video.mp4", [], ProcessingStatus.OK, out_dir)
        assert os.path.isdir(out_dir)
        assert os.path.isfile(path)

    def test_output_write_error_on_permission_failure(self, tmp_path):
        writer = OutputWriter()
        with patch("output_writer.open", side_effect=PermissionError("sin permisos")):
            with pytest.raises(OutputWriteError):
                writer.write("video.mp4", [], ProcessingStatus.OK, str(tmp_path))

    def test_header_format(self, tmp_path):
        writer = OutputWriter()
        path = writer.write("mi_video.mp4", [], ProcessingStatus.OK, str(tmp_path))
        lines = _lines(path)
        assert lines[0].startswith("Video: mi_video.mp4")
        assert lines[1].startswith("Procesado: ")
        assert lines[2].startswith("Estado: OK")
        assert lines[3] == ""

    def test_segment_with_diarization(self, tmp_path):
        segs = [_seg(0.0, 65.0, "hola mundo", speaker="Speaker 1")]
        writer = OutputWriter()
        path = writer.write("v.mp4", segs, ProcessingStatus.OK, str(tmp_path), diarization_enabled=True)
        lines = _lines(path)
        assert lines[4] == "[00:00 - 01:05] Speaker 1: hola mundo"

    def test_segment_without_diarization(self, tmp_path):
        segs = [_seg(0.0, 30.0, "texto sin hablante")]
        writer = OutputWriter()
        path = writer.write("v.mp4", segs, ProcessingStatus.OK, str(tmp_path), diarization_enabled=False)
        lines = _lines(path)
        assert lines[4] == "[00:00 - 00:30] texto sin hablante"

    def test_status_sin_diarizacion_omits_speaker(self, tmp_path):
        segs = [_seg(0.0, 10.0, "texto", speaker="Speaker 1")]
        writer = OutputWriter()
        path = writer.write(
            "v.mp4", segs,
            ProcessingStatus.TRANSCRIPCION_SIN_DIARIZACION,
            str(tmp_path),
            diarization_enabled=True,
        )
        lines = _lines(path)
        assert "Speaker 1" not in lines[4]
        assert lines[4] == "[00:00 - 00:10] texto"

    def test_utf8_encoding(self, tmp_path):
        segs = [_seg(0.0, 5.0, "日本語テスト áéíóú ñ")]
        writer = OutputWriter()
        path = writer.write("v.mp4", segs, ProcessingStatus.OK, str(tmp_path))
        content = _read(path)
        assert "日本語テスト áéíóú ñ" in content

    def test_filename_format(self, tmp_path):
        writer = OutputWriter()
        path = writer.write("mi video.mp4", [], ProcessingStatus.OK, str(tmp_path))
        filename = os.path.basename(path)
        assert re.match(r"^mi video_\d{8}_\d{6}\.txt$", filename)

    def test_time_format_mm_ss(self, tmp_path):
        segs = [_seg(3661.0, 3723.0, "texto")]  # 61:01 - 62:03
        writer = OutputWriter()
        path = writer.write("v.mp4", segs, ProcessingStatus.OK, str(tmp_path), diarization_enabled=False)
        lines = _lines(path)
        assert lines[4] == "[61:01 - 62:03] texto"


# ---------------------------------------------------------------------------
# Test de propiedad P6 — 9.2
# ---------------------------------------------------------------------------

_status_strategy = st.sampled_from(list(ProcessingStatus))

_seg_strategy = st.builds(
    lambda start, duration, text, speaker: _seg(start, start + duration, text, speaker),
    start=st.floats(min_value=0.0, max_value=3600.0, allow_nan=False, allow_infinity=False),
    duration=st.floats(min_value=0.1, max_value=60.0, allow_nan=False, allow_infinity=False),
    text=st.text(min_size=1, max_size=100, alphabet=st.characters(blacklist_categories=("Cs",))),
    speaker=st.one_of(st.none(), st.just("Speaker 1"), st.just("Speaker 2")),
)

_SEGMENT_WITH_SPEAKER = re.compile(r"^\[\d{2,}:\d{2} - \d{2,}:\d{2}\] .+: .+$")
_SEGMENT_WITHOUT_SPEAKER = re.compile(r"^\[\d{2,}:\d{2} - \d{2,}:\d{2}\] .+$")


@settings(max_examples=100)
@given(
    video_name=st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=" _-."
    )).filter(lambda s: s.strip()),
    segments=st.lists(_seg_strategy, min_size=0, max_size=10),
    status=_status_strategy,
    diarization_enabled=st.booleans(),
)
def test_p6_output_file_format(video_name, segments, status, diarization_enabled):
    """
    Propiedad 6: el archivo generado cumple:
    (a) primeras tres líneas son Video:, Procesado:, Estado:
    (b) cuarta línea en blanco
    (c) cada línea de segmento sigue la estructura correcta
    (d) codificación UTF-8
    """
    tmp_dir = tempfile.mkdtemp()
    writer = OutputWriter()
    path = writer.write(
        video_name + ".mp4",
        segments,
        status,
        tmp_dir,
        diarization_enabled=diarization_enabled,
    )

    # (d) UTF-8
    with open(path, "rb") as f:
        raw = f.read()
    content = raw.decode("utf-8")  # lanza si no es UTF-8 válido
    lines = content.splitlines()

    # (a) encabezado
    assert lines[0].startswith("Video: "), f"línea 0 inválida: {lines[0]!r}"
    assert lines[1].startswith("Procesado: "), f"línea 1 inválida: {lines[1]!r}"
    assert lines[2].startswith("Estado: "), f"línea 2 inválida: {lines[2]!r}"

    # (b) línea en blanco
    assert lines[3] == "", f"línea 3 debe ser vacía, got: {lines[3]!r}"

    # (c) segmentos
    use_speaker = (
        diarization_enabled
        and status != ProcessingStatus.TRANSCRIPCION_SIN_DIARIZACION
    )
    seg_lines = lines[4:]
    valid_segs = [seg for seg in segments if seg.text.strip()]
    assert len(seg_lines) == len(valid_segs), (
        f"número de líneas de segmento ({len(seg_lines)}) != segmentos válidos ({len(valid_segs)})"
    )
    for i, seg in enumerate(valid_segs):
        line = seg_lines[i]
        if use_speaker and seg.speaker:
            assert _SEGMENT_WITH_SPEAKER.match(line), f"formato con hablante inválido: {line!r}"
        else:
            assert _SEGMENT_WITHOUT_SPEAKER.match(line), f"formato sin hablante inválido: {line!r}"
