# Tests de integración con mocks de ffmpeg y Whisper — Task 12.2

import io
import os
import wave
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration_seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    n_frames = int(duration_seconds * sample_rate)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


def _mock_whisper_model(segments):
    model = MagicMock()
    model.transcribe.return_value = {"segments": segments}
    return model


def _make_subprocess_mock(wav_bytes: bytes, fail_patterns=None):
    """
    Retorna un mock de subprocess.run que:
    - Responde a `ffmpeg -version` con éxito (verificación del Procesador)
    - Para comandos ffmpeg de extracción: escribe wav_bytes en el output_path
      salvo que el input_file coincida con algún patrón en fail_patterns
    """
    fail_patterns = fail_patterns or []

    def fake_run(cmd, stdout=None, stderr=None):
        r = MagicMock()
        r.stdout = b""

        # Verificación de ffmpeg: `ffmpeg -version`
        if len(cmd) == 2 and cmd[1] == "-version":
            r.returncode = 0
            r.stderr = b""
            return r

        # Extracción de audio: `ffmpeg -y -i <input> ... <output>`
        if "-i" in cmd:
            input_file = cmd[cmd.index("-i") + 1]
            output_path = cmd[-1]

            for pattern in fail_patterns:
                if pattern in input_file:
                    r.returncode = 1
                    r.stderr = b"some unexpected error"
                    return r

            with open(output_path, "wb") as f:
                f.write(wav_bytes)
            r.returncode = 0
            r.stderr = b""
            return r

        r.returncode = 0
        r.stderr = b""
        return r

    return fake_run


# ---------------------------------------------------------------------------
# Escenario 1: flujo completo exitoso (sin diarización)
# ---------------------------------------------------------------------------

def test_integration_full_pipeline_no_diarization(tmp_path):
    """
    Flujo completo: detección → extracción → transcripción → escritura.
    Diarización deshabilitada. Verifica archivo de salida con contenido correcto.
    """
    input_dir = tmp_path / "videos"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "reunion.mp4").write_bytes(b"fake_video")

    wav_bytes = _make_wav_bytes(2.0)
    whisper_model = _mock_whisper_model([
        {"start": 0.0, "end": 1.0, "text": "Hola buenos días"},
        {"start": 1.0, "end": 2.0, "text": "Comenzamos la reunión"},
    ])

    with patch("subprocess.run", side_effect=_make_subprocess_mock(wav_bytes)), \
         patch("processor.Processor._load_whisper_model", return_value=whisper_model):

        exit_code = main([
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--diarization", "false",
        ])

    assert exit_code == 0

    out_files = list(output_dir.glob("reunion_*.txt"))
    assert len(out_files) == 1

    content = out_files[0].read_text(encoding="utf-8")
    lines = content.splitlines()
    assert lines[0] == "Video: reunion.mp4"
    assert lines[1].startswith("Procesado: ")
    assert lines[2] == "Estado: OK"
    assert lines[3] == ""
    assert "[00:00 - 00:01] Hola buenos días" in content
    assert "[00:01 - 00:02] Comenzamos la reunión" in content


# ---------------------------------------------------------------------------
# Escenario 2: error en video individual — el lote continúa
# ---------------------------------------------------------------------------

def test_integration_error_in_one_video_batch_continues(tmp_path):
    """
    Lote con dos videos: uno falla en ffmpeg, el otro se procesa correctamente.
    Exit code debe ser 0 (al menos uno exitoso).
    """
    input_dir = tmp_path / "videos"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "bueno.mp4").write_bytes(b"fake")
    (input_dir / "malo.mkv").write_bytes(b"fake")

    wav_bytes = _make_wav_bytes(2.0)
    whisper_model = _mock_whisper_model([
        {"start": 0.0, "end": 1.0, "text": "texto válido"},
    ])

    with patch("subprocess.run", side_effect=_make_subprocess_mock(wav_bytes, fail_patterns=["malo"])), \
         patch("processor.Processor._load_whisper_model", return_value=whisper_model):

        exit_code = main([
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--diarization", "false",
        ])

    assert exit_code == 0
    assert len(list(output_dir.glob("bueno_*.txt"))) == 1
    assert len(list(output_dir.glob("malo_*.txt"))) == 0


# ---------------------------------------------------------------------------
# Escenario 3: diarización deshabilitada — archivo sin etiquetas de hablante
# ---------------------------------------------------------------------------

def test_integration_diarization_disabled_no_speaker_labels(tmp_path):
    """
    Con --diarization false, el archivo de salida no debe contener etiquetas de hablante.
    """
    input_dir = tmp_path / "videos"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "video.mp4").write_bytes(b"fake")

    wav_bytes = _make_wav_bytes(2.0)
    whisper_model = _mock_whisper_model([
        {"start": 0.0, "end": 2.0, "text": "texto de prueba"},
    ])

    with patch("subprocess.run", side_effect=_make_subprocess_mock(wav_bytes)), \
         patch("processor.Processor._load_whisper_model", return_value=whisper_model):

        exit_code = main([
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--diarization", "false",
        ])

    assert exit_code == 0
    content = list(output_dir.glob("video_*.txt"))[0].read_text(encoding="utf-8")
    assert "Speaker" not in content
    assert "[00:00 - 00:02] texto de prueba" in content


# ---------------------------------------------------------------------------
# Escenario 4: diarización habilitada pero token ausente → TRANSCRIPCION_SIN_DIARIZACION
# ---------------------------------------------------------------------------

def test_integration_diarization_enabled_token_missing(tmp_path, monkeypatch):
    """
    Con --diarization true pero sin HUGGINGFACE_TOKEN, el video se procesa,
    el archivo se genera con Estado: TRANSCRIPCION_SIN_DIARIZACION y sin etiquetas.
    """
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    input_dir = tmp_path / "videos"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "video.mp4").write_bytes(b"fake")

    wav_bytes = _make_wav_bytes(2.0)
    whisper_model = _mock_whisper_model([
        {"start": 0.0, "end": 2.0, "text": "texto sin diarización"},
    ])

    with patch("subprocess.run", side_effect=_make_subprocess_mock(wav_bytes)), \
         patch("processor.Processor._load_whisper_model", return_value=whisper_model):

        exit_code = main([
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--diarization", "true",
        ])

    assert exit_code == 0
    content = list(output_dir.glob("video_*.txt"))[0].read_text(encoding="utf-8")
    assert "Estado: TRANSCRIPCION_SIN_DIARIZACION" in content
    assert "Speaker" not in content
    assert "[00:00 - 00:02] texto sin diarización" in content
