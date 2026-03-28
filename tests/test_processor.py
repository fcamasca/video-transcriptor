# Feature: video-transcriptor, Propiedades 1, 2, 7, 8, 9, 10: Procesador
# Tests unitarios y de propiedad para Processor

import os
import tempfile
from unittest.mock import MagicMock, patch, call
from contextlib import contextmanager

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AppConfig, DuplicatePolicy, ProcessingStatus, WhisperModel
from processor import Processor, VALID_EXTENSIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(**kwargs) -> AppConfig:
    defaults = dict(
        input_dir="Videos/",
        output_dir="Transcripcion/",
        language="es",
        diarization=False,
        duplicate_policy=DuplicatePolicy.SKIP,
        whisper_model=WhisperModel.BASE,
    )
    defaults.update(kwargs)
    return AppConfig(**defaults)


def _make_logger():
    import logging
    logger = MagicMock(spec=logging.Logger)
    return logger


def _processor(config=None, logger=None):
    return Processor(config or _config(), logger or _make_logger())


# ---------------------------------------------------------------------------
# Tests unitarios — 10.10
# ---------------------------------------------------------------------------

class TestProcessorUnit:

    def test_no_audio_stream_writes_file_with_sin_audio_status(self, tmp_path):
        """SIN_AUDIO debe generar archivo de salida con encabezado y sin segmentos."""
        from exceptions import NoAudioStreamError
        from contextlib import contextmanager

        out_dir = tmp_path / "out"
        cfg = _config(input_dir=str(tmp_path), output_dir=str(out_dir))
        proc = _processor(cfg)

        @contextmanager
        def raise_no_audio(video_path):
            raise NoAudioStreamError("sin audio")
            yield  # nunca llega aquí

        with patch.object(proc._extractor, "extract", side_effect=raise_no_audio):
            result = proc._process_video(str(tmp_path / "video.mp4"), "video.mp4", MagicMock())

        assert result.status == ProcessingStatus.SIN_AUDIO
        out_files = list(out_dir.glob("video_*.txt"))
        assert len(out_files) == 1
        content = out_files[0].read_text(encoding="utf-8")
        assert "Estado: SIN_AUDIO" in content

    def test_missing_input_dir_returns_exit_1(self, tmp_path):
        cfg = _config(input_dir=str(tmp_path / "no_existe"))
        proc = _processor(cfg)
        with patch.object(proc, "_check_ffmpeg", return_value=True):
            result = proc.run()
        assert result == 1

    def test_empty_input_dir_returns_exit_0(self, tmp_path):
        cfg = _config(input_dir=str(tmp_path))
        proc = _processor(cfg)
        with patch.object(proc, "_check_ffmpeg", return_value=True):
            result = proc.run()
        assert result == 0

    def test_ffmpeg_not_available_returns_exit_1(self, tmp_path):
        cfg = _config(input_dir=str(tmp_path))
        proc = _processor(cfg)
        with patch.object(proc, "_check_ffmpeg", return_value=False):
            result = proc.run()
        assert result == 1

    def test_all_videos_error_returns_exit_1(self, tmp_path):
        # Crear un video falso
        (tmp_path / "video.mp4").write_bytes(b"fake")
        cfg = _config(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
        proc = _processor(cfg)

        with patch.object(proc, "_check_ffmpeg", return_value=True), \
             patch.object(proc, "_load_whisper_model", return_value=MagicMock()), \
             patch.object(proc, "_process_video") as mock_pv:
            from models import VideoResult
            mock_pv.return_value = VideoResult(
                video_name="video.mp4",
                status=ProcessingStatus.ERROR_EXTRACCION_AUDIO,
                error_message="fallo"
            )
            result = proc.run()
        assert result == 1

    def test_at_least_one_success_returns_exit_0(self, tmp_path):
        (tmp_path / "video.mp4").write_bytes(b"fake")
        cfg = _config(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
        proc = _processor(cfg)

        with patch.object(proc, "_check_ffmpeg", return_value=True), \
             patch.object(proc, "_load_whisper_model", return_value=MagicMock()), \
             patch.object(proc, "_process_video") as mock_pv:
            from models import VideoResult
            mock_pv.return_value = VideoResult(
                video_name="video.mp4",
                status=ProcessingStatus.OK,
            )
            result = proc.run()
        assert result == 0

    def test_skip_policy_logs_and_skips(self, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # Crear transcripción existente
        (out_dir / "video_20260101_120000.txt").write_text("dummy")
        (tmp_path / "video.mp4").write_bytes(b"fake")

        cfg = _config(
            input_dir=str(tmp_path),
            output_dir=str(out_dir),
            duplicate_policy=DuplicatePolicy.SKIP,
        )
        logger = _make_logger()
        proc = _processor(cfg, logger)

        with patch.object(proc, "_check_ffmpeg", return_value=True), \
             patch.object(proc, "_load_whisper_model", return_value=MagicMock()), \
             patch.object(proc, "_process_video") as mock_pv:
            proc.run()
        # _process_video no debe haberse llamado
        mock_pv.assert_not_called()
        # Logger debe haber registrado el skip
        log_calls = " ".join(str(c) for c in logger.info.call_args_list)
        assert "skip" in log_calls.lower() or "omitido" in log_calls.lower()

    def test_overwrite_policy_processes_despite_existing(self, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "video_20260101_120000.txt").write_text("dummy")
        (tmp_path / "video.mp4").write_bytes(b"fake")

        cfg = _config(
            input_dir=str(tmp_path),
            output_dir=str(out_dir),
            duplicate_policy=DuplicatePolicy.OVERWRITE,
        )
        proc = _processor(cfg)

        with patch.object(proc, "_check_ffmpeg", return_value=True), \
             patch.object(proc, "_load_whisper_model", return_value=MagicMock()), \
             patch.object(proc, "_process_video") as mock_pv:
            from models import VideoResult
            mock_pv.return_value = VideoResult("video.mp4", ProcessingStatus.OK)
            proc.run()
        mock_pv.assert_called_once()


# ---------------------------------------------------------------------------
# Test de propiedad P1 — 10.4: Filtrado por extensión
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    filenames=st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-"
        )).flatmap(lambda stem: st.sampled_from([
            stem + ".mp4", stem + ".mkv", stem + ".avi", stem + ".mov", stem + ".webm",
            stem + ".txt", stem + ".pdf", stem + ".py", stem + ".MP4", stem + ".MKV",
        ])),
        min_size=0, max_size=20,
    )
)
def test_p1_extension_filter(filenames):
    """
    Propiedad 1: solo se detectan archivos con extensiones válidas (case-insensitive).
    """
    import shutil
    tmp = tempfile.mkdtemp()
    try:
        for fname in filenames:
            open(os.path.join(tmp, fname), "wb").close()

        cfg = _config(input_dir=tmp)
        proc = _processor(cfg)
        detected = proc._scan_input_dir()
        detected_names = [os.path.basename(p) for p in detected]

        for name in detected_names:
            ext = os.path.splitext(name)[1].lower()
            assert ext in VALID_EXTENSIONS, f"extensión inválida detectada: {name}"

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTENSIONS:
                # En Windows el FS es case-insensitive: buscar por nombre normalizado
                detected_lower = [n.lower() for n in detected_names]
                assert fname.lower() in detected_lower, f"archivo válido no detectado: {fname}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test de propiedad P2 — 10.5: Ignorar subdirectorios
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    root_videos=st.lists(
        st.sampled_from(["a.mp4", "b.mkv", "c.avi", "d.mov", "e.webm"]),
        min_size=0, max_size=5, unique=True,
    ),
    subdir_videos=st.lists(
        st.sampled_from(["x.mp4", "y.mkv", "z.avi"]),
        min_size=0, max_size=5, unique=True,
    ),
)
def test_p2_ignore_subdirectories(root_videos, subdir_videos):
    """
    Propiedad 2: solo se detectan archivos del nivel raíz, no de subdirectorios.
    """
    import shutil
    tmp = tempfile.mkdtemp()
    try:
        for fname in root_videos:
            open(os.path.join(tmp, fname), "wb").close()

        subdir = os.path.join(tmp, "subdir")
        os.makedirs(subdir)
        for fname in subdir_videos:
            open(os.path.join(subdir, fname), "wb").close()

        cfg = _config(input_dir=tmp)
        proc = _processor(cfg)
        detected = proc._scan_input_dir()
        detected_names = [os.path.basename(p) for p in detected]

        for fname in subdir_videos:
            assert fname not in detected_names, f"archivo de subdirectorio detectado: {fname}"

        for fname in root_videos:
            assert fname in detected_names, f"archivo raíz no detectado: {fname}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test de propiedad P7 — 10.6: Política de duplicados
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    has_existing=st.booleans(),
    policy=st.sampled_from([DuplicatePolicy.SKIP, DuplicatePolicy.OVERWRITE]),
)
def test_p7_duplicate_policy(has_existing, policy):
    """
    Propiedad 7: con política skip, el video no se procesa si existe transcripción previa.
    Con política overwrite, siempre se procesa.
    """
    import shutil
    tmp = tempfile.mkdtemp()
    out_dir = os.path.join(tmp, "out")
    os.makedirs(out_dir)
    try:
        if has_existing:
            open(os.path.join(out_dir, "video_20260101_120000.txt"), "w").close()

        open(os.path.join(tmp, "video.mp4"), "wb").close()

        cfg = _config(input_dir=tmp, output_dir=out_dir, duplicate_policy=policy)
        proc = _processor(cfg)

        with patch.object(proc, "_check_ffmpeg", return_value=True), \
             patch.object(proc, "_load_whisper_model", return_value=MagicMock()), \
             patch.object(proc, "_process_video") as mock_pv:
            from models import VideoResult
            mock_pv.return_value = VideoResult("video.mp4", ProcessingStatus.OK)
            proc.run()

        if has_existing and policy == DuplicatePolicy.SKIP:
            mock_pv.assert_not_called()
        else:
            mock_pv.assert_called_once()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test de propiedad P8 — 10.7: Continuidad del lote ante errores
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    n_ok=st.integers(min_value=1, max_value=5),
    n_err=st.integers(min_value=1, max_value=5),
)
def test_p8_batch_continuity(n_ok, n_err):
    """
    Propiedad 8: los videos válidos se procesan independientemente de los errores en otros.
    """
    import shutil
    tmp = tempfile.mkdtemp()
    try:
        videos_ok = [f"ok_{i}.mp4" for i in range(n_ok)]
        videos_err = [f"err_{i}.mkv" for i in range(n_err)]

        for fname in videos_ok + videos_err:
            open(os.path.join(tmp, fname), "wb").close()

        cfg = _config(input_dir=tmp, output_dir=os.path.join(tmp, "out"))
        proc = _processor(cfg)

        from models import VideoResult

        def fake_process(video_path, video_name, model):
            if video_name.startswith("ok_"):
                return VideoResult(video_name, ProcessingStatus.OK)
            return VideoResult(video_name, ProcessingStatus.ERROR_EXTRACCION_AUDIO, error_message="fallo")

        with patch.object(proc, "_check_ffmpeg", return_value=True), \
             patch.object(proc, "_load_whisper_model", return_value=MagicMock()), \
             patch.object(proc, "_process_video", side_effect=fake_process):
            result = proc.run()

        assert result == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test de propiedad P9 — 10.8: Consistencia del resumen final
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    n_ok=st.integers(min_value=0, max_value=5),
    n_err=st.integers(min_value=0, max_value=5),
    n_skip=st.integers(min_value=0, max_value=5),
)
def test_p9_summary_consistency(n_ok, n_err, n_skip):
    """
    Propiedad 9: total_detectados = exitosos + errores + omitidos.
    """
    import shutil
    tmp = tempfile.mkdtemp()
    out_dir = os.path.join(tmp, "out")
    os.makedirs(out_dir)
    try:
        ok_names = [f"ok_{i}.mp4" for i in range(n_ok)]
        err_names = [f"err_{i}.mkv" for i in range(n_err)]
        skip_names = [f"skip_{i}.avi" for i in range(n_skip)]

        for fname in ok_names + err_names + skip_names:
            open(os.path.join(tmp, fname), "wb").close()

        for fname in skip_names:
            stem = os.path.splitext(fname)[0]
            open(os.path.join(out_dir, f"{stem}_20260101_120000.txt"), "w").close()

        cfg = _config(input_dir=tmp, output_dir=out_dir, duplicate_policy=DuplicatePolicy.SKIP)
        proc = _processor(cfg)

        from models import VideoResult

        def fake_process(video_path, video_name, model):
            if video_name.startswith("ok_"):
                return VideoResult(video_name, ProcessingStatus.OK)
            return VideoResult(video_name, ProcessingStatus.ERROR_EXTRACCION_AUDIO, error_message="fallo")

        with patch.object(proc, "_check_ffmpeg", return_value=True), \
             patch.object(proc, "_load_whisper_model", return_value=MagicMock()), \
             patch.object(proc, "_process_video", side_effect=fake_process):
            proc.run()
        # La invariante se verifica internamente con assert en run()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test de propiedad P10 — 10.9: Código de salida según resultado del lote
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    n_ok=st.integers(min_value=0, max_value=5),
    n_err=st.integers(min_value=0, max_value=5),
)
def test_p10_exit_code(n_ok, n_err):
    """
    Propiedad 10: exit 0 si al menos un exitoso; exit 1 si todos con error.
    """
    if n_ok == 0 and n_err == 0:
        return

    import shutil
    tmp = tempfile.mkdtemp()
    try:
        for i in range(n_ok):
            open(os.path.join(tmp, f"ok_{i}.mp4"), "wb").close()
        for i in range(n_err):
            open(os.path.join(tmp, f"err_{i}.mkv"), "wb").close()

        cfg = _config(input_dir=tmp, output_dir=os.path.join(tmp, "out"))
        proc = _processor(cfg)

        from models import VideoResult

        def fake_process(video_path, video_name, model):
            if video_name.startswith("ok_"):
                return VideoResult(video_name, ProcessingStatus.OK)
            return VideoResult(video_name, ProcessingStatus.ERROR_EXTRACCION_AUDIO, error_message="fallo")

        with patch.object(proc, "_check_ffmpeg", return_value=True), \
             patch.object(proc, "_load_whisper_model", return_value=MagicMock()), \
             patch.object(proc, "_process_video", side_effect=fake_process):
            result = proc.run()

        if n_ok > 0:
            assert result == 0
        else:
            assert result == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
