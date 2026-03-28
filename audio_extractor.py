# audio_extractor.py — Extracción de audio con ffmpeg

import os
import subprocess
import tempfile
from contextlib import contextmanager

from exceptions import AudioExtractionError, FfmpegNotFoundError, NoAudioStreamError


class AudioExtractor:
    """Extrae el audio de un archivo de video a WAV mono 16 kHz usando ffmpeg."""

    @contextmanager
    def extract(self, video_path: str):
        """
        Context manager que extrae el audio a un WAV temporal y garantiza
        su eliminación al salir del bloque (éxito o error).

        Usage:
            with extractor.extract(video_path) as wav_path:
                # usar wav_path

        Raises:
            FfmpegNotFoundError: si ffmpeg no está disponible.
            NoAudioStreamError: si el video no contiene pista de audio.
            AudioExtractionError: ante cualquier otro fallo de ffmpeg.
        """
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(tmp_fd)

            try:
                self._run_ffmpeg(video_path, tmp_path)
            except FileNotFoundError:
                raise FfmpegNotFoundError("ffmpeg no está disponible en el sistema.")
            except (NoAudioStreamError, AudioExtractionError, FfmpegNotFoundError):
                raise
            except Exception as exc:
                raise AudioExtractionError(f"Error inesperado durante la extracción: {exc}") from exc

            yield tmp_path

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _run_ffmpeg(self, video_path: str, output_path: str) -> None:
        """Invoca ffmpeg y traduce errores a excepciones del dominio."""
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_path,
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stderr_text = result.stderr.decode("utf-8", errors="replace")

        if result.returncode != 0:
            stderr_lower = stderr_text.lower()
            if (
                "no audio" in stderr_lower
                or "does not contain any stream" in stderr_lower
                or "invalid data found" in stderr_lower
            ):
                raise NoAudioStreamError(f"No se encontró pista de audio en: {video_path}")
            raise AudioExtractionError(
                f"ffmpeg falló (código {result.returncode}): {stderr_text}"
            )
