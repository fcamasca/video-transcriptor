# transcriber.py — Transcripción de audio con openai-whisper

import sys
import threading
import time
import warnings
import wave
from typing import List

from exceptions import AudioTooShortError, TranscriptionError
from models import TranscriptionSegment


class Transcriber:
    """Transcribe un archivo WAV usando una instancia de modelo Whisper ya cargada."""

    def transcribe(self, wav_path: str, model, language: str = "es") -> List[TranscriptionSegment]:
        """
        Transcribe el audio WAV y retorna una lista de TranscriptionSegment.

        Args:
            wav_path: Ruta al archivo WAV temporal.
            model: Instancia del modelo Whisper ya cargada (proporcionada por el Procesador).
            language: Código de idioma para Whisper.

        Returns:
            Lista de TranscriptionSegment con start, end y text.

        Raises:
            AudioTooShortError: si el audio tiene duración < 1 segundo.
            TranscriptionError: ante cualquier fallo del motor Whisper.
        """
        duration = self._validate_duration(wav_path)

        # Estado compartido entre el hilo de progreso y el callback de Whisper
        progress_state = {"last_end": 0.0, "done": False}

        def _render_bar(pos, elapsed, final=False):
            pct = 100.0 if final else min(pos / duration * 100, 99.9) if duration > 0 else 0.0
            bar_filled = int(pct / 5)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            return (
                f"  [{bar}] {pct:5.1f}%  "
                f"audio: {_fmt(pos)}/{_fmt(duration)}  "
                f"transcurrido: {_fmt_wall(elapsed)}"
            )

        def _progress_loop():
            # Reservar línea fija para la barra: imprimir una vez con \n
            # Luego usar \033[1A para subir y sobreescribir siempre esa línea
            sys.stderr.write(_render_bar(0, 0) + "\n")
            sys.stderr.flush()
            start_wall = time.time()
            while not progress_state["done"]:
                elapsed = time.time() - start_wall
                pos = progress_state["last_end"]
                # Subir una línea (\033[1A), ir al inicio (\r), sobreescribir, bajar (\n)
                sys.stderr.write(f"\033[1A\r{_render_bar(pos, elapsed)}\n")
                sys.stderr.flush()
                time.sleep(0.5)
            # Línea final al 100%
            elapsed = time.time() - start_wall
            sys.stderr.write(f"\033[1A\r{_render_bar(duration, elapsed, final=True)}\n")
            sys.stderr.flush()

        def _segment_callback(seg):
            progress_state["last_end"] = float(seg.get("end", 0))

        thread = threading.Thread(target=_progress_loop, daemon=True)
        thread.start()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.transcribe(
                    wav_path,
                    language=language,
                    verbose=False,  # False muestra tqdm de frames debajo de nuestra barra
                )
            for seg in result.get("segments", []):
                _segment_callback(seg)
        except Exception as exc:
            progress_state["done"] = True
            thread.join(timeout=1)
            raise TranscriptionError(f"Error en el motor Whisper: {exc}") from exc
        finally:
            progress_state["done"] = True
            thread.join(timeout=2)

        segments = []
        for seg in result.get("segments", []):
            text = seg["text"].strip()
            if not text:
                continue
            segments.append(
                TranscriptionSegment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=text,
                )
            )

        return segments

    def _validate_duration(self, wav_path: str) -> float:
        """Lanza AudioTooShortError si el audio dura menos de 1 segundo. Retorna la duración."""
        try:
            with wave.open(wav_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / rate if rate > 0 else 0.0
        except Exception as exc:
            raise TranscriptionError(f"No se pudo leer el archivo WAV: {exc}") from exc

        if duration < 1.0:
            raise AudioTooShortError(
                f"El audio tiene duración {duration:.3f}s, mínimo requerido: 1s"
            )
        return duration


def _fmt(seconds: float) -> str:
    """Formatea segundos a MM:SS."""
    t = int(seconds)
    return f"{t // 60:02d}:{t % 60:02d}"


def _fmt_wall(seconds: float) -> str:
    """Formatea segundos de reloj a HH:MM:SS o MM:SS."""
    t = int(seconds)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
