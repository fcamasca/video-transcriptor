# output_writer.py — Generación del archivo de salida estructurado

import os
import re
from datetime import datetime, timezone
from typing import List

from exceptions import OutputWriteError
from models import ProcessingStatus, TranscriptionSegment

# Caracteres que splitlines() reconoce como separadores de línea
_LINE_SEP_RE = re.compile(r"[\r\n\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]+")


class OutputWriter:
    """Genera el archivo de transcripción estructurado en la carpeta de salida."""

    def write(
        self,
        video_name: str,
        segments: List[TranscriptionSegment],
        status: ProcessingStatus,
        output_dir: str,
        diarization_enabled: bool = True,
    ) -> str:
        """
        Escribe el archivo de salida y retorna su ruta.

        Nombre del archivo: {nombre_video_sin_extension}_{YYYYMMDD_HHMMSS}.txt
        Formato:
            Video: {nombre_archivo_video}
            Procesado: {fecha_hora_ISO8601}
            Estado: {estado}
            <línea en blanco>
            [MM:SS - MM:SS] Hablante: texto   (con diarización)
            [MM:SS - MM:SS] texto              (sin diarización)

        Raises:
            OutputWriteError: ante cualquier fallo de escritura.
        """
        now_utc = datetime.now(tz=timezone.utc)
        timestamp = now_utc.strftime("%Y%m%d_%H%M%S")
        stem = os.path.splitext(video_name)[0]
        filename = f"{stem}_{timestamp}.txt"

        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as exc:
            raise OutputWriteError(f"No se pudo crear la carpeta de salida '{output_dir}': {exc}") from exc

        output_path = os.path.join(output_dir, filename)

        use_speaker = diarization_enabled and status != ProcessingStatus.TRANSCRIPCION_SIN_DIARIZACION

        lines = [
            f"Video: {video_name}",
            f"Procesado: {now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            f"Estado: {status.value}",
            "",
        ]

        for seg in segments:
            text = _LINE_SEP_RE.sub(" ", seg.text).strip()
            if not text:
                continue
            time_range = f"[{self._fmt_time(seg.start)} - {self._fmt_time(seg.end)}]"
            if use_speaker and seg.speaker:
                lines.append(f"{time_range} {seg.speaker}: {text}")
            else:
                lines.append(f"{time_range} {text}")

        content = "\n".join(lines) + "\n"

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as exc:
            raise OutputWriteError(f"Error al escribir '{output_path}': {exc}") from exc

        return output_path

    def write_srt(
        self,
        video_name: str,
        segments: List[TranscriptionSegment],
        status: ProcessingStatus,
        output_dir: str,
        diarization_enabled: bool = True,
    ) -> str:
        """
        Escribe el archivo .srt de subtítulos y retorna su ruta.
        El nombre del archivo usa el mismo timestamp que el .txt correspondiente.

        Raises:
            OutputWriteError: ante cualquier fallo de escritura.
        """
        now_utc = datetime.now(tz=timezone.utc)
        timestamp = now_utc.strftime("%Y%m%d_%H%M%S")
        stem = os.path.splitext(video_name)[0]
        filename = f"{stem}_{timestamp}.srt"
        output_path = os.path.join(output_dir, filename)

        use_speaker = diarization_enabled and status != ProcessingStatus.TRANSCRIPCION_SIN_DIARIZACION

        lines = []
        index = 1
        for seg in segments:
            text = _LINE_SEP_RE.sub(" ", seg.text).strip()
            if not text:
                continue
            if use_speaker and seg.speaker:
                text = f"{seg.speaker}: {text}"
            lines.append(str(index))
            lines.append(f"{self._fmt_srt_time(seg.start)} --> {self._fmt_srt_time(seg.end)}")
            lines.append(text)
            lines.append("")
            index += 1

        content = "\n".join(lines)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as exc:
            raise OutputWriteError(f"Error al escribir '{output_path}': {exc}") from exc

        return output_path

    @staticmethod
    def _fmt_srt_time(seconds: float) -> str:
        """Convierte segundos al formato SRT: HH:MM:SS,mmm"""
        ms = int(round(seconds * 1000))
        hh = ms // 3_600_000
        ms %= 3_600_000
        mm = ms // 60_000
        ms %= 60_000
        ss = ms // 1_000
        ms %= 1_000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Convierte segundos a MM:SS con cero a la izquierda."""
        total = int(seconds)
        mm = total // 60
        ss = total % 60
        return f"{mm:02d}:{ss:02d}"
