# processor.py — Orquestador del pipeline de transcripción

import logging
import os
import subprocess
from typing import List, Optional

from audio_extractor import AudioExtractor
from diarizer import Diarizer
from exceptions import (
    AudioExtractionError,
    AudioTooShortError,
    DiarizationError,
    FfmpegNotFoundError,
    MissingHuggingFaceTokenError,
    NoAudioStreamError,
    OutputWriteError,
    TranscriptionError,
)
from models import AppConfig, DuplicatePolicy, ProcessingStatus, VideoResult
from output_writer import OutputWriter
from transcriber import Transcriber

VALID_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


class Processor:
    """Orquesta el pipeline completo: escaneo → extracción → transcripción → diarización → salida."""

    def __init__(self, config: AppConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._extractor = AudioExtractor()
        self._transcriber = Transcriber()
        self._diarizer = Diarizer()
        self._writer = OutputWriter()

    def run(self) -> int:
        """
        Ejecuta el pipeline completo.

        Returns:
            0 si al menos un video fue procesado exitosamente.
            1 si error crítico o todos los videos fallaron.
        """
        self.logger.info(
            f"Inicio de ejecución | input_dir={self.config.input_dir} "
            f"output_dir={self.config.output_dir} language={self.config.language} "
            f"diarization={self.config.diarization} "
            f"duplicate_policy={self.config.duplicate_policy.value} "
            f"whisper_model={self.config.whisper_model.value}"
        )

        # 10.1 — Verificar ffmpeg
        if not self._check_ffmpeg():
            self.logger.error("ffmpeg no está disponible en el sistema. Abortando.")
            return 1

        # 10.1 — Verificar diarización anticipada (antes de transcribir)
        if self.config.diarization:
            self._check_diarization()

        # 10.1 — Verificar carpeta de entrada
        if not os.path.isdir(self.config.input_dir):
            self.logger.error(f"La carpeta de entrada '{self.config.input_dir}' no existe.")
            return 1

        # 10.1 — Escanear y filtrar archivos
        video_files = self._scan_input_dir()
        if not video_files:
            self.logger.warning(f"No se encontraron archivos de video en '{self.config.input_dir}'.")
            return 0

        # 10.2 — Inicializar modelo Whisper una sola vez
        whisper_model = self._load_whisper_model()

        # 10.2 — Procesar cada video
        total = len(video_files)
        successful = 0
        errors = 0
        skipped = 0

        for video_path in video_files:
            video_name = os.path.basename(video_path)
            self.logger.info(f"Archivo detectado: {video_name}")

            # Política de duplicados
            if self._is_duplicate(video_name):
                if self.config.duplicate_policy == DuplicatePolicy.SKIP:
                    self.logger.info(f"Omitido por política 'skip': {video_name}")
                    skipped += 1
                    continue

            result = self._process_video(video_path, video_name, whisper_model)

            if result.status in (
                ProcessingStatus.OK,
                ProcessingStatus.TRANSCRIPCION_SIN_DIARIZACION,
                ProcessingStatus.SIN_AUDIO,
            ):
                successful += 1
            else:
                errors += 1

        # 10.3 — Resumen final
        self.logger.info(
            f"Resumen final | detectados={total} exitosos={successful} "
            f"errores={errors} omitidos={skipped}"
        )

        assert total == successful + errors + skipped, (
            f"Invariante violada: {total} != {successful} + {errors} + {skipped}"
        )

        return 0 if successful > 0 else 1

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _check_ffmpeg(self) -> bool:
        """Verifica que ffmpeg esté disponible en el PATH."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.logger.info("ffmpeg disponible en el sistema.")
            return True
        except FileNotFoundError:
            return False

    def _check_diarization(self) -> None:
        """
        Verifica anticipadamente que la diarización es posible:
        token presente y modelo accesible en HuggingFace.
        Loguea warning si no es posible, sin abortar.
        """
        import os as _os
        token = _os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            self.logger.warning(
                "Diarización solicitada pero HUGGINGFACE_TOKEN no está configurado. "
                "Los videos se transcribirán sin identificación de hablantes."
            )
            return
        try:
            import requests
            resp = requests.get(
                "https://huggingface.co/api/models/pyannote/speaker-diarization-3.1",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5,
            )
            if resp.status_code == 200:
                self.logger.info("Diarización disponible: token y modelo verificados.")
            elif resp.status_code == 401:
                self.logger.warning(
                    "Diarización solicitada pero el token HuggingFace es inválido o expiró. "
                    "Los videos se transcribirán sin identificación de hablantes."
                )
            elif resp.status_code == 403:
                self.logger.warning(
                    "Diarización solicitada pero no tienes acceso al modelo pyannote/speaker-diarization-3.1. "
                    "Acepta los términos en huggingface.co/pyannote/speaker-diarization-3.1"
                )
            else:
                self.logger.warning(
                    f"No se pudo verificar el acceso a la diarización (HTTP {resp.status_code}). "
                    "Se intentará igualmente al procesar cada video."
                )
        except Exception:
            self.logger.warning(
                "No se pudo verificar el acceso a la diarización (sin conexión). "
                "Se intentará igualmente al procesar cada video."
            )

    def _scan_input_dir(self) -> List[str]:
        """Escanea el nivel raíz de input_dir y retorna rutas de videos válidos, ordenados."""
        files = []
        for entry in os.scandir(self.config.input_dir):
            if entry.is_file():
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in VALID_EXTENSIONS:
                    files.append(entry.path)
        return sorted(files)

    def _is_duplicate(self, video_name: str) -> bool:
        """Retorna True si ya existe al menos un archivo de transcripción para este video."""
        stem = os.path.splitext(video_name)[0]
        if not os.path.isdir(self.config.output_dir):
            return False
        for fname in os.listdir(self.config.output_dir):
            if fname.startswith(stem + "_") and fname.endswith(".txt"):
                return True
        return False

    def _load_whisper_model(self):
        """Carga el modelo Whisper una sola vez."""
        try:
            import whisper
            model = whisper.load_model(self.config.whisper_model.value)
            self.logger.info(f"Modelo Whisper '{self.config.whisper_model.value}' cargado.")
            return model
        except Exception as exc:
            self.logger.error(f"No se pudo cargar el modelo Whisper: {exc}")
            return None

    def _process_video(self, video_path: str, video_name: str, whisper_model) -> VideoResult:
        """Ejecuta el pipeline completo para un video individual."""
        self.logger.info(f"Iniciando procesamiento: {video_name}")

        segments = []
        status = ProcessingStatus.OK

        # Etapa 1: Extracción de audio
        try:
            with self._extractor.extract(video_path) as wav_path:
                # Etapa 2: Transcripción
                try:
                    segments = self._transcriber.transcribe(
                        wav_path, whisper_model, language=self.config.language
                    )
                except (AudioTooShortError, TranscriptionError) as exc:
                    self.logger.error(f"Error en transcripción [{video_name}]: {exc}")
                    status = ProcessingStatus.ERROR_TRANSCRIPCION
                    return VideoResult(video_name=video_name, status=status, error_message=str(exc))

                # Etapa 3: Diarización (opcional)
                if self.config.diarization and segments:
                    try:
                        segments = self._diarizer.diarize(wav_path, segments)
                    except (MissingHuggingFaceTokenError, DiarizationError) as exc:
                        self.logger.error(f"Diarización fallida [{video_name}]: {exc}. Continuando sin diarización.")
                        status = ProcessingStatus.TRANSCRIPCION_SIN_DIARIZACION

        except NoAudioStreamError as exc:
            self.logger.error(f"Sin pista de audio [{video_name}]: {exc}")
            status = ProcessingStatus.SIN_AUDIO
            # Continúa a escritura: genera archivo con encabezado y Estado: SIN_AUDIO, sin segmentos
        except (FfmpegNotFoundError, AudioExtractionError) as exc:
            self.logger.error(f"Error en extracción de audio [{video_name}]: {exc}")
            status = ProcessingStatus.ERROR_EXTRACCION_AUDIO
            return VideoResult(video_name=video_name, status=status, error_message=str(exc))

        # Etapa 4: Escritura de salida
        try:
            output_path = self._writer.write(
                video_name=video_name,
                segments=segments,
                status=status,
                output_dir=self.config.output_dir,
                diarization_enabled=self.config.diarization,
            )
            self._writer.write_srt(
                video_name=video_name,
                segments=segments,
                status=status,
                output_dir=self.config.output_dir,
                diarization_enabled=self.config.diarization,
            )
            self.logger.info(f"Procesamiento exitoso: {video_name} → {output_path}")
        except OutputWriteError as exc:
            self.logger.error(f"Error al escribir salida [{video_name}]: {exc}")
            return VideoResult(video_name=video_name, status=ProcessingStatus.ERROR_TRANSCRIPCION, error_message=str(exc))

        return VideoResult(video_name=video_name, status=status, segments=segments)
