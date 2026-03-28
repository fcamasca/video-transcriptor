# models.py — Dataclasses y enums compartidos del dominio

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class ProcessingStatus(str, Enum):
    OK = "OK"
    ERROR_EXTRACCION_AUDIO = "ERROR_EXTRACCION_AUDIO"
    ERROR_TRANSCRIPCION = "ERROR_TRANSCRIPCION"
    SIN_AUDIO = "SIN_AUDIO"
    TRANSCRIPCION_SIN_DIARIZACION = "TRANSCRIPCION_SIN_DIARIZACION"


class DuplicatePolicy(str, Enum):
    SKIP = "skip"
    OVERWRITE = "overwrite"


class WhisperModel(str, Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class TranscriptionSegment:
    start: float          # segundos
    end: float            # segundos
    text: str
    speaker: Optional[str] = None  # "Speaker 1", "Speaker Desconocido", o None


@dataclass
class AppConfig:
    input_dir: str = "Videos/"
    output_dir: str = "Transcripcion/"
    language: str = "es"
    diarization: bool = True
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.SKIP
    log_file: Optional[str] = None
    whisper_model: WhisperModel = WhisperModel.BASE


@dataclass
class VideoResult:
    video_name: str
    status: ProcessingStatus
    segments: list = field(default_factory=list)
    error_message: Optional[str] = None
