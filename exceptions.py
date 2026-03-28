# exceptions.py — Excepciones tipadas del dominio


class AudioExtractionError(Exception):
    """Error genérico durante la extracción de audio con ffmpeg."""
    pass


class NoAudioStreamError(AudioExtractionError):
    """El video no contiene pista de audio."""
    pass


class TranscriptionError(Exception):
    """Error genérico durante la transcripción con Whisper."""
    pass


class AudioTooShortError(TranscriptionError):
    """El audio tiene duración menor a 1 segundo."""
    pass


class DiarizationError(Exception):
    """Error genérico durante la diarización con pyannote."""
    pass


class MissingHuggingFaceTokenError(DiarizationError):
    """El token HUGGINGFACE_TOKEN no está configurado en el entorno."""
    pass


class OutputWriteError(Exception):
    """Error al escribir el archivo de salida."""
    pass


class FfmpegNotFoundError(Exception):
    """ffmpeg no está disponible en el sistema."""
    pass
