# logger.py — Configuración del logger estándar de Python

import logging
import sys
from datetime import timezone


class _UTCFormatter(logging.Formatter):
    """Formatter con timestamps en UTC e ISO 8601."""

    converter = None  # no usar time.localtime ni time.gmtime

    def formatTime(self, record, datefmt=None):
        from datetime import datetime
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def format(self, record):
        record.asctime = self.formatTime(record)
        return f"{record.asctime} | {record.levelname} | {record.getMessage()}"


def setup_logger(level: int = logging.INFO, log_file: str = None) -> logging.Logger:
    """
    Configura y retorna el logger de la aplicación.

    - StreamHandler (stdout) siempre activo.
    - FileHandler (UTF-8) activado solo si se recibe log_file.
    - Formato: {timestamp_ISO8601} | {NIVEL} | {mensaje}
    - Timestamps en UTC.
    """
    logger = logging.getLogger("video-transcriptor")
    logger.setLevel(level)

    # Evitar duplicar handlers si se llama más de una vez
    if logger.handlers:
        logger.handlers.clear()

    formatter = _UTCFormatter()

    # Handler stdout — siempre activo
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Handler archivo — solo si se recibe ruta
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
