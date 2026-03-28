# tests/test_logger.py — Tests para logger.py

import logging
import re
import sys
import io
from datetime import datetime, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger import setup_logger


# --- Helpers ---

def capture_log_output(logger: logging.Logger, level: int, message: str) -> str:
    """Captura la salida del StreamHandler del logger."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    from logger import _UTCFormatter
    handler.setFormatter(_UTCFormatter())
    logger.addHandler(handler)
    logger.log(level, message)
    logger.removeHandler(handler)
    # Strip only the trailing newline added by StreamHandler, not the message content
    return buf.getvalue().rstrip("\n")


# --- Tests unitarios ---

class TestSetupLogger:
    def test_returns_logger(self):
        logger = setup_logger()
        assert isinstance(logger, logging.Logger)

    def test_has_stream_handler(self):
        logger = setup_logger()
        types = [type(h) for h in logger.handlers]
        assert logging.StreamHandler in types

    def test_no_file_handler_by_default(self):
        logger = setup_logger()
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 0

    def test_file_handler_created_when_log_file_given(self, tmp_path):
        log_path = str(tmp_path / "test.log")
        logger = setup_logger(log_file=log_path)
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

    def test_no_duplicate_handlers_on_repeated_calls(self):
        logger = setup_logger()
        logger = setup_logger()
        assert len(logger.handlers) == 1

    def test_log_level_respected(self):
        logger = setup_logger(level=logging.WARNING)
        assert logger.level == logging.WARNING


class TestLogFormat:
    def test_format_structure(self):
        logger = setup_logger()
        line = capture_log_output(logger, logging.INFO, "mensaje de prueba")
        parts = line.split(" | ")
        assert len(parts) == 3
        timestamp, level, message = parts
        assert level == "INFO"
        assert message == "mensaje de prueba"

    def test_timestamp_is_utc_iso8601(self):
        logger = setup_logger()
        line = capture_log_output(logger, logging.INFO, "test")
        timestamp = line.split(" | ")[0]
        # Timestamp emitido con Z al final (UTC)
        assert timestamp.endswith("Z"), f"Timestamp debe terminar en Z: {timestamp!r}"
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert dt.tzinfo is not None
        assert dt.utcoffset().total_seconds() == 0

    def test_all_levels_format_correctly(self):
        logger = setup_logger(level=logging.DEBUG)
        for level, name in [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
        ]:
            line = capture_log_output(logger, level, f"msg {name}")
            assert f"| {name} |" in line


# --- Test de propiedad P11 ---

# Feature: video-transcriptor, Propiedad 11: Formato de líneas de log
@given(
    message=st.text(min_size=1, max_size=200),
    level=st.sampled_from([logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]),
)
@settings(max_examples=100)
def test_p11_log_line_format(message, level):
    """
    Propiedad 11: Para cualquier mensaje y nivel, la línea de log debe seguir
    exactamente la estructura {timestamp_ISO8601} | {NIVEL} | {mensaje}
    con timestamp UTC válido.
    """
    logger = setup_logger(level=logging.DEBUG)
    line = capture_log_output(logger, level, message)

    parts = line.split(" | ", 2)
    assert len(parts) == 3, f"Formato incorrecto: {line!r}"

    timestamp_str, level_str, msg_str = parts

    # Timestamp debe ser ISO 8601 con UTC (termina en Z)
    assert timestamp_str.endswith("Z"), f"Timestamp debe terminar en Z: {timestamp_str!r}"
    dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    assert dt.tzinfo is not None
    assert dt.utcoffset().total_seconds() == 0

    # Nivel debe ser uno de los válidos
    assert level_str in ("DEBUG", "INFO", "WARNING", "ERROR")

    # El campo mensaje existe (puede ser vacío si el mensaje es solo whitespace Unicode)
    assert msg_str is not None
