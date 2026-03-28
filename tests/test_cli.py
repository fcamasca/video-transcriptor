# tests/test_cli.py — Tests unitarios y de propiedad para main.py

import sys
import os
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import parse_args, build_parser
from models import AppConfig, DuplicatePolicy, WhisperModel


# --- Tests unitarios ---

class TestHelp:
    def test_help_exits_with_code_0(self):
        with pytest.raises(SystemExit) as exc:
            parse_args(["--help"])
        assert exc.value.code == 0


class TestEnumValidation:
    def test_invalid_language_exits_with_code_2(self):
        with pytest.raises(SystemExit) as exc:
            parse_args(["--language", "fr"])
        assert exc.value.code == 2

    def test_invalid_duplicate_policy_exits_with_code_2(self):
        with pytest.raises(SystemExit) as exc:
            parse_args(["--duplicate-policy", "delete"])
        assert exc.value.code == 2

    def test_invalid_whisper_model_exits_with_code_2(self):
        with pytest.raises(SystemExit) as exc:
            parse_args(["--whisper-model", "xlarge"])
        assert exc.value.code == 2

    def test_invalid_diarization_exits_with_code_2(self):
        with pytest.raises(SystemExit) as exc:
            parse_args(["--diarization", "yes"])
        assert exc.value.code == 2


class TestDefaultValues:
    def test_defaults_build_correct_appconfig(self):
        config = parse_args([])
        assert isinstance(config, AppConfig)
        assert config.input_dir == "Videos/"
        assert config.output_dir == "Transcripcion/"
        assert config.language == "es"
        assert config.diarization is True
        assert config.duplicate_policy == DuplicatePolicy.SKIP
        assert config.log_file is None
        assert config.whisper_model == WhisperModel.BASE


class TestValidParams:
    def test_valid_language_en(self):
        config = parse_args(["--language", "en"])
        assert config.language == "en"

    def test_valid_duplicate_policy_overwrite(self):
        config = parse_args(["--duplicate-policy", "overwrite"])
        assert config.duplicate_policy == DuplicatePolicy.OVERWRITE

    def test_valid_whisper_model_large(self):
        config = parse_args(["--whisper-model", "large"])
        assert config.whisper_model == WhisperModel.LARGE

    def test_diarization_false(self):
        config = parse_args(["--diarization", "false"])
        assert config.diarization is False

    def test_custom_dirs(self):
        config = parse_args(["--input-dir", "in/", "--output-dir", "out/"])
        assert config.input_dir == "in/"
        assert config.output_dir == "out/"

    def test_log_file(self):
        config = parse_args(["--log-file", "app.log"])
        assert config.log_file == "app.log"

    def test_all_whisper_models_valid(self):
        for model in ("tiny", "base", "small", "medium", "large"):
            config = parse_args(["--whisper-model", model])
            assert config.whisper_model == WhisperModel(model)


# --- Test de propiedad P12 ---

_VALID_LANGUAGES = ["es", "en"]
_VALID_DUPLICATE_POLICIES = ["skip", "overwrite"]
_VALID_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]


def _is_invalid(value: str, valid_values: list) -> bool:
    return value not in valid_values


# Feature: video-transcriptor, Propiedad 12: Validación de parámetros enum en CLI
@given(
    language=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Ll",))),
    duplicate_policy=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Ll",))),
    whisper_model=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Ll",))),
)
@settings(max_examples=100)
def test_p12_invalid_enum_exits_with_code_2(language, duplicate_policy, whisper_model):
    """
    Propiedad 12: Para cualquier valor fuera del conjunto válido en parámetros enum,
    la CLI debe rechazarlo con exit code 2 sin iniciar procesamiento.
    """
    if _is_invalid(language, _VALID_LANGUAGES):
        with pytest.raises(SystemExit) as exc:
            parse_args(["--language", language])
        assert exc.value.code == 2

    if _is_invalid(duplicate_policy, _VALID_DUPLICATE_POLICIES):
        with pytest.raises(SystemExit) as exc:
            parse_args(["--duplicate-policy", duplicate_policy])
        assert exc.value.code == 2

    if _is_invalid(whisper_model, _VALID_WHISPER_MODELS):
        with pytest.raises(SystemExit) as exc:
            parse_args(["--whisper-model", whisper_model])
        assert exc.value.code == 2
