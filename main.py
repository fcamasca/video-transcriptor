# main.py — Punto de entrada CLI de la aplicación video-transcriptor

import argparse
import sys

from models import AppConfig, DuplicatePolicy, WhisperModel


# Valores válidos para los enums de la CLI
_VALID_LANGUAGES = ["es", "en"]
_VALID_DUPLICATE_POLICIES = [p.value for p in DuplicatePolicy]
_VALID_WHISPER_MODELS = [m.value for m in WhisperModel]


def _enum_validator(valid_values: list, param_name: str):
    """Retorna una función de validación para argparse que rechaza valores fuera del conjunto."""
    def validate(value: str) -> str:
        if value not in valid_values:
            raise argparse.ArgumentTypeError(
                f"Valor inválido '{value}' para {param_name}. "
                f"Valores aceptados: {', '.join(valid_values)}"
            )
        return value
    return validate


def build_parser() -> argparse.ArgumentParser:
    """Construye y retorna el parser de argumentos de la CLI."""
    parser = argparse.ArgumentParser(
        prog="video-transcriptor",
        description="Transcribe y diariza archivos de video de forma local.",
    )

    parser.add_argument(
        "--input-dir",
        default="Videos/",
        metavar="RUTA",
        help="Carpeta de entrada con los archivos de video (default: Videos/)",
    )
    parser.add_argument(
        "--output-dir",
        default="Transcripcion/",
        metavar="RUTA",
        help="Carpeta de salida para los archivos de transcripción (default: Transcripcion/)",
    )
    parser.add_argument(
        "--language",
        default="es",
        type=_enum_validator(_VALID_LANGUAGES, "--language"),
        metavar="{" + ",".join(_VALID_LANGUAGES) + "}",
        help="Idioma para el motor de transcripción (default: es)",
    )
    parser.add_argument(
        "--diarization",
        default="true",
        choices=["true", "false"],
        metavar="{true,false}",
        help="Habilita o deshabilita la diarización de hablantes (default: true)",
    )
    parser.add_argument(
        "--duplicate-policy",
        default="skip",
        type=_enum_validator(_VALID_DUPLICATE_POLICIES, "--duplicate-policy"),
        metavar="{" + ",".join(_VALID_DUPLICATE_POLICIES) + "}",
        help="Política ante transcripciones existentes (default: skip)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="RUTA",
        help="Ruta opcional al archivo de log en disco",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        type=_enum_validator(_VALID_WHISPER_MODELS, "--whisper-model"),
        metavar="{" + ",".join(_VALID_WHISPER_MODELS) + "}",
        help="Modelo de Whisper a utilizar (default: base)",
    )

    return parser


def parse_args(argv=None) -> AppConfig:
    """
    Parsea los argumentos de la CLI y retorna un AppConfig.
    Ante valor enum inválido, argparse imprime el error y sale con código 2.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    return AppConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        language=args.language,
        diarization=(args.diarization == "true"),
        duplicate_policy=DuplicatePolicy(args.duplicate_policy),
        log_file=args.log_file,
        whisper_model=WhisperModel(args.whisper_model),
    )


def main(argv=None) -> int:
    """Punto de entrada principal. Conecta CLI → Logger → Procesador."""
    import logging
    from logger import setup_logger
    from processor import Processor

    config = parse_args(argv)

    logger = setup_logger(
        level=logging.INFO,
        log_file=config.log_file,
    )

    processor = Processor(config=config, logger=logger)
    return processor.run()


if __name__ == "__main__":
    sys.exit(main())
