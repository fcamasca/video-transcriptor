# diarizer.py — Diarización de hablantes con pyannote.audio

import os
import re
from typing import List

from exceptions import DiarizationError, MissingHuggingFaceTokenError
from models import TranscriptionSegment

try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pyannote.audio import Pipeline
except ImportError:
    Pipeline = None  # type: ignore


class Diarizer:
    """Asigna etiquetas de hablante a segmentos de transcripción usando pyannote.audio."""

    def diarize(
        self,
        wav_path: str,
        segments: List[TranscriptionSegment],
    ) -> List[TranscriptionSegment]:
        """
        Ejecuta la diarización y asigna speaker a cada TranscriptionSegment.

        Args:
            wav_path: Ruta al archivo WAV.
            segments: Lista de TranscriptionSegment producidos por el Transcriptor.

        Returns:
            La misma lista con el campo `speaker` asignado en cada segmento.

        Raises:
            MissingHuggingFaceTokenError: si HUGGINGFACE_TOKEN no está en el entorno.
            DiarizationError: ante cualquier fallo de pyannote.
        """
        token = os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            raise MissingHuggingFaceTokenError(
                "La variable de entorno HUGGINGFACE_TOKEN no está configurada."
            )

        if Pipeline is None:
            raise DiarizationError(
                "pyannote.audio no está instalado o no pudo importarse. "
                "Instálalo con: pip install pyannote.audio"
            )

        try:
            import warnings
            import os as _os
            import torch
            import soundfile as sf
            # HF_TOKEN es necesario para que pyannote no haga llamadas sin autenticar
            _os.environ.setdefault("HF_TOKEN", token)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=token,
                )
            # Leemos con soundfile para evitar torchcodec (falla en Windows sin DLLs FFmpeg)
            data, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(data.T)  # (channels, samples)
            duration = waveform.shape[1] / sample_rate

            from tqdm import tqdm
            pbar = tqdm(total=100, desc="diarización", unit="%", bar_format="{l_bar}{bar}| {n:.0f}%")
            last_pct = [0]

            def _progress_hook(step_name, step_artifact, file=None, total=None, completed=None):
                if total and completed is not None:
                    pct = int(completed / total * 100)
                    delta = pct - last_pct[0]
                    if delta > 0:
                        pbar.update(delta)
                        last_pct[0] = pct

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                diarization = pipeline(
                    {"waveform": waveform, "sample_rate": sample_rate},
                    hook=_progress_hook,
                )
            pbar.n = 100
            pbar.refresh()
            pbar.close()
        except (MissingHuggingFaceTokenError, DiarizationError):
            raise
        except Exception as exc:
            raise DiarizationError(f"Error en pyannote.audio: {exc}") from exc

        # pyannote 4.x devuelve DiarizeOutput; la Annotation está en .speaker_diarization
        annotation = diarization.speaker_diarization if hasattr(diarization, "speaker_diarization") else diarization
        diar_segments = list(annotation.itertracks(yield_label=True))
        return self._assign_speakers(segments, diar_segments)

    def _assign_speakers(
        self,
        segments: List[TranscriptionSegment],
        diar_segments: list,
    ) -> List[TranscriptionSegment]:
        """
        Alinea temporalmente los segmentos de transcripción con los de diarización.
        Asigna la etiqueta del hablante con mayor overlap; "Speaker Desconocido" si ninguno.
        Normaliza etiquetas a "Speaker 1", "Speaker 2", etc.

        En caso de empate exacto de overlap entre dos hablantes, gana el primero
        encontrado en la lista de diar_segments (comportamiento determinista: `>`
        estricto, no `>=`).
        """
        # Construir mapa de etiquetas raw → normalizadas
        raw_labels = []
        for _, _, label in diar_segments:
            if label not in raw_labels:
                raw_labels.append(label)
        label_map = {raw: f"Speaker {i + 1}" for i, raw in enumerate(raw_labels)}

        for seg in segments:
            best_label = None
            best_overlap = 0.0

            for turn, _, label in diar_segments:
                overlap = max(0.0, min(seg.end, turn.end) - max(seg.start, turn.start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label = label

            seg.speaker = label_map.get(best_label, "Speaker Desconocido") if best_overlap > 0 else "Speaker Desconocido"

        return segments
