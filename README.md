# video-transcriptor

Transcripción local de archivos de video con identificación de hablantes. Sin APIs externas, sin envío de datos a la nube.

Extrae el audio con **ffmpeg**, transcribe con **Whisper** (OpenAI) y opcionalmente diariza con **pyannote.audio**. El resultado es un archivo `.txt` por video con timestamps y etiquetas de hablante.

---

## Instalación

### Opción A — Setup automático (recomendado)

Ejecuta el script incluido desde una red sin restricciones (casa, hotspot):

```
setup_and_download.bat
```

El script instala ffmpeg si falta, instala todas las dependencias Python, descarga los modelos de Whisper y pyannote, y al final indica qué carpeta copiar al PC de oficina si trabajas en red corporativa.

---

### Opción B — Instalación manual

#### 1. Requisitos del sistema

- Python 3.10+ — [python.org/downloads](https://www.python.org/downloads/)
- [ffmpeg](https://ffmpeg.org/download.html) instalado y disponible en PATH
- [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) (Windows — requerido por torch)

**Instalar ffmpeg en Windows:**
```powershell
winget install Gyan.FFmpeg --source winget
```
Cierra y vuelve a abrir la terminal después. Verifica con:
```bash
ffmpeg -version
```

#### 2. Dependencias Python

En redes sin restricciones:
```bash
py -m pip install -r requirements.txt
```

**En redes corporativas con proxy** (las descargas grandes pueden cortarse):

```bash
# 1. Instalar sympy por separado primero
py -m pip install sympy --timeout 120

# 2. Descargar torch manualmente desde https://download.pytorch.org/whl/cpu/torch/
#    y luego instalar desde el archivo local:
py -m pip install torch-2.10.0+cpu-cp313-cp313-win_amd64.whl --timeout 120

# 3. Instalar el resto
py -m pip install openai-whisper --timeout 120
py -m pip install pyannote.audio --timeout 120
py -m pip install hypothesis pytest
```

> El wheel de torch para Python 3.13 CPU se puede descargar desde otra red y copiar al equipo.

#### 3. Token HuggingFace (solo si usas diarización)

La diarización requiere:

1. Crear cuenta en [huggingface.co](https://huggingface.co)
2. Aceptar los términos de uso en ambos modelos:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Crear un token en Settings → Access Tokens con permiso: `Read access to contents of public gated repos`
4. Configurar la variable de entorno:

```bash
# Linux / macOS
export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxx

# Windows CMD
set HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxx

# Windows PowerShell
$env:HUGGINGFACE_TOKEN="hf_xxxxxxxxxxxx"
```

Si el token no está presente o es inválido, la aplicación lo detecta al inicio y avisa inmediatamente — el video se transcribe igualmente sin etiquetas de hablante (`Estado: TRANSCRIPCION_SIN_DIARIZACION`).

#### 4. Redes corporativas — copiar modelos desde otro equipo

Si la red corporativa bloquea HuggingFace, descarga los modelos desde casa con `setup_and_download.bat` y copia la carpeta de caché al PC de oficina por USB:

```
%USERPROFILE%\.cache\   →   C:\Users\TU_USUARIO\.cache\
```

Contiene tanto el modelo de Whisper como el de pyannote. Una vez copiada, no se necesita conexión para transcribir ni diarizar.

---

## Uso

```bash
python main.py [opciones]
```

### Parámetros

| Parámetro | Default | Descripción |
|---|---|---|
| `--input-dir` | `Videos/` | Carpeta con los archivos de video |
| `--output-dir` | `Transcripcion/` | Carpeta de salida para los `.txt` |
| `--language` | `es` | Idioma de transcripción (`es`, `en`) |
| `--diarization` | `true` | Habilitar identificación de hablantes (`true`, `false`) |
| `--duplicate-policy` | `skip` | Qué hacer si ya existe transcripción (`skip`, `overwrite`) |
| `--whisper-model` | `base` | Modelo Whisper (`tiny`, `base`, `small`, `medium`, `large`) |
| `--log-file` | — | Ruta opcional para guardar el log en disco |

### Ejemplos

```bash
# Uso básico — transcribe todo lo que haya en Videos/
py main.py

# Sin diarización, modelo más preciso
py main.py --whisper-model medium --diarization false

# Videos en inglés, sobreescribir transcripciones existentes
py main.py --language en --duplicate-policy overwrite

# Guardar log en archivo
py main.py --log-file transcripcion.log

# Carpetas personalizadas
py main.py --input-dir /media/grabaciones --output-dir /media/textos
```

### Formatos de video soportados

`.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`

---

## Ejemplo de salida

Archivo generado: `Transcripcion/reunion_20260319_150923.txt` y `Transcripcion/reunion_20260319_150923.srt`

```
Video: reunion.mp4
Procesado: 2026-03-19T15:09:23Z
Estado: OK

[00:00 - 00:08] Speaker 1: Buenos días a todos, empezamos con el punto uno del orden del día.
[00:08 - 00:21] Speaker 2: Gracias. Como comentaba la semana pasada, el proyecto está en fase de revisión.
[00:21 - 00:35] Speaker 1: Perfecto. ¿Hay alguna observación antes de continuar?
[00:35 - 00:42] Speaker 3: Sí, tengo una pregunta sobre el presupuesto.
```

Por cada video se generan dos archivos en `Transcripcion/`:
- `.txt` — transcripción estructurada con timestamps y hablantes
- `.srt` — subtítulos estándar para VLC, YouTube, editores de video (cárgalo en VLC con `Subtítulos → Añadir archivo de subtítulos`)

```
Video: reunion.mp4
Procesado: 2026-03-19T15:09:23Z
Estado: OK

[00:00 - 00:08] Buenos días a todos, empezamos con el punto uno del orden del día.
[00:08 - 00:21] Como comentaba la semana pasada, el proyecto está en fase de revisión.
```

### Estados posibles

| Estado | Significado |
|---|---|
| `OK` | Transcripción y diarización completadas |
| `TRANSCRIPCION_SIN_DIARIZACION` | Transcripción OK, diarización fallida (sin token, error de red, etc.) |
| `SIN_AUDIO` | El video no contiene pista de audio |
| `ERROR_EXTRACCION_AUDIO` | ffmpeg no pudo procesar el archivo |
| `ERROR_TRANSCRIPCION` | Whisper falló o el audio es menor a 1 segundo |

---

## Estructura del proyecto

```
video-transcriptor/
├── main.py              # Punto de entrada CLI (argparse + wiring)
├── processor.py         # Orquestador del pipeline completo
├── audio_extractor.py   # Extracción de audio con ffmpeg (context manager)
├── transcriber.py       # Transcripción con Whisper
├── diarizer.py          # Diarización con pyannote.audio
├── output_writer.py     # Generación del archivo .txt de salida
├── logger.py            # Logger UTC con formato ISO 8601
├── models.py            # Dataclasses y enums del dominio
├── exceptions.py        # Jerarquía de excepciones tipadas
├── requirements.txt
├── setup_and_download.bat  # Setup automático + descarga de modelos
├── Videos/              # Carpeta de entrada (por defecto)
├── Transcripcion/       # Carpeta de salida (por defecto)
└── tests/
    ├── test_audio_extractor.py
    ├── test_cli.py
    ├── test_diarizer.py
    ├── test_integration.py
    ├── test_logger.py
    ├── test_models_exceptions.py
    ├── test_output_writer.py
    ├── test_processor.py
    └── test_transcriber.py
```

---

## Ejecución de tests

```bash
# Suite completa
py -m pytest tests/ -v

# Un módulo específico
py -m pytest tests/test_processor.py -v

# Solo tests de integración
py -m pytest tests/test_integration.py -v
```

La suite incluye 97 tests: unitarios por componente y tests de propiedad con [Hypothesis](https://hypothesis.readthedocs.io/) que verifican invariantes con datos generados aleatoriamente.

---

## Limitaciones conocidas

- Solo escanea el nivel raíz de `--input-dir`, no subdirectorios
- Los videos se procesan en serie, sin paralelismo
- Audios menores a 1 segundo se descartan (`ERROR_TRANSCRIPCION`)
- La diarización descarga el modelo de pyannote en el primer uso (~1GB, requiere conexión). El setup automático descarga todos los modelos necesarios incluyendo dependencias internas (`speaker-diarization-community-1`, `wespeaker-voxceleb-resnet34-LM`)
- En Windows, la detección de duplicados es case-insensitive (`video.mp4` y `video.MP4` se tratan como el mismo archivo)
- Los modelos Whisper grandes (`medium`, `large`) requieren GPU o tiempo considerable en CPU (modelo `base` en CPU: ~7 min por hora de video)
- La GPU integrada Intel UHD no es compatible con torch — solo se aprovecha GPU NVIDIA (CUDA)
