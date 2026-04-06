# 🎙️ Video Transcriptor (Local, sin nube)

Transcripción local de archivos de video con identificación de
hablantes.\
Sin APIs externas, sin envío de datos a la nube.

Extrae audio con **ffmpeg**, transcribe con **Whisper** (OpenAI) y
opcionalmente diariza con **pyannote.audio**.

------------------------------------------------------------------------

## 🎯 Motivación

Este proyecto nace para resolver un problema real en entornos
corporativos:

-   restricciones de red que impiden usar servicios externos\
-   necesidad de mantener la información sensible dentro de la
    organización\
-   requerimientos de trazabilidad y control del procesamiento

La solución implementa un pipeline completamente **local, reproducible y
desacoplado**, capaz de operar incluso sin conexión a internet.

------------------------------------------------------------------------

## ⚙️ Características

-   Transcripción local con Whisper (sin APIs externas)
-   Identificación de hablantes con pyannote.audio
-   Extracción de audio con ffmpeg
-   Generación de archivos `.txt` y `.srt`
-   Compatible con entornos offline (incluyendo redes corporativas
    restringidas)
-   Manejo explícito de estados y errores del proceso
-   Suite de pruebas automatizadas (97 tests)

------------------------------------------------------------------------

## 🏗️ Arquitectura

El sistema sigue un pipeline modular:

    Video → AudioExtractor → Transcriber → Diarizer → OutputWriter

Cada componente es independiente y testeable, lo que permite:

-   reemplazo de modelos\
-   ejecución parcial del pipeline\
-   pruebas unitarias por módulo\
-   mantenimiento y extensión sencilla

------------------------------------------------------------------------

## Instalación

### Opción A --- Setup automático (recomendado)

Ejecuta el script incluido desde una red sin restricciones (casa,
hotspot):

    setup_and_download.bat

El script instala ffmpeg si falta, instala todas las dependencias
Python, descarga los modelos de Whisper y pyannote, y al final indica
qué carpeta copiar al PC de oficina si trabajas en red corporativa.

------------------------------------------------------------------------

### Opción B --- Instalación manual

#### 1. Requisitos del sistema

-   Python 3.10+ --- https://www.python.org/downloads/
-   https://ffmpeg.org/download.html instalado y disponible en PATH
-   https://aka.ms/vs/17/release/vc_redist.x64.exe (Windows ---
    requerido por torch)

**Instalar ffmpeg en Windows:**

``` powershell
winget install Gyan.FFmpeg --source winget
```

Cierra y vuelve a abrir la terminal después. Verifica con:

``` bash
ffmpeg -version
```

------------------------------------------------------------------------

#### 2. Dependencias Python

En redes sin restricciones:

``` bash
py -m pip install -r requirements.txt
```

**En redes corporativas con proxy:**

``` bash
py -m pip install sympy --timeout 120
py -m pip install torch-2.10.0+cpu-cp313-cp313-win_amd64.whl --timeout 120
py -m pip install openai-whisper --timeout 120
py -m pip install pyannote.audio --timeout 120
py -m pip install hypothesis pytest
```

------------------------------------------------------------------------

#### 3. Token HuggingFace (solo si usas diarización)

Configurar variable de entorno:

``` bash
# Windows PowerShell
$env:HUGGINGFACE_TOKEN="hf_xxxxxxxxxxxx"
```

Si el token no está presente:

-   la transcripción funciona\
-   pero sin identificación de hablantes

Estado: `TRANSCRIPCION_SIN_DIARIZACION`

------------------------------------------------------------------------

#### 4. Redes corporativas --- copiar modelos

    %USERPROFILE%\.cache\ → C:\Users\TU_USUARIO\.cache\

------------------------------------------------------------------------

## Uso

``` bash
python main.py [opciones]
```

### Ejemplos

``` bash
# Uso básico con defaults
python main.py

# Especificar carpetas y modelo más preciso
python main.py --input-dir Reuniones/ --output-dir Salida/ --whisper-model small

# Transcribir en inglés sin diarización
python main.py --language en --diarization false

# Sobreescribir transcripciones existentes y guardar log
python main.py --duplicate-policy overwrite --log-file transcriptor.log
```

### Formatos de video soportados

`.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`

### Parámetros

| Parámetro | Default | Valores posibles | Descripción |
|---|---|---|---|
| `--input-dir` | `Videos/` | cualquier ruta | Carpeta con los archivos de video |
| `--output-dir` | `Transcripcion/` | cualquier ruta | Carpeta de salida |
| `--language` | `es` | `es`, `en` | Idioma para la transcripción |
| `--diarization` | `true` | `true`, `false` | Habilitar identificación de hablantes |
| `--duplicate-policy` | `skip` | `skip`, `overwrite` | Manejo de archivos ya transcritos |
| `--whisper-model` | `base` | `tiny`, `base`, `small`, `medium`, `large` | Modelo Whisper a utilizar |
| `--log-file` | — | cualquier ruta | Ruta opcional al archivo de log |


------------------------------------------------------------------------

## Ejemplo de salida

    [00:00 - 00:08] Speaker 1: Buenos días a todos...

------------------------------------------------------------------------

## Estados posibles

| Estado | Cuándo ocurre | Significado |
|---|---|---|
| `OK` | Transcripción y diarización completadas | Todo funcionó correctamente |
| `TRANSCRIPCION_SIN_DIARIZACION` | Token ausente, inválido, o error en pyannote | Transcripción disponible, sin identificación de hablantes |
| `SIN_AUDIO` | El video no contiene pista de audio | Se genera el archivo de salida indicando el estado |
| `ERROR_EXTRACCION_AUDIO` | ffmpeg falló al procesar el archivo | No se genera transcripción |
| `ERROR_TRANSCRIPCION` | Whisper falló o el audio es demasiado corto | No se genera transcripción |

------------------------------------------------------------------------

## Estructura del proyecto

    video-transcriptor/
    ├── main.py
    ├── processor.py
    ├── audio_extractor.py
    ├── transcriber.py
    ├── diarizer.py
    ├── output_writer.py
    ├── logger.py
    ├── models.py
    ├── exceptions.py
    ├── tests/

------------------------------------------------------------------------

## Ejecución de tests

``` bash
py -m pytest tests/ -v
```

------------------------------------------------------------------------

## Limitaciones conocidas

-   No subdirectorios\
-   Procesamiento secuencial\
-   Requiere descarga inicial de modelos\
-   GPU Intel no soportada
