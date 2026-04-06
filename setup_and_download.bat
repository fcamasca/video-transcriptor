@echo off
echo ============================================
echo  Video Transcriptor - Setup inicial
echo ============================================
echo.

REM Verificar Python
py --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado.
    echo Descargalo desde https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python encontrado
py --version

REM Verificar ffmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [!] ffmpeg no esta instalado. Instalando con winget...
    winget install Gyan.FFmpeg --source winget
    echo.
    echo Reinicia esta ventana despues de instalar ffmpeg y vuelve a ejecutar este script.
    pause
    exit /b 1
)
echo [OK] ffmpeg encontrado

REM Instalar dependencias Python
echo.
echo Instalando dependencias Python...
py -m pip install sympy --timeout 120
py -m pip install openai-whisper --timeout 120
py -m pip install pyannote.audio --timeout 120
py -m pip install soundfile --timeout 120
py -m pip install hypothesis pytest --timeout 120

REM Descargar modelo Whisper base
echo.
echo Descargando modelo Whisper 'base' (~150MB)...
py -c "import whisper; whisper.load_model('base')"
echo [OK] Modelo Whisper descargado

REM Descargar modelo pyannote
echo.
set /p TOKEN="Ingresa tu HUGGINGFACE_TOKEN (o Enter para omitir diarizacion): "
if "%TOKEN%"=="" (
    echo [!] Sin token - diarizacion no disponible. Puedes agregar el token despues.
    goto :done
)

REM Guardar token como variable de entorno de usuario (persistente)
setx HUGGINGFACE_TOKEN "%TOKEN%"
echo [OK] HUGGINGFACE_TOKEN guardado como variable de entorno de usuario

echo Descargando modelo pyannote (~1GB)...
py -c "from huggingface_hub import snapshot_download; snapshot_download('pyannote/speaker-diarization-3.1', token='%TOKEN%')"
py -c "from huggingface_hub import snapshot_download; snapshot_download('pyannote/segmentation-3.0', token='%TOKEN%')"
py -c "from huggingface_hub import snapshot_download; snapshot_download('pyannote/speaker-diarization-community-1', token='%TOKEN%')"
py -c "from huggingface_hub import snapshot_download; snapshot_download('pyannote/wespeaker-voxceleb-resnet34-LM', token='%TOKEN%')"
echo [OK] Modelos pyannote descargados

:done
echo.
echo ============================================
echo  Setup completado.
echo  Copia la carpeta .cache a tu PC de oficina:
echo  %USERPROFILE%\.cache\
echo ============================================
pause
