@echo off
setlocal
cd /d "%~dp0"
title ClipRush Gamer AI v3 - Workspace Local

where ffmpeg >nul 2>nul
if errorlevel 1 goto :missing_ffmpeg
where ffprobe >nul 2>nul
if errorlevel 1 goto :missing_ffmpeg
py -3.11 --version >nul 2>nul
if errorlevel 1 goto :missing_python

if not exist "venv\Scripts\python.exe" (
  echo [ClipRush] Criando ambiente virtual Python 3.11 x64...
  py -3.11 -m venv venv
  if errorlevel 1 goto :missing_python
)

if not exist "venv\cliprush-v3-workspace.ok" (
  echo [ClipRush] Instalando dependencias fixas do workspace. A primeira instalacao requer internet.
  "venv\Scripts\python.exe" -m pip install --upgrade pip
  if errorlevel 1 goto :install_error
  "venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :install_error
  "venv\Scripts\python.exe" -c "import numpy, scipy, soundfile, faster_whisper, ctranslate2, onnxruntime, streamlit, pandas, pyarrow"
  if errorlevel 1 goto :install_error
  "venv\Scripts\python.exe" scripts\validate_library.py
  if errorlevel 1 goto :library_error
  echo workspace v3>"venv\cliprush-v3-workspace.ok"
)

for %%D in (uploads output temp data logs biblioteca) do if not exist "%%D" mkdir "%%D"
if not exist "output\sessoes" mkdir "output\sessoes"
if not exist "temp\cache" mkdir "temp\cache"

if /I "%~1"=="--app" goto :run_app

echo.
echo ClipRush Gamer AI - Motor local
if not "%~1"=="" (
  set "VIDEO=%~1"
) else (
  set /p "VIDEO=Cole o caminho da gameplay MP4/MKV e pressione Enter: "
)
set "VIDEO=%VIDEO:"=%"
if "%VIDEO%"=="" goto :no_video

echo.
echo [ClipRush] Analisando toda a linha do tempo. O progresso fica em output\sessoes.
"venv\Scripts\python.exe" analyzer.py "%VIDEO%" --mode quick --debug
if errorlevel 1 goto :analysis_error
echo.
echo Analise concluida. Use analysis.json no painel ou renderer.py para renderizar.
pause
exit /b 0

:run_app
echo.
echo [ClipRush] Abrindo o workspace local em http://127.0.0.1:8501 ...
"venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1
if errorlevel 1 goto :analysis_error
exit /b 0

:no_video
echo Nenhum caminho recebido. Arraste um MP4/MKV para este arquivo .bat ou cole o caminho acima.
pause
exit /b 0
:missing_ffmpeg
echo ERRO: ffmpeg/ffprobe nao encontrados no PATH. Instale FFmpeg full e reinicie o terminal.
pause
exit /b 1
:missing_python
echo ERRO: Python 3.11 x64/launcher py.exe nao encontrado. Instale de python.org.
pause
exit /b 1
:install_error
echo ERRO ao instalar dependencias. Confira internet, Python 3.11 x64 e tente novamente.
pause
exit /b 1
:library_error
echo ERRO: a biblioteca editorial falhou na validacao. Consulte scripts\validate_library.py.
pause
exit /b 1
:analysis_error
echo ERRO no ClipRush. Consulte logs\app.log ou logs\queue_worker.log e o README.md.
pause
exit /b 1
