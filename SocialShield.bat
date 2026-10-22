@echo off
setlocal EnableDelayedExpansion
title SocialShield
cd /d "%~dp0"
set "MODEL=qwen2.5:3b"
set "OLLAMA_KEEP_ALIVE=1h"

echo ==========================================
echo   SocialShield is starting
echo   Keep this window open while you use the app.
echo   Close it to stop SocialShield.
echo ==========================================
echo.

rem ---- 1. One-time setup on a new laptop
if exist ".venv\Scripts\activate.bat" goto :find_ollama
echo First run: installing everything. This takes about 5 minutes, only once.
call setup.bat
if not exist ".venv\Scripts\activate.bat" goto :fail

rem ---- 2. Find Ollama
:find_ollama
set "OLLAMA_EXE="
for /f "delims=" %%i in ('where ollama 2^>nul') do if not defined OLLAMA_EXE set "OLLAMA_EXE=%%i"
if not defined OLLAMA_EXE if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not defined OLLAMA_EXE (
  echo [AI] Ollama is not installed. SocialShield will run without the AI second opinion.
  goto :start_app
)

rem ---- 3. Start Ollama in the background if it is not already running
curl -s -o nul -m 2 http://localhost:11434/api/tags && goto :ollama_ready
echo [AI] Starting Ollama in the background...
start "Ollama for SocialShield" /min "%OLLAMA_EXE%" serve
set /a tries=0
:wait_ollama
timeout /t 1 /nobreak >nul
curl -s -o nul -m 2 http://localhost:11434/api/tags && goto :ollama_ready
set /a tries+=1
if !tries! lss 30 goto :wait_ollama
echo [AI] Ollama did not start. SocialShield will run without the AI second opinion.
goto :start_app

:ollama_ready
echo [AI] Ollama is running.

rem ---- 4. Make sure the AI model is downloaded
"%OLLAMA_EXE%" list | findstr /i /c:"%MODEL%" >nul && goto :warm_up
echo [AI] Downloading the AI model %MODEL%, about 2 GB. This happens only once.
"%OLLAMA_EXE%" pull %MODEL%

rem ---- 5. Load the model now so the first scan in the app is fast
:warm_up
echo [AI] Loading the AI model in the background...
> "%TEMP%\socialshield_warmup.json" echo {"model":"%MODEL%","prompt":"hi","stream":false,"keep_alive":"1h"}
start "" /b curl -s -o nul http://localhost:11434/api/generate -d @"%TEMP%\socialshield_warmup.json"

rem ---- 6. Open the app
:start_app
echo.
echo Opening SocialShield in your browser...
call .venv\Scripts\activate.bat
streamlit run app.py
goto :eof

:fail
echo.
echo Setup did not finish. Copy the error above and send it to Claude.
pause
