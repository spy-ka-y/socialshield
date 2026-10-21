@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo   SocialShield setup (one time, about 5 min)
echo ============================================
set PY=python
%PY% --version >nul 2>&1 || set PY=py -3
%PY% --version || (echo. & echo Python was not found. Install Python from python.org and tick "Add to PATH". & pause & exit /b 1)

if not exist .venv (
  echo Creating an isolated environment in .venv ...
  %PY% -m venv .venv || (echo Could not create the environment. & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || (echo. & echo Library install FAILED. Copy the error above and send it to Claude. & pause & exit /b 1)

echo.
echo Training models and preparing data ...
python src\train_account_model.py || goto fail
python src\train_post_model.py || goto fail
python src\evaluate.py || goto fail
python src\precompute.py || goto fail
echo.
echo Setup complete. Double-click run_app.bat to start SocialShield.
pause
exit /b 0

:fail
echo.
echo A step FAILED. Copy the error above and send it to Claude.
pause
exit /b 1
