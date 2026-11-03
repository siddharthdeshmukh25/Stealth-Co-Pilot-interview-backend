@echo off
setlocal
cd /d "%~dp0"
title Stealth Interview Co-Pilot - Backend

if exist ".venv\Scripts\python.exe" goto venv_ready

set "BASE_PYTHON="
where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 --version >nul 2>&1
    if not errorlevel 1 set "BASE_PYTHON=py -3.11"
)
if not defined BASE_PYTHON (
    where python >nul 2>&1
    if not errorlevel 1 set "BASE_PYTHON=python"
)
if not defined BASE_PYTHON (
    echo ERROR: Python was not found. Install Python 3.11 or newer and try again.
    goto failed
)

echo Creating the backend virtual environment...
%BASE_PYTHON% -m venv .venv
if errorlevel 1 (
    echo ERROR: Could not create backend\.venv.
    goto failed
)

:venv_ready
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: The backend virtual environment is missing its Python executable.
    goto failed
)

where curl.exe >nul 2>&1
if not errorlevel 1 (
    curl.exe --silent --fail http://127.0.0.1:8000/health | findstr /C:"status" >nul
    if not errorlevel 1 (
        echo Backend is already running at http://127.0.0.1:8000
        echo API docs: http://127.0.0.1:8000/docs
        echo Phone on the same Wi-Fi should use one of these backend URLs:
        "%~dp0.venv\Scripts\python.exe" "%~dp0lan_urls.py"
        echo If your phone still says Reconnecting, run scripts\allow_lan_firewall.bat once as administrator.
        echo If you just updated the app, close the old backend window and run this file again.
        exit /b 0
    )
)

"%~dp0.venv\Scripts\python.exe" -c "import fastapi, uvicorn, sqlmodel, fitz, groq, dotenv, multipart" >nul 2>&1
if errorlevel 1 (
    echo Installing backend requirements. This may take a few minutes the first time...
    "%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo ERROR: Dependency installation failed. Check the messages above and your internet connection.
        goto failed
    )
)

if not exist ".env" echo NOTE: backend\.env was not found. Add GROQ_API_KEY to enable AI features.
echo GROQ_API_KEY powers resume parsing, interview answers, and fallback speech transcription.
echo.
echo Starting the backend at http://127.0.0.1:8000 ^(bound to 0.0.0.0 so the phone can connect^)
echo API docs: http://127.0.0.1:8000/docs
echo Phone on the same Wi-Fi should use one of these backend URLs:
"%~dp0.venv\Scripts\python.exe" "%~dp0lan_urls.py"
echo If your phone cannot connect, run scripts\allow_lan_firewall.bat once as administrator.
echo Keep this window open while using the app. Press Ctrl+C to stop the backend.
echo.
"%~dp0.venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
set "BACKEND_EXIT=%ERRORLEVEL%"
if not "%BACKEND_EXIT%"=="0" echo Backend stopped with exit code %BACKEND_EXIT%.
echo.
pause
exit /b %BACKEND_EXIT%

:failed
echo.
pause
exit /b 1
