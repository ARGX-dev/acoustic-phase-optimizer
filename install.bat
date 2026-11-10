@echo off
setlocal
title Acoustic Phase Optimizer - Setup
cd /d "%~dp0"

echo.
echo  Acoustic Phase Optimizer - setup
echo  ================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo  Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/
    echo  and tick "Add Python to PATH", then run this again.
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist venv (
    echo  Creating a private Python environment...
    python -m venv venv || goto :fail
)

echo  Installing (first time takes a minute)...
call venv\Scripts\python -m pip install --upgrade pip >nul
call venv\Scripts\python -m pip install -e . || goto :fail

echo.
echo  Done. Start the app with "Acoustic Phase Optimizer.bat".
echo.
pause
exit /b 0

:fail
echo.
echo  Setup failed. Scroll up for the error.
pause
exit /b 1
