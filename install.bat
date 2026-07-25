@echo off
setlocal enabledelayedexpansion

title Acoustic Phase Optimizer - Setup

echo.
echo ============================================
echo  Acoustic Phase Optimizer - Windows Setup
echo ============================================
echo.

:: ---- Check Python ----
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [STEP 1/5] Python not found. Downloading Python 3.12...
    echo.
    echo  =^> Download from: https://www.python.org/downloads/
    echo  =^> Make sure to check "Add Python to PATH" during install
    echo.
    pause
    start https://www.python.org/downloads/
    echo.
    echo  Press any key AFTER installing Python...
    pause >nul
    python --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Python not detected after install. Please restart this script.
        pause
        exit /b 1
    )
) else (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set pyver=%%v
    echo [OK] Python found: %pyver%
)

:: ---- Create virtual environment ----
echo.
echo [STEP 2/5] Creating virtual environment...
if exist venv (
    echo  =^> Existing venv found, removing...
    rmdir /s /q venv
)
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo  [OK] Virtual environment created.

:: ---- Upgrade pip ----
echo.
echo [STEP 3/5] Upgrading pip...
call venv\Scripts\pip install --upgrade pip >nul 2>&1
echo  [OK] pip upgraded.

:: ---- Install package with extras ----
echo.
echo [STEP 4/5] Installing Acoustic Phase Optimizer with GUI + dev extras...
echo  (this may take a few minutes on first run)
echo.

call venv\Scripts\pip install -e ".[gui,dev]"
if %errorlevel% neq 0 (
    echo [WARN] Some dependencies failed. Retrying without GUI extras...
    call venv\Scripts\pip install -e ".[dev]"
)

echo.
echo ============================================
echo  Setup Complete!
echo ============================================
echo.
echo  To run the program:
echo.
echo      venv\Scripts\activate
echo      python -m acoustic_phase_optimizer --gui
echo.
echo  Or run from anywhere with this shortcut:
echo.
echo      venv\Scripts\python -m acoustic_phase_optimizer --gui
echo.
echo  Headless (no GUI):
echo      venv\Scripts\python -m acoustic_phase_optimizer --headless
echo.
pause
