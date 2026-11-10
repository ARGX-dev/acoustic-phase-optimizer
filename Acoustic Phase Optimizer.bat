@echo off
cd /d "%~dp0"
if exist venv\Scripts\pythonw.exe (
    start "" venv\Scripts\pythonw.exe -m acoustic_phase_optimizer
) else (
    echo Run install.bat first.
    pause
)
