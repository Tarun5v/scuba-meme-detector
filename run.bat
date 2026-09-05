@echo off
rem Launch the scuba meme detector on Windows.
rem Run from the project root.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [!] Virtual environment not found. Run:
    echo     py -3.11 -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt
    exit /b 1
)

".venv\Scripts\python.exe" src\main.py %*