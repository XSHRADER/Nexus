@echo off
REM ===========================================================
REM  NEXUS AI - one-click start
REM
REM  Double-click this file, or run it with any run.py flag:
REM      start.bat --check     preflight only
REM      start.bat --eval      score retrieval quality
REM      start.bat --rebuild   force a full re-index first
REM      start.bat --pull      download a model if none installed
REM
REM  Creates the virtual environment and installs dependencies on
REM  first run, then hands over to run.py, which starts Ollama if
REM  it isn't running, indexes new documents, proves the pipeline
REM  end to end, and opens the UI.
REM ===========================================================

cd /d "%~dp0"
title NEXUS AI

echo.
echo  ==========================================
echo    NEXUS AI
echo  ==========================================
echo.

set "VENV=%~dp0nexus-env"
set "PY=%VENV%\Scripts\python.exe"

REM --- 1. virtual environment --------------------------------
if not exist "%PY%" (
    echo  [setup] No virtual environment found.
    where python >nul 2>&1
    if errorlevel 1 (
        echo  [!!] Python is not on PATH.
        echo       Install Python 3.10+ from https://python.org and re-run.
        goto :setup_failed
    )
    echo  [setup] Creating nexus-env...
    python -m venv "%VENV%"
    if errorlevel 1 goto :setup_failed
    echo  [setup] Installing dependencies. First run only - this takes a few minutes.
    "%PY%" -m pip install --upgrade pip --quiet
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto :setup_failed
    echo  [setup] Done.
    echo.
)

REM --- 2. are the dependencies actually importable? ----------
"%PY%" -c "import streamlit, chromadb, sentence_transformers, rank_bm25" >nul 2>&1
if errorlevel 1 (
    echo  [setup] Installing missing dependencies...
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto :setup_failed
    echo.
)

REM --- 3. hand over to run.py --------------------------------
REM run.py does the rest: starts Ollama if needed, indexes new
REM documents, runs an end-to-end smoke test, launches the UI.
"%PY%" run.py %*
set "RC=%errorlevel%"

if not "%RC%"=="0" (
    echo.
    echo  NEXUS exited with code %RC%.
    pause
)
exit /b %RC%

:setup_failed
echo.
echo  ------------------------------------------
echo   Setup failed. See the messages above.
echo  ------------------------------------------
pause
exit /b 1
