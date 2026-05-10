@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo [ERROR] venv not found. Create it first:
  echo   py -3.11 -m venv venv
  echo   venv\Scripts\pip.exe install -r requirements.txt
  pause
  exit /b 1
)

venv\Scripts\python.exe run_ui.py
endlocal

