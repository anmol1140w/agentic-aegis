@echo off
setlocal
cd /d "%~dp0.."

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher ^(py^) was not found. Install Python 3.10+ and enable the launcher.
  exit /b 1
)

py -3 -m venv .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -e ".[dev]"
if errorlevel 1 exit /b 1

echo AEGIS environment ready.
echo Activate with: .venv\Scripts\activate.bat
echo For Docker support, use PowerShell: .\scripts\setup.ps1 -Extras 'dev,sandbox'
echo Install Ollama separately and pull the tags in config\models.yaml.
