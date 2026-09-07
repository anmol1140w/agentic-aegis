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

where cmake >nul 2>nul
if not errorlevel 1 (
  cmake -S native -B native\build -DCMAKE_BUILD_TYPE=Release
  if errorlevel 1 exit /b 1
  cmake --build native\build --config Release
  if errorlevel 1 exit /b 1
  if not exist native\bin mkdir native\bin
  for /r native\build %%F in (aegis-exec.exe) do copy /y "%%F" native\bin\aegis-exec.exe >nul
)

echo AEGIS environment ready.
echo Activate with: .venv\Scripts\activate.bat
echo For Docker support, use PowerShell: .\scripts\setup.ps1 -Extras 'dev,sandbox'
echo Install Ollama separately and pull the tags in config\models.yaml.
