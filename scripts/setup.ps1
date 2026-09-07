param(
    [string]$Python = "py -3",
    [string]$Venv = ".venv",
    [string]$Extras = "dev"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

function Invoke-Python([string[]]$Arguments) {
    & cmd.exe /c "$Python -m $($Arguments -join ' ')"
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE" }
}

if (-not (Get-Command py -ErrorAction SilentlyContinue) -and
    -not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.10+ was not found. Install it from https://www.python.org/downloads/windows/"
}

Invoke-Python @("venv", $Venv)
$VenvPython = Join-Path $Venv "Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $VenvPython -m pip install -e ".[${Extras}]"
if ($LASTEXITCODE -ne 0) { throw "AEGIS installation failed" }

Write-Host "AEGIS environment ready. Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Optional sandbox install: .\scripts\setup.ps1 -Extras 'dev,sandbox'"
Write-Host "Install Ollama separately and pull the tags in config\models.yaml."
