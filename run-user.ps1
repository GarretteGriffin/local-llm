param(
  [int]$Port = 7860
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

$py = Join-Path $repo 'venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
  Write-Host 'Virtualenv not found. Creating venv…'
  python -m venv venv
  $py = Join-Path $repo 'venv\Scripts\python.exe'
}

Write-Host 'Ensuring dependencies are installed…'
& $py -m pip install -r requirements.txt

Write-Host "Starting user service on http://localhost:$Port" 
$env:AUTH_ENABLED = 'false'
$env:ENVIRONMENT = 'production'
& $py main.py --port $Port
