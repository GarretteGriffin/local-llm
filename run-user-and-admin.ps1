param(
  [int]$Port = 7860,
  [int]$AdminPort = 7861
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
Write-Host "Starting admin service on http://localhost:$AdminPort" 

$env:AUTH_ENABLED = 'false'
$env:ADMIN_AUTH_ENABLED = 'false'
$env:ENVIRONMENT = 'production'

& $py main.py --admin --port $Port --admin-port $AdminPort
