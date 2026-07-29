$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$stopScript = Join-Path $root 'stop-app.ps1'
$venvPython = Join-Path $root '.venv\Scripts\python.exe'
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { (Get-Command python).Source }
$webDir = Join-Path $root 'web'

Write-Host "Ensuring existing Sheepdog servers are stopped..."
if (Test-Path $stopScript) {
    try {
        & $stopScript
    } catch {
        Write-Host "Warning during stop: $_"
    }
}

Start-Sleep -Seconds 1

Write-Host "Launching Backend API Server (Port 8000) in new PowerShell terminal..."
Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; Write-Host '=== Sheepdog Backend API Server (Port 8000) ===' -ForegroundColor Cyan; & '$pythonExe' -m sheepdog.server"

Write-Host "Launching Frontend Web Server (Port 5173) in new PowerShell terminal..."
Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", "Set-Location '$webDir'; Write-Host '=== Sheepdog Web UI Server (Port 5173) ===' -ForegroundColor Green; npm run dev"

Write-Host "Successfully launched both servers in separate PowerShell terminal windows!" -ForegroundColor Green
