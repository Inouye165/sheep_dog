$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$stopScript = Join-Path $root 'stop-app.ps1'
$startScript = Join-Path $root 'start-app.ps1'

if (-not (Test-Path $stopScript)) {
    throw "Missing script: $stopScript"
}
if (-not (Test-Path $startScript)) {
    throw "Missing script: $startScript"
}

Write-Host 'Restarting sheepdog services...'
& $stopScript
& $startScript

Write-Host 'Restart complete.'
