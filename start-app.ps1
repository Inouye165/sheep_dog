$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$pythonExe = (Get-Command python).Source
$npmCmd = (Get-Command npm.cmd).Source
$webDir = Join-Path $root 'web'
$artifactDir = Join-Path $root 'artifacts\startup'
$backendOut = Join-Path $artifactDir 'backend.out.log'
$backendErr = Join-Path $artifactDir 'backend.err.log'
$webOut = Join-Path $artifactDir 'web.out.log'
$webErr = Join-Path $artifactDir 'web.err.log'
$pidFile = Join-Path $artifactDir 'pids.json'

New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null

function Invoke-CommandChecked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [string]$WorkingDirectory = $root
    )

    Push-Location $WorkingDirectory
    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed: $FilePath $($Arguments -join ' ')"
        }
    }
    finally {
        Pop-Location
    }
}

function Start-DetachedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$StandardOutput,
        [Parameter(Mandatory = $true)][string]$StandardError
    )

    return Start-Process `
        -FilePath $FilePath `
        -ArgumentList $Arguments `
        -WorkingDirectory $WorkingDirectory `
        -WindowStyle Hidden `
        -PassThru `
        -RedirectStandardOutput $StandardOutput `
        -RedirectStandardError $StandardError
}

Write-Host 'Checking backend dependencies...'
try {
    Invoke-CommandChecked -FilePath $pythonExe -Arguments @('-c', 'import sheepdog')
}
catch {
    Write-Host 'Installing backend dependencies...'
    Invoke-CommandChecked -FilePath $pythonExe -Arguments @('-m', 'pip', 'install', '-e', '.[dev]')
}

if (-not (Test-Path (Join-Path $webDir 'node_modules'))) {
    Write-Host 'Installing web dependencies...'
    Invoke-CommandChecked -FilePath 'npm' -Arguments @('install') -WorkingDirectory $webDir
}

Write-Host 'Starting backend API on http://127.0.0.1:8000'
$backend = Start-DetachedProcess `
    -FilePath $pythonExe `
    -Arguments @('-m', 'sheepdog.server') `
    -WorkingDirectory $root `
    -StandardOutput $backendOut `
    -StandardError $backendErr

Write-Host 'Starting web viewer on http://127.0.0.1:5173'
$web = Start-DetachedProcess `
    -FilePath $npmCmd `
    -Arguments @('run', 'dev', '--', '--host', '127.0.0.1', '--port', '5173') `
    -WorkingDirectory $webDir `
    -StandardOutput $webOut `
    -StandardError $webErr

[pscustomobject]@{
    startedAt = (Get-Date).ToString('o')
    backendPid = $backend.Id
    webPid = $web.Id
    backendUrl = 'http://127.0.0.1:8000'
    webUrl = 'http://127.0.0.1:5173'
    backendLog = $backendOut
    webLog = $webOut
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $pidFile

Write-Host 'Startup complete.'
Write-Host "Backend logs: $backendOut"
Write-Host "Web logs: $webOut"
Write-Host "Process ids: $pidFile"