param(
    # Skip opening the browser once the web UI is ready.
    [switch]$NoBrowser,
    # How long to keep reporting on backend initialization (PyTorch, checkpoint restore, etc.).
    # Exceeding this is only a warning while the backend process is still alive: large runs
    # can take several minutes to restore before the API binds port 8000.
    [int]$BackendTimeoutSeconds = 900,
    # Maximum time to wait for the Vite dev server.
    [int]$WebTimeoutSeconds = 60
)

$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$stopScript = Join-Path $root 'stop-app.ps1'
$venvPython = Join-Path $root '.venv\Scripts\python.exe'
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { (Get-Command python).Source }
$webDir = Join-Path $root 'web'
$backendPort = 8000
$webPort = 5173
# Readiness probe: 503 + Retry-After while training state restores, 200 once the API is usable.
$backendReadyUrl = "http://127.0.0.1:$backendPort/api/ready"
# Liveness probe: always 200 while serving; includes the current restore step for progress output.
$backendHealthUrl = "http://127.0.0.1:$backendPort/api/health"
# Probe and open 127.0.0.1 explicitly; "localhost" may resolve to ::1 while Vite listens on IPv4.
$webUrl = "http://127.0.0.1:$webPort/"

function Test-HttpReady {
    param([Parameter(Mandatory = $true)][string]$Url)

    try {
        $null = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

function Test-TerminalChildAlive {
    # The service terminals run with -NoExit, so the terminal stays open even if the
    # server inside it crashes. Check for a live child process instead.
    param([Parameter(Mandatory = $true)][int]$TerminalPid)

    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $TerminalPid" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne 'conhost.exe' }
    return $null -ne $children -and @($children).Count -gt 0
}

function Get-ServiceState {
    # Returns 'ready', 'exited', or 'starting' for a service hosted in a terminal window.
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$TerminalProcess,
        [Parameter(Mandatory = $true)][double]$ElapsedSeconds
    )

    if (Test-HttpReady -Url $Url) {
        return 'ready'
    }
    if ($TerminalProcess.HasExited) {
        return 'exited'
    }
    # Give the shell a moment to spawn its child before treating "no child" as a crash.
    if ($ElapsedSeconds -gt 5 -and -not (Test-TerminalChildAlive -TerminalPid $TerminalProcess.Id)) {
        return 'exited'
    }
    return 'starting'
}

function Wait-ForPortRelease {
    param(
        [Parameter(Mandatory = $true)][int]$Port,
        [int]$TimeoutSeconds = 15
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Where-Object { $_.OwningProcess -gt 4 }
        if ($null -eq $listeners -or @($listeners).Count -eq 0) {
            return
        }
        Start-Sleep -Milliseconds 200
    }
    $holder = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.OwningProcess -gt 4 } |
        Select-Object -First 1 -ExpandProperty OwningProcess
    if ($null -ne $holder) {
        throw "Port $Port is still held by process $holder after ${TimeoutSeconds}s. Run .\stop-app.ps1 and retry."
    }
}

Write-Host "Ensuring existing Sheepdog servers are stopped..."
if (Test-Path $stopScript) {
    try {
        & $stopScript
    } catch {
        Write-Host "Warning during stop: $_"
    }
}

Wait-ForPortRelease -Port $backendPort
Wait-ForPortRelease -Port $webPort

$artifactDir = Join-Path $root 'artifacts\startup'
if (-not (Test-Path $artifactDir)) { New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null }
$pidFile = Join-Path $artifactDir 'pids.json'

function Save-Pids {
    param([System.Diagnostics.Process]$Backend, [System.Diagnostics.Process]$Web)

    $pids = [ordered]@{
        backendTerminalPid = if ($Backend) { $Backend.Id } else { $null }
        webTerminalPid = if ($Web) { $Web.Id } else { $null }
        createdAt = (Get-Date).ToString("o")
    }
    $pids | ConvertTo-Json | Set-Content $pidFile -Encoding UTF8
}

# Launch both services up front. The Vite dev server does not depend on the backend
# (it only proxies /api), so it must never be gated on backend readiness.
Write-Host "Launching Backend API Server (Port $backendPort) in new PowerShell terminal..."
$backendProc = Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; Write-Host '=== Sheepdog Backend API Server (Port $backendPort) ===' -ForegroundColor Cyan; & '$pythonExe' -m sheepdog.server" -PassThru
# Record the backend pid immediately so stop-app.ps1 can clean up if startup is aborted.
Save-Pids -Backend $backendProc

Write-Host "Launching Frontend Web Server (Port $webPort) in new PowerShell terminal..."
# --strictPort makes Vite fail loudly instead of silently moving to 5174 when 5173 is taken.
$webProc = Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", "Set-Location '$webDir'; Write-Host '=== Sheepdog Web UI Server (Port $webPort) ===' -ForegroundColor Green; npm run dev -- --host 127.0.0.1 --port $webPort --strictPort" -PassThru
Save-Pids -Backend $backendProc -Web $webProc

Write-Host "Waiting for web UI (timeout ${WebTimeoutSeconds}s) and backend API (reporting for up to ${BackendTimeoutSeconds}s)..." -ForegroundColor Cyan

$timer = [System.Diagnostics.Stopwatch]::StartNew()
$webState = 'starting'
$backendState = 'starting'
$browserOpened = $false
$nextProgressAt = 15
# Back off backend readiness probes during long restores (0.5s, 0.75s, ... capped at 5s).
$backendProbeInterval = 0.5
$nextBackendProbeAt = 0.0

while ($true) {
    $elapsed = $timer.Elapsed.TotalSeconds

    if ($webState -eq 'starting') {
        $webState = Get-ServiceState -Url $webUrl -TerminalProcess $webProc -ElapsedSeconds $elapsed
        if ($webState -eq 'starting' -and $elapsed -ge $WebTimeoutSeconds) {
            $webState = 'timeout'
        }
        switch ($webState) {
            'ready' { Write-Host ("Web UI is ready after {0:n1}s at $webUrl" -f $elapsed) -ForegroundColor Green }
            'exited' { Write-Host "Web UI process exited before becoming ready. Check its terminal window for errors." -ForegroundColor Red }
            'timeout' { Write-Host "Web UI did not become ready at $webUrl within ${WebTimeoutSeconds}s. Check its terminal window." -ForegroundColor Red }
        }
    }

    if ($backendState -eq 'starting' -and $elapsed -ge $nextBackendProbeAt) {
        $backendState = Get-ServiceState -Url $backendReadyUrl -TerminalProcess $backendProc -ElapsedSeconds $elapsed
        if ($backendState -eq 'starting' -and $elapsed -ge $BackendTimeoutSeconds) {
            $backendState = 'slow'
        }
        $nextBackendProbeAt = $elapsed + $backendProbeInterval
        $backendProbeInterval = [Math]::Min($backendProbeInterval * 1.5, 5.0)
        switch ($backendState) {
            'ready' { Write-Host ("Backend API is ready after {0:n1}s." -f $elapsed) -ForegroundColor Green }
            'exited' { Write-Host "Backend API process exited before becoming ready. Check its terminal window for errors." -ForegroundColor Red }
            'slow' { Write-Host "Backend API is still restoring after ${BackendTimeoutSeconds}s. Its process is alive, so it was left running; the UI shows progress and connects when it is ready." -ForegroundColor Yellow }
        }
    }

    # Open the browser as soon as the UI is up; it polls and reconnects to the backend on its own.
    if ($webState -eq 'ready' -and -not $browserOpened -and -not $NoBrowser) {
        Write-Host "Opening browser at $webUrl" -ForegroundColor Cyan
        Start-Process $webUrl
        $browserOpened = $true
    }

    if ($webState -ne 'starting' -and $backendState -ne 'starting') {
        break
    }

    if ($backendState -eq 'starting' -and $elapsed -ge $nextProgressAt) {
        # Liveness answers 200 during the restore and reports which step is running.
        $detail = 'backend process starting'
        try {
            $health = Invoke-RestMethod -Uri $backendHealthUrl -TimeoutSec 3 -ErrorAction Stop
            $detail = if ($health.step) { "API is up; restoring: $($health.step)" } else { 'API is up; restoring training state' }
        }
        catch {
            # Not listening yet (process still importing modules); keep the default detail.
        }
        Write-Host ("  ...{0} ({1:n0}s elapsed)" -f $detail, $elapsed)
        $nextProgressAt += 15
    }

    Start-Sleep -Milliseconds 250
}

$webOk = $webState -eq 'ready'
$backendOk = $backendState -in @('ready', 'slow')

if ($webOk -and $backendState -eq 'ready') {
    Write-Host "Successfully launched both servers in separate PowerShell terminal windows!" -ForegroundColor Green
    exit 0
}
if ($webOk -and $backendOk) {
    Write-Host "Web UI is up; backend is still initializing in its terminal window." -ForegroundColor Yellow
    exit 0
}

Write-Host "Startup incomplete (web=$webState, backend=$backendState). Terminal windows were left open for inspection." -ForegroundColor Red
exit 1
