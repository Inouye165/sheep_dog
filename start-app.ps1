$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$venvPython = Join-Path $root '.venv\Scripts\python.exe'
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { (Get-Command python).Source }
$npmCmd = (Get-Command npm.cmd).Source
$webDir = Join-Path $root 'web'
$artifactDir = Join-Path $root 'artifacts\startup'
$backendOut = Join-Path $artifactDir 'backend.out.log'
$backendErr = Join-Path $artifactDir 'backend.err.log'
$webOut = Join-Path $artifactDir 'web.out.log'
$webErr = Join-Path $artifactDir 'web.err.log'
$watchdogLog = Join-Path $artifactDir 'process-watchdog.log'
$watchdogScript = Join-Path $artifactDir 'watch-service-health.ps1'
$pidFile = Join-Path $artifactDir 'pids.json'
$backendPort = 8000
$webPort = 5173
$backendStartupTimeoutSeconds = 180
$webStartupTimeoutSeconds = 45

New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null

function Test-HttpReady {
    param(
        [Parameter(Mandatory = $true)][string]$Url
    )

    try {
        $null = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return $true
    }
    catch {
        return $false
    }
}

function Get-LogTail {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [int]$Lines = 30
    )

    if (-not (Test-Path $Path)) {
        return "(log file missing: $Path)"
    }
    $content = Get-Content $Path -Tail $Lines -ErrorAction SilentlyContinue
    if ($null -eq $content -or $content.Count -eq 0) {
        return "(log is empty: $Path)"
    }
    return ($content -join [Environment]::NewLine)
}

function Wait-ForService {
    param(
        [Parameter(Mandatory = $true)][int]$ProcessId,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$StdOutLog,
        [Parameter(Mandatory = $true)][string]$StdErrLog,
        [int]$TimeoutSeconds = 30
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $processExited = $false
    while ((Get-Date) -lt $deadline) {
        if (Test-HttpReady -Url $Url) {
            return
        }

        if (-not $processExited) {
            $processExited = $null -eq (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
        }

        Start-Sleep -Milliseconds 250
    }

    $outTail = Get-LogTail -Path $StdOutLog
    $errTail = Get-LogTail -Path $StdErrLog
    $processStatus = if ($processExited) { 'launcher process exited' } else { 'launcher process still running' }
    throw "$Name did not become ready at $Url within $TimeoutSeconds seconds ($processStatus).`n--- $Name stdout ---`n$outTail`n--- $Name stderr ---`n$errTail"
}

function Get-ListenerPid {
    param(
        [Parameter(Mandatory = $true)][int]$Port
    )

    $listenerPid = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty OwningProcess
    if ($null -eq $listenerPid) {
        return $null
    }
    return [int]$listenerPid
}

function Ensure-WatchdogScript {
    param(
        [Parameter(Mandatory = $true)][string]$Path
    )

    $scriptContent = @'
param(
    [Parameter(Mandatory = $true)][int]$BackendPort,
    [Parameter(Mandatory = $true)][int]$WebPort,
    [Parameter(Mandatory = $true)][string]$BackendUrl,
    [Parameter(Mandatory = $true)][string]$WebUrl,
    [Parameter(Mandatory = $true)][string]$LogPath,
    [int]$PollMs = 1000
)

$ErrorActionPreference = 'SilentlyContinue'

function Write-WatchdogLine {
    param([Parameter(Mandatory = $true)][string]$Message)

    for ($i = 0; $i -lt 50; $i++) {
        try {
            $stream = [System.IO.File]::Open($LogPath, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
            $writer = New-Object System.IO.StreamWriter($stream, [System.Text.Encoding]::UTF8)
            $writer.WriteLine($Message)
            $writer.Flush()
            $writer.Dispose()
            $stream.Dispose()
            return
        }
        catch {
            Start-Sleep -Milliseconds 50
        }
    }
}

function Get-ListenerPid {
    param([Parameter(Mandatory = $true)][int]$Port)

    $listenerPid = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty OwningProcess
    if ($null -eq $listenerPid) {
        return $null
    }
    return [int]$listenerPid
}

function Test-Ready {
    param([Parameter(Mandatory = $true)][string]$Url)

    try {
        $null = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return $true
    }
    catch {
        return $false
    }
}

$targets = @(
    [pscustomobject]@{ Name = 'backend'; Port = $BackendPort; Url = $BackendUrl },
    [pscustomobject]@{ Name = 'web'; Port = $WebPort; Url = $WebUrl }
)

$state = @{}
foreach ($target in $targets) {
    $state[$target.Name] = [pscustomobject]@{
        LastPid = $null
        LastUp = $null
    }
}

Write-WatchdogLine ("{0}`tWATCHDOG_ARMED`tbackendPort={1}`twebPort={2}" -f (Get-Date).ToString('o'), $BackendPort, $WebPort)

while ($true) {
    foreach ($target in $targets) {
        $currentPid = Get-ListenerPid -Port $target.Port
        $ready = $false
        if ($null -ne $currentPid) {
            $ready = Test-Ready -Url $target.Url
        }

        $up = ($null -ne $currentPid) -and $ready
        $s = $state[$target.Name]

        if ($s.LastPid -ne $currentPid) {
            Write-WatchdogLine ("{0}`tPID_CHANGE`t{1}`told={2}`tnew={3}" -f (Get-Date).ToString('o'), $target.Name, $s.LastPid, $currentPid)
        }

        if ($s.LastUp -ne $up) {
            $reason = if ($null -eq $currentPid) { 'no-listener' } elseif (-not $ready) { 'http-not-ready' } else { 'healthy' }
            $status = if ($up) { 'UP' } else { 'DOWN' }
            Write-WatchdogLine ("{0}`tSTATUS_CHANGE`t{1}`tstatus={2}`tpid={3}`treason={4}" -f (Get-Date).ToString('o'), $target.Name, $status, $currentPid, $reason)
        }

        $s.LastPid = $currentPid
        $s.LastUp = $up
    }

    Start-Sleep -Milliseconds $PollMs
}
'@

    Set-Content -Path $Path -Value $scriptContent -Encoding UTF8
}

function Start-ServiceWatchdog {
    param(
        [Parameter(Mandatory = $true)][int]$BackendPort,
        [Parameter(Mandatory = $true)][int]$WebPort,
        [Parameter(Mandatory = $true)][string]$BackendUrl,
        [Parameter(Mandatory = $true)][string]$WebUrl,
        [Parameter(Mandatory = $true)][string]$ScriptPath,
        [Parameter(Mandatory = $true)][string]$LogPath
    )

    return Start-Process `
        -FilePath 'powershell.exe' `
        -ArgumentList @('-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', $ScriptPath, '-BackendPort', "$BackendPort", '-WebPort', "$WebPort", '-BackendUrl', $BackendUrl, '-WebUrl', $WebUrl, '-LogPath', $LogPath, '-PollMs', '1000') `
        -WindowStyle Hidden `
        -PassThru
}

function Stop-ProcessTree {
    param(
        [Parameter(Mandatory = $true)][int]$Id
    )

    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $Id" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Stop-ProcessTree -Id $child.ProcessId
    }

    $process = Get-Process -Id $Id -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        Stop-Process -Id $Id -Force -ErrorAction SilentlyContinue
    }
}

function Stop-ProcessIfRunning {
    param(
        [Parameter(Mandatory = $true)][int]$Id
    )

    Stop-ProcessTree -Id $Id
}

function Stop-ListenerOnPort {
    param(
        [Parameter(Mandatory = $true)][int]$Port
    )

    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($listenerPid in ($listeners | Where-Object { $_ -ne $null })) {
        Stop-ProcessIfRunning -Id $listenerPid
    }
}

function Stop-ExistingServices {
    if (Test-Path $pidFile) {
        try {
            $existing = Get-Content $pidFile -Raw | ConvertFrom-Json
            if ($existing.backendPid) {
                Stop-ProcessIfRunning -Id ([int]$existing.backendPid)
            }
            if ($existing.webPid) {
                Stop-ProcessIfRunning -Id ([int]$existing.webPid)
            }
            if ($existing.watchdogPid) {
                Stop-ProcessIfRunning -Id ([int]$existing.watchdogPid)
            }
        }
        catch {
            Write-Host 'Ignoring unreadable startup pid file.'
        }
    }

    Stop-ListenerOnPort -Port $backendPort
    Stop-ListenerOnPort -Port $webPort
}

function Wait-ForPortRelease {
    param(
        [Parameter(Mandatory = $true)][int]$Port,
        [int]$TimeoutSeconds = 10
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($null -eq (Get-ListenerPid -Port $Port)) {
            return
        }
        Start-Sleep -Milliseconds 100
    }

    $listenerPid = Get-ListenerPid -Port $Port
    throw "Port $Port is still held by process $listenerPid after ${TimeoutSeconds}s."
}

function Get-TrainingSessionState {
    param(
        [Parameter(Mandatory = $true)][string]$Path
    )

    if (-not (Test-Path $Path)) {
        return $null
    }

    try {
        return Get-Content $Path -Raw | ConvertFrom-Json
    }
    catch {
        return $null
    }
}

function ConvertTo-ResumeRequest {
    param(
        [Parameter(Mandatory = $true)][object]$TrainingRequest,
        [Parameter(Mandatory = $true)][int]$RemainingEpisodes
    )

    $resumeRequest = [ordered]@{}
    foreach ($property in $TrainingRequest.PSObject.Properties) {
        $resumeRequest[$property.Name] = $property.Value
    }
    $resumeRequest['episodes'] = $RemainingEpisodes
    $resumeRequest['resume'] = $true
    return $resumeRequest
}

function Invoke-JsonPost {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][object]$Body
    )

    return Invoke-RestMethod -Method Post -Uri $Uri -ContentType 'application/json' -Body (
        $Body | ConvertTo-Json -Depth 10 -Compress
    )
}

function Get-ResumeTrainingRequest {
    param(
        [AllowNull()][object]$SessionState = $null
    )

    if ($null -eq $SessionState) {
        return $null
    }

    $sessionState = [string]$SessionState.state
    if ($sessionState -notin @('paused', 'stopped')) {
        return $null
    }

    $remainingEpisodes = 0
    if ($null -ne $SessionState.remaining_episodes) {
        $remainingEpisodes = [int]$SessionState.remaining_episodes
    }
    $trainingRequest = $SessionState.training_request
    if ($remainingEpisodes -le 0 -or $null -eq $trainingRequest) {
        return $null
    }

    $prompt = if ($sessionState -eq 'paused') {
        "A paused training session has $remainingEpisodes episodes remaining. Resume it after launch? [y/N]"
    }
    else {
        "A stopped training session has $remainingEpisodes episodes remaining. Resume it after launch? [y/N]"
    }

    $answer = Read-Host $prompt
    if ($answer -match '^(?i)y(es)?$') {
        return [pscustomobject]@{
            request = (ConvertTo-ResumeRequest -TrainingRequest $trainingRequest -RemainingEpisodes $remainingEpisodes)
            remainingEpisodes = $remainingEpisodes
            state = $sessionState
        }
    }

    return $null
}

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

function Start-ManagedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$StandardOutput,
        [Parameter(Mandatory = $true)][string]$StandardError
    )

    if (Test-Path $StandardOutput) {
        Clear-Content -Path $StandardOutput -ErrorAction SilentlyContinue
    }
    else {
        New-Item -ItemType File -Path $StandardOutput -Force | Out-Null
    }

    if (Test-Path $StandardError) {
        Clear-Content -Path $StandardError -ErrorAction SilentlyContinue
    }
    else {
        New-Item -ItemType File -Path $StandardError -Force | Out-Null
    }

    $process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $Arguments `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $StandardOutput `
        -RedirectStandardError $StandardError `
        -WindowStyle Hidden `
        -PassThru

    if ($null -eq $process -or $process.Id -le 0) {
        throw "Failed to start process: $FilePath"
    }

    return $process
}

$startupMutex = New-Object System.Threading.Mutex($false, 'Local\SheepdogAppStartup')
$startupMutexAcquired = $false
try {
    try {
        $startupMutexAcquired = $startupMutex.WaitOne(0)
    }
    catch [System.Threading.AbandonedMutexException] {
        $startupMutexAcquired = $true
    }
    if (-not $startupMutexAcquired) {
        throw 'Another sheepdog startup is already in progress. Wait for it to complete before retrying.'
    }

$startupTimer = [System.Diagnostics.Stopwatch]::StartNew()

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

Write-Host 'Stopping any existing sheepdog services...'
Stop-ExistingServices
Wait-ForPortRelease -Port $backendPort
Wait-ForPortRelease -Port $webPort

$trainingSessionPath = Join-Path $artifactDir 'training-session.json'
$resumeTraining = Get-ResumeTrainingRequest -SessionState (Get-TrainingSessionState -Path $trainingSessionPath)

Ensure-WatchdogScript -Path $watchdogScript
Set-Content -Path $watchdogLog -Value ("{0}`tWATCHDOG_READY" -f (Get-Date).ToString('o')) -Encoding UTF8

Write-Host 'Starting backend API on http://127.0.0.1:8000'
$backend = Start-ManagedProcess `
    -FilePath $pythonExe `
    -Arguments @('-u', '-m', 'sheepdog.server') `
    -WorkingDirectory $root `
    -StandardOutput $backendOut `
    -StandardError $backendErr

Write-Host 'Starting web viewer on http://127.0.0.1:5173'
$web = Start-ManagedProcess `
    -FilePath $npmCmd `
    -Arguments @('run', 'dev', '--', '--host', '127.0.0.1', '--port', '5173') `
    -WorkingDirectory $webDir `
    -StandardOutput $webOut `
    -StandardError $webErr

try {
    Write-Host "Waiting for backend readiness (timeout ${backendStartupTimeoutSeconds}s)..."
    Wait-ForService `
        -ProcessId $backend.Id `
        -Name 'Backend API' `
        -Url "http://127.0.0.1:$backendPort/api/training/status" `
        -StdOutLog $backendOut `
        -StdErrLog $backendErr `
        -TimeoutSeconds $backendStartupTimeoutSeconds
    Write-Host ("Backend ready after {0:n1}s." -f $startupTimer.Elapsed.TotalSeconds)
    if ($null -ne $resumeTraining) {
        Write-Host "Resuming training with $($resumeTraining.remainingEpisodes) remaining episodes..."
        $resumeResponse = Invoke-JsonPost -Uri "http://127.0.0.1:$backendPort/api/training/start" -Body $resumeTraining.request
        $msg = $resumeResponse.message
        if ($null -eq $msg) { $msg = 'Training resume requested.' }
        Write-Host $msg
    }

    Write-Host "Waiting for web readiness (timeout ${webStartupTimeoutSeconds}s)..."
    Wait-ForService `
        -ProcessId $web.Id `
        -Name 'Web viewer' `
        -Url "http://127.0.0.1:$webPort/" `
        -StdOutLog $webOut `
        -StdErrLog $webErr `
        -TimeoutSeconds $webStartupTimeoutSeconds
    Write-Host ("Web ready after {0:n1}s total." -f $startupTimer.Elapsed.TotalSeconds)
}
catch {
    Write-Host 'Startup failed. Cleaning up launched services...'
    Stop-ProcessIfRunning -Id $backend.Id
    if ($web -ne $null) {
        Stop-ProcessIfRunning -Id $web.Id
    }
    throw
}

$backendListenerPid = Get-ListenerPid -Port $backendPort
$webListenerPid = Get-ListenerPid -Port $webPort
$watchdog = Start-ServiceWatchdog `
    -BackendPort $backendPort `
    -WebPort $webPort `
    -BackendUrl "http://127.0.0.1:$backendPort/api/training/status" `
    -WebUrl "http://127.0.0.1:$webPort/" `
    -ScriptPath $watchdogScript `
    -LogPath $watchdogLog

[pscustomobject]@{
    startedAt = (Get-Date).ToString('o')
    backendPid = $backend.Id
    webPid = $web.Id
    watchdogPid = $watchdog.Id
    backendListenerPid = $backendListenerPid
    webListenerPid = $webListenerPid
    backendUrl = "http://127.0.0.1:$backendPort"
    webUrl = "http://127.0.0.1:$webPort"
    backendLog = $backendOut
    webLog = $webOut
    watchdogLog = $watchdogLog
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $pidFile

Write-Host 'Startup complete.'
Write-Host "Backend logs: $backendOut"
Write-Host "Backend errors: $backendErr"
Write-Host "Web logs: $webOut"
Write-Host "Web errors: $webErr"
Write-Host "Watchdog log: $watchdogLog"
Write-Host "Process ids: $pidFile"
}
finally {
    if ($startupMutexAcquired) {
        $startupMutex.ReleaseMutex()
    }
    $startupMutex.Dispose()
}