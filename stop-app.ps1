$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$artifactDir = Join-Path $root 'artifacts\startup'
$pidFile = Join-Path $artifactDir 'pids.json'
$backendPort = 8000
$webPort = 5173
$stoppedPidSet = @{}

function Invoke-JsonPost {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [object]$Body = @{}
    )

    return Invoke-RestMethod -Method Post -Uri $Uri -ContentType 'application/json' -TimeoutSec 2 -Body (
        $Body | ConvertTo-Json -Depth 10 -Compress
    )
}

function Wait-ForTrainingShutdown {
    param(
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $status = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:$backendPort/api/training/status" -TimeoutSec 2
            if ($null -eq $status.running -or -not [bool]$status.running) {
                return $true
            }
        }
        catch {
            return $true
        }

        Start-Sleep -Milliseconds 500
    }

    return $false
}

function Stop-ProcessIfRunning {
    param(
        [Parameter(Mandatory = $true)][int]$Id
    )

    $process = Get-Process -Id $Id -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return $false
    }

    Stop-Process -Id $Id -Force -ErrorAction SilentlyContinue
    return $true
}

function Stop-UniqueProcessIfRunning {
    param(
        [Parameter(Mandatory = $true)][int]$Id
    )

    if ($stoppedPidSet.ContainsKey($Id)) {
        return $false
    }

    $didStop = Stop-ProcessIfRunning -Id $Id
    if ($didStop) {
        $stoppedPidSet[$Id] = $true
    }
    return $didStop
}

function Stop-ListenerOnPort {
    param(
        [Parameter(Mandatory = $true)][int]$Port
    )

    $stopped = @()
    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique

    foreach ($listenerPid in ($listeners | Where-Object { $_ -ne $null })) {
        if (Stop-UniqueProcessIfRunning -Id $listenerPid) {
            $stopped += [int]$listenerPid
        }
    }

    return $stopped
}

$stoppedBackend = $false
$stoppedWeb = $false
$watchdogPid = $null
$stoppedByPort = @()

Write-Host 'Stopping sheepdog services...'

try {
    Invoke-JsonPost -Uri "http://127.0.0.1:$backendPort/api/training/stop" | Out-Null
    if (-not (Wait-ForTrainingShutdown -TimeoutSeconds 30)) {
        Write-Host 'Training stop request timed out; forcing process shutdown.'
    }
}
catch {
    Write-Host 'Unable to request a graceful training stop; falling back to process shutdown.'
}

if (Test-Path $pidFile) {
    try {
        $existing = Get-Content $pidFile -Raw | ConvertFrom-Json
        if ($existing.backendListenerPid) {
            $stoppedBackend = Stop-UniqueProcessIfRunning -Id ([int]$existing.backendListenerPid)
        }
        if ($existing.webListenerPid) {
            $stoppedWeb = Stop-UniqueProcessIfRunning -Id ([int]$existing.webListenerPid)
        }
        if ($existing.watchdogPid) {
            $watchdogPid = [int]$existing.watchdogPid
        }
        if ($existing.backendPid) {
            if (-not $stoppedBackend) {
                $stoppedBackend = Stop-UniqueProcessIfRunning -Id ([int]$existing.backendPid)
            }
        }
        if ($existing.webPid) {
            if (-not $stoppedWeb) {
                $stoppedWeb = Stop-UniqueProcessIfRunning -Id ([int]$existing.webPid)
            }
        }
    }
    catch {
        Write-Host 'Ignoring unreadable startup pid file.'
    }
}

$stoppedByPort += Stop-ListenerOnPort -Port $backendPort
$stoppedByPort += Stop-ListenerOnPort -Port $webPort

if ($null -ne $watchdogPid) {
    Start-Sleep -Milliseconds 800
    Stop-UniqueProcessIfRunning -Id $watchdogPid | Out-Null
}

if (Test-Path $pidFile) {
    Remove-Item -Path $pidFile -Force -ErrorAction SilentlyContinue
}

if (-not $stoppedBackend -and -not $stoppedWeb -and $stoppedByPort.Count -eq 0) {
    Write-Host 'No running sheepdog services were found.'
    return
}

if ($stoppedBackend) {
    Write-Host "Stopped backend process from pid file."
}
if ($stoppedWeb) {
    Write-Host "Stopped web process from pid file."
}
if ($stoppedByPort.Count -gt 0) {
    $uniquePids = $stoppedByPort | Sort-Object -Unique
    Write-Host "Stopped additional listener processes by port: $($uniquePids -join ', ')"
}

Write-Host 'Services stopped.'
