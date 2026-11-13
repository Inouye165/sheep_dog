# Startup Guide

Updated: 2026-04-21 08:42:44 -07:00

## How to Start the App

Run this single command from the repository root:

```powershell
.\start-app.ps1
```

That command will:

- install the backend in editable mode if `sheepdog` is not already importable
- install web dependencies if `web/node_modules` is missing
- start the Python API server at `http://127.0.0.1:8000`
- start the Vite viewer at `http://127.0.0.1:5173`
- write process ids and log file paths to `artifacts/startup/pids.json`

Open the web viewer after both services report that they are running.

## Stop and Restart

Stop both backend and web services:

```powershell
.\stop-app.ps1
```

Restart both services in one command:

```powershell
.\restart-app.ps1
```

Start both services in dedicated, visible PowerShell terminal windows:

```powershell
.\start-terminals.ps1
```

`stop-app.ps1` stops tracked PIDs from `artifacts/startup/pids.json` and also
cleans up any listeners still bound to ports `8000` and `5173`.

When a training job is active, `stop-app.ps1` now requests a graceful pause/stop first so the last completed checkpoint is persisted before shutdown. On the next `start-app.ps1` launch, you will be asked whether to resume that saved session. If the process was interrupted unexpectedly while training was still running, the backend now resumes automatically from the last persisted safe episode or checkpoint instead of waiting for a prompt.

## Intermittent Vite connection refusals

`ECONNREFUSED 127.0.0.1:8000` does not always mean the backend exited.
Check `/api/health` and the port listener before restarting. Python 3.12's
HTTP server defaults to only five pending connections; dashboard request bursts
can exhaust that queue while training delays the accept loop. The backend now
uses a backlog of 128, covered by an isolated 24-connection regression test.
An already-running backend must be restarted to pick up this change. Stop active
training gracefully and wait for checkpoint persistence before restarting it.
This does not make a stopped backend available or guarantee unlimited concurrency.

## Manual Start Commands

If you need the separate commands the launcher uses, they are:

```powershell
python -m sheepdog.server
cd web
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

## Running Log

| Timestamp | Issue | Fix | Prevention |
| --- | --- | --- | --- |
| 2026-04-21 06:43:40 -07:00 | The repo had split startup steps, so there was no single command to boot the backend and web viewer together. | Added `start-app.ps1` at the repo root and documented it here as the one command to launch both services. | Keep this launcher and the startup guide in sync whenever ports, entrypoints, or dependency setup changes. |
| 2026-04-21 06:53:42 -07:00 | The first launcher version tried to start npm through a PowerShell wrapper, which Windows rejected with a Win32 application error. | Switched the launcher to the real `python.exe` and `node.exe` executables and invoked npm through Node's `npm-cli.js`. | Keep the launcher pinned to executable paths instead of shell wrappers when the command needs to run detached. |
| 2026-04-21 06:57:57 -07:00 | The Node-based npm launch still failed because the spaced path was passed through to Node as a module argument. | Switched the web launcher to the native `npm.cmd` wrapper so the command line is quoted correctly by Windows. | Prefer the native npm wrapper for detached launches when the command includes paths with spaces. |
| 2026-04-21 08:42:44 -07:00 | The previous startup artifacts were stale, so the recorded PIDs no longer matched live services. | Re-ran `./start-app.ps1` from the repo root and verified `http://127.0.0.1:8000/api/health` and `http://127.0.0.1:5173` both responded successfully. | Treat `artifacts/startup/pids.json` as a point-in-time record and confirm the URLs after a fresh launch when validating startup. |
| 2026-04-22 11:00:00 -07:00 | Re-running the launcher could leave an older backend bound to port 8000, so the UI hit stale routes like `/api/training/clear` and got 404s. | Updated `start-app.ps1` to stop prior tracked processes and any listeners already bound to ports `8000` and `5173` before launching fresh services. | Make the launcher responsible for replacing old services instead of assuming the previous processes are already gone. |
| 2026-04-22 11:05:00 -07:00 | The launcher could still pick a global Python from PATH instead of the repo's virtual environment, which risks starting a different installed `sheepdog` package. | Updated `start-app.ps1` to prefer `.venv\Scripts\python.exe` when that interpreter exists. | Prefer the repository interpreter for startup so the running backend matches the code and dependencies in the workspace. |
| 2026-04-22 11:00:00 -07:00 | Re-running the launcher could leave an older backend bound to port 8000, so the UI hit stale routes like `/api/training/clear` and got 404s. | Updated `start-app.ps1` to stop prior tracked processes and any listeners already bound to ports `8000` and `5173` before launching fresh services. | Make the launcher responsible for replacing old services instead of assuming the previous processes are already gone. |
| 2026-10-06 06:50:00 -07:00 | `start-terminals.ps1` only launched the web server after the backend was ready. Restoring training state now takes about 3.5 minutes before port 8000 opens, so the 180s timeout ran out and the script exited before it ever launched the web server. The backend came up later on its own. | Both launchers now start the web server right away and check it separately from the backend. Readiness checks use the lightweight `/api/health` endpoint (`/api/training/status` returns ~200 KB). A backend that is slow but still alive gets a warning instead of a failure. A crashed process is reported immediately. Vite runs with `--strictPort`. The backend now logs how long the restore took. | Never make the web server wait for the backend: it only proxies `/api`. Check the `training state restored in Xs` line in the backend log if startup gets slower. |
| 2026-10-06 06:51:00 -07:00 | Vite was listening on 5173 but did not answer. It pinned a CPU core because its file watcher was crawling `web/public/generated` (34k files, 20 GB of backend exports) and `web/dist` (10 GB). | Added `server.watch.ignored` for `public/generated`, `dist` and `artifacts` in `web/vite.config.ts`. `/generated` is already proxied to the backend. | Keep generated or bulky output out of Vite's watch set. Consider pruning old exports in `web/public/generated`. |
| 2026-10-06 07:20:00 -07:00 | Port 8000 stayed closed for the whole multi-minute restore, so a slow start looked the same as a crash and the UI just showed "Backend disconnected". | `sheepdog.server.main()` now opens port 8000 right away and restores training state on a background thread. During the restore, `/api/health` answers `503 {"status":"starting"}`, other API routes answer `503 BACKEND_STARTING`, and `/generated/*` and `/api/training/history` keep working. The UI shows an amber "Restoring state…" pill and retries every 2s. A failed restore still makes the process exit with code 1. The port is now bound exclusively on Windows, so a second backend can't share it. | New routes that don't use `TrainingManager` can be added to `_STARTUP_EXEMPT_PATHS`. Everything else is blocked automatically until the restore finishes. Covered by `tests/test_server_startup.py`. || 2026-10-06 13:20:00 -07:00 | The backend log filled with `GET /api/health 503` / `GET /api/training/status 503` lines about twice a second during the multi-minute restore. Probes ran at a fixed rate, and liveness and readiness were mixed together. Running py-spy on the slow restore showed `_resolve_precedence_state` opening and parsing every `artifacts/checkpoints/checkpoint-*.json` (14,417 files, 1.9 GB) on every start. | Added `sheepdog.checkpoints.summary_cache`, a sidecar cache at `artifacts/cache/checkpoint-summaries.json`. It is keyed by file size and mtime, so only new or changed checkpoints get parsed. Split the probes: `/api/health` is liveness (always 200, reports `ready`/`step`), and `/api/ready` is readiness (503 with `Retry-After` until restored). Startup 503s and probes no longer appear in the access log. The server now prints the restore steps, checkpoint-index progress, and a 15s heartbeat. Both launchers poll `/api/ready` with backoff (capped at 5s). Early 503 replies now drain the POST body first to avoid WinError 10053 resets. |
