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

`stop-app.ps1` stops tracked PIDs from `artifacts/startup/pids.json` and also
cleans up any listeners still bound to ports `8000` and `5173`.

When a training job is active, `stop-app.ps1` now requests a graceful pause/stop first so the last completed checkpoint is persisted before shutdown. On the next `start-app.ps1` launch, you will be asked whether to resume that saved session. If the process was interrupted unexpectedly while training was still running, the backend now resumes automatically from the last persisted safe episode or checkpoint instead of waiting for a prompt.

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