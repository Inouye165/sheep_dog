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