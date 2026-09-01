"""Regression tests for the PowerShell application launcher."""

from pathlib import Path


def test_launcher_serializes_and_starts_services_before_waiting() -> None:
    """Ensure startup prevents races and launches both services concurrently."""

    launcher = Path(__file__).parents[1] / "start-app.ps1"
    launcher_text = launcher.read_text(encoding="utf-8")

    assert "Local\\SheepdogAppStartup" in launcher_text
    assert "Wait-ForPortRelease -Port $backendPort" in launcher_text
    assert launcher_text.index("Starting web viewer") < launcher_text.index(
        "Waiting for backend readiness"
    )
    readiness_function = launcher_text[
        launcher_text.index("function Wait-ForService") : launcher_text.index(
            "function Get-ListenerPid"
        )
    ]
    assert readiness_function.index("Test-HttpReady") < readiness_function.index(
        "Get-Process"
    )
