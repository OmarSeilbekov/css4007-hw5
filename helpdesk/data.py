"""Loading of the supplied fictional fixtures in data/."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def force_utf8_stdio() -> None:
    """Windows consoles/pipes default to cp1251/cp1252, which cannot print Kazakh text."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _load(name: str) -> dict:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


@lru_cache
def helpdesk() -> dict:
    return _load("helpdesk.json")


@lru_cache
def service_status() -> dict:
    return _load("service_status.json")


@lru_cache
def scenarios() -> dict:
    return _load("scenarios.json")


@lru_cache
def ticket_template() -> dict:
    return _load("ticket_template.json")


def status_profiles() -> list[str]:
    return list(service_status()["profiles"].keys())


def storage_dir() -> Path:
    """Directory for live memory (checkpoints, user store). Ignored by git."""
    path = Path(os.environ.get("HELPDESK_STORAGE_DIR", REPO_ROOT / "local-storage"))
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def topics_in(text: str) -> list[str]:
    """Return helpdesk topics whose aliases appear in text (case-insensitive)."""
    lowered = text.lower()
    found = []
    for topic, aliases in helpdesk()["topic_aliases"].items():
        if any(alias in lowered for alias in aliases):
            found.append(topic)
    return found
