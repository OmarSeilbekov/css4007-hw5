"""Part 2 — the three reusable helpdesk tools (+ one memory tool).

Every tool is a plain Python function that works without the agent:

    >>> find_instructions("wifi")["instruction_id"]
    'GUIDE-WIFI'
    >>> check_service_status("wifi", profile="wifi_outage")["status"]
    'outage'

`make_tools(ctx)` wraps them as LangChain tools for the model. Values the model
must NOT choose (status snapshot, user_id, thread_id, ticket folder) are bound
from the ToolContext by the application, not passed as model arguments.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from langchain_core.tools import StructuredTool

from . import data

VALID_STATUS_VALUES = {"operational", "outage", "unknown", "not_checked"}
SECRET_PATTERN = re.compile(
    r"(password\s*(is|=|:)\s*\S+|recovery\s*code\s*(is|=|:)?\s*\S*\d|api[_ ]?key\s*(is|=|:)\s*\S+|sk-[A-Za-z0-9]{10,})",
    re.IGNORECASE,
)


def _normalise_topic(topic: str) -> Optional[str]:
    t = (topic or "").strip().lower()
    if t in data.helpdesk()["instructions"]:
        return t
    found = data.topics_in(t)
    if len(found) == 1:
        return found[0]
    if t in ("account_recovery", "account recovery", "recovery", "login", "account"):
        return "password"
    return None


# ------------------------------------------------------------------ tools
def find_instructions(topic: str) -> dict:
    """Retrieve Wi-Fi or password instructions from data/helpdesk.json."""
    key = _normalise_topic(topic)
    if key is None:
        return {
            "found": False,
            "topic": topic,
            "message": "No local instructions for this topic. Supported topics: wifi, password.",
        }
    guide = data.helpdesk()["instructions"][key]
    return {
        "found": True,
        "topic": key,
        "instruction_id": guide["instruction_id"],
        "title": guide["title"],
        "steps": list(guide["steps"]),
        "source": "data/helpdesk.json",
    }


def check_service_status(service: str, profile: str = "normal") -> dict:
    """Look up a service in the selected fictional status snapshot.

    `service` may be a service name (wifi, account_recovery) or a topic (password).
    """
    profiles = data.service_status()["profiles"]
    if profile not in profiles:
        raise ValueError(f"Unknown status profile {profile!r}; choose from {list(profiles)}")
    name = (service or "").strip().lower()
    topic = _normalise_topic(name)
    if name not in profiles[profile]:
        name = data.helpdesk()["topic_services"].get(topic or "", name)
    entry = profiles[profile].get(name)
    if entry is None:
        return {
            "service": service,
            "status": "unknown",
            "message": "This service is not listed in the demo status data.",
            "status_profile": profile,
            "source": "data/service_status.json",
        }
    return {
        "service": name,
        "status": entry["status"],
        "message": entry["message"],
        "status_profile": profile,
        "source": "data/service_status.json",
    }


def prepare_ticket(
    category: str,
    summary: str,
    steps_tried: list[str],
    service_status: str = "not_checked",
    *,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    status_profile: Optional[str] = None,
    ticket_dir: Path | str = "local-storage/tickets",
) -> dict:
    """Write a LOCAL ticket draft (JSON). Nothing is sent anywhere.

    Refuses (writes nothing) when the problem description or steps tried are
    missing, or when the text appears to contain a secret.
    """
    summary = (summary or "").strip()
    steps = [s.strip() for s in (steps_tried or []) if s and s.strip()]
    missing = []
    if not summary:
        missing.append("problem description")
    if not steps:
        missing.append("steps already tried")
    if missing:
        return {
            "created": False,
            "missing": missing,
            "message": "Ask the user for: " + ", ".join(missing) + ". Do not invent details.",
        }
    if SECRET_PATTERN.search(" ".join([summary, *steps])):
        return {
            "created": False,
            "message": "The text looks like it contains a password, recovery code or key. "
            "Remove it; tickets must never contain secrets.",
        }
    status = (service_status or "not_checked").strip().lower().replace(" ", "_")
    if status not in VALID_STATUS_VALUES:
        status = "not_checked"
    topic = _normalise_topic(category) or (category or "other").strip().lower()
    ref = data.helpdesk()["instructions"].get(topic, {}).get("instruction_id")

    ticket = json.loads(json.dumps(data.ticket_template()["ticket"]))
    now = datetime.now(timezone.utc)
    ticket.update(
        ticket_id=f"DRAFT-{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
        status="draft",
        user_id=user_id,
        thread_id=thread_id,
        category=topic,
        summary=summary,
        steps_tried=steps,
        service_status=status,
        status_profile=status_profile,
        reference_ids=[ref] if ref else [],
    )
    ticket_dir = Path(ticket_dir)
    ticket_dir.mkdir(parents=True, exist_ok=True)
    path = ticket_dir / f"{ticket['ticket_id']}.json"
    path.write_text(json.dumps({"ticket": ticket}, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        shown = str(path.resolve().relative_to(data.REPO_ROOT))
    except ValueError:
        shown = str(path)
    return {
        "created": True,
        "sent": False,
        "ticket_id": ticket["ticket_id"],
        "file": shown,
        "ticket": ticket,
        "message": "Local draft saved. Nothing was sent to a helpdesk.",
    }


# ------------------------------------------------------------ LangChain wrappers
@dataclass
class ToolContext:
    """Session values chosen by the application, never by the model."""

    status_profile: str = "normal"
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    ticket_dir: Path = field(default_factory=lambda: data.storage_dir() / "tickets")
    save_preference: Optional[Callable[[str, str], dict]] = None  # (key, value) -> result


def make_tools(ctx: ToolContext) -> list[StructuredTool]:
    def _find_instructions(topic: str) -> str:
        """Look up local helpdesk instructions. topic: 'wifi' or 'password'."""
        return json.dumps(find_instructions(topic), ensure_ascii=False)

    def _check_service_status(service: str) -> str:
        """Check the demo status of a service: 'wifi' or 'account_recovery' (password topic).
        Status values: operational, outage, unknown (unknown is NOT operational)."""
        return json.dumps(check_service_status(service, profile=ctx.status_profile), ensure_ascii=False)

    def _prepare_ticket(
        category: str, summary: str, steps_tried: list[str], service_status: str = "not_checked"
    ) -> str:
        """Save a LOCAL ticket draft (nothing is sent). category: wifi|password.
        summary: the user's problem in their words. steps_tried: only steps the user said
        they tried. service_status: operational|outage|unknown|not_checked, taken from a
        check_service_status result in this conversation, otherwise not_checked."""
        return json.dumps(
            prepare_ticket(
                category, summary, steps_tried, service_status,
                user_id=ctx.user_id, thread_id=ctx.thread_id,
                status_profile=ctx.status_profile, ticket_dir=ctx.ticket_dir,
            ),
            ensure_ascii=False,
        )

    def _save_user_preference(language: str) -> str:
        """Persist the CURRENT user's preferred reply language (e.g. 'Kazakh') so later
        conversations can use it. Call only when the user explicitly asks to remember it."""
        if ctx.save_preference is None:
            return json.dumps({"saved": False, "message": "No user memory configured."})
        return json.dumps(ctx.save_preference("language", language.strip()), ensure_ascii=False)

    return [
        StructuredTool.from_function(_find_instructions, name="find_instructions"),
        StructuredTool.from_function(_check_service_status, name="check_service_status"),
        StructuredTool.from_function(_prepare_ticket, name="prepare_ticket"),
        StructuredTool.from_function(_save_user_preference, name="save_user_preference"),
    ]
