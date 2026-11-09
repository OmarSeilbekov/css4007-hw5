"""Part 3 — persistent storage.

Two separate kinds of memory, in two separate SQLite files under HELPDESK_STORAGE_DIR:

* conversations.sqlite — LangGraph SqliteSaver checkpointer. Key: thread_id.
  Holds the whole graph state of one conversation (messages, current topic, ...).
* user_memory.sqlite   — LangGraph SqliteStore. Namespace: ("users", user_id).
  Holds long-term preferences (reply language) shared by all threads of that user.
"""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from . import data

CHECKPOINT_FILE = "conversations.sqlite"
USER_MEMORY_FILE = "user_memory.sqlite"


def open_checkpointer(storage: Path) -> SqliteSaver:
    storage.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(storage / CHECKPOINT_FILE, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


class UserMemory:
    """Long-term, per-user memory. The user_id always comes from the application."""

    def __init__(self, storage: Path):
        storage.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(storage / USER_MEMORY_FILE, check_same_thread=False, isolation_level=None)
        self.store = SqliteStore(conn)
        self.store.setup()

    @staticmethod
    def _ns(user_id: str) -> tuple[str, str]:
        if not user_id:
            raise ValueError("user_id is required")
        return ("users", user_id)

    def get_preferences(self, user_id: str) -> dict:
        item = self.store.get(self._ns(user_id), "preferences")
        return dict(item.value) if item else {}

    def set_preference(self, user_id: str, key: str, value: str) -> dict:
        prefs = self.get_preferences(user_id)
        prefs[key] = value
        self.store.put(self._ns(user_id), "preferences", prefs)
        return {"saved": True, "user_id": user_id, key: value, "storage": USER_MEMORY_FILE}


def reset_storage(storage: Path | None = None) -> Path:
    """Delete all saved conversations, user preferences and local tickets in storage."""
    storage = storage or data.storage_dir()
    if storage.exists():
        shutil.rmtree(storage)
    storage.mkdir(parents=True, exist_ok=True)
    return storage
