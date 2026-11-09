"""Terminal helpdesk (Parts 2–3).

Interactive:
    python -m helpdesk.cli --user demo-user-a --thread t1 --status normal --model openai
One-shot (each launch is a new process using the same saved storage):
    python -m helpdesk.cli --user demo-user-a --thread t1 -m "My issue is Wi-Fi..."
Reset all saved memory:
    python -m helpdesk.cli --reset
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import data
from .agent import HelpdeskSession
from .memory import reset_storage


def print_trace(trace: dict) -> None:
    print(f"  · model: {trace['model']}")
    src = trace["sources"]
    print(f"  · conversation memory (thread {src['conversation_memory']['checkpoint_thread_id']}): "
          f"{src['conversation_memory']['prior_messages_loaded']} prior messages")
    print(f"  · user memory ({trace['user_id']}): {src['user_memory']['preferences_loaded'] or 'none'}")
    for step in trace["steps"]:
        for m in step["update"].get("messages(appended)", []):
            if m["type"] == "model_tool_request":
                for c in m["tool_calls"]:
                    print(f"  · [{step['node']}] model requests {c['name']}({json.dumps(c['args'], ensure_ascii=False)})")
            elif m["type"] == "tool_result":
                c = m["content"]
                brief = {k: c[k] for k in ("status", "instruction_id", "file", "created", "saved", "missing")
                         if isinstance(c, dict) and k in c}
                print(f"  · [{step['node']}] {m['name']} -> {brief or c}")
    if trace["sources"]["reference_file"]:
        print(f"  · reference file: {', '.join(trace['sources']['reference_file'])}")
    if trace["ticket_files"]:
        print(f"  · ticket draft: {', '.join(trace['ticket_files'])}")
    print(f"  · tool rounds: {trace['tool_rounds']}  loop limit reached: {trace['loop_limit_reached']}")


def main() -> None:
    p = argparse.ArgumentParser(description="HW5 LangGraph IT helpdesk (fictional data)")
    p.add_argument("--user", default="demo-user-a", help="invented user identifier")
    p.add_argument("--thread", default="default", help="conversation / thread identifier")
    p.add_argument("--status", default="normal", choices=data.status_profiles(), help="service-status snapshot")
    p.add_argument("--model", default=os.environ.get("HELPDESK_MODEL", "openai"), choices=["openai", "fake"])
    p.add_argument("--storage", help="storage directory (default: $HELPDESK_STORAGE_DIR or local-storage)")
    p.add_argument("-m", "--message", action="append", help="send message(s) and exit")
    p.add_argument("--no-trace", action="store_true")
    p.add_argument("--json", action="store_true", help="print full JSON traces")
    p.add_argument("--reset", action="store_true", help="delete saved conversations, preferences and tickets")
    args = p.parse_args()

    if args.storage:
        os.environ["HELPDESK_STORAGE_DIR"] = args.storage
    storage = data.storage_dir()
    if args.reset:
        print(f"Cleared {reset_storage(storage)}")
        return

    session = HelpdeskSession(args.user, args.thread, args.status, args.model, storage=Path(storage))
    print(f"[helpdesk] pid={os.getpid()} user={args.user} thread={args.thread} "
          f"status={args.status} storage={storage}")

    def handle(text: str) -> None:
        trace = session.send(text)
        print(f"\nYou: {text}\nHelpdesk: {trace['reply']}")
        if args.json:
            print(json.dumps(trace, indent=2, ensure_ascii=False, default=str))
        elif not args.no_trace:
            print_trace(trace)

    if args.message:
        for text in args.message:
            handle(text)
        return
    print("Type a message, or /quit to exit.")
    while True:
        try:
            text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in ("/quit", "/exit"):
            break
        if text:
            handle(text)


if __name__ == "__main__":
    main()
