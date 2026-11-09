"""Run the supplied scenarios S01–S14 from data/scenarios.json and save logs in outputs/.

    python -m helpdesk.scenarios --model openai            # all scenarios, real model
    python -m helpdesk.scenarios --model fake --only S11   # one scenario, fake model

* reset_before=true  -> the scenario's storage directory is deleted first.
* restart_process    -> events are split into segments; EACH segment runs in a NEW
                        Python process (subprocess) against the same storage. The
                        child PIDs are written to the log as evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import data, part1
from .memory import reset_storage

PART1_LOG_DIR = data.OUTPUTS_DIR / "part1"


def _segments(events: list[dict]) -> list[list[dict]]:
    segs: list[list[dict]] = [[]]
    for e in events:
        if e["action"] == "restart_process":
            segs.append([])
        else:
            segs[-1].append(e)
    return [s for s in segs if s]


def _rel(value: str) -> str:
    try:
        return str(Path(value).resolve().relative_to(data.REPO_ROOT))
    except (ValueError, OSError):
        return value


def _scenario(sid: str) -> dict:
    return next(s for s in data.scenarios()["scenarios"] if s["id"] == sid)


def scenario_storage(sid: str, model: str) -> Path:
    return data.storage_dir() / "scenarios" / model / sid


# ---------------------------------------------------------------- worker (child process)
def run_worker(sid: str, segment: int, model: str, storage: Path, ticket_dir: Path) -> None:
    from .agent import HelpdeskSession

    sc = _scenario(sid)
    traces = []
    sessions: dict[tuple, HelpdeskSession] = {}
    for event in _segments(sc["events"])[segment]:
        key = (event["user_id"], event["thread_id"])
        if key not in sessions:
            sessions[key] = HelpdeskSession(event["user_id"], event["thread_id"], sc["status_profile"],
                                            model, storage=storage, ticket_dir=ticket_dir)
        trace = sessions[key].send(event["message"])
        trace["pid"] = os.getpid()
        traces.append(trace)
    print("@@TRACES@@" + json.dumps(traces, ensure_ascii=False, default=str))


# ---------------------------------------------------------------- parent
def run_part1(sc: dict) -> dict:
    variant = sc.get("graph_variant", "original")
    msg = sc["events"][0]["message"]
    result = part1.run_with_trace(msg, variant)
    log = {"scenario": sc["id"], "title": sc["title"], "part": 1, "graph_variant": variant,
           "status_profile": sc["status_profile"], "user_id": sc["events"][0]["user_id"],
           "thread_id": sc["events"][0]["thread_id"],
           "command": f'python -m helpdesk.part1 --variant {variant} "{msg}"',
           "model": "none (Part 1 has no LLM)", **result}
    PART1_LOG_DIR.mkdir(parents=True, exist_ok=True)
    (PART1_LOG_DIR / f"{sc['id']}.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [f"# {sc['id']} — {sc['title']}", "", f"- Graph variant: `{variant}`",
          f"- Snapshot: `{sc['status_profile']}` (not used in Part 1)",
          f"- User / thread: `{log['user_id']}` / `{log['thread_id']}`",
          f"- Command: `{log['command']}`", "- Model: none", "",
          f"**Route:** {' → '.join(result['route'])}", "", "| Node | Update | State after node |", "|---|---|---|"]
    for s in result["steps"]:
        md.append(f"| {s['node']} | `{json.dumps(s['update'], ensure_ascii=False)}` | "
                  f"`{json.dumps(s['state_after'], ensure_ascii=False)}` |")
    md += ["", "**Final reply:**", "", "```", result["final_state"]["reply"], "```", ""]
    (PART1_LOG_DIR / f"{sc['id']}.md").write_text("\n".join(md), encoding="utf-8")
    return log


def run_agent_scenario(sc: dict, model: str, log_dir: Path) -> dict:
    storage = scenario_storage(sc["id"], model)
    if sc.get("reset_before"):
        reset_storage(storage)
    ticket_dir = log_dir / "tickets"
    turns, commands = [], []
    segments = _segments(sc["events"])
    for i, _ in enumerate(segments):
        cmd = [sys.executable, "-m", "helpdesk.scenarios", "--worker", sc["id"], "--segment", str(i),
               "--model", model, "--storage", str(storage), "--ticket-dir", str(ticket_dir)]
        commands.append(" ".join(["python"] + [_rel(c) for c in cmd[1:]]))
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=data.REPO_ROOT, env=os.environ.copy())
        if proc.returncode != 0:
            raise RuntimeError(f"{sc['id']} segment {i} failed:\n{proc.stderr}")
        payload = proc.stdout.split("@@TRACES@@", 1)[1]
        for t in json.loads(payload):
            t["process_segment"] = i
            turns.append(t)
    log = {"scenario": sc["id"], "title": sc["title"], "part": sc["part"], "status_profile": sc["status_profile"],
           "model": turns[0]["model"], "reset_before": sc.get("reset_before"),
           "process_restarts": len(segments) - 1, "commands": commands, "turns": turns}
    (log_dir / f"{sc['id']}.json").write_text(json.dumps(log, indent=2, ensure_ascii=False, default=str),
                                             encoding="utf-8")
    (log_dir / f"{sc['id']}.md").write_text(_agent_markdown(log), encoding="utf-8")
    return log


def _agent_markdown(log: dict) -> str:
    md = [f"# {log['scenario']} — {log['title']}", "",
          f"- Model: **{log['model']}**", f"- Snapshot: `{log['status_profile']}`",
          f"- Reset before: {log['reset_before']}; process restarts: {log['process_restarts']}",
          "- Commands (one new process each):"]
    md += [f"  - `{c}`" for c in log["commands"]]
    for n, t in enumerate(log["turns"], 1):
        cm, um = t["sources"]["conversation_memory"], t["sources"]["user_memory"]
        md += ["", f"## Turn {n} — process segment {t['process_segment']} (pid {t['pid']})", "",
               f"- User / thread: `{t['user_id']}` / `{t['thread_id']}`",
               f"- **User:** {t['user_message']}",
               f"- Read from conversation memory (checkpoint `{cm['checkpoint_thread_id']}`): "
               f"{cm['prior_messages_loaded']} prior messages; prior user messages: "
               f"{json.dumps(cm['prior_user_messages'], ensure_ascii=False)}; topic before: `{cm['current_topic_before']}`",
               f"- Read from user memory (`{um['user_id']}`): `{json.dumps(um['preferences_loaded'], ensure_ascii=False)}`",
               f"- Read from reference files: {', '.join(t['sources']['reference_file']) or 'nothing'}",
               "", "| # | Node | What happened |", "|---|---|---|"]
        for i, s in enumerate(t["steps"], 1):
            parts = []
            for k, v in s["update"].items():
                if k == "messages(appended)":
                    for m in v:
                        if m["type"] == "model_tool_request":
                            parts += [f"model requests `{c['name']}({json.dumps(c['args'], ensure_ascii=False)})`"
                                      for c in m["tool_calls"]]
                        elif m["type"] == "tool_result":
                            c = dict(m["content"]) if isinstance(m["content"], dict) else m["content"]
                            if isinstance(c, dict):
                                c.pop("ticket", None)
                                c.pop("steps", None)
                            parts.append(f"tool result `{m['name']}`: `{json.dumps(c, ensure_ascii=False)}`")
                        else:
                            parts.append("model answer")
                else:
                    parts.append(f"`{k}` = `{json.dumps(v, ensure_ascii=False)}`")
            md.append(f"| {i} | {s['node']} | {'<br>'.join(parts).replace('|', '/')} |")
        md += ["", "**Final reply:**", "", "```", t["reply"], "```", "",
               f"- Ticket files: {', '.join(t['ticket_files']) or 'none'}",
               f"- User memory after turn: `{json.dumps(t['user_memory_after'], ensure_ascii=False)}`",
               f"- Tool rounds: {t['tool_rounds']}; loop limit reached: {t['loop_limit_reached']}"]
    return "\n".join(md) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["openai", "fake"], default="openai")
    p.add_argument("--only", nargs="*", help="scenario ids, e.g. S06 S07")
    p.add_argument("--worker")
    p.add_argument("--segment", type=int)
    p.add_argument("--storage")
    p.add_argument("--ticket-dir")
    args = p.parse_args()
    if args.worker:
        run_worker(args.worker, args.segment, args.model, Path(args.storage), Path(args.ticket_dir))
        return
    log_dir = data.OUTPUTS_DIR / args.model
    log_dir.mkdir(parents=True, exist_ok=True)
    for sc in data.scenarios()["scenarios"]:
        if args.only and sc["id"] not in args.only:
            continue
        if sc["part"] == 1:
            log = run_part1(sc)
            print(f"{sc['id']}: {' -> '.join(log['route'])}")
        else:
            log = run_agent_scenario(sc, args.model, log_dir)
            last = log["turns"][-1]
            print(f"{sc['id']}: {len(log['turns'])} turns, restarts={log['process_restarts']}, "
                  f"tickets={sum(len(t['ticket_files']) for t in log['turns'])} :: {last['reply'][:90]!r}")


if __name__ == "__main__":
    main()
