"""Part 1 — a rule-based helpdesk graph with no LLM.

Original graph:
    START -> read_request -> (conditional edge: route_request)
                               ├─ wifi_help     -> END
                               ├─ password_help -> END
                               └─ other_help    -> END

Changed graph ("with_clarification") adds one branch:
                               └─ clarify_problem -> END   (both topics present)
"""
from __future__ import annotations

import argparse
import json
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from . import data

GraphVariant = Literal["original", "with_clarification"]


class HelpdeskState(TypedDict, total=False):
    message: str    # the user's original message (never modified)
    category: str   # set by read_request: wifi | password | both | other
    reply: str      # set by exactly one help node


# ---------------------------------------------------------------- nodes
def read_request(state: HelpdeskState) -> dict:
    """Reads `message`; writes `category`. Pure Python keyword rules."""
    text = state["message"].lower()
    has_wifi = "wifi" in text or "wi-fi" in text
    has_password = "password" in text
    if has_wifi and has_password:
        category = "both"
    elif has_wifi:
        category = "wifi"
    elif has_password:
        category = "password"
    else:
        category = "other"
    return {"category": category}


def _format_instructions(topic: str) -> str:
    guide = data.helpdesk()["instructions"][topic]
    steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(guide["steps"], 1))
    return f"{guide['title']} [{guide['instruction_id']}]\n{steps}"


def wifi_help(state: HelpdeskState) -> dict:
    """Reads nothing but the reference file; writes `reply`."""
    return {"reply": _format_instructions("wifi")}


def password_help(state: HelpdeskState) -> dict:
    return {"reply": _format_instructions("password")}


def other_help(state: HelpdeskState) -> dict:
    return {
        "reply": "I can only help with Wi-Fi or password problems. "
        "Does your problem concern Wi-Fi or a password?"
    }


def clarify_problem(state: HelpdeskState) -> dict:
    return {
        "reply": "Your message mentions both Wi-Fi and a password. "
        "Is the problem connecting to Wi-Fi, or recovering your account password? "
        "Which one should we solve first?"
    }


# ---------------------------------------------------------------- routing
def route_original(state: HelpdeskState) -> str:
    """Conditional edge of the original graph. Reads `category`, returns a node name.

    Rule order from the brief: Wi-Fi is checked first, so 'both' goes to wifi_help.
    """
    category = state["category"]
    if category in ("wifi", "both"):
        return "wifi_help"
    if category == "password":
        return "password_help"
    return "other_help"


def route_with_clarification(state: HelpdeskState) -> str:
    if state["category"] == "both":
        return "clarify_problem"
    return route_original(state)


def build_graph(variant: GraphVariant = "original"):
    builder = StateGraph(HelpdeskState)
    builder.add_node("read_request", read_request)
    builder.add_node("wifi_help", wifi_help)
    builder.add_node("password_help", password_help)
    builder.add_node("other_help", other_help)
    targets = ["wifi_help", "password_help", "other_help"]
    router = route_original
    if variant == "with_clarification":
        builder.add_node("clarify_problem", clarify_problem)
        targets.append("clarify_problem")
        router = route_with_clarification
    builder.add_edge(START, "read_request")
    builder.add_conditional_edges("read_request", router, targets)
    for node in targets:
        builder.add_edge(node, END)
    return builder.compile()


def run_with_trace(message: str, variant: GraphVariant = "original") -> dict:
    """Run the graph, recording the order of nodes and the full state after each node."""
    graph = build_graph(variant)
    state: dict = {"message": message}
    steps = [{"node": "START", "update": {}, "state_after": dict(state)}]
    for chunk in graph.stream({"message": message}, stream_mode="updates"):
        for node, update in chunk.items():
            state.update(update or {})
            steps.append({"node": node, "update": update, "state_after": dict(state)})
    steps.append({"node": "END", "update": {}, "state_after": dict(state)})
    return {
        "graph_variant": variant,
        "route": [s["node"] for s in steps],
        "steps": steps,
        "final_state": state,
    }


def mermaid(variant: GraphVariant = "original") -> str:
    return build_graph(variant).get_graph().draw_mermaid()


def main() -> None:
    data.force_utf8_stdio()
    p = argparse.ArgumentParser(description="Part 1 rule-based helpdesk graph (no LLM)")
    p.add_argument("message", nargs="?", help="message to route")
    p.add_argument("--variant", choices=["original", "with_clarification"], default="original")
    p.add_argument("--mermaid", action="store_true", help="print the graph as Mermaid")
    args = p.parse_args()
    if args.mermaid:
        print(mermaid(args.variant))
        return
    if not args.message:
        p.error("message is required unless --mermaid is given")
    result = run_with_trace(args.message, args.variant)
    print("Route:", " -> ".join(result["route"]))
    for step in result["steps"]:
        print(f"\n[{step['node']}] update={json.dumps(step['update'], ensure_ascii=False)}")
        print("  state:", json.dumps(step["state_after"], ensure_ascii=False))
    print("\nReply:\n" + result["final_state"]["reply"])


if __name__ == "__main__":
    main()
