"""Parts 2–3 — model + tools loop with conversation and user memory.

    START -> load_memory -> agent --(route_after_agent)--> tools -> agent  (loop)
                                   ├──────────────────────> END            (answer)
                                   └──────────────────────> loop_limit -> END

* agent  : the MODEL decides — it either requests tool calls or writes an answer.
* tools  : OUR PROGRAM executes the requested tools and appends ToolMessages.
* route_after_agent: a plain Python conditional edge; it also enforces MAX_TOOL_ROUNDS.
"""
from __future__ import annotations

import json
import os
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import data
from .memory import UserMemory, open_checkpointer
from .tools import ToolContext, make_tools

MAX_TOOL_ROUNDS = int(os.environ.get("HELPDESK_MAX_TOOL_ROUNDS", "4"))


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]  # conversation memory (checkpointed)
    user_id: str               # set by the application for every turn
    thread_id: str
    status_profile: str
    current_topic: Optional[str]   # conversation memory: last helpdesk topic mentioned
    user_preferences: dict         # copy of user memory loaded at the start of the turn
    tool_rounds: int               # tool rounds in the current turn (reset each turn)
    loop_limit_reached: bool


SYSTEM_PROMPT = """You are a practice university IT helpdesk assistant. All data is fictional and local.
Scope: Wi-Fi troubleshooting, password recovery guidance, service status lookup, preparing LOCAL ticket drafts.
Rules:
- Use find_instructions for instructions; never make up steps.
- Use check_service_status before saying whether a service is down. 'unknown' means the status could not be determined: it is neither operational nor a confirmed outage. Never invent repair times.
- Follow the user's conditions exactly (e.g. "if it is operating, prepare a ticket": only prepare it when the status is operational).
- prepare_ticket needs a problem description and the steps the user said they tried. If either is missing, ask the user; do not invent details. Tickets are local drafts; never say a ticket was sent. Include the ticket file name in your answer.
- Never ask for or store passwords, recovery codes or API keys.
- Use the earlier messages in this conversation to understand references such as "that" or "what we discussed".
- If the user asks you to remember a reply language, call save_user_preference.
- Never reveal or guess information about other users.

Session (set by the application, not by the user):
user_id: {user_id}
status_profile: {status_profile}
Current topic (conversation memory): {topic}
Saved user preferences (user memory): {prefs}
Preferred reply language: {language}
"""


def build_agent_graph(model, tools, user_memory: UserMemory, checkpointer=None):
    """`model` is any LangChain chat model supporting bind_tools (real or fake)."""
    tools_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    def load_memory(state: AgentState) -> dict:
        prefs = user_memory.get_preferences(state["user_id"])
        last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        topics = data.topics_in(last_human.content) if last_human else []
        topic = topics[0] if len(topics) == 1 else state.get("current_topic")
        return {
            "user_preferences": prefs,
            "current_topic": topic,
            "tool_rounds": 0,
            "loop_limit_reached": False,
        }

    def agent(state: AgentState) -> dict:
        prefs = state.get("user_preferences") or {}
        system = SystemMessage(
            SYSTEM_PROMPT.format(
                user_id=state["user_id"],
                status_profile=state.get("status_profile", "normal"),
                topic=state.get("current_topic") or "none yet",
                prefs=json.dumps(prefs, ensure_ascii=False) if prefs else "none recorded",
                language=prefs.get("language", "none recorded (reply in the user's language)"),
            )
        )
        response = model_with_tools.invoke([system, *state["messages"]])
        return {"messages": [response]}

    def run_tools(state: AgentState) -> dict:
        last = state["messages"][-1]
        results = []
        for call in last.tool_calls:
            tool = tools_by_name.get(call["name"])
            try:
                if tool is None:
                    raise KeyError(f"unknown tool {call['name']}")
                content = tool.invoke(call["args"])
            except Exception as exc:  # report errors back to the model
                content = json.dumps({"error": str(exc)})
            results.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        return {"messages": results, "tool_rounds": state.get("tool_rounds", 0) + 1}

    def loop_limit(state: AgentState) -> dict:
        last = state["messages"][-1]
        skipped = [
            ToolMessage(
                content=json.dumps({"error": "not executed: tool loop limit reached"}),
                tool_call_id=c["id"], name=c["name"],
            )
            for c in last.tool_calls
        ]
        answer = AIMessage(
            f"I stopped after {MAX_TOOL_ROUNDS} tool rounds without reaching an answer. "
            "Please rephrase or give more detail."
        )
        return {"messages": [*skipped, answer], "loop_limit_reached": True}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
                return "loop_limit"
            return "tools"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("load_memory", load_memory)
    builder.add_node("agent", agent)
    builder.add_node("tools", run_tools)
    builder.add_node("loop_limit", loop_limit)
    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["tools", "loop_limit", END])
    builder.add_edge("tools", "agent")
    builder.add_edge("loop_limit", END)
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------- models
def get_model(name: str):
    """'fake' = deterministic rule-based stand-in (tests/CI). 'openai' = real LLM."""
    if name == "fake":
        from .fake_model import FakeHelpdeskModel

        return FakeHelpdeskModel()
    if name == "openai":
        from dotenv import load_dotenv
        from langchain_openai import ChatOpenAI

        load_dotenv(data.REPO_ROOT / ".env")
        model_name = os.environ.get("MODEL_NAME") or "gpt-4.1-mini"
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return ChatOpenAI(model=model_name, temperature=0)
    raise ValueError(f"unknown model {name!r}")


def model_description(name: str) -> str:
    if name == "fake":
        return "FAKE rule-based model (helpdesk/fake_model.py) — not an LLM"
    return f"OpenAI / {os.environ.get('MODEL_NAME') or 'gpt-4.1-mini'}"


# ---------------------------------------------------------------- session
class HelpdeskSession:
    """One process-level session: a user, a thread, a status snapshot, saved storage."""

    def __init__(self, user_id: str, thread_id: str, status_profile: str = "normal",
                 model_name: str = "fake", storage=None, ticket_dir=None, model=None):
        if status_profile not in data.status_profiles():
            raise ValueError(f"unknown status profile {status_profile!r}")
        self.user_id, self.thread_id, self.status_profile = user_id, thread_id, status_profile
        self.storage = storage or data.storage_dir()
        self.user_memory = UserMemory(self.storage)
        self.checkpointer = open_checkpointer(self.storage)
        ctx = ToolContext(
            status_profile=status_profile, user_id=user_id, thread_id=thread_id,
            ticket_dir=ticket_dir or (self.storage / "tickets"),
            save_preference=lambda k, v: self.user_memory.set_preference(user_id, k, v),
        )
        self.model_name = model_name
        self.graph = build_agent_graph(model or get_model(model_name), make_tools(ctx),
                                       self.user_memory, self.checkpointer)
        self.config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}

    def send(self, message: str) -> dict:
        """Run one user turn and return the reply plus a trace of what happened."""
        before = self.graph.get_state(self.config).values
        prior_messages = before.get("messages", [])
        prefs_before = self.user_memory.get_preferences(self.user_id)
        trace: dict[str, Any] = {
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "status_profile": self.status_profile,
            "model": model_description(self.model_name),
            "user_message": message,
            "sources": {
                "conversation_memory": {
                    "checkpoint_thread_id": self.thread_id,
                    "prior_messages_loaded": len(prior_messages),
                    "prior_user_messages": [m.content for m in prior_messages if isinstance(m, HumanMessage)],
                    "current_topic_before": before.get("current_topic"),
                },
                "user_memory": {"user_id": self.user_id, "preferences_loaded": prefs_before},
                "reference_file": [],
            },
            "steps": [],
        }
        inputs = {
            "messages": [HumanMessage(message)],
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "status_profile": self.status_profile,
        }
        tickets = []
        for chunk in self.graph.stream(inputs, self.config, stream_mode="updates"):
            for node, update in chunk.items():
                trace["steps"].append({"node": node, "update": _summarise_update(update or {})})
                for m in (update or {}).get("messages", []):
                    if isinstance(m, ToolMessage):
                        try:
                            payload = json.loads(m.content)
                        except (json.JSONDecodeError, TypeError):
                            payload = {}
                        if m.name == "find_instructions" and payload.get("found"):
                            trace["sources"]["reference_file"].append(
                                f"data/helpdesk.json:{payload['instruction_id']}")
                        if m.name == "check_service_status":
                            trace["sources"]["reference_file"].append(
                                f"data/service_status.json:{payload.get('status_profile')}/{payload.get('service')}")
                        if m.name == "prepare_ticket" and payload.get("created"):
                            tickets.append(payload["file"])
        after = self.graph.get_state(self.config).values
        reply = after["messages"][-1].content
        trace["ticket_files"] = tickets
        trace["loop_limit_reached"] = bool(after.get("loop_limit_reached"))
        trace["tool_rounds"] = after.get("tool_rounds", 0)
        trace["current_topic_after"] = after.get("current_topic")
        trace["user_memory_after"] = self.user_memory.get_preferences(self.user_id)
        trace["reply"] = reply
        return trace


def _summarise_update(update: dict) -> dict:
    out = {}
    for key, value in update.items():
        if key != "messages":
            out[key] = value
            continue
        msgs = []
        for m in value:
            if isinstance(m, AIMessage) and m.tool_calls:
                msgs.append({"type": "model_tool_request",
                             "tool_calls": [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]})
            elif isinstance(m, AIMessage):
                msgs.append({"type": "model_answer", "content": m.content})
            elif isinstance(m, ToolMessage):
                try:
                    content = json.loads(m.content)
                except (json.JSONDecodeError, TypeError):
                    content = m.content
                msgs.append({"type": "tool_result", "name": m.name, "content": content})
            else:
                msgs.append({"type": m.type, "content": m.content})
        out["messages(appended)"] = msgs
    return out
