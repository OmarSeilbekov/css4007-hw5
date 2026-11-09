"""A clearly labelled FAKE chat model used by tests, CI and offline demos.

It is NOT an LLM. It is a small deterministic rule set that reads the same inputs a
real model receives (system prompt + conversation messages + tool results) and
emits either tool calls or an answer, so the LangGraph loop, tools and memory can
be tested without an API key. Its Kazakh output is template-based.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from . import data

STEP_PATTERNS = [
    (r"airplane", "Turned off airplane mode"),
    (r"\benabled?\b", "Enabled Wi-Fi"),
    (r"forgot (the )?saved network|forget (the )?saved network", "Forgot the saved network"),
    (r"reconnect", "Tried reconnecting"),
    (r"forgot password|choose forgot password", "Used Forgot password"),
]

KK = {
    "language": "Сіз маған қазақ тілінде жауап беруді сұрадыңыз.",
    "saved": "Жауаптарды қазақ тілінде беру туралы қалауыңызды сақтадым.",
    "scope": "Мен көмектесе алатын мәселелер: Wi-Fi ақауларын жою, құпия сөзді қалпына келтіру бойынша нұсқаулық, қызмет күйін тексеру және жергілікті өтінім жобасын дайындау.",
    "recall": "Біз {topic} мәселесін талқыладық. Сіз жасап көрген қадамдар: {steps}.",
    "guide": "Жергілікті нұсқаулық ({ref}):",
}


def _call(name: str, args: dict) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}])


def _steps_from(texts: list[str]) -> list[str]:
    steps: list[str] = []
    for text in texts:
        low = text.lower()
        for pattern, label in STEP_PATTERNS:
            if re.search(pattern, low) and label not in steps:
                if label == "Enabled Wi-Fi" and "wi-fi" not in low and "wifi" not in low and " it" not in low:
                    continue
                steps.append(label)
    return steps


def _topic_name(topic: Optional[str]) -> str:
    return {"wifi": "Wi-Fi", "password": "password"}.get(topic or "", "unknown")


class FakeHelpdeskModel(BaseChatModel):
    """Deterministic stand-in for a tool-calling chat model. Label: FAKE."""

    @property
    def _llm_type(self) -> str:
        return "fake-helpdesk-rules"

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._decide(messages))])

    # ------------------------------------------------------------ decision rules
    def _decide(self, messages: list[BaseMessage]) -> AIMessage:
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        lang_match = re.search(r"Preferred reply language: (\w+)", system)
        language = lang_match.group(1) if lang_match and lang_match.group(1) != "none" else None
        topic_match = re.search(r"Current topic \(conversation memory\): (\w+)", system)
        topic = topic_match.group(1) if topic_match and topic_match.group(1) != "none" else None

        idx = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        human = messages[idx].content
        low = human.lower()
        history = messages[1:idx]
        prev_humans = [m.content for m in history if isinstance(m, HumanMessage)]
        prev_ai = next((m.content for m in reversed(history) if isinstance(m, AIMessage) and m.content), "")
        results: dict[str, dict] = {}
        for m in messages[idx + 1:]:
            if isinstance(m, ToolMessage):
                try:
                    results[m.name] = json.loads(m.content)
                except json.JSONDecodeError:
                    results[m.name] = {"raw": m.content}

        kk = language == "Kazakh"
        out: list[str] = []

        # 1. Explicit request to remember a preference -> user-memory tool.
        pref = re.search(r"prefer (?:replies|answers) in (\w+)", human, re.IGNORECASE)
        if pref and "save_user_preference" not in results:
            return _call("save_user_preference", {"language": pref.group(1).capitalize()})
        if "save_user_preference" in results and results["save_user_preference"].get("saved"):
            language = results["save_user_preference"].get("language", language)
            kk = language == "Kazakh"
            out.append(KK["saved"] if kk else f"I saved your preference: replies in {language}.")

        # 2. Questions answered from memory (no tools needed).
        asks_language = "which language" in low or "preferred language" in low
        asks_history = any(p in low for p in ("already try", "already tried?", "did we discuss",
                                              "previous problem", "which steps", "which problem"))
        if asks_language or asks_history or "what can you help" in low:
            if asks_language:
                if language:
                    out.append(KK["language"] if kk else f"You asked me to reply in {language}.")
                    if kk:
                        out.append(f"(English: you asked me to reply in {language}.)")
                else:
                    out.append("No preferred language has been recorded for you.")
            if asks_history:
                steps = _steps_from(prev_humans)
                if prev_humans and (topic or steps):
                    s = "; ".join(steps) if steps else "none mentioned"
                    if kk:
                        out.append(KK["recall"].format(topic=_topic_name(topic), steps=s))
                    out.append(f"We discussed a {_topic_name(topic)} problem. Steps you already tried: {s}.")
                else:
                    out.append("No previous problem has been recorded in this conversation. "
                               "What problem are you having?")
            if "what can you help" in low:
                out.append(KK["scope"] if kk else "I can help with: " + ", ".join(data.helpdesk()["scope"]) + ".")
            return AIMessage("\n".join(out))

        topics_now = data.topics_in(human)
        turn_topic = topics_now[0] if len(topics_now) == 1 else topic
        service = data.helpdesk()["topic_services"].get(turn_topic or "", None)
        answering_ticket_question = "what have you already tried" in prev_ai.lower()
        wants_ticket = "ticket" in low or answering_ticket_question
        conditional = bool(re.search(r"if (it is|it's)? ?(operating|working|up)", low))
        wants_status = any(p in low for p in ("is it down", "is down", "service status", "known problem",
                                              "whether wi-fi is down", "check whether"))

        # 3. Status check (S07/S08/S14 and conditional ticket requests).
        if (wants_status or conditional) and service and "check_service_status" not in results:
            return _call("check_service_status", {"service": service})
        status = results.get("check_service_status")

        if wants_ticket:
            all_texts = prev_humans + [human]
            steps = _steps_from([human] + (prev_humans if answering_ticket_question else []))
            if not steps:
                steps = _steps_from(all_texts)
            if turn_topic is None or not steps:
                return AIMessage("What problem are you having, and what have you already tried?")
            if conditional and status:
                st = status.get("status")
                if st == "outage":
                    return AIMessage(
                        f"The demo status source reports a {_topic_name(turn_topic)} outage: {status['message']} "
                        "Your request said to prepare a ticket only if the service was operating, "
                        "so I have not prepared a ticket. The data gives no repair time.")
                if st != "operational":
                    return AIMessage(
                        f"I could not determine the {_topic_name(turn_topic)} status ({status['message']}). "
                        "I cannot confirm that it is operating, so I have not prepared a ticket. "
                        "Would you like a ticket draft explaining that the status could not be checked?")
            if "prepare_ticket" not in results:
                summary = (f"Cannot connect to the Campus-Demo Wi-Fi network; it still fails after the steps tried."
                           if turn_topic == "wifi" and "campus-demo" in " ".join(all_texts).lower()
                           else f"{_topic_name(turn_topic)} problem still fails after the steps tried.")
                return _call("prepare_ticket", {
                    "category": turn_topic, "summary": summary, "steps_tried": steps,
                    "service_status": status["status"] if status else "not_checked"})
            t = results["prepare_ticket"]
            if not t.get("created"):
                return AIMessage("I could not prepare the draft: " + t.get("message", ""))
            status_text = (f"The demo status source reports {_topic_name(turn_topic)} is {status['status']}. "
                           if status else "Service status: not checked. ")
            return AIMessage(status_text + f"I prepared a local ticket draft {t['ticket_id']} "
                             f"(file: {t['file']}) with the steps you tried: {'; '.join(t['ticket']['steps_tried'])}. "
                             "Nothing has been sent.")

        if status:
            if status["status"] == "outage":
                return AIMessage(f"Yes. The demo status source reports an outage of {status['service']}: "
                                 f"{status['message']} I have no repair time to give.")
            if status["status"] == "unknown":
                return AIMessage(f"I could not determine the status of {status['service']}: {status['message']}")
            return AIMessage(f"The demo status source reports {status['service']} is operating: {status['message']}")

        # 4. Instructions from the reference file.
        if turn_topic and "find_instructions" not in results:
            return _call("find_instructions", {"topic": turn_topic})
        guide = results.get("find_instructions")
        if guide and guide.get("found"):
            tried = _steps_from([human])
            body = "\n".join(f"{i}. {s}" for i, s in enumerate(guide["steps"], 1))
            head = KK["guide"].format(ref=guide["instruction_id"]) if kk else \
                f"Here are the instructions from the local guide [{guide['instruction_id']}] — {guide['title']}:"
            out.append(head + "\n" + body)
            if tried:
                out.append("You said you already tried: " + "; ".join(tried) +
                           ". If it still fails, I can check the service status or prepare a local ticket draft.")
            return AIMessage("\n".join(out))

        if out:
            return AIMessage("\n".join(out))
        return AIMessage("I can help with Wi-Fi or password problems. Which one is it, and what have you tried?")
