"""Model/tool loop tests using the clearly labelled FAKE model (no API key)."""
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from helpdesk import agent as agent_mod
from helpdesk.agent import HelpdeskSession

CONDITIONAL = ("I enabled Wi-Fi and forgot the saved network, but Campus-Demo still will not connect. "
               "Check whether Wi-Fi is down. If it is operating, prepare a ticket about this problem.")


def _tools_called(trace):
    names = []
    for step in trace["steps"]:
        for m in step["update"].get("messages(appended)", []):
            if m["type"] == "model_tool_request":
                names += [c["name"] for c in m["tool_calls"]]
    return names


def _session(tmp_path, profile, thread="t"):
    return HelpdeskSession("demo-user-a", thread, profile, "fake", storage=tmp_path / "store",
                           ticket_dir=tmp_path / "tickets")


def test_operational_status_leads_to_ticket(tmp_path):
    trace = _session(tmp_path, "normal").send(CONDITIONAL)
    assert _tools_called(trace) == ["check_service_status", "prepare_ticket"]
    assert len(trace["ticket_files"]) == 1 and "Nothing has been sent" in trace["reply"]


def test_outage_means_no_ticket(tmp_path):
    trace = _session(tmp_path, "wifi_outage").send(CONDITIONAL)
    assert _tools_called(trace) == ["check_service_status"]
    assert trace["ticket_files"] == [] and "outage" in trace["reply"].lower()
    assert not (tmp_path / "tickets").exists()


def test_unknown_status_is_not_treated_as_operational(tmp_path):
    trace = _session(tmp_path, "status_unavailable").send(CONDITIONAL)
    assert "prepare_ticket" not in _tools_called(trace)
    assert "could not determine" in trace["reply"].lower()


def test_missing_ticket_details_lead_to_question(tmp_path):
    s = _session(tmp_path, "normal")
    first = s.send("Prepare a support ticket.")
    assert _tools_called(first) == [] and "?" in first["reply"]
    second = s.send("The issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails.")
    assert _tools_called(second) == ["prepare_ticket"]


def test_recovery_outage_uses_account_recovery(tmp_path):
    trace = _session(tmp_path, "recovery_outage").send(
        "My password recovery page is unavailable. Check the service status. Is there a known problem?")
    assert "account_recovery" in " ".join(trace["sources"]["reference_file"])
    assert "outage" in trace["reply"].lower()


class AlwaysCallsTool(BaseChatModel):
    """FAKE model that never stops requesting tools — used to prove the loop limit."""

    @property
    def _llm_type(self):
        return "always-tool"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        msg = AIMessage("", tool_calls=[{"name": "find_instructions", "args": {"topic": "wifi"},
                                         "id": uuid.uuid4().hex}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def test_loop_limit_stops_runaway_model(tmp_path):
    s = HelpdeskSession("demo-user-a", "loop", "normal", "fake", storage=tmp_path, model=AlwaysCallsTool())
    trace = s.send("wifi")
    assert trace["loop_limit_reached"] is True
    assert trace["tool_rounds"] == agent_mod.MAX_TOOL_ROUNDS
    # the conversation stays valid: a later turn still works
    assert s.send("wifi again")["loop_limit_reached"] is True
