"""The three tools called directly, without the agent or a model."""
import json

from helpdesk.tools import ToolContext, check_service_status, find_instructions, make_tools, prepare_ticket


def test_find_instructions_wifi_and_password():
    wifi = find_instructions("wifi")
    assert wifi["found"] and wifi["instruction_id"] == "GUIDE-WIFI"
    assert any("Campus-Demo" in s for s in wifi["steps"])
    assert find_instructions("Wi-Fi")["instruction_id"] == "GUIDE-WIFI"
    assert find_instructions("password")["instruction_id"] == "GUIDE-PASSWORD"


def test_find_instructions_unknown_topic_is_not_invented():
    result = find_instructions("printer")
    assert result["found"] is False and "steps" not in result


def test_check_service_status_uses_selected_snapshot():
    assert check_service_status("wifi", "normal")["status"] == "operational"
    assert check_service_status("wifi", "wifi_outage")["status"] == "outage"
    assert check_service_status("wifi", "status_unavailable")["status"] == "unknown"
    # password topic maps to the account_recovery service
    rec = check_service_status("password", "recovery_outage")
    assert rec["service"] == "account_recovery" and rec["status"] == "outage"


def test_check_service_status_unknown_service_is_unknown_not_operational():
    assert check_service_status("printing", "normal")["status"] == "unknown"


def test_prepare_ticket_writes_local_draft(tmp_path):
    r = prepare_ticket("wifi", "Cannot connect to Campus-Demo", ["Enabled Wi-Fi"], "operational",
                       user_id="demo-user-a", thread_id="t1", status_profile="normal", ticket_dir=tmp_path)
    assert r["created"] and r["sent"] is False
    saved = json.loads((tmp_path / f"{r['ticket_id']}.json").read_text())["ticket"]
    assert saved["status"] == "draft"
    assert saved["steps_tried"] == ["Enabled Wi-Fi"]
    assert saved["reference_ids"] == ["GUIDE-WIFI"]
    assert set(saved) == {"ticket_id", "status", "user_id", "thread_id", "category", "summary",
                          "steps_tried", "service_status", "status_profile", "reference_ids"}


def test_prepare_ticket_refuses_missing_details(tmp_path):
    r = prepare_ticket("wifi", "", [], ticket_dir=tmp_path)
    assert r["created"] is False
    assert set(r["missing"]) == {"problem description", "steps already tried"}
    assert list(tmp_path.iterdir()) == []


def test_prepare_ticket_refuses_secrets(tmp_path):
    r = prepare_ticket("password", "Login fails, my password is hunter2", ["Tried recovery"], ticket_dir=tmp_path)
    assert r["created"] is False and list(tmp_path.iterdir()) == []


def test_langchain_wrappers_bind_session_values(tmp_path):
    tools = {t.name: t for t in make_tools(ToolContext(status_profile="wifi_outage", user_id="u",
                                                       thread_id="t", ticket_dir=tmp_path))}
    assert set(tools) >= {"find_instructions", "check_service_status", "prepare_ticket"}
    # the model cannot pass a profile: the bound snapshot is used
    assert json.loads(tools["check_service_status"].invoke({"service": "wifi"}))["status"] == "outage"
    t = json.loads(tools["prepare_ticket"].invoke(
        {"category": "wifi", "summary": "x fails", "steps_tried": ["Enabled Wi-Fi"]}))
    assert t["ticket"]["user_id"] == "u" and t["ticket"]["status_profile"] == "wifi_outage"
