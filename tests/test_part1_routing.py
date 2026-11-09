"""Part 1 routing tests (no model, no API key)."""
import pytest

from helpdesk.part1 import run_with_trace

CASES = [
    ("I cannot connect to Wi-Fi.", "original", "wifi_help"),
    ("I forgot my PASSWORD.", "original", "password_help"),
    ("My laptop screen is broken.", "original", "other_help"),
    # Rule order: the Wi-Fi rule is checked first, so it wins in the original graph.
    ("My Wi-Fi password does not work.", "original", "wifi_help"),
    ("My Wi-Fi password does not work.", "with_clarification", "clarify_problem"),
    ("wifi keeps dropping", "with_clarification", "wifi_help"),
    ("PaSsWoRd reset", "with_clarification", "password_help"),
]


@pytest.mark.parametrize("message,variant,expected_node", CASES)
def test_route(message, variant, expected_node):
    result = run_with_trace(message, variant)
    assert result["route"] == ["START", "read_request", expected_node, "END"]


def test_state_keeps_original_message_and_sets_fields():
    result = run_with_trace("I forgot my PASSWORD.")
    after_read = result["steps"][1]
    assert after_read["node"] == "read_request"
    assert after_read["state_after"] == {"message": "I forgot my PASSWORD.", "category": "password"}
    final = result["final_state"]
    assert final["message"] == "I forgot my PASSWORD."
    assert "GUIDE-PASSWORD" in final["reply"]


def test_help_nodes_use_reference_file():
    assert "Campus-Demo" in run_with_trace("wifi")["final_state"]["reply"]


def test_other_help_asks_for_topic():
    reply = run_with_trace("My laptop screen is broken.")["final_state"]["reply"]
    assert "Wi-Fi" in reply and "password" in reply and "?" in reply
