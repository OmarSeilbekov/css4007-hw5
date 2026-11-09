"""Memory: survives a real process restart; threads and users stay separate."""
import json
import subprocess
import sys

from helpdesk.agent import HelpdeskSession
from helpdesk.data import REPO_ROOT


def _run_cli(storage, user, thread, message):
    proc = subprocess.run(
        [sys.executable, "-m", "helpdesk.cli", "--model", "fake", "--storage", str(storage),
         "--user", user, "--thread", thread, "--json", "-m", message],
        capture_output=True, text=True, encoding="utf-8", cwd=REPO_ROOT, check=True)
    out = proc.stdout
    return json.loads(out[out.index("\n{") + 1:])


def test_conversation_survives_process_restart(tmp_path):
    _run_cli(tmp_path, "demo-user-a", "restart-a",
             "My issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails.")
    trace = _run_cli(tmp_path, "demo-user-a", "restart-a", "Which problem and steps did we discuss?")
    cm = trace["sources"]["conversation_memory"]
    assert cm["prior_messages_loaded"] > 0
    assert "Wi-Fi" in trace["reply"] and "Tried reconnecting" in trace["reply"]


def test_preference_survives_restart_in_new_thread(tmp_path):
    _run_cli(tmp_path, "demo-user-a", "p1", "Please remember that I prefer replies in Kazakh.")
    trace = _run_cli(tmp_path, "demo-user-a", "p2", "Which language did I ask you to use?")
    assert trace["sources"]["conversation_memory"]["prior_messages_loaded"] == 0  # new thread
    assert trace["sources"]["user_memory"]["preferences_loaded"] == {"language": "Kazakh"}
    assert "Kazakh" in trace["reply"]


def test_users_do_not_share_preferences(tmp_path):
    a = HelpdeskSession("demo-user-a", "isolation-a", "normal", "fake", storage=tmp_path)
    a.send("Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem.")
    b = HelpdeskSession("demo-user-b", "isolation-b", "normal", "fake", storage=tmp_path)
    trace = b.send("Have I told you a preferred language or a previous problem?")
    assert trace["sources"]["user_memory"]["preferences_loaded"] == {}
    assert trace["sources"]["conversation_memory"]["prior_messages_loaded"] == 0
    assert "Kazakh" not in trace["reply"] and "No preferred language" in trace["reply"]


def test_threads_have_separate_histories(tmp_path):
    s1 = HelpdeskSession("demo-user-a", "t1", "normal", "fake", storage=tmp_path)
    s1.send("I cannot connect to Wi-Fi. I enabled it and forgot the saved network.")
    s2 = HelpdeskSession("demo-user-a", "t2", "normal", "fake", storage=tmp_path)
    trace = s2.send("Which steps did I already try?")
    assert trace["sources"]["conversation_memory"]["prior_messages_loaded"] == 0
    assert "No previous problem" in trace["reply"]


def test_earlier_message_resolves_that(tmp_path):
    s = HelpdeskSession("demo-user-a", "s10", "normal", "fake", storage=tmp_path)
    s.send("I cannot connect to Wi-Fi. I enabled it and forgot the saved network, but reconnecting failed.")
    trace = s.send("Which steps did I already try?")
    assert "Forgot the saved network" in trace["reply"]
