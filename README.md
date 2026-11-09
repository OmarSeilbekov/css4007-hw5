# HW5 — University IT helpdesk with LangGraph

A terminal helpdesk built in four stages with **LangGraph**. All data is fictional
(`data/`); nothing connects to real university systems and tickets are only local drafts.
The original assignment brief is in [ASSIGNMENT.md](ASSIGNMENT.md); my answers are in
[SUBMISSION.md](SUBMISSION.md).

**Python version:** 3.13 (CI also runs 3.11). Exact package versions: [requirements.txt](requirements.txt).

## 1. Install

```bash
git clone https://github.com/<your-account>/css4007-hw5.git
cd css4007-hw5
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 2. Configure a model (only for the real-model runs)

Part 1, the tests and the `--model fake` runs need **no API key**.

```bash
cp .env.example .env               # Windows: copy .env.example .env
# edit .env:
#   OPENAI_API_KEY=sk-...
#   MODEL_NAME=gpt-4.1-mini        # any tool-calling OpenAI model
```

`.env` is ignored by git. The provider is OpenAI through `langchain-openai`
(`helpdesk/agent.py:get_model`).

## 3. Run

| What | Command |
| --- | --- |
| Part 1, original graph | `python -m helpdesk.part1 "I cannot connect to Wi-Fi."` |
| Part 1, changed graph | `python -m helpdesk.part1 --variant with_clarification "My Wi-Fi password does not work."` |
| Part 1 graph drawing (Mermaid) | `python -m helpdesk.part1 --mermaid --variant with_clarification` |
| Call each tool directly | `python -m helpdesk.tools_demo` |
| Interactive helpdesk (real model) | `python -m helpdesk.cli --user demo-user-a --thread t1 --status normal --model openai` |
| One message per launch | `python -m helpdesk.cli --user demo-user-a --thread t1 --status wifi_outage -m "Is Wi-Fi down?"` |
| Same, with the fake model | add `--model fake` |
| Full JSON trace | add `--json` |
| All scenarios S01–S14, real model | `python -m helpdesk.scenarios --model openai` |
| All scenarios, fake model | `python -m helpdesk.scenarios --model fake` |
| Some scenarios | `python -m helpdesk.scenarios --model openai --only S06 S07 S08` |
| Tests | `python -m pytest -v` |

CLI options: `--user` (invented id, e.g. `demo-user-a`), `--thread` (conversation id),
`--status` (`normal`, `wifi_outage`, `recovery_outage`, `status_unavailable`),
`--model` (`openai` or `fake`), `--storage DIR`. In interactive mode type `/quit` to exit.

Example of a real restart (two separate processes, same storage):

```bash
python -m helpdesk.cli --model openai --user demo-user-a --thread restart-a \
  -m "My issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails."
python -m helpdesk.cli --model openai --user demo-user-a --thread restart-a \
  -m "Which problem and steps did we discuss?"
```

The scenario runner does the same automatically: each block of events between two
`restart_process` events runs in a **new Python process** (PIDs are written to the log).

## 4. Where memory is saved and how to clear it

| File | Contents | Key |
| --- | --- | --- |
| `local-storage/conversations.sqlite` | LangGraph `SqliteSaver` checkpoints (conversation memory) | `thread_id` |
| `local-storage/user_memory.sqlite` | LangGraph `SqliteStore` (user preferences) | namespace `("users", user_id)` |
| `local-storage/tickets/*.json` | Ticket drafts from the CLI | ticket id |
| `local-storage/scenarios/<model>/<Sxx>/` | Separate storage per scenario run | — |

Change the folder with `HELPDESK_STORAGE_DIR` or `--storage`. `local-storage/` is git-ignored.

Clear everything: `python -m helpdesk.cli --reset` (or delete the `local-storage` folder).
The scenario runner resets a scenario's own folder automatically when `reset_before` is true.

## 5. Logs

* `outputs/part1/Sxx.md|json` — Part 1 traces (S01–S05), node order and state after each node.
* `outputs/fake/Sxx.md|json` — S06–S14 with the **FAKE** rule-based model (for reproducibility/CI).
* `outputs/openai/Sxx.md|json` — S06–S14 with the real model (created by `--model openai`).
* `outputs/<model>/tickets/` — ticket drafts created during the scenario runs (invented data only).
* `outputs/part2_direct_tool_calls.txt` — each tool called directly, without the agent.

## 6. Project layout

```
helpdesk/
  part1.py        Part 1 rule-based graph (+ clarify_problem variant), tracing
  tools.py        find_instructions, check_service_status, prepare_ticket (+ save_user_preference)
  agent.py        model/tool loop graph, loop limit, HelpdeskSession
  memory.py       SqliteSaver checkpointer + SqliteStore user memory, reset
  fake_model.py   FAKE deterministic model for tests/CI (not an LLM)
  cli.py          terminal application
  scenarios.py    runs S01–S14 and writes outputs/
  tools_demo.py   direct tool calls
tests/            pytest suite (no API key needed)
.github/workflows/tests.yml   CI
```
