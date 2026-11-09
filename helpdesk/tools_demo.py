"""Call each tool directly, without the agent:  python -m helpdesk.tools_demo"""
import json
import tempfile

from .tools import check_service_status, find_instructions, prepare_ticket


def main() -> None:
    calls = [
        ("find_instructions('wifi')", lambda: find_instructions("wifi")),
        ("find_instructions('printer')", lambda: find_instructions("printer")),
        *[(f"check_service_status('wifi', profile='{p}')", lambda p=p: check_service_status("wifi", profile=p))
          for p in ("normal", "wifi_outage", "status_unavailable")],
        ("check_service_status('password', profile='recovery_outage')",
         lambda: check_service_status("password", profile="recovery_outage")),
        ("prepare_ticket('wifi', '', [])", lambda: prepare_ticket("wifi", "", [], ticket_dir=tempfile.mkdtemp())),
        ("prepare_ticket('wifi', 'Campus-Demo will not connect', ['Enabled Wi-Fi'], 'operational', user_id='demo-user-a', thread_id='direct')",
         lambda: prepare_ticket("wifi", "Campus-Demo will not connect", ["Enabled Wi-Fi"], "operational",
                                user_id="demo-user-a", thread_id="direct", status_profile="normal",
                                ticket_dir=tempfile.mkdtemp())),
    ]
    for label, fn in calls:
        print(f">>> {label}\n{json.dumps(fn(), indent=2, ensure_ascii=False)}\n")


if __name__ == "__main__":
    main()
