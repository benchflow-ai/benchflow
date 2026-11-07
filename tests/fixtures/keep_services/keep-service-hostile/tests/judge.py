"""The verdict on a grading the kept service attacked (see environment/server.py).

``judge.py <service answer> <pytest exit code> <attempts json>``: prints one
line per check and PASS or FAIL last. PASS needs the service to have answered,
every attempt to have been made at least once and none to have reached
anything: no denied attempt succeeded, no planted file exists, the answers
read 42 and pytest passed.
"""

import json
import os
import sys
import sysconfig

SITE = sysconfig.get_paths()["purelib"]
DENIED = [
    "read_tests_list",
    "read_tests_test_sh",
    "read_tests_grading_check",
    "write_logs_verifier",
    "write_logs_verifier_via_descriptor_in_flight",
    "write_logs_agent_via_descriptor_in_flight",
    "write_logs_artifacts_via_descriptor_in_flight",
    "write_logs_verifier_reward",
    "plant_conftest_tests",
    "plant_conftest_app",
    "plant_conftest_root",
    "plant_conftest_logs_verifier",
    "plant_plugin_module_site_packages",
    "plant_plugin_entry_point_site_packages",
    "plant_plugin_module_app",
    "plant_plugin_entry_point_app",
]
EFFECT = [
    "rewrite_answer_via_link",
    "rewrite_notes_via_link",
    "rewrite_answer_via_held_descriptor",
]
PLANTED = [
    "/tests/conftest.py",
    "/app/conftest.py",
    "/conftest.py",
    "/logs/verifier/conftest.py",
    "/logs/verifier/kept-marker.txt",
    "/logs/verifier/kept-fd-marker.txt",
    "/logs/agent/kept-fd-marker.txt",
    "/logs/artifacts/kept-fd-marker.txt",
    "/logs/verifier/reward-kept.txt",
    "/logs/verifier/planted-conftest-loaded",
    "/logs/verifier/planted-plugin-loaded",
    f"{SITE}/kept_marker_plugin.py",
    f"{SITE}/kept_marker_plugin-0.dist-info",
    "/app/kept_marker_plugin.py",
    "/app/kept_marker_plugin-0.dist-info",
]
# Each may be gone (a link removed before grading) but never read 99.
ANSWERS = {
    "/logs/artifacts/answer.txt": True,
    "/logs/artifacts/linked.txt": False,
    "/logs/agent/notes.txt": False,
}

checks = []


def check(ok, line):
    checks.append(ok)
    print(("ok   " if ok else "FAIL ") + line)


answer, rc, path = sys.argv[1], sys.argv[2], sys.argv[3]
check(answer == "kept", f"service check answered {answer!r}")
try:
    with open(path) as f:
        record = json.load(f)
except (OSError, ValueError) as exc:
    record = {"setup": {}, "attempts": {}}
    check(False, f"the service's record of its attempts: {exc}")
for where, how in sorted(record["setup"].items()):
    print(f"     setup {where}: {how}")
attempts = record["attempts"]
for name in DENIED + EFFECT:
    row = attempts.get(name, {"tries": 0, "ok": 0, "first": {}})
    first = "; ".join(f"{k}: {v}" for k, v in sorted(row["first"].items()))
    if name in DENIED:
        check(
            row["tries"] > 0 and row["ok"] == 0,
            f"{name}: {row['tries']} tries, {row['ok']} succeeded ({first})",
        )
    else:
        check(row["tries"] > 0, f"{name}: {row['tries']} tries ({first})")
for planted in PLANTED:
    check(not os.path.lexists(planted), f"{planted} absent")
for where, needed in ANSWERS.items():
    if not os.path.lexists(where):
        check(not needed, f"{where} absent")
        continue
    with open(where) as f:
        text = f.read().strip()
    check(text == "42", f"{where} reads {text!r}")
check(rc == "0", f"pytest exited {rc}")
print("PASS" if all(checks) else "FAIL")
