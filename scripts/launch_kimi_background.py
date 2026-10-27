"""Run the Kimi collection independently of the interactive terminal."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import time


BASE = Path(__file__).resolve().parent
EVALUATION = BASE / "research_benchmark_legal_edit_v1" / "evaluation"
STATUS_PATH = EVALUATION / "kimi_background_status.json"
LOG_PATH = EVALUATION / "kimi_background.log"
RUN_REPORT = EVALUATION / "kimi_batch_nonthinking_run.json"


def write_status(**values: object) -> None:
    current = {}
    if STATUS_PATH.exists():
        try:
            current = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    current.update(values)
    current["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = STATUS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATUS_PATH)


def run_checked(script: str, env: dict[str, str] | None = None) -> int:
    return subprocess.run([sys.executable, "-u", script], cwd=BASE, env=env, check=False).returncode


def worker() -> int:
    key = os.environ.pop("KIMI_API_KEY", "")
    if not key:
        write_status(status="failed", reason="missing_api_key")
        return 1
    write_status(status="running", pid=os.getpid(), started_at_utc=datetime.now(timezone.utc).isoformat())
    environment = os.environ.copy()
    environment["KIMI_API_KEY"] = key
    key = ""
    for attempt in range(1, 11):
        write_status(collection_attempt=attempt)
        result = subprocess.run(
            [sys.executable, "-u", "run_kimi_edit.py", "--workers", "1",
             "--request-interval", "10", "--flush-every", "10"],
            cwd=BASE, env=environment, check=False,
        )
        if result.returncode == 0:
            break
        try:
            report = json.loads(RUN_REPORT.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = {}
        if report.get("status") != "paused_rate_limited":
            write_status(status="failed", reason="collection_error", collection_exit_code=result.returncode)
            return result.returncode
        write_status(status="waiting_after_rate_limit", rate_limit_reason=report.get("rate_limit_reason", ""))
        time.sleep(300)
    else:
        write_status(status="failed", reason="repeated_rate_limit")
        return 1
    environment.pop("KIMI_API_KEY", None)
    write_status(status="processing")
    process_code = run_checked("process_claw_edit_outputs.py")
    validate_code = run_checked("validate_claw_edit_dataset.py") if process_code == 0 else 1
    queue_path = BASE / "research_benchmark_legal_edit_v1" / "source_registry" / "generation_queue.csv"
    with queue_path.open(encoding="utf-8-sig", newline="") as handle:
        kimi_rows = [row for row in csv.DictReader(handle) if row["provider"] == "kimi"]
    accepted = sum(row["status"] == "accepted" for row in kimi_rows)
    complete = accepted == len(kimi_rows) and process_code == 0 and validate_code == 0
    write_status(
        status="completed" if complete else "failed",
        reason="" if complete else "incomplete_queue_or_validation",
        accepted=accepted,
        expected=len(kimi_rows),
        processing_exit_code=process_code,
        validation_exit_code=validate_code,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    return 0 if complete else 1


def launch() -> int:
    key = os.getenv("KIMI_API_KEY") or getpass.getpass("Kimi API key: ")
    if not key.strip():
        print("Kimi API key is required", file=sys.stderr)
        return 1
    EVALUATION.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["KIMI_API_KEY"] = key
    write_status(status="starting", log_path=str(LOG_PATH))
    with LOG_PATH.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"],
            cwd=BASE, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    write_status(pid=process.pid)
    print(f"Kimi background PID: {process.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(worker() if "--worker" in sys.argv[1:] else launch())
