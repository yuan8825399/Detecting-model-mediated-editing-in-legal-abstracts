"""Run DeepSeek edit collection independently of the interactive terminal."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys


BASE = Path(__file__).resolve().parent
ROOT = BASE / "research_benchmark_legal_edit_v1"
EVALUATION = ROOT / "evaluation"
STATUS_PATH = EVALUATION / "deepseek_background_status.json"
LOG_PATH = EVALUATION / "deepseek_background.log"
RUN_REPORT = EVALUATION / "deepseek_batch_run.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(**values: object) -> None:
    current = {}
    if STATUS_PATH.exists():
        try:
            current = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    current.update(values)
    current["updated_at_utc"] = now()
    temporary = STATUS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATUS_PATH)


def worker() -> int:
    key = os.environ.pop("DEEPSEEK_API_KEY", "")
    if not key:
        write_status(status="failed", reason="missing_api_key")
        return 1
    write_status(status="running", pid=os.getpid(), started_at_utc=now())
    environment = os.environ.copy()
    environment["DEEPSEEK_API_KEY"] = key
    key = ""
    result = subprocess.run(
        [sys.executable, "-u", "run_deepseek_edit.py", "--request-interval", "2",
         "--flush-every", "10"],
        cwd=BASE, env=environment, check=False,
    )
    environment.pop("DEEPSEEK_API_KEY", None)
    if result.returncode != 0:
        try:
            report = json.loads(RUN_REPORT.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = {}
        write_status(status="paused", reason=report.get("last_error", "collection_error"),
                     collection_exit_code=result.returncode)
        return result.returncode

    write_status(status="processing")
    process_code = subprocess.run([sys.executable, "process_claw_edit_outputs.py"],
                                  cwd=BASE, check=False).returncode
    validate_code = (subprocess.run([sys.executable, "validate_claw_edit_dataset.py"],
                                    cwd=BASE, check=False).returncode if process_code == 0 else 1)
    with (ROOT / "source_registry/generation_queue.csv").open(encoding="utf-8-sig", newline="") as handle:
        queue = [row for row in csv.DictReader(handle) if row["provider"] == "deepseek"]
    accepted = sum(row["status"] == "accepted" for row in queue)
    processed = ROOT / "processed/records.jsonl"
    quality_counts: dict[str, int] = {}
    if process_code == 0:
        with processed.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("provider") == "deepseek":
                    status = row["quality_status"]
                    quality_counts[status] = quality_counts.get(status, 0) + 1
    complete = (accepted == len(queue) == 2124 and
                quality_counts.get("accepted") == 2124 and
                process_code == 0 and validate_code == 0)
    write_status(
        status="completed" if complete else "completed_with_quality_issues",
        reason="" if complete else "incomplete_queue_or_quality_or_validation",
        accepted=accepted,
        expected=len(queue),
        quality_counts=quality_counts,
        processing_exit_code=process_code,
        validation_exit_code=validate_code,
        finished_at_utc=now(),
    )
    return 0 if complete else 1


def launch() -> int:
    key = os.getenv("DEEPSEEK_API_KEY") or getpass.getpass("DeepSeek API key: ")
    if not key.strip():
        print("DeepSeek API key is required", file=sys.stderr)
        return 1
    EVALUATION.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["DEEPSEEK_API_KEY"] = key
    write_status(status="starting", log_path=str(LOG_PATH), reason="")
    with LOG_PATH.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"],
            cwd=BASE, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    write_status(pid=process.pid)
    print(f"DeepSeek background PID: {process.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(worker() if "--worker" in sys.argv[1:] else launch())
