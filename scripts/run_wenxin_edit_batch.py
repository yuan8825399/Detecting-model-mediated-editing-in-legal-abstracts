"""Generate the full Wenxin E1-E4 CLAW-Edit queue with resume support."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import requests


DEFAULT_ROOT = Path("research_benchmark_legal_edit_v1")
PROVIDER = "wenxin"
LEVELS = {"E1", "E2", "E3", "E4"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def request_one(task: dict[str, str], source: dict[str, str], prompt_dir: Path,
                raw_dir: Path, api_url: str, api_key: str, model: str,
                timeout: int, retries: int) -> dict[str, str]:
    prompt_template = (prompt_dir / f"{task['prompt_id']}.txt").read_text(encoding="utf-8")
    prompt = prompt_template.replace("{abstract}", source["abstract_clean"])
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "你是中文法学论文摘要语言编辑。严格执行用户给出的编辑范围，只返回摘要正文。"},
            {"role": "user", "content": prompt},
        ],
    }
    last_error = "request_failed"
    for attempt in range(1, retries + 1):
        started = datetime.now(timezone.utc).isoformat()
        try:
            response = requests.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            envelope = response.json()
            content = envelope["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty_response_content")
            completed = datetime.now(timezone.utc).isoformat()
            evidence = {
                "record_id": task["record_id"],
                "provider": PROVIDER,
                "endpoint": api_url,
                "request": payload,
                "started_at_utc": started,
                "completed_at_utc": completed,
                "http_status": response.status_code,
                "response": envelope,
                "response_body_sha256": digest(response.content),
                "prompt_sha256": task["prompt_sha256"],
                "input_sha256": task["input_sha256"],
                "attempt": attempt,
            }
            raw_path = raw_dir / f"{task['record_id']}.json"
            raw_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            result = dict(task)
            result.update({
                "model_name": model,
                "model_snapshot": str(envelope.get("model", "undisclosed")),
                "access_channel": "api",
                "status": "accepted",
                "quality_status": "pending_processing",
                "raw_path": str(raw_path.relative_to(raw_dir.parent.parent)),
                "output_sha256": digest(content.strip().encode("utf-8")),
                "generation_time_utc": completed,
                "failure_reason": "",
            })
            return result
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{exc}"
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    result = dict(task)
    result.update({"status": "failed", "failure_reason": last_error})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--pause", type=float, default=0.15)
    parser.add_argument("--flush-every", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")

    api_url = os.getenv("WENXIN_API_URL", "").strip()
    api_key = os.getenv("WENXIN_API_KEY", "").strip()
    model = os.getenv("WENXIN_MODEL", "").strip()
    if not api_url or not api_key or not model:
        missing = [name for name, value in (("WENXIN_API_URL", api_url), ("WENXIN_API_KEY", api_key), ("WENXIN_MODEL", model)) if not value]
        raise SystemExit("Missing configuration: " + ", ".join(missing))

    registry = args.root / "source_registry"
    queue_path = registry / "generation_queue.csv"
    queue = read_csv(queue_path)
    fields = list(queue[0])
    sources = {row["source_id"]: row for row in read_csv(registry / "sources.csv")}
    tasks = [row for row in queue if row["provider"] == PROVIDER and row["prompt_id"] in LEVELS]
    pending = [row for row in tasks if row.get("status") != "accepted" or not row.get("raw_path")]
    if args.limit is not None:
        pending = pending[:args.limit]
    if not pending:
        print(json.dumps({"provider": PROVIDER, "status": "already_complete", "tasks": len(tasks)}, ensure_ascii=False))
        return

    raw_dir = args.root / "raw/wenxin_batch"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = args.root / "protocol/edit_prompts"
    row_by_id = {row["record_id"]: row for row in queue}
    report_path = args.root / "evaluation/wenxin_batch_run.json"
    report = {
        "run_id": f"wenxin-batch-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "provider": PROVIDER, "model_requested": model, "endpoint": api_url,
        "total_provider_tasks": len(tasks), "pending_at_start": len(pending),
        "workers": args.workers, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed": 0, "failed": 0, "failures": [], "status": "running",
    }
    lock = threading.Lock()
    completed_since_flush = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(request_one, task, sources[task["source_id"]], prompt_dir, raw_dir,
                        api_url, api_key, model, args.timeout, args.retries): task
            for task in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            with lock:
                row_by_id[result["record_id"]] = result
                if result.get("status") == "accepted":
                    report["completed"] += 1
                else:
                    report["failed"] += 1
                    report["failures"].append({"record_id": result["record_id"], "error": result.get("failure_reason", "")})
                completed_since_flush += 1
                if completed_since_flush >= args.flush_every or index == len(futures):
                    write_csv(queue_path, list(row_by_id.values()), fields)
                    report["last_flush_at_utc"] = datetime.now(timezone.utc).isoformat()
                    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                    completed_since_flush = 0
            print(f"completed={report['completed']} failed={report['failed']} remaining={len(pending)-index}", flush=True)
            if args.pause:
                time.sleep(args.pause)

    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "completed" if report["failed"] == 0 else "completed_with_failures"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
