"""Collect Kimi E1-E4 edits from the existing CLAW-Edit queue."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import requests


ROOT = Path("research_benchmark_legal_edit_v1")
ENDPOINT = "https://api.moonshot.cn/v1/chat/completions"
PROVIDER = "kimi"
LEVELS = {"E1", "E2", "E3", "E4"}


class RequestPacer:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.next_request = 0.0
        self.lock = threading.Lock()

    def wait(self, stop_event: threading.Event) -> bool:
        with self.lock:
            delay = max(0.0, self.next_request - time.monotonic())
            self.next_request = time.monotonic() + delay + self.interval
        return not stop_event.wait(delay)

    def cooldown(self, seconds: float) -> None:
        with self.lock:
            self.next_request = max(self.next_request, time.monotonic() + seconds)


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


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def recover_saved_responses(queue: list[dict[str, str]], root: Path) -> int:
    recovered = 0
    for task in queue:
        if task["provider"] != PROVIDER or task.get("status") == "accepted":
            continue
        for kind in ("test_nonthinking", "batch_nonthinking", "test", "batch"):
            raw_path = root / "raw" / f"kimi_{kind}" / f"{task['record_id']}.json"
            if not raw_path.exists():
                continue
            try:
                evidence = json.loads(raw_path.read_text(encoding="utf-8"))
                content = evidence["response"]["choices"][0]["message"]["content"]
                if (evidence.get("record_id") != task["record_id"] or
                        evidence.get("provider") != PROVIDER or
                        evidence.get("http_status") != 200 or
                        evidence.get("prompt_sha256") != task["prompt_sha256"] or
                        evidence.get("input_sha256") != task["input_sha256"] or
                        not isinstance(content, str) or not content.strip()):
                    continue
                task.update({
                    "model_name": evidence["request"]["model"],
                    "model_snapshot": str(evidence["response"].get("model", "undisclosed")),
                    "access_channel": "api",
                    "status": "accepted",
                    "quality_status": "pending_processing",
                    "raw_path": str(raw_path.relative_to(root)),
                    "output_sha256": sha256(content.strip().encode("utf-8")),
                    "generation_time_utc": evidence["completed_at_utc"],
                    "failure_reason": "",
                })
                recovered += 1
                break
            except (KeyError, IndexError, TypeError, ValueError):
                continue
    return recovered


def has_requested_mode(task: dict[str, str], root: Path, mode: str) -> bool:
    if task.get("status") != "accepted" or not task.get("raw_path"):
        return False
    try:
        evidence = json.loads((root / task["raw_path"]).read_text(encoding="utf-8"))
        actual = evidence["request"].get("thinking", {}).get("type", "enabled")
        return actual == mode
    except (OSError, ValueError, KeyError, TypeError):
        return False


def request_one(task: dict[str, str], source: dict[str, str], root: Path,
                raw_dir: Path, endpoint: str, api_key: str, model: str,
                thinking_mode: str, timeout: int, retries: int, pacer: RequestPacer,
                stop_event: threading.Event) -> dict[str, str] | None:
    if stop_event.is_set():
        return None
    template = (root / "protocol/edit_prompts" / f"{task['prompt_id']}.txt").read_text(encoding="utf-8")
    prompt = template.replace("{abstract}", source["abstract_clean"])
    payload = {
        "model": model,
        "temperature": 0.6 if thinking_mode == "disabled" else 1,
        "messages": [
            {"role": "system", "content": "你是中文法学论文摘要语言编辑。严格执行用户给出的编辑范围，只返回摘要正文。"},
            {"role": "user", "content": prompt},
        ],
    }
    if thinking_mode == "disabled":
        payload["thinking"] = {"type": "disabled"}
    result = dict(task)
    for attempt in range(1, retries + 1):
        if not pacer.wait(stop_event):
            return None
        started = utc_now()
        try:
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            if response.status_code in {400, 401, 402, 403}:
                try:
                    error = response.json().get("error", {})
                    error_code = error.get("type") or error.get("code") or "unknown"
                except ValueError:
                    error_code = "unknown"
                stop_event.set()
                result.update({
                    "status": "failed",
                    "failure_reason": f"HTTP {response.status_code} {error_code}",
                })
                return result
            if response.status_code == 429:
                try:
                    details = response.json().get("error", {})
                    reason = str(details.get("message") or details.get("type") or "rate_limited")[:240]
                except (ValueError, AttributeError):
                    reason = "rate_limited"
                try:
                    retry_after = float(response.headers.get("Retry-After", "0"))
                except ValueError:
                    retry_after = 0.0
                pacer.cooldown(max(retry_after, 30.0 * attempt))
                if attempt == retries:
                    stop_event.set()
                    result.update({"status": "rate_limited", "failure_reason": f"HTTP 429: {reason}"})
                    return result
                continue
            response.raise_for_status()
            envelope = response.json()
            content = envelope["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty_or_unexpected_response_content")
            completed = utc_now()
            evidence = {
                "record_id": task["record_id"],
                "provider": PROVIDER,
                "endpoint": endpoint,
                "request": payload,
                "started_at_utc": started,
                "completed_at_utc": completed,
                "http_status": response.status_code,
                "response": envelope,
                "response_body_sha256": sha256(response.content),
                "prompt_sha256": task["prompt_sha256"],
                "input_sha256": task["input_sha256"],
                "attempt": attempt,
            }
            raw_path = raw_dir / f"{task['record_id']}.json"
            raw_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            result.update({
                "model_name": model,
                "model_snapshot": str(envelope.get("model", "undisclosed")),
                "access_channel": "api",
                "status": "accepted",
                "quality_status": "pending_processing",
                "raw_path": str(raw_path.relative_to(root)),
                "output_sha256": sha256(content.strip().encode("utf-8")),
                "generation_time_utc": completed,
                "failure_reason": "",
            })
            return result
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            error = f"{type(exc).__name__}:{str(exc)[:240]}"
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    result.update({"status": "failed", "failure_reason": error})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model", default="kimi-k2.6")
    parser.add_argument("--thinking-mode", choices=("enabled", "disabled"), default="disabled")
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--source-id", default="china-law-2019-00038")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--request-interval", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--flush-every", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    if args.request_interval < 0 or args.limit < 0:
        parser.error("--request-interval and --limit must be nonnegative")

    registry = args.root / "source_registry"
    queue_path = registry / "generation_queue.csv"
    queue = read_csv(queue_path)
    recovered = recover_saved_responses(queue, args.root)
    if recovered:
        write_csv(queue_path, queue, list(queue[0]))
        print(f"recovered_saved_responses={recovered}", flush=True)
    sources = {row["source_id"]: row for row in read_csv(registry / "sources.csv")}
    all_tasks = [row for row in queue if row["provider"] == PROVIDER and row["prompt_id"] in LEVELS]
    tasks = [row for row in all_tasks if not args.pilot or row["source_id"] == args.source_id]
    pending = [row for row in tasks if not has_requested_mode(row, args.root, args.thinking_mode)]
    if args.limit:
        pending = pending[:args.limit]
    if args.pilot and (len(tasks) != 4 or {task["prompt_id"] for task in tasks} != LEVELS):
        parser.error("pilot source must have exactly one task per E1-E4 level")
    if not pending:
        print(json.dumps({"provider": PROVIDER, "status": "already_complete", "tasks": len(tasks)}))
        return
    api_key = os.getenv("KIMI_API_KEY") or getpass.getpass("Kimi API key: ")
    if not api_key.strip():
        parser.error("Kimi API key is required")

    kind = "test" if args.pilot else "batch"
    if args.thinking_mode == "disabled":
        kind += "_nonthinking"
    raw_dir = args.root / "raw" / f"kimi_{kind}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.root / "evaluation" / f"kimi_{kind}_run.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        previous_id = previous.get("run_id")
        if previous_id:
            if previous.get("status") == "running":
                previous["status"] = "interrupted"
                previous["interruption_reason"] = "process ended before completing the queue"
            archived = report_path.with_name(f"{previous_id}.json")
            archived.write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "run_id": f"kimi-{kind}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "provider": PROVIDER,
        "model_requested": args.model,
        "temperature": 0.6 if args.thinking_mode == "disabled" else 1,
        "thinking_mode": args.thinking_mode,
        "endpoint": args.endpoint,
        "total_provider_tasks": len(all_tasks),
        "pending_at_start": len(pending),
        "workers": 1 if args.pilot else args.workers,
        "request_interval_seconds": args.request_interval,
        "started_at_utc": utc_now(),
        "status": "running",
        "completed": 0,
        "failed": 0,
        "deferred": 0,
        "failures": [],
    }
    if args.pilot:
        report.update({"source_id": args.source_id, "evaluation_role": "pilot_only"})
    row_by_id = {row["record_id"]: row for row in queue}
    stop_event = threading.Event()
    pacer = RequestPacer(args.request_interval)
    processed = 0
    with ThreadPoolExecutor(max_workers=report["workers"]) as pool:
        futures = {
            pool.submit(request_one, task, sources[task["source_id"]], args.root,
                        raw_dir, args.endpoint, api_key, args.model, args.thinking_mode, args.timeout,
                        args.retries, pacer, stop_event): task
            for task in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is None:
                continue
            processed += 1
            if result["status"] == "accepted":
                row_by_id[result["record_id"]] = result
                report["completed"] += 1
            elif result["status"] == "rate_limited":
                report["deferred"] += 1
                report["rate_limit_reason"] = result["failure_reason"]
            else:
                row_by_id[result["record_id"]] = result
                report["failed"] += 1
                report["failures"].append({"record_id": result["record_id"], "error": result["failure_reason"]})
            if processed % args.flush_every == 0 or index == len(pending) or args.pilot:
                write_csv(queue_path, list(row_by_id.values()), list(queue[0]))
                report["last_flush_at_utc"] = utc_now()
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            if processed % 50 == 0 or index == len(pending) or args.pilot:
                print(f"completed={report['completed']} failed={report['failed']} remaining={len(pending)-index}", flush=True)
    write_csv(queue_path, list(row_by_id.values()), list(queue[0]))
    report["finished_at_utc"] = utc_now()
    report["status"] = ("paused_rate_limited" if report["deferred"] else
                        "blocked_auth_or_account_or_request" if stop_event.is_set() else
                        "completed" if report["failed"] == 0 else "completed_with_failures")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["failed"] or report["deferred"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
