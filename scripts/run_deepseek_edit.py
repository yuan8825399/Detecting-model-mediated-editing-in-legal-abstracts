"""Collect DeepSeek E1-E4 edits into the existing CLAW-Edit queue."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import time

import requests

from run_kimi_edit import read_csv, sha256, utc_now, write_csv


ROOT = Path("research_benchmark_legal_edit_v1")
ENDPOINT = "https://api.deepseek.com/chat/completions"
PROVIDER = "deepseek"
LEVELS = {"E1", "E2", "E3", "E4"}


def response_content(evidence: dict) -> str:
    response = evidence["response"]
    if not isinstance(response, dict):
        raise ValueError("invalid_response_envelope")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("missing_choices")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("incomplete_response")
    content = choice["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty_response")
    return content.strip()


def recover_saved_responses(queue: list[dict[str, str]], root: Path) -> int:
    recovered = 0
    for task in queue:
        if task["provider"] != PROVIDER or task["status"] == "accepted":
            continue
        for kind in ("test", "batch"):
            path = root / "raw" / f"deepseek_{kind}" / f"{task['record_id']}.json"
            if not path.exists():
                continue
            try:
                evidence = json.loads(path.read_text(encoding="utf-8"))
                if (evidence["record_id"] != task["record_id"] or
                        evidence["provider"] != PROVIDER or
                        evidence["http_status"] != 200 or
                        evidence["prompt_sha256"] != task["prompt_sha256"] or
                        evidence["input_sha256"] != task["input_sha256"]):
                    continue
                content = response_content(evidence)
                task.update({
                    "model_name": evidence["request"]["model"],
                    "model_snapshot": str(evidence["response"].get("model", "undisclosed")),
                    "access_channel": "api",
                    "status": "accepted",
                    "quality_status": "pending_processing",
                    "raw_path": str(path.relative_to(root)),
                    "output_sha256": sha256(content.encode("utf-8")),
                    "generation_time_utc": evidence["completed_at_utc"],
                    "failure_reason": "",
                })
                recovered += 1
                break
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                continue
    return recovered


def request_one(task: dict[str, str], source: dict[str, str], root: Path,
                raw_dir: Path, key: str, model: str,
                timeout: int, retries: int) -> tuple[dict[str, str] | None, str]:
    template = (root / "protocol/edit_prompts" / f"{task['prompt_id']}.txt").read_text(encoding="utf-8")
    prompt = template.replace("{abstract}", source["abstract_clean"])
    payload = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 2048,
        "messages": [
            {"role": "system", "content": "你是中文法学论文摘要语言编辑。严格执行用户给出的编辑范围，只返回摘要正文。"},
            {"role": "user", "content": prompt},
        ],
    }
    last_error = "unknown_error"
    for attempt in range(1, retries + 1):
        started = utc_now()
        try:
            response = requests.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            if response.status_code in {400, 401, 402, 403}:
                try:
                    detail = response.json().get("error", {})
                    code = detail.get("type") or detail.get("code") or "unknown"
                except (ValueError, AttributeError):
                    code = "unknown"
                return None, f"HTTP {response.status_code}: {code}"
            if response.status_code == 429:
                try:
                    delay = float(response.headers.get("Retry-After", "0"))
                except ValueError:
                    delay = 0.0
                last_error = "HTTP 429: rate_limited"
                if attempt < retries:
                    time.sleep(max(delay, 30.0 * attempt))
                    continue
                return None, last_error
            response.raise_for_status()
            envelope = response.json()
            if isinstance(envelope, dict) and "error" in envelope:
                error = envelope["error"]
                code = (error.get("code") or error.get("type") or "unknown"
                        if isinstance(error, dict) else "unknown")
                last_error = f"API error response: {str(code)[:80]}"
                if str(code).lower() in {"insufficient_balance", "invalid_api_key", "invalid_request_error"}:
                    return None, last_error
                if attempt < retries:
                    time.sleep(min(15 * attempt, 60))
                    continue
                return None, last_error
            completed = utc_now()
            evidence = {
                "record_id": task["record_id"],
                "provider": PROVIDER,
                "endpoint": ENDPOINT,
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
            content = response_content(evidence)
            path = raw_dir / f"{task['record_id']}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
            result = dict(task)
            result.update({
                "model_name": model,
                "model_snapshot": str(envelope.get("model", "undisclosed")),
                "access_channel": "api",
                "status": "accepted",
                "quality_status": "pending_processing",
                "raw_path": str(path.relative_to(root)),
                "output_sha256": sha256(content.encode("utf-8")),
                "generation_time_utc": completed,
                "failure_reason": "",
            })
            return result, ""
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
    return None, last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--source-id", default="china-law-2019-00038")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--request-interval", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--flush-every", type=int, default=10)
    args = parser.parse_args()
    if args.limit < 0 or args.request_interval < 0 or args.retries < 1 or args.flush_every < 1:
        parser.error("invalid limit, interval, retries, or flush frequency")

    queue_path = args.root / "source_registry/generation_queue.csv"
    queue = read_csv(queue_path)
    recovered = recover_saved_responses(queue, args.root)
    if recovered:
        write_csv(queue_path, queue, list(queue[0]))
        print(f"recovered_saved_responses={recovered}", flush=True)
    sources = {row["source_id"]: row for row in read_csv(args.root / "source_registry/sources.csv")}
    all_tasks = [row for row in queue if row["provider"] == PROVIDER and row["prompt_id"] in LEVELS]
    tasks = [row for row in all_tasks if not args.pilot or row["source_id"] == args.source_id]
    if args.pilot and (len(tasks) != 4 or {task["prompt_id"] for task in tasks} != LEVELS):
        parser.error("pilot source must have exactly one task for each E1-E4 level")
    pending = [row for row in tasks if row["status"] != "accepted" or not row["raw_path"]]
    if args.limit:
        pending = pending[:args.limit]
    if not pending:
        print(json.dumps({"provider": PROVIDER, "status": "already_complete", "tasks": len(tasks)}))
        return
    key = os.getenv("DEEPSEEK_API_KEY") or getpass.getpass("DeepSeek API key: ")
    if not key.strip():
        parser.error("DeepSeek API key is required")

    kind = "test" if args.pilot else "batch"
    raw_dir = args.root / "raw" / f"deepseek_{kind}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.root / "evaluation" / f"deepseek_{kind}_run.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("run_id"):
            if previous.get("status") == "running":
                previous["status"] = "interrupted"
                previous["interruption_reason"] = "process ended before completing the queue"
            archive = report_path.with_name(f"{previous['run_id']}.json")
            archive.write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "run_id": f"deepseek-{kind}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "provider": PROVIDER,
        "model_requested": args.model,
        "temperature": 0.2,
        "max_tokens": 2048,
        "endpoint": ENDPOINT,
        "total_provider_tasks": len(all_tasks),
        "pending_at_start": len(pending),
        "request_interval_seconds": args.request_interval,
        "started_at_utc": utc_now(),
        "status": "running",
        "completed": 0,
        "failed": 0,
        "last_error": "",
    }
    if args.pilot:
        report.update({"source_id": args.source_id, "evaluation_role": "pilot_only"})
    by_id = {row["record_id"]: row for row in queue}
    next_request = time.monotonic()
    for index, task in enumerate(pending, 1):
        time.sleep(max(0.0, next_request - time.monotonic()))
        next_request = time.monotonic() + args.request_interval
        result, error = request_one(task, sources[task["source_id"]], args.root,
                                    raw_dir, key, args.model,
                                    args.timeout, args.retries)
        if result is None:
            report["last_error"] = error
            report["status"] = "paused_error"
            report["failed"] += 1
            break
        by_id[result["record_id"]] = result
        report["completed"] += 1
        if index % args.flush_every == 0 or index == len(pending) or args.pilot:
            write_csv(queue_path, list(by_id.values()), list(queue[0]))
            report["last_flush_at_utc"] = utc_now()
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if index % 50 == 0:
            print(f"completed={report['completed']} remaining={len(pending)-index}", flush=True)
    write_csv(queue_path, list(by_id.values()), list(queue[0]))
    report["last_flush_at_utc"] = utc_now()
    report["finished_at_utc"] = utc_now()
    if report["status"] == "running":
        report["status"] = "completed"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
