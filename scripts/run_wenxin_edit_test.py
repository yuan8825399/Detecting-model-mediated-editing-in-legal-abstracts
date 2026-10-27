"""Run one bounded four-level Wenxin edit test against an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import requests


ROOT = Path("research_benchmark_legal_edit_v1")
PROVIDER = "wenxin"
LEVELS = {"E1", "E2", "E3", "E4"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    temp = path.with_name(path.stem + ".tmp" + path.suffix)
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-id", default="china-law-2019-00038")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--pause", type=float, default=0.5)
    args = parser.parse_args()

    api_url = os.getenv("WENXIN_API_URL", "").strip()
    api_key = os.getenv("WENXIN_API_KEY", "").strip()
    model = os.getenv("WENXIN_MODEL", "").strip()
    registry = args.root / "source_registry"
    queue_path = registry / "generation_queue.csv"
    queue = read_csv(queue_path)
    sources = {row["source_id"]: row for row in read_csv(registry / "sources.csv")}
    source = sources.get(args.source_id)
    if source is None:
        raise SystemExit(f"Unknown source_id: {args.source_id}")
    tasks = [row for row in queue if row["provider"] == PROVIDER and row["source_id"] == args.source_id and row["prompt_id"] in LEVELS]
    tasks.sort(key=lambda row: row["intervention_level"])
    if len(tasks) != 4 or {row["prompt_id"] for row in tasks} != LEVELS:
        raise SystemExit(f"Expected exactly one pending Wenxin task for each E1-E4; found {len(tasks)}")
    report_path = args.root / "evaluation/wenxin_test_run.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": f"wenxin-test-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "provider": PROVIDER,
        "source_id": args.source_id,
        "group_id": source["group_id"],
        "evaluation_role": "pilot_only",
        "source_title": source["source_title_clean"],
        "planned_tasks": [row["record_id"] for row in tasks],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "completed": [],
        "failures": [],
    }
    if not api_url or not api_key or not model:
        missing = [name for name, value in (("WENXIN_API_URL", api_url), ("WENXIN_API_KEY", api_key), ("WENXIN_MODEL", model)) if not value]
        report["status"] = "awaiting_configuration"
        report["missing_configuration"] = missing
        report["message"] = "No request was sent; configure the Wenxin OpenAI-compatible endpoint, credential, and model before retrying."
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    prompts_dir = args.root / "protocol/edit_prompts"
    raw_dir = args.root / "raw/wenxin_test"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for index, task in enumerate(tasks):
        prompt_template = (prompts_dir / f"{task['prompt_id']}.txt").read_text(encoding="utf-8")
        prompt = prompt_template.replace("{abstract}", source["abstract_clean"])
        payload = {
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "你是中文法学论文摘要语言编辑。严格执行用户给出的编辑范围，只返回摘要正文。"},
                {"role": "user", "content": prompt},
            ],
        }
        started = datetime.now(timezone.utc).isoformat()
        try:
            response = requests.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=args.timeout,
            )
            response.raise_for_status()
            envelope = response.json()
            content = envelope["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty_or_unexpected_response_content")
            evidence = {
                "record_id": task["record_id"],
                "provider": PROVIDER,
                "endpoint": api_url,
                "request": payload,
                "started_at_utc": started,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "http_status": response.status_code,
                "response": envelope,
                "response_body_sha256": digest(response.content),
                "prompt_sha256": task["prompt_sha256"],
                "input_sha256": task["input_sha256"],
            }
            raw_path = raw_dir / f"{task['record_id']}.json"
            raw_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            task["model_name"] = model
            task["model_snapshot"] = str(envelope.get("model", "undisclosed"))
            task["access_channel"] = "api"
            task["status"] = "accepted"
            task["quality_status"] = "pending_processing"
            task["raw_path"] = str(raw_path.relative_to(args.root))
            task["output_sha256"] = digest(content.strip().encode("utf-8"))
            task["generation_time_utc"] = evidence["completed_at_utc"]
            task["failure_reason"] = ""
            report["completed"].append({"record_id": task["record_id"], "prompt_id": task["prompt_id"], "model_snapshot": task["model_snapshot"], "raw_path": task["raw_path"]})
        except Exception as exc:
            task["status"] = "failed"
            task["failure_reason"] = type(exc).__name__
            report["failures"].append({"record_id": task["record_id"], "error": type(exc).__name__})
            report["status"] = "partial_failure"
            write_csv(queue_path, queue, list(queue[0]))
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            raise SystemExit(1)
        if index + 1 < len(tasks):
            time.sleep(args.pause)

    write_csv(queue_path, queue, list(queue[0]))
    report["status"] = "completed" if len(report["completed"]) == 4 else "partial_failure"
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
