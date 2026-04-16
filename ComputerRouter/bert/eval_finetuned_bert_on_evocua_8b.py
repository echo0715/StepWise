#!/usr/bin/env python3
"""
Evaluate a finetuned (Modern)BERT stuck-step detector on EvoCUA trajectories.

Goal metric (requested):
  Count trajectories that are predicted as "non-stuck" but actually failed.

Definitions used (matches existing analysis conventions in this repo):
- Per-step model output: binary classification (0=not stuck, 1=stuck)
- Per-trajectory "predicted non-stuck": num_pred_stuck_steps == 0
- Per-trajectory "failed":
    - primary: score == 0 in evocua_8b/summary/results.json
    - fallback: last_step.info.fail == True (if present)

Example:
  python BERT_Training/eval_finetuned_bert_on_evocua_8b.py \
    --model_dir /path/to/modernbert-stuck-detector \
    --evocua_root BERT_Training/evocua_8b
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _repo_root_from_here() -> str:
    """
    This file lives in <repo_root>/BERT_Training/.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path_with_repo_root_fallback(path: str) -> str:
    """
    Resolve a user-provided path robustly:
    - First try as-is (relative to cwd if relative).
    - If it doesn't exist and is relative, try resolving relative to repo root.
    """
    if not path:
        return path
    abs_from_cwd = os.path.abspath(path)
    if os.path.exists(abs_from_cwd):
        return abs_from_cwd
    if os.path.isabs(path):
        return abs_from_cwd
    repo_root = _repo_root_from_here()
    abs_from_repo = os.path.abspath(os.path.join(repo_root, path))
    if os.path.exists(abs_from_repo):
        return abs_from_repo
    # Fall back to cwd resolution (best error messages downstream).
    return abs_from_cwd


def _resolve_output_path(path: str) -> str:
    """
    Output paths shouldn't depend on current working directory (to avoid
    accidentally writing to BERT_Training/BERT_Training/...).

    - If absolute: use as-is
    - If relative: resolve relative to repo root
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path
    repo_root = _repo_root_from_here()
    return os.path.abspath(os.path.join(repo_root, path))


def _iter_traj_jsonl_files(evocua_root: str) -> Iterable[str]:
    """
    Finds traj.jsonl files under:
      <evocua_root>/pyautogui/screenshot/EvoCUA/<app>/<task_id>/traj.jsonl
    """
    for root, _dirs, files in os.walk(evocua_root):
        if "traj.jsonl" in files:
            yield os.path.join(root, "traj.jsonl")


def _task_id_and_app_from_traj_path(traj_path: str) -> Tuple[str, str]:
    # .../EvoCUA/<app>/<task_id>/traj.jsonl
    task_dir = os.path.dirname(traj_path)
    task_id = os.path.basename(task_dir)
    app = os.path.basename(os.path.dirname(task_dir))
    return task_id, app


def _load_results_map(results_json_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Map task_id -> result dict with at least {"application","status","score",...}
    """
    with open(results_json_path, "r") as f:
        data = json.load(f)
    out: Dict[str, Dict[str, Any]] = {}
    for row in data:
        tid = row.get("task_id")
        if tid:
            out[str(tid)] = row
    return out


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            steps.append(json.loads(line))
    steps.sort(key=lambda d: int(d.get("step_num", 0)))
    return steps


def _build_step_text_prefix(steps: List[Dict[str, Any]], upto_idx_inclusive: int) -> str:
    """
    DEPRECATED: kept for backward compatibility during development.
    The actual training CSV uses a sliding window of recent steps (see below).
    """
    chunks: List[str] = []
    for i in range(upto_idx_inclusive + 1):
        s = steps[i]
        step_num = int(s.get("step_num", i + 1))
        response = s.get("response", "")
        action = s.get("action", "")
        chunks.append(f"Step {step_num}:\nResponse: {response}\nAction: {action}\n")
    return "\n".join(chunks).strip() + "\n"


def _build_step_text_window(
    steps: List[Dict[str, Any]], end_idx_inclusive: int, window_steps: int
) -> str:
    """
    Mirrors the training CSV style:

    The training examples are NOT full-history; they are a sliding window of the most
    recent steps including the current step.

    Empirically from `bert_training_dataset.csv`:
      - step 1 contains "Step 1"
      - step 21 contains "Step 16 ... Step 21"
    i.e. default window is 6 steps.
    """
    if window_steps <= 0:
        raise ValueError("--window_steps must be >= 1")

    start_idx = max(0, end_idx_inclusive - (window_steps - 1))
    chunks: List[str] = []
    for i in range(start_idx, end_idx_inclusive + 1):
        s = steps[i]
        step_num = int(s.get("step_num", i + 1))
        response = s.get("response", "")
        action = s.get("action", "")
        chunks.append(f"Step {step_num}:\nResponse: {response}\nAction: {action}\n")
    return "\n".join(chunks).strip() + "\n"


@dataclass
class TrajEvalRow:
    task_id: str
    application: str
    score: Optional[float]
    failed: bool
    num_steps: int
    num_pred_stuck_steps: int
    pred_non_stuck: bool
    pred_non_stuck_and_failed: bool
    pred_stuck_step_nums: List[int]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to your finetuned model directory (or HF repo id).",
    )
    parser.add_argument(
        "--evocua_root",
        type=str,
        default="BERT_Training/evocua_8b",
        help="Root directory of evocua_8b run outputs.",
    )
    parser.add_argument(
        "--results_json",
        type=str,
        default="BERT_Training/evocua_8b/summary/results.json",
        help="Path to run summary results.json (contains score per task).",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument(
        "--window_steps",
        type=int,
        default=6,
        help=(
            "How many steps to include in the model input window (including the current step). "
            "This should match the training dataset format (default: 6)."
        ),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Tokenizer max_length (should match fine-tune).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', etc.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="BERT_Training/evocua_8b_finetuned_bert_eval.json",
        help="Where to write the per-trajectory evaluation JSON.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="BERT_Training/evocua_8b_finetuned_bert_eval.csv",
        help="Where to write the per-trajectory evaluation CSV.",
    )
    parser.add_argument(
        "--debug_task_id",
        type=str,
        default=None,
        help="If set, print the exact constructed prompt(s) for this task id and exit.",
    )
    parser.add_argument(
        "--debug_step_num",
        type=int,
        default=None,
        help=(
            "With --debug_task_id, optionally limit to a specific 1-indexed step_num to print "
            "(prints all steps if omitted)."
        ),
    )
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies. Install torch + transformers in your environment."
        ) from e

    device: str
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    evocua_root = _resolve_path_with_repo_root_fallback(args.evocua_root)
    results_json = _resolve_path_with_repo_root_fallback(args.results_json)
    output_json = _resolve_output_path(args.output_json)
    output_csv = _resolve_output_path(args.output_csv)

    results_map = _load_results_map(results_json) if os.path.exists(results_json) else {}

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.model_max_length = int(args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()
    model.to(device)

    traj_paths = sorted(_iter_traj_jsonl_files(evocua_root))
    if not traj_paths:
        raise FileNotFoundError(
            "No traj.jsonl files found.\n"
            f"  provided --evocua_root: {args.evocua_root}\n"
            f"  resolved evocua_root:   {evocua_root}\n"
            "Tip: if you are running from inside BERT_Training/, pass --evocua_root evocua_8b\n"
        )

    # Debug mode: jump straight to the requested trajectory and exit after printing prompts.
    if args.debug_task_id:
        target: Optional[str] = None
        target_app: Optional[str] = None
        for p in traj_paths:
            tid, app = _task_id_and_app_from_traj_path(p)
            if tid == args.debug_task_id:
                target = p
                target_app = app
                break
        if target is None:
            raise FileNotFoundError(
                f"Could not find traj.jsonl for debug_task_id={args.debug_task_id} under {evocua_root}"
            )

        steps = _read_jsonl(target)
        step_nums = [int(s.get("step_num", i + 1)) for i, s in enumerate(steps)]
        print(f"=== DEBUG task_id={args.debug_task_id} app={target_app} num_steps={len(steps)} ===")
        for i, step_num in enumerate(step_nums):
            if args.debug_step_num is not None and int(args.debug_step_num) != step_num:
                continue
            prompt = _build_step_text_window(steps, i, int(args.window_steps))
            print(f"\n--- Prompt for step_num={step_num} (window_steps={args.window_steps}) ---")
            print(prompt)
        return

    rows: List[TrajEvalRow] = []
    total_step_examples = 0

    for idx, traj_path in enumerate(traj_paths, start=1):
        task_id, app = _task_id_and_app_from_traj_path(traj_path)
        steps = _read_jsonl(traj_path)
        if not steps:
            # Degenerate: no steps recorded
            score = results_map.get(task_id, {}).get("score")
            failed = bool(score == 0)
            row = TrajEvalRow(
                task_id=task_id,
                application=app,
                score=score,
                failed=failed,
                num_steps=0,
                num_pred_stuck_steps=0,
                pred_non_stuck=True,
                pred_non_stuck_and_failed=failed,
                pred_stuck_step_nums=[],
            )
            rows.append(row)
            continue

        # Determine "failed" from summary results (preferred) or traj terminal info.
        score = results_map.get(task_id, {}).get("score")
        summary_failed = (score == 0) if score is not None else None

        last_info = steps[-1].get("info") or {}
        traj_failed_flag = bool(last_info.get("fail") is True)

        failed = bool(summary_failed) if summary_failed is not None else traj_failed_flag

        # Build windowed texts for each step (matches training format).
        step_texts = [_build_step_text_window(steps, i, int(args.window_steps)) for i in range(len(steps))]
        total_step_examples += len(step_texts)

        # Batched inference.
        pred_labels: List[int] = []
        for start in range(0, len(step_texts), int(args.batch_size)):
            batch_text = step_texts[start : start + int(args.batch_size)]
            enc = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=int(args.max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
                preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
                pred_labels.extend(int(p) for p in preds)

        pred_stuck_step_nums = [
            int(steps[i].get("step_num", i + 1)) for i, p in enumerate(pred_labels) if int(p) == 1
        ]
        num_pred_stuck_steps = len(pred_stuck_step_nums)
        pred_non_stuck = num_pred_stuck_steps == 0
        pred_non_stuck_and_failed = pred_non_stuck and failed

        rows.append(
            TrajEvalRow(
                task_id=task_id,
                application=app,
                score=score,
                failed=failed,
                num_steps=len(steps),
                num_pred_stuck_steps=num_pred_stuck_steps,
                pred_non_stuck=pred_non_stuck,
                pred_non_stuck_and_failed=pred_non_stuck_and_failed,
                pred_stuck_step_nums=pred_stuck_step_nums,
            )
        )

        # Lightweight progress (every ~25 trajectories).
        if idx == 1 or idx % 25 == 0 or idx == len(traj_paths):
            print(f"[{idx}/{len(traj_paths)}] processed {task_id} ({app})")

    # Aggregate metrics.
    total_trajs = len(rows)
    failed_trajs = sum(1 for r in rows if r.failed)
    pred_non_stuck_trajs = sum(1 for r in rows if r.pred_non_stuck)
    pred_non_stuck_and_failed = sum(1 for r in rows if r.pred_non_stuck_and_failed)

    print("\n=== Finetuned BERT evaluation on evocua_8b ===")
    print(f"Trajectories:                 {total_trajs}")
    print(f"Total step examples scored:   {total_step_examples}")
    print(f"Failed trajectories:          {failed_trajs}")
    print(f"Predicted non-stuck:          {pred_non_stuck_trajs}")
    print(f"Pred non-stuck AND failed:    {pred_non_stuck_and_failed}")

    # Write outputs.
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)
    print(f"\nWrote JSON: {output_json}")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "application",
                "score",
                "failed",
                "num_steps",
                "num_pred_stuck_steps",
                "pred_non_stuck",
                "pred_non_stuck_and_failed",
                "pred_stuck_step_nums",
            ],
        )
        w.writeheader()
        for r in rows:
            d = asdict(r)
            d["pred_stuck_step_nums"] = ",".join(str(x) for x in r.pred_stuck_step_nums)
            w.writerow(d)
    print(f"Wrote CSV: {output_csv}")

    # Print the list of (non-stuck & failed) ids for quick inspection.
    offenders = [r for r in rows if r.pred_non_stuck_and_failed]
    if offenders:
        print("\nTask IDs predicted non-stuck but failed:")
        for r in offenders:
            print(f"  {r.task_id} ({r.application}) score={r.score}")


if __name__ == "__main__":
    main()

