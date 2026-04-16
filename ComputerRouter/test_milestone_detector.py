#!/usr/bin/env python3
"""
Test script: run the ModernBERT milestone detector on a saved trajectory (traj.jsonl).

Typical usage (from ComputerRouter/):

  python3 test_milestone_detector.py \
    --traj "hybrid_results/pyautogui/screenshot/Hybrid-.../libreoffice_calc/<task_id>/traj.jsonl" \
    --instruction "..." \
    --model-path "/gpfs/radev/scratch/cohan/jw3278/modernbert-milestone-detector"

Or load instruction from an evaluation example JSON:

  python3 test_milestone_detector.py \
    --traj ".../traj.jsonl" \
    --example-json "evaluation_examples/examples/libreoffice_calc/035f41ba-....json"

If you don't pass --example-json, the script will try to infer it from the traj path:
  .../<domain>/<task_id>/traj.jsonl  ->  evaluation_examples/examples/<domain>/<task_id>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# Make `mm_agents` importable when running this file directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


class _LocalDummyMilestoneDetector:
    """Fallback detector when torch/transformers aren't available."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or "cpu"

    def check_if_milestone(self, *args, **kwargs) -> Tuple[bool, float, str]:
        return False, 0.0, ""


def _try_create_detector(args: argparse.Namespace):
    """
    Create the configured milestone detector.

    If torch/transformers are not installed, this will:
    - return a local dummy detector if --use-dummy
    - otherwise raise with a helpful error message
    """
    if args.use_dummy:
        return _LocalDummyMilestoneDetector(device=args.device)

    try:
        from mm_agents.milestone_detector import create_milestone_detector  # type: ignore
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            "MilestoneDetector dependencies are missing in this Python environment. "
            f"Missing module: {missing!r}. "
            "Activate the environment that has torch+transformers installed, or re-run with --use-dummy."
        ) from e

    return create_milestone_detector(
        model_path=args.model_path,
        use_dummy=False,
        device=args.device,
        max_length=args.max_length,
        milestone_threshold=args.threshold,
        context_steps=args.context_steps,
    )


@dataclass(frozen=True)
class TrajStep:
    step_num: int
    response: str
    action: Optional[str] = None
    screenshot_file: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


def _safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def load_traj_steps(traj_jsonl: str) -> List[TrajStep]:
    """
    Load steps from traj.jsonl.

    Note: some trajectories contain repeated `step_num` entries (e.g. keyDown/keyUp),
    usually with identical `response`. We keep the **last** occurrence per step number.
    """
    by_step: Dict[int, Dict[str, Any]] = {}
    with open(traj_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = _safe_json_loads(line)
            if not obj:
                continue
            if "Error" in obj:
                # Preserve error line as non-step.
                continue
            step_num = obj.get("step_num")
            if not isinstance(step_num, int):
                continue
            by_step[step_num] = obj

    if not by_step:
        return []

    steps: List[TrajStep] = []
    for step_num in sorted(by_step.keys()):
        obj = by_step[step_num]
        steps.append(
            TrajStep(
                step_num=step_num,
                response=str(obj.get("response", "") or ""),
                action=(str(obj.get("action")) if obj.get("action") is not None else None),
                screenshot_file=(
                    str(obj.get("screenshot_file")) if obj.get("screenshot_file") is not None else None
                ),
                raw=obj,
            )
        )
    return steps


def load_instruction(
    instruction: Optional[str],
    example_json: Optional[str],
    traj_jsonl: Optional[str],
    examples_root: str,
) -> str:
    if instruction is not None:
        return instruction
    if not example_json:
        # Try to infer from traj path: .../<domain>/<task_id>/traj.jsonl
        if traj_jsonl:
            task_id = os.path.basename(os.path.dirname(os.path.abspath(traj_jsonl)))
            domain = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(traj_jsonl))))
            inferred = os.path.join(examples_root, domain, f"{task_id}.json")
            if os.path.exists(inferred):
                example_json = inferred
            else:
                return ""
        else:
            return ""
    with open(example_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # OSWorld evaluation examples use "instruction"
    if isinstance(obj, dict) and isinstance(obj.get("instruction"), str):
        return obj["instruction"]
    # Fallbacks (just in case)
    for key in ("task", "prompt", "goal", "query"):
        if isinstance(obj, dict) and isinstance(obj.get(key), str):
            return obj[key]
    return ""


def format_row(cols: List[str], widths: List[int]) -> str:
    padded = []
    for c, w in zip(cols, widths):
        c = c if c is not None else ""
        s = str(c)
        if len(s) > w:
            s = s[: max(0, w - 1)] + "…"
        padded.append(s.ljust(w))
    return "  ".join(padded).rstrip()


def summarize_predictions(
    predictions: List[Dict[str, Any]],
    top_k: int,
    threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    milestones = [p for p in predictions if bool(p["is_milestone"])]
    top = sorted(predictions, key=lambda x: float(x["milestone_prob"]), reverse=True)[: max(0, top_k)]
    # Recompute milestone boolean if threshold changed at CLI-level
    for p in predictions:
        p["is_milestone"] = float(p["milestone_prob"]) >= threshold
    milestones = [p for p in predictions if bool(p["is_milestone"])]
    return milestones, top


def main() -> int:
    ap = argparse.ArgumentParser(description="Run milestone detector on a traj.jsonl")
    ap.add_argument("--traj", required=True, help="Path to traj.jsonl")
    ap.add_argument("--instruction", default=None, help="Task instruction text (overrides --example-json)")
    ap.add_argument("--example-json", default=None, help="Path to example JSON containing 'instruction'")
    ap.add_argument(
        "--examples-root",
        default=os.path.join(_THIS_DIR, "evaluation_examples", "examples"),
        help="Root directory containing evaluation examples (default: ComputerRouter/evaluation_examples/examples)",
    )

    ap.add_argument(
        "--model-path",
        default="/gpfs/radev/scratch/cohan/jw3278/modernbert-milestone-detector",
        help="Path to fine-tuned ModernBERT milestone detector model",
    )
    ap.add_argument("--device", default=None, help="Device for inference: cuda/cpu (default: auto)")
    ap.add_argument("--max-length", type=int, default=2048, help="Max token length")
    ap.add_argument("--context-steps", type=int, default=5, help="Number of previous steps to include")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for classifying a milestone",
    )
    ap.add_argument(
        "--use-dummy",
        action="store_true",
        help="Use DummyMilestoneDetector (always returns non-milestone)",
    )

    ap.add_argument("--top-k", type=int, default=10, help="Show top-K steps by probability")
    ap.add_argument(
        "--print-formatted-text",
        action="store_true",
        help="Print the formatted detector input text for predicted milestones",
    )
    ap.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save full per-step predictions as JSON",
    )

    args = ap.parse_args()

    traj_path = os.path.abspath(args.traj)
    if not os.path.exists(traj_path):
        print(f"ERROR: traj.jsonl not found: {traj_path}", file=sys.stderr)
        return 2

    task_instruction = load_instruction(
        instruction=args.instruction,
        example_json=args.example_json,
        traj_jsonl=traj_path,
        examples_root=os.path.abspath(args.examples_root),
    )
    if not task_instruction.strip():
        print(
            "WARNING: no instruction provided (pass --instruction or --example-json). "
            "Detector will run with an empty task description.",
            file=sys.stderr,
        )
    else:
        # Print instruction source hint (without dumping the full prompt).
        if args.instruction is not None:
            src = "--instruction"
        elif args.example_json is not None:
            src = f"--example-json {os.path.abspath(args.example_json)}"
        else:
            task_id = os.path.basename(os.path.dirname(traj_path))
            domain = os.path.basename(os.path.dirname(os.path.dirname(traj_path)))
            inferred = os.path.join(os.path.abspath(args.examples_root), domain, f"{task_id}.json")
            src = f"inferred {inferred}"
        snippet = " ".join(task_instruction.strip().split())
        snippet = snippet[:160] + ("…" if len(snippet) > 160 else "")
        print(f"instruction_source: {src}")
        print(f"instruction_snippet: {snippet}")

    steps = load_traj_steps(traj_path)
    if not steps:
        print(f"ERROR: no steps found in: {traj_path}", file=sys.stderr)
        return 2

    try:
        detector = _try_create_detector(args)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    predictions: List[Dict[str, Any]] = []
    step_responses: List[str] = []
    for s in steps:
        step_responses.append(s.response)
        is_milestone, prob, formatted_text = detector.check_if_milestone(
            task_description=task_instruction,
            step_responses=step_responses,
            current_step=s.step_num,
        )
        predictions.append(
            {
                "step_num": s.step_num,
                "milestone_prob": float(prob),
                "is_milestone": bool(is_milestone),
                "action": s.action,
                "screenshot_file": s.screenshot_file,
                "formatted_text": formatted_text,
            }
        )

    # Apply CLI threshold again (in case the detector used a different one)
    milestones, top = summarize_predictions(predictions, top_k=args.top_k, threshold=args.threshold)

    print(f"traj: {traj_path}")
    print(f"steps: {len(steps)}")
    print(f"model_path: {args.model_path}")
    print(f"device: {getattr(detector, 'device', 'unknown')}")
    print(f"context_steps: {args.context_steps}")
    print(f"threshold: {args.threshold}")
    print()

    widths = [6, 10, 9, 34, 28]
    print(format_row(["step", "prob", "milestone", "action", "screenshot_file"], widths))
    print(format_row(["-" * w for w in widths], widths))

    for p in predictions:
        print(
            format_row(
                [
                    str(p["step_num"]),
                    f"{p['milestone_prob']:.4f}",
                    "YES" if p["is_milestone"] else "no",
                    p.get("action") or "",
                    os.path.basename(p.get("screenshot_file") or ""),
                ],
                widths,
            )
        )

    print()
    print(f"predicted milestones (>= {args.threshold}): {len(milestones)}")
    if milestones:
        print("steps:", ", ".join(str(m["step_num"]) for m in milestones))
    print()

    print(f"top-{args.top_k} by probability:")
    for p in top:
        print(f"  step {p['step_num']:>3}: prob={p['milestone_prob']:.4f}  milestone={bool(p['is_milestone'])}")

    if args.print_formatted_text and milestones:
        print()
        print("formatted_text for predicted milestones:")
        for p in milestones:
            print()
            print("=" * 80)
            print(f"STEP {p['step_num']}  prob={p['milestone_prob']:.4f}")
            print("=" * 80)
            print(p["formatted_text"])

    if args.save_json:
        out_path = os.path.abspath(args.save_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Avoid writing extremely large blobs unless requested.
        payload = {
            "traj": traj_path,
            "model_path": args.model_path,
            "device": getattr(detector, "device", None),
            "context_steps": args.context_steps,
            "threshold": args.threshold,
            "instruction": task_instruction,
            "predictions": predictions,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print()
        print(f"saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

