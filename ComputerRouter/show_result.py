# python3 show_result.py --folder evocua_results/pyautogui/screenshot/EvoCUA-S2
# python3 show_result.py --folder hybrid_results_new/pyautogui/screenshot/Hybrid-EvoCUA-S2-claude-sonnet-4-5-20250929
# python3 show_result.py --folder hybrid_kimi_results/pyautogui/screenshot/Hybrid-EvoCUA-S2-kimi-k2.5
# python3 show_result.py --folder hybrid_qwen3_kimi_results/pyautogui/screenshot/Hybrid-qwen3-vl-8b-kimi-k2.5
# python3 show_result.py --folder results/pyautogui/screenshot/kimi-k2.5
# python3 show_result.py --folder kimi_results/pyautogui/screenshot/kimi-k2.5
# python3 show_result.py --folder hybrid_kimi_bounce_results/pyautogui/screenshot/HybridBounce-EvoCUA-S2-kimi-k2.5-k5
# python3 show_result.py --folder results_claude-sonnet-4-5-20250929_50steps/claude_computer_use/screenshot/global.anthropic.claude-sonnet-4-5-20250929-v1:0
# python3 show_result.py --folder qwen3_8b_thinking_results/pyautogui/screenshot/Qwen/Qwen3-VL-8B-Thinking
# python3 show_result.py --folder periodic_verify_results/pyautogui/screenshot/PeriodicVerify-EvoCUA-S2-kimi-k2.5-every5
import os
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple


def _parse_score(raw: str) -> float:
    s = (raw or "").strip()
    if s == "":
        return 0.0
    try:
        return float(s)
    except Exception:
        pass
    # handle common legacy formats
    try:
        return float(eval(s))  # noqa: S307 - legacy result files sometimes contain Python literals
    except Exception:
        return float(bool(s))


def _safe_rate(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs) * 100


def _print_group_rate(label: str, domain_result: Dict[str, List[float]], keys: List[str]) -> None:
    present: List[float] = []
    for k in keys:
        if k in domain_result:
            present += domain_result[k]
    if not present:
        return
    rate = _safe_rate(present)
    if rate is None:
        return
    print(label, "Success Rate:", rate, "%")


def _read_json_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _is_success_result(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        # avoid treating True as 1
        return False
    if isinstance(v, (int, float)):
        return float(v) == 1.0
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return False
        try:
            return float(s) == 1.0
        except Exception:
            return False
    return False


def _default_claude_results_root() -> Optional[str]:
    """
    Default location (relative to this script) for Claude's own run results.
    Returns None if it doesn't exist.
    """
    base_dir = os.path.dirname(__file__)
    cand = os.path.normpath(
        os.path.join(
            base_dir,
            "..",
            "BERT_Training",
            "results_claude-sonnet-4-5-20250929_50steps",
        )
    )
    return cand if os.path.exists(cand) else None


def _load_claude_results_map(claude_root: str) -> Dict[str, Dict[str, float]]:
    """
    Load Claude's own run per-task scores.

    Preferred source:
      <claude_root>/summary/results.json  (list of dicts with application/task_id/score)

    Fallback:
      Walk under <claude_root>/claude_computer_use and parse per-task result.txt paths:
        .../screenshot/<model>/<application>/<task_id>/result.txt
    """
    out: Dict[str, Dict[str, float]] = {}
    if not claude_root:
        return out

    summary_path = os.path.join(claude_root, "summary", "results.json")
    try:
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                for row in data:
                    if not isinstance(row, dict):
                        continue
                    app = row.get("application", None)
                    task_id = row.get("task_id", None)
                    score = row.get("score", None)
                    if not isinstance(app, str) or not isinstance(task_id, str):
                        continue
                    if app not in out:
                        out[app] = {}
                    # normalize to float score
                    if _is_success_result(score):
                        out[app][task_id] = 1.0
                    else:
                        try:
                            out[app][task_id] = float(score)
                        except Exception:
                            out[app][task_id] = 0.0
                return out
    except Exception:
        # fall back to filesystem parsing below
        pass

    base = os.path.join(claude_root, "claude_computer_use")
    if not os.path.exists(base):
        return out

    for root, _dirs, files in os.walk(base):
        if "result.txt" not in files:
            continue
        result_path = os.path.join(root, "result.txt")
        parts = root.split(os.sep)
        # expected: .../screenshot/<model>/<application>/<task_id>
        if "screenshot" not in parts:
            continue
        idx = parts.index("screenshot")
        if idx + 3 >= len(parts):
            continue
        app = parts[idx + 2]
        task_id = parts[idx + 3]
        try:
            raw = open(result_path, "r").read()
            score = _parse_score(raw)
        except Exception:
            score = 0.0
        if app not in out:
            out[app] = {}
        out[app][task_id] = score

    return out


def _compare_stuck_failed_vs_claude(
    domain_stuck_task_ids_failure: Dict[str, List[str]],
    claude_scores: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Compare tasks that are (stuck && failed) in the current run against Claude's own run.

    Returns a dict with per-domain counts and task lists.
    """
    report: Dict[str, Any] = {"by_domain": {}, "_overall": {}}
    total_stuck_failed = 0
    total_found_in_claude = 0
    total_missing_in_claude = 0
    total_success_in_claude = 0

    for domain, failed_ids in sorted(domain_stuck_task_ids_failure.items(), key=lambda x: x[0]):
        uniq_failed = sorted(set(str(x) for x in (failed_ids or [])))
        if not uniq_failed:
            continue
        total_stuck_failed += len(uniq_failed)

        c_domain = claude_scores.get(domain, {})
        successes: List[str] = []
        non_successes: List[str] = []
        missing: List[str] = []

        for tid in uniq_failed:
            if tid not in c_domain:
                missing.append(tid)
                continue
            total_found_in_claude += 1
            if _is_success_result(c_domain.get(tid, 0.0)):
                successes.append(tid)
            else:
                non_successes.append(tid)

        total_missing_in_claude += len(missing)
        total_success_in_claude += len(successes)

        report["by_domain"][domain] = {
            "stuck_failed_total": len(uniq_failed),
            "found_in_claude": len(uniq_failed) - len(missing),
            "missing_in_claude": len(missing),
            "claude_success_count": len(successes),
            "claude_success_task_ids": successes,
            "claude_non_success_task_ids": non_successes,
            "missing_task_ids": missing,
        }

    report["_overall"] = {
        "stuck_failed_total": int(total_stuck_failed),
        "found_in_claude": int(total_found_in_claude),
        "missing_in_claude": int(total_missing_in_claude),
        "claude_success_count": int(total_success_in_claude),
    }
    return report


def _stuck_stats_for_task_dir(task_dir: str, fallback_score: float) -> Tuple[bool, bool]:
    """
    Returns:
      - has_nonempty_stuck_detections
      - is_success (result == 1 or 1.0)
    """
    summary = _read_json_file(os.path.join(task_dir, "hybrid_summary.json"))
    if not summary:
        return False, _is_success_result(fallback_score)
    stuck = summary.get("stuck_detections", [])
    has_stuck = isinstance(stuck, list) and len(stuck) > 0
    result_val = summary.get("result", fallback_score)
    return has_stuck, _is_success_result(result_val)


def _steps_after_stuck_detection_for_task_dir(task_dir: str) -> Optional[int]:
    """
    If stuck detection happened, return the number of steps taken *after* the first stuck detection.

    Uses:
      - hybrid_summary.json["total_steps"]
      - hybrid_summary.json["stuck_detections"][*]["step"]

    Returns None if no stuck detection is present or required fields are missing.
    """
    summary = _read_json_file(os.path.join(task_dir, "hybrid_summary.json"))
    if not summary:
        return None

    stuck = summary.get("stuck_detections", [])
    if not isinstance(stuck, list) or len(stuck) == 0:
        return None

    total_steps = summary.get("total_steps", None)
    if not isinstance(total_steps, int):
        try:
            total_steps = int(total_steps)
        except Exception:
            return None

    stuck_steps: List[int] = []
    for d in stuck:
        if not isinstance(d, dict):
            continue
        s = d.get("step", None)
        if isinstance(s, int):
            stuck_steps.append(s)
            continue
        try:
            stuck_steps.append(int(s))
        except Exception:
            continue

    if not stuck_steps:
        return None

    first_stuck_step = min(stuck_steps)
    # "steps after stuck detection" excludes the detection step itself.
    return max(0, total_steps - first_stuck_step)


def _collect_milestone_switch_stats(target_dir: str) -> Dict[str, Any]:
    """
    Walk through task directories and collect milestone switch statistics.

    A 'milestone switch' occurs when switch_reason == 'milestone_failed' in hybrid_summary.json,
    meaning the agent switched to Claude because a milestone was detected (and judged as failed).

    Also collects stuck switch stats (switch_reason == 'stuck') for consistency.

    Returns a dict with:
      - milestone_total: total milestone switches
      - milestone_success: successful tasks among milestone switches
      - stuck_total: total stuck switches
      - stuck_success: successful tasks among stuck switches
      - by_domain: {domain: {milestone_total, milestone_success, milestone_task_ids, ...,
                              stuck_total, stuck_success, stuck_task_ids, ...}}
    """
    stats: Dict[str, Any] = {
        "milestone_total": 0,
        "milestone_success": 0,
        "stuck_total": 0,
        "stuck_success": 0,
        "by_domain": {},
    }
    if not target_dir or not os.path.exists(target_dir):
        return stats

    def _ensure_domain(domain: str) -> None:
        if domain not in stats["by_domain"]:
            stats["by_domain"][domain] = {
                "milestone_total": 0,
                "milestone_success": 0,
                "milestone_task_ids": [],
                "milestone_success_ids": [],
                "milestone_failure_ids": [],
                "stuck_total": 0,
                "stuck_success": 0,
                "stuck_task_ids": [],
                "stuck_success_ids": [],
                "stuck_failure_ids": [],
            }

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if not os.path.isdir(domain_path):
            continue
        for task_id in os.listdir(domain_path):
            task_path = os.path.join(domain_path, task_id)
            if not os.path.isdir(task_path):
                continue
            summary = _read_json_file(os.path.join(task_path, "hybrid_summary.json"))
            if not summary:
                continue
            switch_reason = summary.get("switch_reason", None)
            if switch_reason not in ("stuck", "milestone_failed"):
                continue

            _ensure_domain(domain)
            d = stats["by_domain"][domain]
            result_val = summary.get("result", 0.0)
            is_success = _is_success_result(result_val)

            if switch_reason == "milestone_failed":
                d["milestone_total"] += 1
                d["milestone_task_ids"].append(task_id)
                stats["milestone_total"] += 1
                if is_success:
                    d["milestone_success"] += 1
                    d["milestone_success_ids"].append(task_id)
                    stats["milestone_success"] += 1
                else:
                    d["milestone_failure_ids"].append(task_id)
            elif switch_reason == "stuck":
                d["stuck_total"] += 1
                d["stuck_task_ids"].append(task_id)
                stats["stuck_total"] += 1
                if is_success:
                    d["stuck_success"] += 1
                    d["stuck_success_ids"].append(task_id)
                    stats["stuck_success"] += 1
                else:
                    d["stuck_failure_ids"].append(task_id)

    return stats


def _collect_old_layout(
    action_space: str, use_model: str, observation_type: str, result_dir: str
) -> Tuple[
    Optional[str],
    Dict[str, List[float]],
    Dict[str, Dict[str, float]],
    List[float],
    int,
    int,
    List[int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, List[int]],
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, int],
    int,
]:
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        return None, {}, {}, [], 0, 0, [], {}, {}, {}, {}, {}, {}, {}, 0

    all_result = []
    domain_result = {}
    all_result_for_analysis = {}
    stuck_nonempty = 0
    stuck_nonempty_success = 0
    steps_after_stuck: List[int] = []
    domain_stuck_nonempty: Dict[str, int] = {}
    domain_stuck_nonempty_success: Dict[str, int] = {}
    domain_steps_after_stuck: Dict[str, List[int]] = {}
    domain_stuck_task_ids: Dict[str, List[str]] = {}
    domain_stuck_task_ids_success: Dict[str, List[str]] = {}
    domain_stuck_task_ids_failure: Dict[str, List[str]] = {}
    domain_failed_not_stuck: Dict[str, int] = {}
    failed_not_stuck_total = 0

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        # empty all files under example_id
                        if domain not in domain_result:
                            domain_result[domain] = []
                        if domain not in domain_stuck_nonempty:
                            domain_stuck_nonempty[domain] = 0
                        if domain not in domain_stuck_nonempty_success:
                            domain_stuck_nonempty_success[domain] = 0
                        if domain not in domain_steps_after_stuck:
                            domain_steps_after_stuck[domain] = []
                        if domain not in domain_stuck_task_ids:
                            domain_stuck_task_ids[domain] = []
                        if domain not in domain_stuck_task_ids_success:
                            domain_stuck_task_ids_success[domain] = []
                        if domain not in domain_stuck_task_ids_failure:
                            domain_stuck_task_ids_failure[domain] = []
                        if domain not in domain_failed_not_stuck:
                            domain_failed_not_stuck[domain] = 0
                        result = open(os.path.join(example_path, "result.txt"), "r").read()
                        domain_result[domain].append(_parse_score(result))

                        if domain not in all_result_for_analysis:
                            all_result_for_analysis[domain] = {}
                        all_result_for_analysis[domain][example_id] = domain_result[domain][-1]

                        try:
                            result = open(os.path.join(example_path, "result.txt"), "r").read()
                            score = _parse_score(result)
                            all_result.append(score)
                            has_stuck, is_success = _stuck_stats_for_task_dir(example_path, fallback_score=score)
                            if (not is_success) and (not has_stuck):
                                domain_failed_not_stuck[domain] += 1
                                failed_not_stuck_total += 1
                            if has_stuck:
                                stuck_nonempty += 1
                                domain_stuck_nonempty[domain] += 1
                                domain_stuck_task_ids[domain].append(str(example_id))
                                if is_success:
                                    stuck_nonempty_success += 1
                                    domain_stuck_nonempty_success[domain] += 1
                                    domain_stuck_task_ids_success[domain].append(str(example_id))
                                else:
                                    domain_stuck_task_ids_failure[domain].append(str(example_id))
                                steps_after = _steps_after_stuck_detection_for_task_dir(example_path)
                                if steps_after is not None:
                                    steps_after_stuck.append(steps_after)
                                    domain_steps_after_stuck[domain].append(steps_after)
                        except:
                            all_result.append(0.0)

    return (
        target_dir,
        domain_result,
        all_result_for_analysis,
        all_result,
        stuck_nonempty,
        stuck_nonempty_success,
        steps_after_stuck,
        domain_stuck_nonempty,
        domain_stuck_nonempty_success,
        domain_steps_after_stuck,
        domain_stuck_task_ids,
        domain_stuck_task_ids_success,
        domain_stuck_task_ids_failure,
        domain_failed_not_stuck,
        failed_not_stuck_total,
    )


def _collect_evocua_layout(
    action_space: str,
    observation_type: str,
    result_dir: str,
    suite: Optional[str] = None,
) -> Tuple[
    Optional[str],
    Dict[str, List[float]],
    Dict[str, Dict[str, float]],
    List[float],
    Optional[str],
    int,
    int,
    List[int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, List[int]],
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, int],
    int,
]:
    """
    Expected layout (observed in this repo):
      evocua_results/<action_space>/<observation_type>/<suite>/<application>/<task_id>/result.txt
    Example:
      evocua_results/pyautogui/screenshot/EvoCUA-S2/chrome/<uuid>/result.txt
    """
    base_dir = os.path.join(result_dir, action_space, observation_type)
    if not os.path.exists(base_dir):
        return None, {}, {}, [], None, 0, 0, [], {}, {}, {}, {}, {}, {}, {}, 0

    suites = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not suites:
        return None, {}, {}, [], None, 0, 0, [], {}, {}, {}, {}, {}, {}, {}, 0

    chosen_suite = suite
    if chosen_suite is None:
        # prefer common suite name if present; else deterministic choice
        chosen_suite = "EvoCUA-S2" if "EvoCUA-S2" in suites else sorted(suites)[0]

    target_dir = os.path.join(base_dir, chosen_suite)
    if not os.path.exists(target_dir):
        return None, {}, {}, [], chosen_suite, 0, 0, [], {}, {}, {}, {}, {}, {}, {}, 0

    all_result: List[float] = []
    domain_result: Dict[str, List[float]] = {}
    all_result_for_analysis: Dict[str, Dict[str, float]] = {}
    stuck_nonempty = 0
    stuck_nonempty_success = 0
    steps_after_stuck: List[int] = []
    domain_stuck_nonempty: Dict[str, int] = {}
    domain_stuck_nonempty_success: Dict[str, int] = {}
    domain_steps_after_stuck: Dict[str, List[int]] = {}
    domain_stuck_task_ids: Dict[str, List[str]] = {}
    domain_stuck_task_ids_success: Dict[str, List[str]] = {}
    domain_stuck_task_ids_failure: Dict[str, List[str]] = {}
    domain_failed_not_stuck: Dict[str, int] = {}
    failed_not_stuck_total = 0

    for application in os.listdir(target_dir):
        app_path = os.path.join(target_dir, application)
        if not os.path.isdir(app_path):
            continue
        for task_id in os.listdir(app_path):
            task_path = os.path.join(app_path, task_id)
            if not os.path.isdir(task_path):
                continue
            result_path = os.path.join(task_path, "result.txt")
            if not os.path.exists(result_path):
                continue

            if application not in domain_result:
                domain_result[application] = []
            if application not in all_result_for_analysis:
                all_result_for_analysis[application] = {}
            if application not in domain_stuck_nonempty:
                domain_stuck_nonempty[application] = 0
            if application not in domain_stuck_nonempty_success:
                domain_stuck_nonempty_success[application] = 0
            if application not in domain_steps_after_stuck:
                domain_steps_after_stuck[application] = []
            if application not in domain_stuck_task_ids:
                domain_stuck_task_ids[application] = []
            if application not in domain_stuck_task_ids_success:
                domain_stuck_task_ids_success[application] = []
            if application not in domain_stuck_task_ids_failure:
                domain_stuck_task_ids_failure[application] = []
            if application not in domain_failed_not_stuck:
                domain_failed_not_stuck[application] = 0

            raw = open(result_path, "r").read()
            score = _parse_score(raw)
            domain_result[application].append(score)
            all_result_for_analysis[application][task_id] = score
            all_result.append(score)
            has_stuck, is_success = _stuck_stats_for_task_dir(task_path, fallback_score=score)
            if (not is_success) and (not has_stuck):
                domain_failed_not_stuck[application] += 1
                failed_not_stuck_total += 1
            if has_stuck:
                stuck_nonempty += 1
                domain_stuck_nonempty[application] += 1
                domain_stuck_task_ids[application].append(str(task_id))
                if is_success:
                    stuck_nonempty_success += 1
                    domain_stuck_nonempty_success[application] += 1
                    domain_stuck_task_ids_success[application].append(str(task_id))
                else:
                    domain_stuck_task_ids_failure[application].append(str(task_id))
                steps_after = _steps_after_stuck_detection_for_task_dir(task_path)
                if steps_after is not None:
                    steps_after_stuck.append(steps_after)
                    domain_steps_after_stuck[application].append(steps_after)

    return (
        target_dir,
        domain_result,
        all_result_for_analysis,
        all_result,
        chosen_suite,
        stuck_nonempty,
        stuck_nonempty_success,
        steps_after_stuck,
        domain_stuck_nonempty,
        domain_stuck_nonempty_success,
        domain_steps_after_stuck,
        domain_stuck_task_ids,
        domain_stuck_task_ids_success,
        domain_stuck_task_ids_failure,
        domain_failed_not_stuck,
        failed_not_stuck_total,
    )


def _collect_suite_dir(
    suite_dir: str,
) -> Tuple[
    Optional[str],
    Dict[str, List[float]],
    Dict[str, Dict[str, float]],
    List[float],
    int,
    int,
    List[int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, List[int]],
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, int],
    int,
]:
    """
    Collect results from a *specific* run directory.

    Expected layout:
      <suite_dir>/<application>/<task_id>/result.txt

    Example:
      hybrid_results/pyautogui/screenshot/Hybrid-EvoCUA-S2-claude-sonnet-4-5-20250929/chrome/<uuid>/result.txt
    """
    if not os.path.exists(suite_dir) or not os.path.isdir(suite_dir):
        return None, {}, {}, [], 0, 0, [], {}, {}, {}, {}, {}, {}, {}, 0

    all_result: List[float] = []
    domain_result: Dict[str, List[float]] = {}
    all_result_for_analysis: Dict[str, Dict[str, float]] = {}
    stuck_nonempty = 0
    stuck_nonempty_success = 0
    steps_after_stuck: List[int] = []
    domain_stuck_nonempty: Dict[str, int] = {}
    domain_stuck_nonempty_success: Dict[str, int] = {}
    domain_steps_after_stuck: Dict[str, List[int]] = {}
    domain_stuck_task_ids: Dict[str, List[str]] = {}
    domain_stuck_task_ids_success: Dict[str, List[str]] = {}
    domain_stuck_task_ids_failure: Dict[str, List[str]] = {}
    domain_failed_not_stuck: Dict[str, int] = {}
    failed_not_stuck_total = 0

    for application in os.listdir(suite_dir):
        app_path = os.path.join(suite_dir, application)
        if not os.path.isdir(app_path):
            continue

        for task_id in os.listdir(app_path):
            task_path = os.path.join(app_path, task_id)
            if not os.path.isdir(task_path):
                continue

            result_path = os.path.join(task_path, "result.txt")
            if not os.path.exists(result_path):
                continue

            if application not in domain_result:
                domain_result[application] = []
            if application not in all_result_for_analysis:
                all_result_for_analysis[application] = {}
            if application not in domain_stuck_nonempty:
                domain_stuck_nonempty[application] = 0
            if application not in domain_stuck_nonempty_success:
                domain_stuck_nonempty_success[application] = 0
            if application not in domain_steps_after_stuck:
                domain_steps_after_stuck[application] = []
            if application not in domain_stuck_task_ids:
                domain_stuck_task_ids[application] = []
            if application not in domain_stuck_task_ids_success:
                domain_stuck_task_ids_success[application] = []
            if application not in domain_stuck_task_ids_failure:
                domain_stuck_task_ids_failure[application] = []
            if application not in domain_failed_not_stuck:
                domain_failed_not_stuck[application] = 0

            raw = open(result_path, "r").read()
            score = _parse_score(raw)
            domain_result[application].append(score)
            all_result_for_analysis[application][task_id] = score
            all_result.append(score)
            has_stuck, is_success = _stuck_stats_for_task_dir(task_path, fallback_score=score)
            if (not is_success) and (not has_stuck):
                domain_failed_not_stuck[application] += 1
                failed_not_stuck_total += 1
            if has_stuck:
                stuck_nonempty += 1
                domain_stuck_nonempty[application] += 1
                domain_stuck_task_ids[application].append(str(task_id))
                if is_success:
                    stuck_nonempty_success += 1
                    domain_stuck_nonempty_success[application] += 1
                    domain_stuck_task_ids_success[application].append(str(task_id))
                else:
                    domain_stuck_task_ids_failure[application].append(str(task_id))
                steps_after = _steps_after_stuck_detection_for_task_dir(task_path)
                if steps_after is not None:
                    steps_after_stuck.append(steps_after)
                    domain_steps_after_stuck[application].append(steps_after)

    return (
        suite_dir,
        domain_result,
        all_result_for_analysis,
        all_result,
        stuck_nonempty,
        stuck_nonempty_success,
        steps_after_stuck,
        domain_stuck_nonempty,
        domain_stuck_nonempty_success,
        domain_steps_after_stuck,
        domain_stuck_task_ids,
        domain_stuck_task_ids_success,
        domain_stuck_task_ids_failure,
        domain_failed_not_stuck,
        failed_not_stuck_total,
    )


def get_result(action_space, use_model, observation_type, result_dir, suite: Optional[str] = None):
    # 1) try old layout first (backwards compatible)
    (
        target_dir,
        domain_result,
        all_result_for_analysis,
        all_result,
        stuck_nonempty,
        stuck_nonempty_success,
        steps_after_stuck,
        domain_stuck_nonempty,
        domain_stuck_nonempty_success,
        domain_steps_after_stuck,
        domain_stuck_task_ids,
        domain_stuck_task_ids_success,
        domain_stuck_task_ids_failure,
        domain_failed_not_stuck,
        failed_not_stuck_total,
    ) = _collect_old_layout(
        action_space=action_space,
        use_model=use_model,
        observation_type=observation_type,
        result_dir=result_dir,
    )

    # 2) otherwise try evocua layout
    chosen_suite = None
    if target_dir is None:
        (
            target_dir,
            domain_result,
            all_result_for_analysis,
            all_result,
            chosen_suite,
            stuck_nonempty,
            stuck_nonempty_success,
            steps_after_stuck,
            domain_stuck_nonempty,
            domain_stuck_nonempty_success,
            domain_steps_after_stuck,
            domain_stuck_task_ids,
            domain_stuck_task_ids_success,
            domain_stuck_task_ids_failure,
            domain_failed_not_stuck,
            failed_not_stuck_total,
        ) = _collect_evocua_layout(
            action_space=action_space,
            observation_type=observation_type,
            result_dir=result_dir,
            suite=suite,
        )

    if target_dir is None:
        print("New experiment, no result yet.")
        return None

    # Collect milestone switch stats (switch_reason based)
    switch_stats = _collect_milestone_switch_stats(target_dir)

    for domain in domain_result:
        print(
            "Domain:",
            domain,
            "Runned:",
            len(domain_result[domain]),
            "Success Rate:",
            sum(domain_result[domain]) / len(domain_result[domain]) * 100,
            "%",
        )
        d_stuck = domain_stuck_nonempty.get(domain, 0)
        d_stuck_success = domain_stuck_nonempty_success.get(domain, 0)
        print(
            "  Stuck detections (switched to Claude):",
            d_stuck,
            "| Among them success (result==1):",
            d_stuck_success,
        )
        d_steps = domain_steps_after_stuck.get(domain, [])
        if d_steps:
            avg_steps_after = sum(d_steps) / len(d_steps)
            print(
                "  Avg steps after stuck detection:",
                avg_steps_after,
                f"(n={len(d_steps)})",
            )
        # Milestone switch stats for this domain
        d_ms = switch_stats["by_domain"].get(domain, {})
        d_ms_total = d_ms.get("milestone_total", 0)
        d_ms_success = d_ms.get("milestone_success", 0)
        print(
            "  Milestone detections (switched to Claude):",
            d_ms_total,
            "| Among them success (result==1):",
            d_ms_success,
        )

    print(">>>>>>>>>>>>>")
    _print_group_rate(
        "Office",
        domain_result,
        ["libreoffice_calc", "libreoffice_impress", "libreoffice_writer"],
    )
    _print_group_rate("Daily", domain_result, ["vlc", "thunderbird", "chrome"])
    _print_group_rate("Professional", domain_result, ["gimp", "vs_code"])

    with open(os.path.join(target_dir, "all_result.json"), "w") as f:
        json.dump(all_result_for_analysis, f, indent=2, sort_keys=True)

    # also write out tasks where stuck/milestone was detected (by domain)
    stuck_out: Dict[str, Any] = {}
    for domain in domain_stuck_task_ids.keys():
        success_ids = domain_stuck_task_ids_success.get(domain, [])
        failure_ids = domain_stuck_task_ids_failure.get(domain, [])
        d_ms = switch_stats["by_domain"].get(domain, {})
        stuck_out[domain] = {
            "stuck_switch": {
                "success": sorted(set(success_ids)),
                "failure": sorted(set(failure_ids)),
            },
            "milestone_switch": {
                "success": sorted(set(d_ms.get("milestone_success_ids", []))),
                "failure": sorted(set(d_ms.get("milestone_failure_ids", []))),
            },
            "failed_not_stuck_count": int(domain_failed_not_stuck.get(domain, 0)),
        }
    # Also include domains that only had milestone switches
    for domain in switch_stats["by_domain"]:
        if domain not in stuck_out:
            d_ms = switch_stats["by_domain"][domain]
            stuck_out[domain] = {
                "stuck_switch": {"success": [], "failure": []},
                "milestone_switch": {
                    "success": sorted(set(d_ms.get("milestone_success_ids", []))),
                    "failure": sorted(set(d_ms.get("milestone_failure_ids", []))),
                },
                "failed_not_stuck_count": int(domain_failed_not_stuck.get(domain, 0)),
            }
    stuck_out["_overall"] = {
        "failed_not_stuck_count": int(failed_not_stuck_total),
        "stuck_switch_total": int(switch_stats["stuck_total"]),
        "stuck_switch_success": int(switch_stats["stuck_success"]),
        "milestone_switch_total": int(switch_stats["milestone_total"]),
        "milestone_switch_success": int(switch_stats["milestone_success"]),
    }
    with open(os.path.join(target_dir, "stuck_tasks_by_domain.json"), "w") as f:
        json.dump(stuck_out, f, indent=2, sort_keys=True)

    # Optional: compare stuck+failed tasks vs Claude's own run.
    # Enabled when CLAUDE_RESULTS_ROOT env var is set, or a default path exists.
    claude_root = os.environ.get("CLAUDE_RESULTS_ROOT", None) or _default_claude_results_root()
    if claude_root and os.path.exists(claude_root):
        claude_scores = _load_claude_results_map(claude_root)
        compare_report = _compare_stuck_failed_vs_claude(domain_stuck_task_ids_failure, claude_scores)
        overall = compare_report.get("_overall", {})
        print(">>>>>>>>>>>>>")
        print("Claude comparison for tasks that are stuck+failed in this run:")
        print(
            "  stuck+failed:",
            overall.get("stuck_failed_total", 0),
            "| found in Claude:",
            overall.get("found_in_claude", 0),
            "| missing in Claude:",
            overall.get("missing_in_claude", 0),
            "| Claude success (score==1):",
            overall.get("claude_success_count", 0),
        )
        # Write alongside other analysis artifacts.
        with open(os.path.join(target_dir, "stuck_failed_vs_claude.json"), "w") as f:
            json.dump(
                {
                    "claude_results_root": claude_root,
                    **compare_report,
                },
                f,
                indent=2,
                sort_keys=True,
            )

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        if chosen_suite is not None:
            print("Suite:", chosen_suite)
        print("Runned:", len(all_result), "Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
        print(
            "Stuck detections (switched to Claude):",
            stuck_nonempty,
            "| Among them success (result==1):",
            stuck_nonempty_success,
        )
        if steps_after_stuck:
            avg_steps_after = sum(steps_after_stuck) / len(steps_after_stuck)
            print(
                "Avg steps after stuck detection:",
                avg_steps_after,
                f"(n={len(steps_after_stuck)})",
            )
        print(
            "Milestone detections (switched to Claude):",
            switch_stats["milestone_total"],
            "| Among them success (result==1):",
            switch_stats["milestone_success"],
        )
        return all_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        default=None,
        help=(
            "Path to a specific run folder that directly contains "
            "<application>/<task_id>/result.txt (e.g. "
            "hybrid_results/pyautogui/screenshot/Hybrid-... )."
        ),
    )
    parser.add_argument(
        "--result_dir",
        default=None,
        help="Path to results directory. If not specified, checks multiple default directories.",
    )
    parser.add_argument("--action_space", default="pyautogui")
    parser.add_argument("--observation_type", default="screenshot")
    # kept for backward compatibility with the old layout
    parser.add_argument("--use_model", default="gpt-4o")
    # evocua layout uses <suite> directory (e.g. EvoCUA-S2)
    parser.add_argument("--suite", default=None)
    parser.add_argument(
        "--claude_results_root",
        default=None,
        help=(
            "Path to Claude's own run results root (e.g. "
            "BERT_Training/results_claude-sonnet-4-5-20250929_50steps). "
            "If provided, will compare stuck+failed tasks vs Claude's success."
        ),
    )
    args = parser.parse_args()

    # Allow CLI to override Claude results root without changing code.
    if args.claude_results_root:
        os.environ["CLAUDE_RESULTS_ROOT"] = args.claude_results_root

    # If folder specified, treat it as the suite/run directory directly.
    if args.folder is not None:
        base_dir = os.path.dirname(__file__)
        folder_path = args.folder
        if not os.path.isabs(folder_path):
            folder_path = os.path.join(base_dir, folder_path)

        (
            target_dir,
            domain_result,
            all_result_for_analysis,
            all_result,
            stuck_nonempty,
            stuck_nonempty_success,
            steps_after_stuck,
            domain_stuck_nonempty,
            domain_stuck_nonempty_success,
            domain_steps_after_stuck,
            domain_stuck_task_ids,
            domain_stuck_task_ids_success,
            domain_stuck_task_ids_failure,
            domain_failed_not_stuck,
            failed_not_stuck_total,
        ) = _collect_suite_dir(folder_path)
        if target_dir is None:
            print("New experiment, no result yet.")
        else:
            # Collect milestone switch stats
            switch_stats = _collect_milestone_switch_stats(target_dir)

            for domain in domain_result:
                print(
                    "Domain:",
                    domain,
                    "Runned:",
                    len(domain_result[domain]),
                    "Success Rate:",
                    sum(domain_result[domain]) / len(domain_result[domain]) * 100,
                    "%",
                )
                d_stuck = domain_stuck_nonempty.get(domain, 0)
                d_stuck_success = domain_stuck_nonempty_success.get(domain, 0)
                print(
                    "  Stuck detections (switched to Claude):",
                    d_stuck,
                    "| Among them success (result==1):",
                    d_stuck_success,
                )
                d_steps = domain_steps_after_stuck.get(domain, [])
                if d_steps:
                    avg_steps_after = sum(d_steps) / len(d_steps)
                    print(
                        "  Avg steps after stuck detection:",
                        avg_steps_after,
                        f"(n={len(d_steps)})",
                    )
                # Milestone switch stats for this domain
                d_ms = switch_stats["by_domain"].get(domain, {})
                d_ms_total = d_ms.get("milestone_total", 0)
                d_ms_success = d_ms.get("milestone_success", 0)
                print(
                    "  Milestone detections (switched to Claude):",
                    d_ms_total,
                    "| Among them success (result==1):",
                    d_ms_success,
                )

            print(">>>>>>>>>>>>>")
            _print_group_rate(
                "Office",
                domain_result,
                ["libreoffice_calc", "libreoffice_impress", "libreoffice_writer"],
            )
            _print_group_rate("Daily", domain_result, ["vlc", "thunderbird", "chrome"])
            _print_group_rate("Professional", domain_result, ["gimp", "vs_code"])

            with open(os.path.join(target_dir, "all_result.json"), "w") as f:
                json.dump(all_result_for_analysis, f, indent=2, sort_keys=True)

            stuck_out: Dict[str, Any] = {}
            for domain in domain_stuck_task_ids.keys():
                success_ids = domain_stuck_task_ids_success.get(domain, [])
                failure_ids = domain_stuck_task_ids_failure.get(domain, [])
                d_ms = switch_stats["by_domain"].get(domain, {})
                stuck_out[domain] = {
                    "stuck_switch": {
                        "success": sorted(set(success_ids)),
                        "failure": sorted(set(failure_ids)),
                    },
                    "milestone_switch": {
                        "success": sorted(set(d_ms.get("milestone_success_ids", []))),
                        "failure": sorted(set(d_ms.get("milestone_failure_ids", []))),
                    },
                    "failed_not_stuck_count": int(domain_failed_not_stuck.get(domain, 0)),
                }
            # Also include domains that only had milestone switches
            for domain in switch_stats["by_domain"]:
                if domain not in stuck_out:
                    d_ms = switch_stats["by_domain"][domain]
                    stuck_out[domain] = {
                        "stuck_switch": {"success": [], "failure": []},
                        "milestone_switch": {
                            "success": sorted(set(d_ms.get("milestone_success_ids", []))),
                            "failure": sorted(set(d_ms.get("milestone_failure_ids", []))),
                        },
                        "failed_not_stuck_count": int(domain_failed_not_stuck.get(domain, 0)),
                    }
            stuck_out["_overall"] = {
                "failed_not_stuck_count": int(failed_not_stuck_total),
                "stuck_switch_total": int(switch_stats["stuck_total"]),
                "stuck_switch_success": int(switch_stats["stuck_success"]),
                "milestone_switch_total": int(switch_stats["milestone_total"]),
                "milestone_switch_success": int(switch_stats["milestone_success"]),
            }
            with open(os.path.join(target_dir, "stuck_tasks_by_domain.json"), "w") as f:
                json.dump(stuck_out, f, indent=2, sort_keys=True)

            # Optional: compare stuck+failed tasks vs Claude's own run (same logic as get_result()).
            claude_root = os.environ.get("CLAUDE_RESULTS_ROOT", None) or _default_claude_results_root()
            if claude_root and os.path.exists(claude_root):
                claude_scores = _load_claude_results_map(claude_root)
                compare_report = _compare_stuck_failed_vs_claude(domain_stuck_task_ids_failure, claude_scores)
                overall = compare_report.get("_overall", {})
                print(">>>>>>>>>>>>>")
                print("Claude comparison for tasks that are stuck+failed in this run:")
                print(
                    "  stuck+failed:",
                    overall.get("stuck_failed_total", 0),
                    "| found in Claude:",
                    overall.get("found_in_claude", 0),
                    "| missing in Claude:",
                    overall.get("missing_in_claude", 0),
                    "| Claude success (score==1):",
                    overall.get("claude_success_count", 0),
                )
                with open(os.path.join(target_dir, "stuck_failed_vs_claude.json"), "w") as f:
                    json.dump(
                        {
                            "claude_results_root": claude_root,
                            **compare_report,
                        },
                        f,
                        indent=2,
                        sort_keys=True,
                    )

            if not all_result:
                print("New experiment, no result yet.")
            else:
                print("Suite:", os.path.basename(os.path.normpath(target_dir)))
                print(
                    "Runned:",
                    len(all_result),
                    "Current Success Rate:",
                    sum(all_result) / len(all_result) * 100,
                    "%",
                )
                print(
                    "Stuck detections (switched to Claude):",
                    stuck_nonempty,
                    "| Among them success (result==1):",
                    stuck_nonempty_success,
                )
                if steps_after_stuck:
                    avg_steps_after = sum(steps_after_stuck) / len(steps_after_stuck)
                    print(
                        "Avg steps after stuck detection:",
                        avg_steps_after,
                        f"(n={len(steps_after_stuck)})",
                    )
                print(
                    "Milestone detections (switched to Claude):",
                    switch_stats["milestone_total"],
                    "| Among them success (result==1):",
                    switch_stats["milestone_success"],
                )
        raise SystemExit(0)

    # If no result_dir specified, check multiple directories
    if args.result_dir is None:
        base_dir = os.path.dirname(__file__)
        result_dirs = [
            ("evocua_results", "EvoCUA-S2"),  # (dir_name, use_model/suite)
            ("qwen3_8b_thinking_results", "Qwen/Qwen3-VL-8B-Thinking"),
            ("results", args.use_model),  # generic results directory
        ]
        
        found_any = False
        for dir_name, model_or_suite in result_dirs:
            full_path = os.path.join(base_dir, dir_name)
            if os.path.exists(full_path):
                print("\n" + "=" * 80)
                print(f"Results from: {dir_name}")
                print("=" * 80)
                
                # Try with suite first (evocua layout), then model (old layout)
                result = get_result(
                    args.action_space,
                    model_or_suite,
                    args.observation_type,
                    full_path,
                    suite=model_or_suite if dir_name == "evocua_results" else None,
                )
                if result is not None:
                    found_any = True
        
        if not found_any:
            print("No results found in any default directories.")
    else:
        # Use specified directory
        get_result(
            args.action_space,
            args.use_model,
            args.observation_type,
            args.result_dir,
            suite=args.suite,
        )
