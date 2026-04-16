"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Copy this script and modify at will, but don't push your changes to the
repository.
"""

import logging
from pathlib import Path
import os

from agentlab.agents.generic_agent import (
    AGENT_LLAMA3_70B,
    AGENT_LLAMA31_70B,
    RANDOM_SEARCH_AGENT,
    AGENT_4o,
    AGENT_4o_MINI,
    AGENT_o3_MINI,
    AGENT_37_SONNET,
    AGENT_CLAUDE_SONNET_35,
    AGENT_GPT5_MINI,

    AGENT_AZURE_5_MINI,
    AGENT_AZURE_5_2,
)
from agentlab.experiments.study import Study

logging.getLogger().setLevel(logging.INFO)

# choose your agent or provide a new agent
# agent_args = [AGENT_4o_MINI]
# agent_args = [AGENT_4o]
# agent_args = [AGENT_GPT5_MINI]

# agent_args = [AGENT_AZURE_5_MINI]
agent_args = [AGENT_AZURE_5_2]

# ## select the benchmark to run on
# benchmark = "miniwob_tiny_test"
# benchmark = "miniwob"
# benchmark = "workarena_l1"
# benchmark = "workarena_l2"
# benchmark = "workarena_l3"
benchmark_name = "webarena_verified"

# Set reproducibility_mode = True for reproducibility
# this will "ask" agents to be deterministic. Also, it will prevent you from launching if you have
# local changes. For your custom agents you need to implement set_reproducibility_mode
reproducibility_mode = False

# Set relaunch = True to relaunch an existing study, this will continue incomplete
# experiments and relaunch errored experiments
relaunch = False

## Number of parallel jobs
n_jobs = 4  # Make sure to use 1 job when debugging in VSCode
# n_jobs = -1  # to use all available cores

start=0
end=812
# Only run tasks whose sites are ALL in this list.
include_sites = ["shopping", "shopping_admin", "reddit", "gitlab"]


def print_study_info(study_dir, benchmark_name, include_sites, start, end):
    """Print task filtering summary and progress. Fast: no study/benchmark loading."""
    import json
    import re

    import pandas as pd

    # Load metadata CSV from installed package
    import browsergym.experiments

    pkg_dir = Path(browsergym.experiments.__file__).parent
    metadata_csv = pkg_dir / "benchmark" / "metadata" / f"{benchmark_name}.csv"
    if not metadata_csv.exists():
        print(f"  Metadata CSV not found: {metadata_csv}")
        return

    md = pd.read_csv(metadata_csv)
    all_task_names = set(md["task_name"])

    # Filter by include_sites and task_id range to get expected tasks
    mask = pd.Series(True, index=md.index)
    if include_sites:
        inc = set(include_sites)
        mask &= md["sites"].apply(lambda s: all(x in inc for x in s.split()))
    task_ids = set(range(start, end + 1))
    mask &= md["task_id"].isin(task_ids)

    expected_tasks = set(md[mask]["task_name"])
    skipped_tasks = all_task_names - expected_tasks
    expected_md = md[md["task_name"].isin(expected_tasks)]
    skipped_md = md[md["task_name"].isin(skipped_tasks)]

    print(f"\n{'='*60}")
    print(f"Benchmark: {benchmark_name}")
    print(f"Total tasks in benchmark: {len(all_task_names)}")
    print(f"Expected: {len(expected_tasks)}  |  Skipped: {len(skipped_tasks)}")
    print(f"Task ID range: {start}-{end}")
    if include_sites:
        print(f"Include sites: {include_sites}")
    print(f"\nExpected tasks by site:")
    for site, count in expected_md["sites"].value_counts().items():
        print(f"  {site}: {count}")
    if skipped_tasks:
        print(f"\nSkipped tasks by site:")
        for site, count in skipped_md["sites"].value_counts().items():
            print(f"  {site}: {count}")

    # --- Scan study dir for progress ---
    study_path = Path(study_dir)
    if not study_path.exists():
        print(f"\nStudy dir does not exist yet. Will create on first run.")
        print(f"{'='*60}\n")
        return

    exp_dirs = [
        d for d in study_path.iterdir()
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    ]

    task_status = {}  # task_name -> "done" | "error" | "incomplete"
    n_success = 0
    for d in exp_dirs:
        m = re.search(r"_on_(.+)_\d+$", d.name)
        if not m:
            continue
        tn = m.group(1)
        sf = d / "summary_info.json"
        if sf.exists():
            try:
                info = json.loads(sf.read_text())
                if info.get("err_msg") is not None:
                    task_status[tn] = "error"
                elif info.get("terminated") or info.get("truncated"):
                    task_status[tn] = "done"
                    if (info.get("cum_reward") or 0) > 0:
                        n_success += 1
                else:
                    task_status[tn] = "incomplete"
            except Exception:
                task_status[tn] = "incomplete"
        else:
            task_status[tn] = "incomplete"

    n_done = sum(1 for s in task_status.values() if s == "done")
    n_err = sum(1 for s in task_status.values() if s == "error")
    not_started = expected_tasks - set(task_status.keys())
    to_run = len(not_started) + sum(1 for s in task_status.values() if s != "done")

    print(f"\nProgress: {n_done}/{len(expected_tasks)} done, {n_err} errors, {len(not_started)} not started, {n_success} successes")
    print(f"Tasks to re-run/start: {to_run}")

    remaining = [(t, task_status.get(t, "not_started")) for t in sorted(
        [t for t, s in task_status.items() if s != "done"] + list(not_started)
    )]
    if 0 < len(remaining) <= 50:
        for t, s in remaining:
            print(f"  [{s}] {t}")
    elif len(remaining) > 50:
        print(f"  (showing first 20 of {len(remaining)})")
        for t, s in remaining[:20]:
            print(f"  [{s}] {t}")
        print(f"  ... and {len(remaining) - 20} more")
    print(f"{'='*60}\n")


if __name__ == "__main__":  # necessary for dask backend

    assert len(agent_args) == 1, "Only one agent is supported"

    import bgym

    benchmark = bgym.DEFAULT_BENCHMARKS[benchmark_name]()

    # Filter to only tasks where every site is in include_sites
    if include_sites:
        include_set = set(include_sites)
        md = benchmark.task_metadata
        allowed_tasks = set(
            md[md["sites"].apply(lambda s: all(x in include_set for x in s.split()))]["task_name"]
        )
        benchmark.env_args_list = [
            ea for ea in benchmark.env_args_list if ea.task_name in allowed_tasks
        ]
    task_ids = set(range(start, end+1))
    benchmark.env_args_list = [
        ea for ea in benchmark.env_args_list
        if int(ea.task_name.split(".")[-2]) in task_ids
    ]

    if reproducibility_mode:
        [a.set_reproducibility_mode() for a in agent_args]

    study_dir = Path(os.environ["AGENTLAB_EXP_ROOT"]) / "results" / f"{agent_args[0].agent_name}_{benchmark_name.replace('_', '-')}_{start}-{end}"

    print(f"Study directory: {study_dir}")

    print_study_info(study_dir, benchmark_name, include_sites, start, end)

    study = Study(agent_args, benchmark, logging_level_stdout=logging.WARNING,
        # ignore_dependencies=True
    )
    study.dir = study_dir

    if relaunch:
        assert study_dir.exists(), f"Study directory {study_dir} does not exist"

        # Save full task list before find_incomplete overwrites it
        full_list = {ea.env_args.task_name: ea for ea in study.exp_args_list}

        # Scan disk: completed→dummy, incomplete/errored→real
        study.find_incomplete(include_errors=True)

        # Drop tasks not in current benchmark (e.g. removed sites)
        dropped = [ea.env_args.task_name for ea in study.exp_args_list if ea.env_args.task_name not in full_list]
        if dropped:
            print(f"Dropping {len(dropped)} tasks not in current include_sites.")
            study.exp_args_list = [ea for ea in study.exp_args_list if ea.env_args.task_name in full_list]

        # Merge back tasks not yet on disk (newly added sites)
        on_disk = {ea.env_args.task_name for ea in study.exp_args_list}
        new_tasks = [ea for name, ea in full_list.items() if name not in on_disk]
        if new_tasks:
            print(f"Found {len(new_tasks)} new tasks not yet on disk, adding to study.")
            study.exp_args_list.extend(new_tasks)

        # Always rebuild dependency graph: disk-loaded tasks have stale
        # depends_on exp_ids from previous runs that may not match current list
        from agentlab.experiments.exp_utils import add_dependencies
        task_deps = benchmark.dependency_graph_over_tasks()
        if task_deps:
            add_dependencies(study.exp_args_list, task_deps)

    study.run(
        n_jobs=n_jobs,
        parallel_backend="ray",
        strict_reproducibility=reproducibility_mode,
        n_relaunch=1 if relaunch else 3,
    )

    if reproducibility_mode:
        study.append_to_journal(strict_reproducibility=True)
