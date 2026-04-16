# Step-level Optimization for Efficient Computer-use Agents

Official code for **"Step-level Optimization for Efficient Computer-use Agents"** (Wei, Ni, Zhao, Gan, Cohan — Yale, UNC, Zhejiang). An event-driven, step-level cascade that runs a small GUI policy by default and escalates to a stronger model only when lightweight learned monitors detect elevated risk.

## Overview

Computer-use agents are expensive because they call a large multimodal model at every step, yet GUI trajectories are heterogeneous: most steps are routine, and failures concentrate at a small number of high-risk moments. We identify two recurring failure modes — **progress stalls** (looping, repeated equivalent actions) and **silent semantic drift** (locally plausible actions that have already deviated from user intent) — and route compute accordingly.

Two lightweight ModernBERT monitors drive the controller:

- **Stuck Monitor** — reads a short window of recent rationale/action pairs and fires when the agent stops making progress. Triggers escalation to the large policy for recovery.
- **Milestone Monitor** — task-conditioned; predicts when the current step completes a semantically meaningful checkpoint. Triggers a sparse, localized verification call by a stronger model; if verification fails, control hands off to the large policy.

The framework is plug-and-play: it layers on top of existing small GUI agents without modifying the base model, and is trained from logged trajectories with simple binary labels (stuck vs. not, milestone vs. not) supervised by a stronger LLM.

## Results (from the paper)

Across **OSWorld** (desktop) and **WebArena-Verified** (web), cascading recovers most of the performance of always-large policies at substantially lower cost and latency — up to **74.6% inference cost reduction** and **45.8% latency reduction**.

Representative cascaded pairs:

| Benchmark | Cascade | Acc. | Cost/Task | Lat./Req. |
|---|---|---|---|---|
| OSWorld | EvoCUA-8B → Kimi K2.5 | 58.2% | $0.051 | 4.5s |
| OSWorld | Qwen3-VL-8B → Kimi K2.5 | 59.3% | $0.078 | 6.5s |
| WebArena | gpt-oss-20b → gpt-5.2 | 57.8% | $0.211 | 12.2s |
| WebArena | AgentTrek-32B → gpt-5.2 | 58.8% | $0.208 | 13.4s |

Ablations show the two detectors are complementary (stuck targets local loops; milestone targets semantic drift), and event-driven escalation is both cheaper and more accurate than fixed-interval checking, especially on short-horizon web tasks. Learned detectors reach 91.5% F1 (stuck) and 62.0% F1 (milestone) on a held-out split.

## Repository layout

```
GUI_router/
├── ComputerRouter/   # Desktop cascade on OSWorld (EvoCUA / Qwen3-VL → Claude / Kimi)
├── WebRouter/       # Web cascade on WebArena-Verified (gpt-oss / AgentTrek → gpt-5-mini / gpt-5.2)
└── Step_level_Optimization_for_Efficient_Computer_use_Agents__2_.pdf
```

- **`ComputerRouter/`** — built on top of [OSWorld](https://github.com/xlang-ai/OSWorld). Contains single-model runners (`run_multienv_evocua.py`, `run_multienv_qwen3vl.py`, `run_multienv_claude.py`, `run_multienv_kimi_k25.py`), hybrid cascade runners (`run_multienv_hybrid*.py`), a periodic-verify baseline, ModernBERT detectors under `bert/`, and training/eval trajectory data under `evaluation_examples/`. See `ComputerRouter/README.md` for setup and commands.
- **`WebRouter/`** — built on top of [AgentLab](https://github.com/ServiceNow/AgentLab) and WebArena-Verified. Contains the cascade pipeline (`scripts/run_webarena_verified_cascade.sh`), vLLM serving scripts for open-weight small models, and the BERT detectors used at runtime. See `WebRouter/README.md` for setup.

## Getting started

For full setup, environment files, provider credentials, and run commands, see the per-module READMEs:

- Desktop cascade: [`ComputerRouter/README.md`](ComputerRouter/README.md)
- Web cascade: [`WebRouter/README.md`](WebRouter/README.md)


Correspondence: Jinbiao Wei (jinbiao.wei@yale.edu).
