"""
Figure: Quantitative analysis of GUI agent trajectory heterogeneity.

Reads trajectory_stats.json (produced by analyze_trajectory_heterogeneity.py)
and generates a publication-quality 2×2 figure:

  (a) Trajectory length distributions – success vs. failure (both models)
  (b) Action repetition rate – success vs. failure (both models)
  (c) Per-application success rates – comparison of both models
  (d) Step budget: fraction of all LLM calls in each category
      (success / stall / active-failing/drift)

Usage:
    python plot_trajectory_analysis.py
    # outputs trajectory_analysis.pdf (and .png)
"""

import json
import statistics
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
BASE   = Path("/gpfs/radev/project/cohan/jw3278/GUI_router/BERT_Training")
STATS  = BASE / "trajectory_stats.json"
OUTPDF = BASE / "trajectory_analysis.pdf"
OUTPNG = BASE / "trajectory_analysis.png"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linewidth":   0.6,
})

# Color palette (color-blind friendly)
C_EVOCUA_SUC = "#2166ac"   # blue
C_EVOCUA_FAI = "#92c5de"   # light blue
C_QWEN_SUC   = "#d6604d"   # red-orange
C_QWEN_FAI   = "#fddbc7"   # light peach
C_NEUTRAL    = "#555555"

MODEL_LABELS = {
    "evocua_8b":         "EvoCUA-8B",
    "qwen3_8b_thinking": "Qwen3-VL-8B",
}
APP_DISPLAY = {
    "chrome":              "Chrome",
    "gimp":                "GIMP",
    "libreoffice_calc":    "Calc",
    "libreoffice_impress": "Impress",
    "libreoffice_writer":  "Writer",
    "multi_apps":          "Multi",
    "os":                  "OS",
    "thunderbird":         "Thunderbird",
    "vlc":                 "VLC",
    "vs_code":             "VS Code",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(STATS) as f:
    data = json.load(f)

trajs = data["trajectories"]
app_stats = data["app_stats"]

def subset(model=None, success=None, failed_only=False):
    out = trajs
    if model:
        out = [t for t in out if t["model"] == model]
    if success is True:
        out = [t for t in out if t["success"]]
    if success is False:
        out = [t for t in out if not t["success"]]
    return out

def vals(ts, key):
    return [t[key] for t in ts if t.get(key) is not None]

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
ax_steps, ax_rep, ax_app, ax_budget = axes.flat

# ===========================================================================
# (a) Trajectory length: violin / box plot
# ===========================================================================
ax = ax_steps

groups = [
    ("EvoCUA-8B\nSuccess",  C_EVOCUA_SUC, vals(subset("evocua_8b", success=True),  "n_steps")),
    ("EvoCUA-8B\nFailed",   C_EVOCUA_FAI, vals(subset("evocua_8b", success=False), "n_steps")),
    ("Qwen3-VL-8B\nSuccess",C_QWEN_SUC,   vals(subset("qwen3_8b_thinking", success=True),  "n_steps")),
    ("Qwen3-VL-8B\nFailed", C_QWEN_FAI,   vals(subset("qwen3_8b_thinking", success=False), "n_steps")),
]

positions = [1, 2, 3.5, 4.5]
labels    = [g[0] for g in groups]
colors    = [g[1] for g in groups]
data_     = [g[2] for g in groups]

vp = ax.violinplot(data_, positions=positions, showmedians=True,
                   showextrema=False, widths=0.7)

for body, color in zip(vp["bodies"], colors):
    body.set_facecolor(color)
    body.set_edgecolor("none")
    body.set_alpha(0.85)
vp["cmedians"].set_color("#222222")
vp["cmedians"].set_linewidth(1.5)

# Overlay mean dots
for pos, d, color in zip(positions, data_, colors):
    ax.scatter([pos], [statistics.mean(d)], color="#111111",
               s=30, zorder=5, marker="D", linewidths=0)

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("Steps per trajectory")
ax.set_title("(a) Trajectory length", loc="left", fontweight="bold")
ax.set_ylim(bottom=0)

# Annotate means
for pos, d in zip(positions, data_):
    ax.text(pos, statistics.mean(d) + 1.5, f"{statistics.mean(d):.1f}",
            ha="center", va="bottom", fontsize=7, color="#111111")

# Vertical separator between models
ax.axvline(2.75, color="#aaaaaa", linewidth=0.8, linestyle="--")

# ===========================================================================
# (b) Action repetition rate
# ===========================================================================
ax = ax_rep

groups_rep = [
    ("EvoCUA-8B\nSuccess",  C_EVOCUA_SUC, vals(subset("evocua_8b", success=True),  "rep_rate")),
    ("EvoCUA-8B\nFailed",   C_EVOCUA_FAI, vals(subset("evocua_8b", success=False), "rep_rate")),
    ("Qwen3-VL-8B\nSuccess",C_QWEN_SUC,   vals(subset("qwen3_8b_thinking", success=True),  "rep_rate")),
    ("Qwen3-VL-8B\nFailed", C_QWEN_FAI,   vals(subset("qwen3_8b_thinking", success=False), "rep_rate")),
]
data_rep = [g[2] for g in groups_rep]

vp2 = ax.violinplot(data_rep, positions=positions, showmedians=True,
                    showextrema=False, widths=0.7)
for body, color in zip(vp2["bodies"], colors):
    body.set_facecolor(color)
    body.set_edgecolor("none")
    body.set_alpha(0.85)
vp2["cmedians"].set_color("#222222")
vp2["cmedians"].set_linewidth(1.5)

for pos, d, color in zip(positions, data_rep, colors):
    ax.scatter([pos], [statistics.mean(d)], color="#111111",
               s=30, zorder=5, marker="D", linewidths=0)

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("Action repetition rate")
ax.set_title("(b) Consecutive-action repetition rate", loc="left", fontweight="bold")
ax.axvline(2.75, color="#aaaaaa", linewidth=0.8, linestyle="--")
ax.set_ylim(bottom=0)

for pos, d in zip(positions, data_rep):
    ax.text(pos, statistics.mean(d) + 0.005, f"{statistics.mean(d):.3f}",
            ha="center", va="bottom", fontsize=7, color="#111111")

# ===========================================================================
# (c) Per-app success rates
# ===========================================================================
ax = ax_app

apps = sorted(app_stats["evocua_8b"].keys())
app_short = [APP_DISPLAY.get(a, a) for a in apps]
x = np.arange(len(apps))
width = 0.35

evocua_rates = [app_stats["evocua_8b"][a]["success_rate"] for a in apps]
qwen_rates   = [app_stats["qwen3_8b_thinking"][a]["success_rate"] for a in apps]

bars1 = ax.bar(x - width/2, evocua_rates, width, label="EvoCUA-8B",
               color=C_EVOCUA_SUC, alpha=0.85, linewidth=0)
bars2 = ax.bar(x + width/2, qwen_rates,   width, label="Qwen3-VL-8B",
               color=C_QWEN_SUC,   alpha=0.85, linewidth=0)

ax.set_xticks(x)
ax.set_xticklabels(app_short, rotation=30, ha="right", fontsize=7.5)
ax.set_ylabel("Success rate")
ax.set_ylim(0, 1.05)
ax.set_title("(c) Per-application success rate", loc="left", fontweight="bold")
ax.legend(loc="upper right", framealpha=0.9)

# Dashed line at overall success rate
evocua_overall = sum(t["success"] for t in subset("evocua_8b")) / len(subset("evocua_8b"))
qwen_overall   = sum(t["success"] for t in subset("qwen3_8b_thinking")) / len(subset("qwen3_8b_thinking"))
ax.axhline(evocua_overall, color=C_EVOCUA_SUC, linewidth=1.0, linestyle=":",
           label=f"EvoCUA mean ({evocua_overall:.2f})")
ax.axhline(qwen_overall,   color=C_QWEN_SUC,   linewidth=1.0, linestyle=":",
           label=f"Qwen3 mean ({qwen_overall:.2f})")

# ===========================================================================
# (d) Failure mode breakdown: trajectories (not steps)
#     stacked bar: success | DONE-but-wrong (drift) | explicit-FAIL | step-limit
#     split by stall / no-stall within each failure mode
# ===========================================================================
ax = ax_budget

# Category colors
C_SUC       = "#4dac26"   # green   – success
C_DRIFT_NS  = "#f1a340"   # orange  – no-stall + DONE-but-wrong (silent drift)
C_FAIL_NS   = "#d7191c"   # red     – no-stall + explicit FAIL
C_SLIM_NS   = "#fdae61"   # yellow  – no-stall + step limit
C_DRIFT_S   = "#abd9e9"   # light blue – stall + DONE-but-wrong
C_FAIL_S    = "#2c7bb6"   # blue    – stall + explicit FAIL
C_SLIM_S    = "#74add1"   # med blue – stall + step limit

model_keys  = ["evocua_8b", "qwen3_8b_thinking"]
model_names = ["EvoCUA-8B", "Qwen3-VL-8B"]
x_pos = np.array([0.0, 1.0])
bar_w = 0.55

# Count trajectory fractions per model
seg_order = [
    ("no_stall", "drift",         C_DRIFT_NS, "DONE-but-wrong\n(no stall)"),
    ("no_stall", "explicit_fail", C_FAIL_NS,  "Explicit FAIL\n(no stall)"),
    ("no_stall", "step_limit",    C_SLIM_NS,  "Step limit\n(no stall)"),
    ("stall",    "drift",         C_DRIFT_S,  "DONE-but-wrong\n(stall)"),
    ("stall",    "explicit_fail", C_FAIL_S,   "Explicit FAIL\n(stall)"),
    ("stall",    "step_limit",    C_SLIM_S,   "Step limit\n(stall)"),
]

fracs_traj = {}
for mk in model_keys:
    ts    = [t for t in trajs if t["model"] == mk]
    total = len(ts)
    suc   = sum(1 for t in ts if t["success"])
    fracs_traj[mk] = {"success": suc / total}
    for stall_tag, term, _, _ in seg_order:
        stalled = (stall_tag == "stall")
        cnt = sum(1 for t in ts
                  if not t["success"]
                  and (t["stall_count"] > 0) == stalled
                  and t["terminal_reason"] == term)
        fracs_traj[mk][f"{stall_tag}_{term}"] = cnt / total

# Draw stacked bars
bottoms = np.zeros(2)
patches = []

# success first
fv = np.array([fracs_traj[mk]["success"] for mk in model_keys])
b  = ax.bar(x_pos, fv, bar_w, bottom=bottoms, color=C_SUC, linewidth=0)
patches.append(mpatches.Patch(color=C_SUC, label="Success"))
for i, (v, bot) in enumerate(zip(fv, bottoms)):
    if v > 0.04:
        ax.text(x_pos[i], bot + v/2, f"{100*v:.0f}%",
                ha="center", va="center", fontsize=7.5, fontweight="bold", color="white")
bottoms += fv

for stall_tag, term, color, label in seg_order:
    key = f"{stall_tag}_{term}"
    fv  = np.array([fracs_traj[mk].get(key, 0) for mk in model_keys])
    b   = ax.bar(x_pos, fv, bar_w, bottom=bottoms, color=color, linewidth=0.3,
                 edgecolor="white")
    patches.append(mpatches.Patch(color=color, label=label))
    for i, (v, bot) in enumerate(zip(fv, bottoms)):
        if v > 0.04:
            ax.text(x_pos[i], bot + v/2, f"{100*v:.0f}%",
                    ha="center", va="center", fontsize=7, fontweight="bold", color="white")
    bottoms += fv

ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, fontsize=8.5)
ax.set_ylabel("Fraction of trajectories")
ax.set_ylim(0, 1.02)
ax.set_xlim(-0.5, 1.5)
ax.set_title("(d) Failure mode taxonomy", loc="left", fontweight="bold")
ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1.0),
          fontsize=6.5, framealpha=0.9, title="Category", title_fontsize=7)
ax.grid(axis="x", alpha=0)

# ---------------------------------------------------------------------------
# Finish
# ---------------------------------------------------------------------------
fig.tight_layout(pad=1.5, h_pad=2.0, w_pad=1.5)

fig.savefig(str(OUTPDF), bbox_inches="tight")
fig.savefig(str(OUTPNG), bbox_inches="tight", dpi=200)
print(f"Saved: {OUTPDF}")
print(f"Saved: {OUTPNG}")

# ---------------------------------------------------------------------------
# Print key statistics for paper
# ---------------------------------------------------------------------------
print("\n===== Key statistics for paper =====")
for model, label in MODEL_LABELS.items():
    suc     = subset(model, success=True)
    fai     = subset(model, success=False)
    stalled = [t for t in fai if t["stall_count"] > 0]
    drift   = [t for t in fai if t["stall_count"] == 0]
    sb      = data["step_budget"][model]
    total   = sb["total_steps"]

    print(f"\n{label}:")
    print(f"  Success rate: {len(suc)}/{len(suc)+len(fai)} = {len(suc)/(len(suc)+len(fai)):.2f}")
    print(f"  Mean steps  (success / failed): {statistics.mean(vals(suc,'n_steps')):.1f} / {statistics.mean(vals(fai,'n_steps')):.1f}")
    print(f"  Step ratio  (failed/success): {statistics.mean(vals(fai,'n_steps'))/statistics.mean(vals(suc,'n_steps')):.2f}x")
    print(f"  Mean rep_rate (success / failed): {statistics.mean(vals(suc,'rep_rate')):.3f} / {statistics.mean(vals(fai,'rep_rate')):.3f}")
    print(f"  Stalled trajectories:       {len(stalled)}/{len(fai)} ({100*len(stalled)/max(len(fai),1):.1f}%)")
    print(f"  Active-failing (drift):     {len(drift)}/{len(fai)}   ({100*len(drift)/max(len(fai),1):.1f}%)")
    if stalled:
        print(f"  Median stall onset (norm):  {statistics.median(t['first_stall_pos'] for t in stalled):.2f}")
    print(f"  LLM call budget:")
    print(f"    Success steps:          {sb['success_steps']:5d} / {total} ({100*sb['success_steps']/total:.1f}%)")
    print(f"    Active-failing (drift): {sb['active_fail_steps']:5d} / {total} ({100*sb['active_fail_steps']/total:.1f}%)")
    print(f"    Stall steps:            {sb['stall_steps']:5d} / {total} ({100*sb['stall_steps']/total:.1f}%)")
