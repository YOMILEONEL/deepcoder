"""Regenerate the best-effort (unsolved-tasks) DeepCoder chart across all
five seeds, showing each seed as its own bar for the 'with training'
configuration (which now varies by seed since the solve rate itself varies).
'without training' stays a single bar since that configuration, and hence
which tasks remain unsolved, is fully deterministic and identical across
seeds. Uses the thesis's own short metric names (POS/PPS/PSS/PES).
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = Path("C:/BA/deepcoder/deepcoder_pipeline/output/runs")
SEEDS = [0, 1, 2, 3, 4]

METRIC_COLS = [
    "best_effort_operation_score",
    "best_effort_position_score",
    "best_effort_sequence_score",
    "best_effort_edit_score",
]
LABELS = ["POS", "PPS", "PSS", "PES"]
SEED_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
NO_TRAIN_COLOR = "tab:gray"


def load_rows(seed, config):
    path = RUNS_DIR / f"T2_gas1000_epochs10_seed{seed}" / f"metrics_{config}_predictor.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def unsolved_stats(rows):
    unsolved = [r for r in rows if r["solved"] == "False"]
    with_candidate = [r for r in unsolved if r["best_effort_program"]]
    # Tasks without a best-effort candidate score 0.0 on every metric (same
    # zero-convention as unsolved tasks elsewhere in this chapter), so the
    # mean is taken over all unsolved tasks, not only those with a candidate.
    vals = [sum(float(r[m]) for r in unsolved) / len(unsolved) for m in METRIC_COLS]
    return len(unsolved), len(with_candidate), vals


def main():
    no_n, no_cand, no_vals = unsolved_stats(load_rows(SEEDS[0], "no"))

    with_stats = [unsolved_stats(load_rows(s, "with")) for s in SEEDS]
    with_by_seed = [[with_stats[i][2][j] for i in range(len(SEEDS))] for j in range(len(LABELS))]

    n_bars_per_group = 1 + len(SEEDS)
    x = np.arange(len(LABELS))
    group_width = 0.82
    bar_width = group_width / n_bars_per_group

    fig, ax = plt.subplots(figsize=(11, 5.5))

    offset0 = -group_width / 2 + bar_width / 2
    ax.bar(x + offset0, no_vals, bar_width, color=NO_TRAIN_COLOR,
           label=f"without training ({no_cand}/{no_n} with candidate)")

    for i, seed in enumerate(SEEDS):
        vals = [with_by_seed[j][i] for j in range(len(LABELS))]
        n, cand, _ = with_stats[i]
        offset = offset0 + (i + 1) * bar_width
        ax.bar(x + offset, vals, bar_width, color=SEED_COLORS[i],
               label=f"with training, seed {seed} ({cand}/{n} with candidate)")

    ax.set_ylim(0, 0.4)
    ax.set_ylabel("Mean")
    ax.set_title("Unsolved Tasks Only: Best-Effort Structural Metrics, per Seed (T=2, gas=1000)")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=20, ha="right")
    ax.legend(ncol=2, fontsize=7.5)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    outpath = Path("C:/Users/Lenovo PC/Downloads/Bachelorarbeit/repo/bachelorarbeit/arbeit/Latex_Template/Template 4/images/deepcoder_metric_comparison_unsolved.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", outpath)


if __name__ == "__main__":
    main()
