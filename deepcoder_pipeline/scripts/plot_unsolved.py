"""Regenerate the two best-effort DeepCoder charts:

1. deepcoder_metric_comparison_unsolved.png - all unsolved tasks, tasks with
   no stored best-effort candidate scored 0 (the same zero-convention used
   for unsolved tasks throughout this chapter).
2. deepcoder_metric_comparison_besteffort.png - only the unsolved tasks that
   actually have a stored best-effort candidate, so the mean reflects how
   close a captured candidate is, not how often the search keeps one at all.

The 'with training' configuration (which varies by seed since the solve rate
itself varies) is collapsed into a single mean +/- std bar across its five
seeds, matching the simplified result tables in the thesis. 'without
training' stays a single bar since that configuration, and hence which
tasks remain unsolved, is fully deterministic and identical across seeds.
Uses the thesis's own short metric names (POS/PPS/PSS/PES).
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
NO_TRAIN_COLOR = "tab:gray"
WITH_TRAIN_COLOR = "tab:blue"


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
    all_vals = [sum(float(r[m]) for r in unsolved) / len(unsolved) for m in METRIC_COLS]
    # Candidate-only mean: restricted to the tasks that actually have a
    # stored best-effort candidate, so this reflects how close a captured
    # candidate is, not how often the search keeps one at all.
    cand_vals = [sum(float(r[m]) for r in with_candidate) / len(with_candidate) for m in METRIC_COLS]
    return len(unsolved), len(with_candidate), all_vals, cand_vals


def bar_chart(no_vals, with_mean, with_std, no_n, no_cand, mean_n, mean_cand, title, outpath):
    x = np.arange(len(LABELS))
    group_width = 0.5
    bar_width = group_width / 2

    fig, ax = plt.subplots(figsize=(7, 5.5))

    offset0 = -group_width / 2 + bar_width / 2
    ax.bar(x + offset0, no_vals, bar_width, color=NO_TRAIN_COLOR,
           label=f"without training ({no_cand}/{no_n} with candidate)")
    ax.bar(x + offset0 + bar_width, with_mean, bar_width, yerr=with_std, capsize=4,
           color=WITH_TRAIN_COLOR,
           label=f"with training, mean $\\pm$ std (Ø {mean_cand:.1f}/{mean_n:.1f} with candidate)")

    ax.set_ylim(0, 0.25)
    ax.set_ylabel("Mean")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=20, ha="right")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", outpath)


def main():
    no_n, no_cand, no_all_vals, no_cand_vals = unsolved_stats(load_rows(SEEDS[0], "no"))

    with_stats = [unsolved_stats(load_rows(s, "with")) for s in SEEDS]
    mean_n = np.mean([s[0] for s in with_stats])
    mean_cand = np.mean([s[1] for s in with_stats])

    def collapse(idx):
        series = [[with_stats[i][idx][j] for i in range(len(SEEDS))] for j in range(len(LABELS))]
        return [np.mean(v) for v in series], [np.std(v) for v in series]

    with_all_mean, with_all_std = collapse(2)
    with_cand_mean, with_cand_std = collapse(3)

    images_dir = Path("C:/Users/Lenovo PC/Downloads/Bachelorarbeit/repo/bachelorarbeit/arbeit/Latex_Template/Template 4/images")

    bar_chart(
        no_all_vals, with_all_mean, with_all_std, no_n, no_cand, mean_n, mean_cand,
        "Unsolved Tasks Only: Best-Effort Structural Metrics (T=2, gas=1000)",
        images_dir / "deepcoder_metric_comparison_unsolved.png",
    )
    bar_chart(
        no_cand_vals, with_cand_mean, with_cand_std, no_n, no_cand, mean_n, mean_cand,
        "Unsolved Tasks with a Stored Candidate Only (T=2, gas=1000)",
        images_dir / "deepcoder_metric_comparison_besteffort.png",
    )


if __name__ == "__main__":
    main()
