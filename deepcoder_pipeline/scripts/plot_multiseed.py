"""Regenerate the DeepCoder comparison charts so that each of the five seeds
used for the 'with training' configuration is shown as its own bar, instead
of being collapsed into a single mean +/- std bar. 'without training' stays
a single bar since that configuration is fully deterministic (identical
across all five seeds). Chart labels follow the thesis's own terminology
(BSS/POS/PPS/PSS/PES) instead of the old pipeline names.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "output" / "runs"
SEEDS = [0, 1, 2, 3, 4]
CONFIGS = ["no_predictor", "with_predictor"]

METRIC_COLS = [
    "program_operation_score",
    "program_position_score",
    "program_sequence_score",
    "program_edit_score",
]

SEED_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
NO_TRAIN_COLOR = "tab:gray"


def load_rows(seed, config):
    path = RUNS_DIR / f"T2_gas1000_epochs10_seed{seed}" / f"metrics_{config}.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def aggregate():
    agg = {c: {"bss": [], "exact_match": [], "all": {m: [] for m in METRIC_COLS},
               "solved": {m: [] for m in METRIC_COLS}} for c in CONFIGS}
    for c in CONFIGS:
        for s in SEEDS:
            rows = load_rows(s, c)
            n = len(rows)
            solved_rows = [r for r in rows if r["solved"] == "True"]
            solved_n = len(solved_rows)
            agg[c]["bss"].append(solved_n / n)
            em = sum(1 for r in rows if float(r["exact_match"]) == 1.0)
            agg[c]["exact_match"].append(em / n)
            for m in METRIC_COLS:
                allvals = [float(r[m]) for r in rows]
                agg[c]["all"][m].append(sum(allvals) / n)
                if solved_n:
                    sv = [float(r[m]) for r in solved_rows]
                    agg[c]["solved"][m].append(sum(sv) / solved_n)
                else:
                    agg[c]["solved"][m].append(0.0)
    return agg


def grouped_bar_chart(labels, no_train_vals, with_train_by_seed, title, outpath, figsize):
    """with_train_by_seed: list (len(labels)) of lists (len(SEEDS))."""
    n_bars_per_group = 1 + len(SEEDS)
    x = np.arange(len(labels))
    group_width = 0.82
    bar_width = group_width / n_bars_per_group

    fig, ax = plt.subplots(figsize=figsize)

    offset0 = -group_width / 2 + bar_width / 2
    ax.bar(x + offset0, no_train_vals, bar_width, color=NO_TRAIN_COLOR,
           label="without training")

    for i, seed in enumerate(SEEDS):
        vals = [with_train_by_seed[j][i] for j in range(len(labels))]
        offset = offset0 + (i + 1) * bar_width
        ax.bar(x + offset, vals, bar_width, color=SEED_COLORS[i],
               label=f"with training, seed {seed}")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(ncol=2, fontsize=8)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", outpath)


def main_chart(agg, outpath):
    labels = ["BSS", "Exact Match", "POS", "PPS", "PSS", "PES"]

    no_train_vals = [
        agg["no_predictor"]["bss"][0],
        agg["no_predictor"]["exact_match"][0],
    ] + [agg["no_predictor"]["all"][m][0] for m in METRIC_COLS]

    with_train_by_seed = [
        agg["with_predictor"]["bss"],
        agg["with_predictor"]["exact_match"],
    ] + [agg["with_predictor"]["all"][m] for m in METRIC_COLS]

    grouped_bar_chart(
        labels, no_train_vals, with_train_by_seed,
        "DeepCoder Metrics: Without vs. With Training, per Seed (T=2, gas=1000)",
        outpath, figsize=(13, 5.5),
    )


def solved_chart(agg, outpath):
    labels = ["POS", "PPS", "PSS", "PES"]

    no_train_vals = [agg["no_predictor"]["solved"][m][0] for m in METRIC_COLS]
    with_train_by_seed = [agg["with_predictor"]["solved"][m] for m in METRIC_COLS]

    grouped_bar_chart(
        labels, no_train_vals, with_train_by_seed,
        "Solved Tasks Only: Structural Metrics, per Seed (T=2, gas=1000)",
        outpath, figsize=(12, 5.5),
    )


if __name__ == "__main__":
    agg = aggregate()
    images_dir = Path("C:/Users/Lenovo PC/Downloads/Bachelorarbeit/repo/bachelorarbeit/arbeit/Latex_Template/Template 4/images")
    main_chart(agg, images_dir / "deepcoder_metric_comparison.png")
    solved_chart(agg, images_dir / "deepcoder_metric_comparison_solved.png")
