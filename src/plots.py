"""
Turn results/results.csv into figures.

  1. results/fig_degradation.png - one panel per graded attack (jpeg,
     downscale, crop, reshare): Recall@1 vs attack strength, one line per method.
  2. results/fig_single.png - grouped bars for one-shot attacks
     (hflip, caption_bar, text_overlay).

Usage: python src/plots.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
STYLE = {"phash": ("#888888", "s"), "clip": ("#1f77b4", "o"), "dinov2": ("#d62728", "^")}


def main():
    df = pd.read_csv(RESULTS / "results.csv")
    methods = list(dict.fromkeys(df["method"]))

    graded = ["jpeg", "downscale", "crop", "reshare"]
    fig, axes = plt.subplots(1, len(graded), figsize=(4 * len(graded), 3.6), sharey=True)
    for ax, attack in zip(axes, graded):
        for m in methods:
            sub = df[(df.method == m) & (df.attack == attack)]
            color, marker = STYLE.get(m, (None, "o"))
            ax.plot(sub["level"], sub["R@1"], marker=marker, color=color, label=m)
        ax.set_title(attack)
        ax.set_xlabel("weaker  →  stronger")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Recall@1")
    axes[0].legend()
    fig.suptitle("Finding the original after re-share damage (500 queries, 2,500-image gallery)")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_degradation.png", dpi=150)

    single = ["hflip", "caption_bar", "text_overlay"]
    sub = df[df.attack.isin(single)].pivot(index="attack", columns="method", values="R@1")
    sub = sub.loc[single, methods]
    ax = sub.plot.bar(figsize=(6, 3.6), rot=0,
                      color=[STYLE.get(m, ("C0",))[0] for m in methods])
    ax.set_ylabel("Recall@1")
    ax.set_ylim(0, 1.05)
    ax.set_title("One-shot edits")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "fig_single.png", dpi=150)
    print("saved fig_degradation.png and fig_single.png")


if __name__ == "__main__":
    main()
