#!/usr/bin/env python
"""Two charts for the POOLED 10-transcript study.

The per-session charts in charts_binary.py describe one lesson. Nothing in the
repo drew the cross-lesson picture, which is the level the findings actually
live at, so these two exist:

  A  indicator wobble ranked across both scales - which indicators move, and
     whether the two scales agree that they move
  B  band stability as a lesson x section grid - where a coaching report's
     High/Medium/Low verdict is decided by which pass happened to run
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from wobble_eval.charts import (BASELINE, GRID, INK, INK2, MUTED, SERIES, STATUS,
                                SURFACE, caption, legend_below, save, style, titles)

OUT = "findings_package/charts"
SUB = ("claude-opus-5 · 10 transcripts (8 distinct lessons) · 10 runs each · "
       "both scales")


def chart_wobble_ranked(outdir):
    m = pd.read_csv("scale_comparison/indicator_comparison.csv")
    m["flip_avg"] = m[["mean_flip_rate_14", "mean_flip_rate_bin"]].mean(axis=1)
    m = m.sort_values("flip_avg")
    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(9.6, 0.30 * len(m) + 2.4))
    ax.axvspan(0.10, 0.30, color="#f7ecea", zorder=0)
    ax.axvline(0.10, color=BASELINE, lw=1.2, ls=(0, (5, 3)), zorder=1)
    ax.text(0.102, len(m) - .4, "excluded as unreliable  →", color=MUTED,
            fontsize=8.5, va="top")
    for i, r in zip(y, m.itertuples()):
        col = SERIES[r.section]
        ax.plot([r.mean_flip_rate_14, r.mean_flip_rate_bin], [i, i], color=col,
                lw=1.4, alpha=.30, zorder=2)
        ax.plot(r.mean_flip_rate_14, i, "o", ms=6, color=SURFACE, mec=col, mew=1.8,
                zorder=3)
        ax.plot(r.mean_flip_rate_bin, i, "o", ms=6.5, color=col, mec=SURFACE,
                mew=1.4, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.code}  {r.indicator[:40]}" for r in m.itertuples()],
                       fontsize=8)
    for t, s in zip(ax.get_yticklabels(), m.section):
        t.set_color(SERIES[s])
    ax.set_ylim(-.8, len(m) - .2)
    ax.set_xlim(-0.005, 0.29)
    ax.set_xticks([0, .05, .10, .15, .20, .25])
    ax.set_xticklabels(["0", "5%", "10%", "15%", "20%", "25%"])
    ax.set_xlabel("flip rate — share of passes contradicting the majority verdict")
    style(ax, xgrid=True, spines=("bottom",))
    titles(ax, "Which indicators move, on both scales", sub=SUB)
    legend_below(fig, ax, pts=52, ncol=3, handles=[
        Patch(facecolor=MUTED, label="filled = binary"),
        Patch(facecolor=SURFACE, edgecolor=MUTED, label="hollow = 1–4 collapsed at ≥3"),
        Patch(facecolor="#f7ecea", label="≥10% — the seven removed")])
    caption(fig, ax,
            "Each indicator carries two points, one per scale; a short connector means the "
            "two scales agree about how much it moves. The seven beyond the dashed line "
            "were 19% of the framework but carried 50% of its total wobble. Row colour is "
            "the section (B blue · C orange · D aqua · F violet).", pts=88)
    save(fig, outdir, "01_indicator_wobble_ranked")
    plt.close(fig)


def chart_band_grid(outdir):
    b = pd.read_csv("scale_comparison_fresh/band_comparison.csv")
    ids = sorted(b.session_id.unique())
    lmap = {s: f"Lesson-{i:02d}" for i, s in enumerate(ids, 1)}
    secs = ["B", "C", "D", "F", "ALL"]
    grid = np.zeros((len(ids), len(secs)))
    for i, s in enumerate(ids):
        for j, sec in enumerate(secs):
            r = b[(b.session_id == s) & (b.section == sec)]
            if not len(r):
                grid[i, j] = np.nan; continue
            grid[i, j] = int(r.band_unstable_14.iloc[0]) + int(r.band_unstable_bin.iloc[0])

    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(["#e8efe9", "#f3e2c9", "#e8b9ac"]); cmap.set_bad("#eceae4")
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(ids) + 2.8))
    ax.pcolormesh(np.arange(len(secs) + 1), np.arange(len(ids) + 1),
                  np.ma.masked_invalid(grid)[::-1], cmap=cmap, norm=norm,
                  edgecolors=SURFACE, linewidth=2.4)
    lbl = {0: "stable", 1: "1 scale", 2: "both"}
    for i, row in enumerate(grid[::-1]):
        for j, v in enumerate(row):
            if v != v:
                continue
            ax.text(j + .5, i + .5, lbl[int(v)], ha="center", va="center",
                    fontsize=7.6, color=INK if v else MUTED,
                    fontweight="semibold" if v else "normal")
    ax.set_xticks(np.arange(len(secs)) + .5)
    ax.set_xticklabels(["B\nPlan\nFidelity", "C\nHigh-\nLeverage", "D\nStudent\nEngagement",
                        "F\nSubject\nKnowledge", "ALL\nsections"], fontsize=8.4,
                       color=INK2, linespacing=1.5)
    ax.set_yticks(np.arange(len(ids)) + .5)
    ax.set_yticklabels([lmap[s] for s in ids][::-1], fontsize=8.4)
    ax.set_xlim(0, len(secs)); ax.set_ylim(0, len(ids))
    style(ax, spines=())
    titles(ax, "Where the reported band is decided by luck", sub=SUB)
    legend_below(fig, ax, pts=40, ncol=3, handles=[
        Patch(facecolor="#e8efe9", label="band held across all 10 passes"),
        Patch(facecolor="#f3e2c9", label="band moved on one scale"),
        Patch(facecolor="#e8b9ac", label="band moved on both scales")])
    caption(fig, ax,
            "A cell is one lesson's section score, banded ≥85% High · 60–84% Medium · "
            "<60% Low. Shaded means the band changed between identical reruns, so which "
            "pass you happen to save decides what the coaching report says. The ALL column "
            "is stable everywhere — report the overall figure, not Sections B or D, from a "
            "single pass.", pts=72)
    save(fig, outdir, "02_band_stability_grid")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    chart_wobble_ranked(OUT)
    chart_band_grid(OUT)
    print("wrote", OUT + "/01_indicator_wobble_ranked.png")
    print("wrote", OUT + "/02_band_stability_grid.png")
