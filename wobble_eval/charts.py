"""The seven wobble figures. Same specs, palette and layout as the notebook,
wrapped as functions over the analysis result.

Palette is validated for colour-vision deficiency (all-pairs for the four
section hues; single-hue ordinal ramp for the 1-4 scale).
"""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")          # write PNGs without a display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .framework import CODE2NAME, CODE2SECTION, FRAMEWORK
from .stats import LEVELS

SUBTITLE = ""      # each chart function rebinds this before drawing

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SURFACE, PLANE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

SERIES  = {"B": "#2a78d6", "C": "#eb6834", "D": "#1baf7a", "F": "#4a3aa7"}   # validated all-pairs
ORDINAL = ["#86b6ef", "#3987e5", "#256abf", "#0d366b"]                       # scores 1,2,3,4
STATUS  = {"stable": "#0ca30c", "minor": "#fab219", "material": "#ec835a", "critical": "#d03b3b"}
STATUS["severe"] = STATUS["critical"]
GRADE_MARK = {"stable": "●", "minor": "▲", "material": "◆", "severe": "✕"}
LEVEL_NAME = ["Not observed", "Developing", "Proficient", "Highly effective"]
SECTION_SHORT = {"B": "Lesson Plan Fidelity", "C": "High-Leverage Practices",
                 "D": "Student Engagement", "F": "Teacher Subject Knowledge"}

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9.5,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200,
    "savefig.bbox": "tight", "axes.titlelocation": "left", "axes.titlepad": 14,
    "axes.titlesize": 12.5, "axes.titleweight": "semibold",
})

def style(ax, xgrid=False, ygrid=False, spines=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in spines)
    ax.xaxis.grid(xgrid); ax.yaxis.grid(ygrid)
    return ax

# Offsets are computed in POINTS, not axes fractions: these figures range from 4 to 14 inches
# tall, and an axes-fraction offset that looks right on one silently collapses on the other.
def _axh(fig, ax):
    return max(ax.get_position().height * fig.get_figheight(), 0.1)

def titles(ax, title, sub=None):
    ax.set_title(title, pad=30)
    ax.annotate(SUBTITLE if sub is None else sub, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 11), textcoords="offset points", color=MUTED, fontsize=8.5,
                va="bottom", ha="left", annotation_clip=False)

def legend_below(fig, ax, handles=None, pts=40, ncol=4, **kw):
    off = (pts / 72.0) / _axh(fig, ax)
    kw.setdefault("fontsize", 8.5)
    args = dict(loc="upper left", bbox_to_anchor=(0, -off), ncol=ncol, **kw)
    return ax.legend(handles=handles, **args) if handles is not None else ax.legend(**args)

def caption(fig, ax, text, pts=72):
    pos = ax.get_position()
    fig.text(pos.x0, pos.y0 - (pts / 72.0) / fig.get_figheight(),
             textwrap.fill(text, int(fig.get_figwidth() * 13.5)),
             color=MUTED, fontsize=8.5, ha="left", va="top", linespacing=1.5)

def save(fig, outdir, name):
    fig.savefig(os.path.join(outdir, name + ".png"))
    return fig


def make_subtitle(cfg, session_meta, n_runs):
    return (f"{cfg.MODEL} · effort={cfg.EFFORT} · thinking={cfg.THINKING} · "
            f"{n_runs} runs · {cfg.SCORING_MODE} · "
            f"session {session_meta['session_id'][:8]}")


def chart_headline(res, cfg, session_meta, outdir):
    global SUBTITLE
    SUBTITLE = make_subtitle(cfg, session_meta, res['matrix'].shape[1])
    CFG = cfg
    HEADLINE = res['headline']
    ind_stats = res['ind_stats']
    MATRIX = res['matrix']
    IND_CODES = res['ind_codes']
    reliability = res['reliability']
    sec_stats = res['sec_stats']
    sec_run = res['sec_run']
    overall_run = res['overall_run']
    n_runs = MATRIX.shape[1]
    tiles = [
        ("Overall mean score",     f"{HEADLINE['overall_mean']:.2f}",
         f"95% CI {HEADLINE['overall_ci'][0]:.2f}–{HEADLINE['overall_ci'][1]:.2f}  ·  out of 4.00", None),
        ("Krippendorff α (ordinal)", f"{HEADLINE['kripp_alpha_ordinal']:.2f}",
         "≥.80 dependable · .67–.80 tentative",
         "stable" if HEADLINE['kripp_alpha_ordinal'] >= .80 else
         "minor"  if HEADLINE['kripp_alpha_ordinal'] >= .67 else "critical"),
        ("Mean indicator SD",      f"{HEADLINE['mean_indicator_sd']:.2f}",
         "rubric points of run-to-run noise",
         "stable" if HEADLINE['mean_indicator_sd'] < .25 else
         "minor"  if HEADLINE['mean_indicator_sd'] < .50 else
         "material" if HEADLINE['mean_indicator_sd'] < .80 else "critical"),
        ("Fully stable indicators", f"{HEADLINE['pct_indicators_fully_stable']*100:.0f}%",
         f"identical in all {CFG.N_ITERATIONS} runs",
         "stable" if HEADLINE['pct_indicators_fully_stable'] >= .8 else
         "minor"  if HEADLINE['pct_indicators_fully_stable'] >= .5 else "material"),
        ("Significant wobble",     f"{HEADLINE['pct_indicators_sig_wobble']*100:.0f}%",
         f"Holm-corrected q<{CFG.ALPHA} vs a ≤5% noise floor",
         "stable" if HEADLINE['pct_indicators_sig_wobble'] <= .1 else
         "material" if HEADLINE['pct_indicators_sig_wobble'] <= .4 else "critical"),
        ("Proficiency verdict flips", f"{HEADLINE['n_indicators_flipping_proficiency']}",
         f"of {len(ind_stats)} indicators cross the ≥3 line",
         "stable" if HEADLINE['n_indicators_flipping_proficiency'] == 0 else
         "minor"  if HEADLINE['n_indicators_flipping_proficiency'] <= 3 else "critical"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 4.6))
    fig.subplots_adjust(wspace=0.30, hspace=0.55)
    for ax, (label, value, sub, st) in zip(axes.ravel(), tiles):
        ax.set_axis_off(); ax.set_facecolor(SURFACE)
        ax.text(0.055, 1.00, label.upper(), color=MUTED, fontsize=8.2, va="top", transform=ax.transAxes)
        ax.text(0.055, 0.80, value, color=INK, fontsize=27, va="top", ha="left",
                transform=ax.transAxes, fontweight="medium")
        ax.text(0.055, 0.30, textwrap.fill(sub, 34), color=INK2, fontsize=8.0, va="top",
                linespacing=1.5, transform=ax.transAxes)
        if st:                                    # status = icon + colour + label, never colour alone
            ax.text(0.055, -0.02, f"{GRADE_MARK.get(st, '●')} {st}", color=STATUS[st], fontsize=9,
                    va="top", transform=ax.transAxes, fontweight="semibold")
        ax.plot([0, 0], [-0.06, 1.04], transform=ax.transAxes, color=GRID, lw=1.2, clip_on=False)
    fig.suptitle("Scoring wobble — headline", x=0.008, ha="left", y=1.10, fontsize=13.5,
                 fontweight="semibold")
    fig.text(0.008, 1.015, SUBTITLE, color=MUTED, fontsize=8.5, ha="left")
    save(fig, outdir, "01_headline")
    plt.close(fig)
    return fig


def chart_score_matrix(res, cfg, session_meta, outdir):
    global SUBTITLE
    SUBTITLE = make_subtitle(cfg, session_meta, res['matrix'].shape[1])
    CFG = cfg
    HEADLINE = res['headline']
    ind_stats = res['ind_stats']
    MATRIX = res['matrix']
    IND_CODES = res['ind_codes']
    reliability = res['reliability']
    sec_stats = res['sec_stats']
    sec_run = res['sec_run']
    overall_run = res['overall_run']
    n_runs = MATRIX.shape[1]
    n_runs = MATRIX.shape[1]
    fig, ax = plt.subplots(figsize=(max(7.0, 2.0 + 0.55 * n_runs), 0.30 * len(IND_CODES) + 1.8))

    cmap = ListedColormap(ORDINAL); cmap.set_bad("#eceae4")
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    ax.pcolormesh(np.arange(n_runs + 1), np.arange(len(IND_CODES) + 1),
                  np.ma.masked_invalid(MATRIX)[::-1], cmap=cmap, norm=norm,
                  edgecolors=SURFACE, linewidth=2.0)

    if len(IND_CODES) * n_runs <= 480:                       # label only when the cells fit
        for i, row in enumerate(MATRIX[::-1]):
            for j, v in enumerate(row):
                if np.isnan(v):
                    ax.text(j + .5, i + .5, "NA", ha="center", va="center", color=MUTED, fontsize=7)
                else:
                    ax.text(j + .5, i + .5, f"{int(v)}", ha="center", va="center",
                            color="#ffffff" if v >= 3 else INK, fontsize=7.5)

    ax.set_yticks(np.arange(len(IND_CODES)) + .5)
    ax.set_yticklabels([f"{c}  {CODE2NAME[c][:42]}" for c in IND_CODES][::-1], fontsize=8)
    for t, c in zip(ax.get_yticklabels(), IND_CODES[::-1]):
        t.set_color(SERIES[CODE2SECTION[c]])
    ax.set_xticks(np.arange(n_runs) + .5)
    ax.set_xticklabels([f"r{i+1}" for i in range(n_runs)], fontsize=8)
    ax.set_xlim(0, n_runs); ax.set_ylim(0, len(IND_CODES))
    style(ax, spines=())
    titles(ax, "Score matrix — where the wobble actually is")
    legend_below(fig, ax, pts=34, ncol=5,
                 handles=[Patch(facecolor=ORDINAL[i], label=f"{i+1} — {LEVEL_NAME[i]}")
                          for i in range(4)] + [Patch(facecolor="#eceae4", edgecolor=BASELINE, lw=.8,
                                                  label="NA / unparsed")])
    caption(fig, ax, "A row of one colour is a stable indicator. Any change of shade along a row is "
                     "pure sampling noise: the transcript, rubric and prompt were identical in every "
                     "run. Row labels are coloured by section (B blue · C orange · D aqua · F violet).",
            pts=64)
    save(fig, outdir, "02_score_matrix")
    plt.close(fig)
    return fig


def chart_mean_ci(res, cfg, session_meta, outdir):
    global SUBTITLE
    SUBTITLE = make_subtitle(cfg, session_meta, res['matrix'].shape[1])
    CFG = cfg
    HEADLINE = res['headline']
    ind_stats = res['ind_stats']
    MATRIX = res['matrix']
    IND_CODES = res['ind_codes']
    reliability = res['reliability']
    sec_stats = res['sec_stats']
    sec_run = res['sec_run']
    overall_run = res['overall_run']
    n_runs = MATRIX.shape[1]
    d = ind_stats.dropna(subset=["mean"]).copy()
    d["ypos"] = np.arange(len(d))[::-1]

    fig, ax = plt.subplots(figsize=(9.5, 0.30 * len(d) + 1.9))
    ax.axvspan(CFG.PROFICIENCY_CUT, 4.35, color="#f4f3ef", zorder=0)
    ax.axvline(CFG.PROFICIENCY_CUT, color=BASELINE, lw=1.4, ls=(0, (5, 3)), zorder=1)
    ax.text(CFG.PROFICIENCY_CUT + .05, len(d) - .4, "Proficient ≥ 3", color=MUTED, fontsize=8.5,
            va="top")

    for r in d.itertuples():
        col = SERIES[r.section]
        ax.plot([r.min, r.max], [r.ypos, r.ypos], color=col, lw=0.9, alpha=.30, zorder=2)
        ax.plot([r.ci_lo, r.ci_hi], [r.ypos, r.ypos], color=col, lw=2.4, solid_capstyle="round",
                alpha=.55, zorder=3)
        ax.plot(r.mean, r.ypos, "o", ms=7.5, color=col, mec=SURFACE, mew=2.0, zorder=4)

    ax.set_yticks(d.ypos)
    ax.set_yticklabels([f"{r.code}  {r.indicator[:40]}" for r in d.itertuples()], fontsize=8)
    for t, s in zip(ax.get_yticklabels(), d.section):
        t.set_color(SERIES[s])
    ax.set_ylim(-.8, len(d) - .2)
    ax.set_xlim(0.7, 4.35); ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("score  (1 = not observed  →  4 = highly effective)")
    style(ax, xgrid=True, spines=("bottom",))
    titles(ax, "Per-indicator mean and 95% bootstrap CI")
    legend_below(fig, ax, pts=52, ncol=3,
                 handles=[Line2D([], [], marker="o", ls="none", ms=7, color=SERIES[s], mec=SURFACE,
                                 mew=1.6, label=f"Section {s} — {SECTION_SHORT[s]}")
                          for s in CFG.SECTIONS]
                 + [Line2D([], [], color=MUTED, lw=2.4, alpha=.55, label="95% CI of the mean"),
                    Line2D([], [], color=MUTED, lw=0.9, alpha=.4, label="observed min–max")])
    caption(fig, ax, "Thick bar = bootstrap 95% CI of the mean; hairline = full observed range; "
                     "dot = mean. Any CI straddling the dashed line is an indicator whose Proficient "
                     "verdict is not resolvable at these settings — raise N_ITERATIONS to narrow it, "
                     "or lower TEMPERATURE to shrink the underlying wobble.", pts=102)
    save(fig, outdir, "03_mean_ci")
    plt.close(fig)
    return fig


def chart_wobble_sd(res, cfg, session_meta, outdir):
    global SUBTITLE
    SUBTITLE = make_subtitle(cfg, session_meta, res['matrix'].shape[1])
    CFG = cfg
    HEADLINE = res['headline']
    ind_stats = res['ind_stats']
    MATRIX = res['matrix']
    IND_CODES = res['ind_codes']
    reliability = res['reliability']
    sec_stats = res['sec_stats']
    sec_run = res['sec_run']
    overall_run = res['overall_run']
    n_runs = MATRIX.shape[1]
    d = ind_stats.sort_values("sd", ascending=True, na_position="first").copy()
    d["ypos"] = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(9.0, 0.29 * len(d) + 1.9))
    for xv, lab in ((0.25, "negligible"), (0.50, "material"), (0.80, "severe")):
        ax.axvline(xv, color=BASELINE, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(xv, len(d) - .35, f" {lab}", color=MUTED, fontsize=8, va="bottom")

    ax.barh(d.ypos, d.sd.fillna(0), height=0.62, color=[STATUS[g] for g in d.grade], zorder=2,
            edgecolor=SURFACE, linewidth=2.0)
    for r in d.itertuples():
        txt = f"{GRADE_MARK[r.grade]} {r.sd:.2f}" if pd.notna(r.sd) else f"{GRADE_MARK[r.grade]} no data"
        ax.text((r.sd if pd.notna(r.sd) else 0) + 0.012, r.ypos, txt, va="center", fontsize=7.8,
                color=INK2)

    ax.set_yticks(d.ypos)
    ax.set_yticklabels([f"{r.code}  {r.indicator[:40]}" for r in d.itertuples()], fontsize=8)
    ax.set_xlabel("SD of the score across runs  (rubric points)")
    _sdmax = float(np.nanmax(d.sd.to_numpy(float))) if d.sd.notna().any() else 0.0
    ax.set_xlim(0, max(0.95, _sdmax * 1.28)); ax.set_ylim(-.8, len(d) - .2)
    style(ax, xgrid=True, spines=("bottom",))
    titles(ax, "How much each indicator wobbles")
    legend_below(fig, ax, pts=52, ncol=2,
                 handles=[Patch(facecolor=STATUS[g], label=f"{GRADE_MARK[g]} {g}"
                                + {"stable": " — identical in every run",
                                   "minor": " — ≥80% modal, spread ≤1 band",
                                   "material": " — ≥60% modal, spread ≤2 bands",
                                   "severe": " — below that, or mostly NA"}[g])
                          for g in ("stable", "minor", "material", "severe")])
    caption(fig, ax, "Grade combines modal agreement with spread, so it separates “always 3” from "
                     "“3 six times out of ten”. Each bar carries its own icon and value, so the grade "
                     "never depends on colour alone. SD around 0.5 means a report built on a single "
                     "run is roughly a coin-flip away from a different band on that indicator.",
            pts=102)
    save(fig, outdir, "04_wobble_sd")
    plt.close(fig)
    return fig


def chart_distribution(res, cfg, session_meta, outdir):
    global SUBTITLE
    SUBTITLE = make_subtitle(cfg, session_meta, res['matrix'].shape[1])
    CFG = cfg
    HEADLINE = res['headline']
    ind_stats = res['ind_stats']
    MATRIX = res['matrix']
    IND_CODES = res['ind_codes']
    reliability = res['reliability']
    sec_stats = res['sec_stats']
    sec_run = res['sec_run']
    overall_run = res['overall_run']
    n_runs = MATRIX.shape[1]
    d = ind_stats.copy()
    props = np.zeros((len(d), 5))                       # columns: 1, 2, 3, 4, NA
    for i, code_ in enumerate(d.code):
        xs = MATRIX[IND_CODES.index(code_)]
        valid = np.isfinite(xs)
        if valid.any():
            for j, lv in enumerate(LEVELS):
                props[i, j] = (xs[valid] == lv).mean() * valid.mean()
        props[i, 4] = 1.0 - props[i, :4].sum()

    ypos = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 0.29 * len(d) + 1.9))
    left = np.zeros(len(d))
    for j, (col, lab) in enumerate(zip(ORDINAL + ["#eceae4"],
                                       [f"{i+1} {LEVEL_NAME[i]}" for i in range(4)] + ["NA"])):
        ax.barh(ypos, props[:, j], left=left, height=0.66, color=col, label=lab,
                edgecolor=SURFACE, linewidth=2.0, zorder=2)
        left += props[:, j]

    for i, r in enumerate(d.itertuples()):              # label the modal share only, never every cell
        ax.text(1.015, ypos[i], f"{r.modal_share*100:.0f}% @ {int(r.mode)}" if r.n else "no data",
                va="center", fontsize=7.8, color=INK2)

    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r.code}  {r.indicator[:38]}" for r in d.itertuples()], fontsize=8)
    for t, s in zip(ax.get_yticklabels(), d.section):
        t.set_color(SERIES[s])
    ax.set_ylim(-.8, len(d) - .2)
    ax.set_xlim(0, 1.0); ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of runs")
    style(ax, xgrid=True, spines=("bottom",))
    titles(ax, "Which levels the model actually chose")
    legend_below(fig, ax, pts=52, ncol=5)
    caption(fig, ax, "A single full-width block is a decided indicator. Two adjacent shades is "
                     "ordinary boundary uncertainty. Non-adjacent shades (1 and 3, or 2 and 4) mean "
                     "the model is not reading the same evidence twice — check its evidence strings in "
                     "§8 before trusting the mean.", pts=102)
    save(fig, outdir, "05_distribution")
    plt.close(fig)
    return fig


def chart_section_runs(res, cfg, session_meta, outdir):
    global SUBTITLE
    SUBTITLE = make_subtitle(cfg, session_meta, res['matrix'].shape[1])
    CFG = cfg
    HEADLINE = res['headline']
    ind_stats = res['ind_stats']
    MATRIX = res['matrix']
    IND_CODES = res['ind_codes']
    reliability = res['reliability']
    sec_stats = res['sec_stats']
    sec_run = res['sec_run']
    overall_run = res['overall_run']
    n_runs = MATRIX.shape[1]
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    jit = np.random.default_rng(CFG.STATS_SEED).uniform(-.13, .13, size=(len(CFG.SECTIONS), n_runs))

    for i, s in enumerate(CFG.SECTIONS):
        xs, col = sec_run.loc[s].to_numpy(float), SERIES[s]
        row = sec_stats[sec_stats.section == s].iloc[0]
        ax.plot(np.full(n_runs, i) + jit[i], xs, "o", ms=6, color=col, alpha=.42, mec=SURFACE,
                mew=1.6, zorder=2)
        ax.plot([i, i], [row.ci_lo, row.ci_hi], color=col, lw=9, alpha=.20, solid_capstyle="round",
                zorder=3)
        ax.plot([i - .30, i + .30], [row["mean"]] * 2, color=col, lw=2.6, solid_capstyle="round",
                zorder=4)
        ax.text(i + .36, row["mean"], f"{row['mean']:.2f}\n±{row.ci_width/2:.2f}", fontsize=8.5,
                color=col, va="center", fontweight="semibold")

    allrow = sec_stats[sec_stats.section == "ALL"].iloc[0]
    ax.axhline(allrow["mean"], color=BASELINE, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(len(CFG.SECTIONS) - .55, allrow["mean"], f"all-section mean {allrow['mean']:.2f} ",
            color=MUTED, fontsize=8, va="bottom", ha="right")

    ax.set_xticks(range(len(CFG.SECTIONS)))
    ax.set_xticklabels([f"{s}\n{FRAMEWORK[s]['title'][:22]}" for s in CFG.SECTIONS], fontsize=8.5,
                       color=INK2)
    ax.set_xlim(-.55, len(CFG.SECTIONS) - .25)
    ax.set_ylabel("section mean score"); ax.set_ylim(1, 4); ax.set_yticks([1, 2, 3, 4])
    style(ax, ygrid=True, spines=("left",))
    titles(ax, "Section means — one dot per run")
    caption(fig, ax, "Faint dots are the individual runs, the bar is the mean, the band is its "
                     "bootstrap 95% CI. Averaging 7–12 indicators cancels a lot of noise, which is why "
                     "section means are far tighter than the indicators inside them — report at this "
                     "level when the indicator CIs are too wide to defend.", pts=54)
    save(fig, outdir, "06_section_runs")
    plt.close(fig)
    return fig


def chart_drift(res, cfg, session_meta, outdir):
    global SUBTITLE
    SUBTITLE = make_subtitle(cfg, session_meta, res['matrix'].shape[1])
    CFG = cfg
    HEADLINE = res['headline']
    ind_stats = res['ind_stats']
    MATRIX = res['matrix']
    IND_CODES = res['ind_codes']
    reliability = res['reliability']
    sec_stats = res['sec_stats']
    sec_run = res['sec_run']
    overall_run = res['overall_run']
    n_runs = MATRIX.shape[1]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(1, n_runs + 1)
    ax.plot(x, overall_run.to_numpy(float), "-", lw=1.4, color=MUTED, alpha=.85, label="all sections",
            zorder=2)
    for s in CFG.SECTIONS:
        ys = sec_run.loc[s].to_numpy(float)
        ax.plot(x, ys, "-o", lw=2.0, ms=6, color=SERIES[s], mec=SURFACE, mew=1.8,
                label=f"Section {s}", zorder=3)
        ax.text(x[-1] + .14, ys[-1], s, color=SERIES[s], fontsize=9, va="center",
                fontweight="semibold")

    ax.set_xticks(x); ax.set_xlabel("iteration"); ax.set_ylabel("mean score")
    ax.set_xlim(.7, n_runs + .6); ax.set_ylim(1, 4); ax.set_yticks([1, 2, 3, 4])
    style(ax, ygrid=True, spines=("left", "bottom"))
    titles(ax, "Run-to-run drift")
    legend_below(fig, ax, pts=48, ncol=5)
    fr_p, ic_p = reliability.loc[0, "friedman_p"], reliability.loc[0, "p_raters_drift"]
    caption(fig, ax, f"Friedman p={fr_p:.4f} · ICC rater F-test p={ic_p:.4f}. "
            + ("Runs differ systematically — the noise has a direction, so a single run is biased, "
               "not merely imprecise, and averaging converges on that bias."
               if pd.notna(fr_p) and fr_p < CFG.ALPHA else
               "No significant systematic difference between runs: the noise is unbiased, so "
               "averaging across runs converges on the right answer at roughly √N."), pts=76)
    save(fig, outdir, "07_drift")
    plt.close(fig)
    return fig



ALL_CHARTS = [chart_headline, chart_score_matrix, chart_mean_ci, chart_wobble_sd,
              chart_distribution, chart_section_runs, chart_drift]


def render_all(res, cfg, session_meta, outdir):
    os.makedirs(outdir, exist_ok=True)
    made = []
    for fn in ALL_CHARTS:
        try:
            fn(res, cfg, session_meta, outdir)
            made.append(fn.__name__)
        except Exception as exc:              # one bad chart must not kill the run
            print(f"  ! {fn.__name__} failed: {type(exc).__name__}: {exc}")
    return made
