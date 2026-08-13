"""The seven figures for the binary (YES/NO) run.

Shares the palette, typography and layout helpers with `charts.py` so the two
report styles cannot drift apart. Two deliberate departures:

* YES/NO uses the single-hue ordinal ramp (dark = YES, light = NO) rather than
  two competing hues — the section colours already own blue/orange/aqua/violet,
  and a two-hue verdict scale would collide with them. Every cell also carries
  a Y / N glyph, so the verdict never depends on colour alone.
* Chart 06 has no 1-4 counterpart: it exists to show Krippendorff's alpha next
  to Gwet's AC1 and raw agreement, because on a binary scale those three can
  disagree wildly and only the gap between them tells you why.
"""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .charts import (BASELINE, GRADE_MARK, GRID, INK, INK2, MUTED, ORDINAL,
                     SECTION_SHORT, SERIES, STATUS, SURFACE, caption,
                     legend_below, save, style, titles)
from .framework import CODE2NAME, CODE2SECTION, FRAMEWORK

YESC, NOC, NAC = ORDINAL[3], ORDINAL[0], "#eceae4"
BAND_FILL = {"High": "#eef4ec", "Medium": "#f7f4ea", "Low": "#f8eeea"}


def subtitle(cfg, meta, n_runs, yes_at):
    return (f"{cfg.MODEL} · effort={cfg.EFFORT} · thinking={cfg.THINKING} · {n_runs} runs · "
            f"YES bar = level {yes_at}+ · session {meta['session_id'][:8]}")


def _ctx(res, cfg, meta):
    return subtitle(cfg, meta, res["matrix"].shape[1], res["yes_at"])


# ------------------------------------------------------------------ 01 headline
def chart_headline(res, cfg, meta, outdir):
    H, ind = res["headline"], res["ind_stats"]
    sub = _ctx(res, cfg, meta)
    n = len(ind)
    paradox = H["prevalence_paradox"]

    tiles = [
        ("Overall YES rate", f"{H['overall_yes_rate']*100:.0f}%",
         f"95% CI {H['overall_yes_ci'][0]*100:.0f}–{H['overall_yes_ci'][1]*100:.0f}%  ·  "
         f"band {H['fidelity_band']}", None),
        ("Gwet's AC1", f"{H['gwet_ac1']:.2f}",
         f"α={H['kripp_alpha']:.2f} · raw agree {H['pairwise_exact_agreement']*100:.0f}%"
         + ("  · prevalence paradox" if paradox else ""),
         "stable" if H["gwet_ac1"] >= .80 else
         "minor" if H["gwet_ac1"] >= .67 else "critical"),
        ("Verdict flips", f"{H['n_indicators_flipping_verdict']}",
         f"of {n} indicators answered both YES and NO",
         "stable" if H["n_indicators_flipping_verdict"] == 0 else
         "minor" if H["n_indicators_flipping_verdict"] <= 3 else
         "material" if H["n_indicators_flipping_verdict"] <= 8 else "critical"),
        ("Unanimous indicators", f"{H['pct_indicators_unanimous']*100:.0f}%",
         f"same verdict in all {cfg.N_ITERATIONS} runs",
         "stable" if H["pct_indicators_unanimous"] >= .8 else
         "minor" if H["pct_indicators_unanimous"] >= .5 else "material"),
        ("Single-pass verdict error", f"{H['mean_single_pass_verdict_error']*100:.0f}%",
         "chance one run contradicts the majority",
         "stable" if H["mean_single_pass_verdict_error"] < .05 else
         "minor" if H["mean_single_pass_verdict_error"] < .12 else
         "material" if H["mean_single_pass_verdict_error"] < .25 else "critical"),
        ("Coin-flip indicators", f"{H['n_indicators_coinflip']}",
         "verdict indistinguishable from chance",
         "stable" if H["n_indicators_coinflip"] == 0 else
         "minor" if H["n_indicators_coinflip"] <= 2 else "critical"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 4.6))
    fig.subplots_adjust(wspace=0.30, hspace=0.55)
    for ax, (label, value, note, st) in zip(axes.ravel(), tiles):
        ax.set_axis_off(); ax.set_facecolor(SURFACE)
        ax.text(0.055, 1.00, label.upper(), color=MUTED, fontsize=8.2, va="top",
                transform=ax.transAxes)
        ax.text(0.055, 0.80, value, color=INK, fontsize=27, va="top", ha="left",
                transform=ax.transAxes, fontweight="medium")
        ax.text(0.055, 0.30, textwrap.fill(note, 34), color=INK2, fontsize=8.0, va="top",
                linespacing=1.5, transform=ax.transAxes)
        if st:
            ax.text(0.055, -0.02, f"{GRADE_MARK.get(st, '●')} {st}", color=STATUS[st],
                    fontsize=9, va="top", transform=ax.transAxes, fontweight="semibold")
        ax.plot([0, 0], [-0.06, 1.04], transform=ax.transAxes, color=GRID, lw=1.2,
                clip_on=False)
    fig.suptitle("Binary verdict wobble — headline", x=0.008, ha="left", y=1.10,
                 fontsize=13.5, fontweight="semibold")
    fig.text(0.008, 1.015, sub, color=MUTED, fontsize=8.5, ha="left")
    save(fig, outdir, "01_headline")
    plt.close(fig)
    return fig


# ------------------------------------------------------------ 02 verdict matrix
def chart_verdict_matrix(res, cfg, meta, outdir):
    M, codes = res["matrix"], res["ind_codes"]
    n_runs = M.shape[1]
    fig, ax = plt.subplots(figsize=(max(7.0, 2.0 + 0.55 * n_runs), 0.30 * len(codes) + 1.8))

    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap([NOC, YESC]); cmap.set_bad(NAC)
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax.pcolormesh(np.arange(n_runs + 1), np.arange(len(codes) + 1),
                  np.ma.masked_invalid(M)[::-1], cmap=cmap, norm=norm,
                  edgecolors=SURFACE, linewidth=2.0)

    if len(codes) * n_runs <= 600:
        for i, row in enumerate(M[::-1]):
            for j, v in enumerate(row):
                if np.isnan(v):
                    ax.text(j + .5, i + .5, "NA", ha="center", va="center",
                            color=MUTED, fontsize=7)
                else:
                    ax.text(j + .5, i + .5, "Y" if v else "N", ha="center", va="center",
                            color="#ffffff" if v else INK, fontsize=7.5,
                            fontweight="semibold")

    ax.set_yticks(np.arange(len(codes)) + .5)
    ax.set_yticklabels([f"{c}  {CODE2NAME[c][:42]}" for c in codes][::-1], fontsize=8)
    for t, c in zip(ax.get_yticklabels(), codes[::-1]):
        t.set_color(SERIES[CODE2SECTION[c]])
    ax.set_xticks(np.arange(n_runs) + .5)
    ax.set_xticklabels([f"r{i+1}" for i in range(n_runs)], fontsize=8)
    ax.set_xlim(0, n_runs); ax.set_ylim(0, len(codes))
    style(ax, spines=())
    titles(ax, "Verdict matrix — where the verdict actually moves", sub=_ctx(res, cfg, meta))
    legend_below(fig, ax, pts=34, ncol=3,
                 handles=[Patch(facecolor=YESC, label="Y — YES, bar clearly met"),
                          Patch(facecolor=NOC, label="N — NO"),
                          Patch(facecolor=NAC, edgecolor=BASELINE, lw=.8,
                                label="NA / unparsed")])
    caption(fig, ax,
            "A single-colour row is a settled verdict. Any mixed row is a coaching decision that "
            "changes with the run — and unlike the 1-4 scale there is no adjacent band to soften "
            "it: every change of shade here flips what the coach is told. Row labels are coloured "
            "by section (B blue · C orange · D aqua · F violet).", pts=64)
    save(fig, outdir, "02_verdict_matrix")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------- 03 YES rate
def chart_yes_rate(res, cfg, meta, outdir):
    d = res["ind_stats"].dropna(subset=["p_yes"]).copy()
    d["ypos"] = np.arange(len(d))[::-1]

    fig, ax = plt.subplots(figsize=(9.5, 0.30 * len(d) + 2.0))
    ax.axvspan(0.5, 1.02, color="#f4f3ef", zorder=0)
    ax.axvline(0.5, color=BASELINE, lw=1.4, ls=(0, (5, 3)), zorder=1)
    ax.text(0.505, len(d) - .4, "verdict = YES →", color=MUTED, fontsize=8.5, va="top")
    ax.text(0.495, len(d) - .4, "← verdict = NO", color=MUTED, fontsize=8.5, va="top",
            ha="right")

    for r in d.itertuples():
        col = SERIES[r.section]
        ax.plot([r.ci_lo, r.ci_hi], [r.ypos, r.ypos], color=col, lw=2.4,
                solid_capstyle="round", alpha=.55, zorder=3)
        ax.plot(r.p_yes, r.ypos, "o", ms=7.5, color=col, mec=SURFACE, mew=2.0, zorder=4)
        if r.flip:
            ax.text(1.035, r.ypos, f"{int(r.n_yes)}/{int(r.n)}", va="center", fontsize=7.6,
                    color=INK2)

    ax.set_yticks(d.ypos)
    ax.set_yticklabels([f"{r.code}  {r.indicator[:40]}" for r in d.itertuples()], fontsize=8)
    for t, s in zip(ax.get_yticklabels(), d.section):
        t.set_color(SERIES[s])
    ax.set_ylim(-.8, len(d) - .2)
    ax.set_xlim(-0.02, 1.02); ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of runs answering YES")
    style(ax, xgrid=True, spines=("bottom",))
    titles(ax, "Per-indicator YES rate and 95% Wilson interval", sub=_ctx(res, cfg, meta))
    legend_below(fig, ax, pts=52, ncol=3,
                 handles=[Line2D([], [], marker="o", ls="none", ms=7, color=SERIES[s],
                                 mec=SURFACE, mew=1.6,
                                 label=f"Section {s} — {SECTION_SHORT[s]}")
                          for s in cfg.SECTIONS]
                 + [Line2D([], [], color=MUTED, lw=2.4, alpha=.55, label="95% Wilson CI")])
    caption(fig, ax,
            "Dots pinned at 0 or 1 are settled indicators; the Wilson interval still has width "
            "there, which is the honest statement that N runs cannot prove certainty. Any "
            "interval crossing the dashed line is an indicator whose YES/NO verdict this run "
            "cannot resolve — the fraction beside it is the actual YES count. Raise N to narrow "
            "the interval; if the dot itself sits near 0.5, more runs will not help.", pts=102)
    save(fig, outdir, "03_yes_rate")
    plt.close(fig)
    return fig


# -------------------------------------------------------- 04 verdict stability
def chart_verdict_stability(res, cfg, meta, outdir):
    d = res["ind_stats"].sort_values("modal_share", ascending=True,
                                     na_position="first").copy()
    d["ypos"] = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(9.2, 0.29 * len(d) + 2.0))
    for xv, lab in ((0.70, "material"), (0.90, "minor"), (1.00, "unanimous")):
        ax.axvline(xv, color=BASELINE, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(xv, len(d) - .35, f" {lab}", color=MUTED, fontsize=8, va="bottom")

    # fillna(0.5) not 0: a no-data row draws a zero-width bar at the floor rather than a
    # full-width one running off the left edge of the axis
    ax.barh(d.ypos, d.modal_share.fillna(0.5) - 0.5, left=0.5, height=0.62,
            color=[STATUS[g] for g in d.grade], zorder=2, edgecolor=SURFACE, linewidth=2.0)
    for r in d.itertuples():
        if not r.n:
            txt = f"{GRADE_MARK[r.grade]} no data"
        elif r.votes_needed != r.votes_needed:            # NaN — never converges
            txt = f"{GRADE_MARK[r.grade]} {r.modal_share*100:.0f}%  ·  voting won't settle it"
        else:
            v = int(r.votes_needed)
            txt = (f"{GRADE_MARK[r.grade]} {r.modal_share*100:.0f}%  ·  "
                   f"{v} pass{'' if v == 1 else 'es'}")
        ax.text((r.modal_share if r.modal_share == r.modal_share else 0.5) + 0.008,
                r.ypos, txt, va="center", fontsize=7.6, color=INK2)

    ax.set_yticks(d.ypos)
    ax.set_yticklabels([f"{r.code}  {r.indicator[:40]}" for r in d.itertuples()], fontsize=8)
    ax.set_xlabel("share of runs agreeing with the majority verdict")
    ax.set_xlim(0.5, 1.28); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["50%\n(coin flip)", "60%", "70%", "80%", "90%", "100%"])
    ax.set_ylim(-.8, len(d) - .2)
    style(ax, xgrid=True, spines=("bottom",))
    titles(ax, "How settled each verdict is — and what it would cost to settle it",
           sub=_ctx(res, cfg, meta))
    legend_below(fig, ax, pts=54, ncol=2,
                 handles=[Patch(facecolor=STATUS[g], label=f"{GRADE_MARK[g]} {g}" + txt)
                          for g, txt in (("stable", " — identical verdict in every run"),
                                         ("minor", " — ≥90% agree with the majority"),
                                         ("material", " — ≥70% agree"),
                                         ("severe", " — below 70%, or mostly NA"))])
    caption(fig, ax,
            "The bar starts at 50% because that is the floor: a binary indicator can never agree "
            "with its own majority less than half the time. The number after the percentage is "
            "how many independent passes you would have to run and majority-vote to reproduce "
            "this verdict 95% of the time — 1 means one pass is enough, a dash means the "
            "indicator is too close to a coin flip for voting to ever converge. Above about "
            "seven passes, fix the rubric wording rather than buying more runs.", pts=102)
    save(fig, outdir, "04_verdict_stability")
    plt.close(fig)
    return fig


# --------------------------------------------------------- 05 section fidelity
def chart_section_fidelity(res, cfg, meta, outdir):
    sec_run, sec_stats = res["sec_run"], res["sec_stats"]
    n_runs = res["matrix"].shape[1]
    high = getattr(cfg, "HIGH_BAND", 0.85)
    med = getattr(cfg, "MED_BAND", 0.60)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for lo, hi, name in ((high, 1.0, "High"), (med, high, "Medium"), (0.0, med, "Low")):
        ax.axhspan(lo, hi, color=BAND_FILL[name], zorder=0)
        ax.text(-0.52, (lo + hi) / 2, name, color=MUTED, fontsize=8.5, va="center", ha="left")
    for y in (med, high):
        ax.axhline(y, color=BASELINE, lw=1.0, ls=(0, (4, 3)), zorder=1)

    jit = np.random.default_rng(cfg.STATS_SEED).uniform(-.13, .13,
                                                        size=(len(cfg.SECTIONS), n_runs))
    for i, s in enumerate(cfg.SECTIONS):
        xs, col = sec_run.loc[s].to_numpy(float), SERIES[s]
        row = sec_stats[sec_stats.section == s].iloc[0]
        ax.plot(np.full(n_runs, i) + jit[i], xs, "o", ms=6, color=col, alpha=.45,
                mec=SURFACE, mew=1.6, zorder=3)
        ax.plot([i, i], [row.ci_lo, row.ci_hi], color=col, lw=9, alpha=.20,
                solid_capstyle="round", zorder=2)
        ax.plot([i - .30, i + .30], [row.yes_rate] * 2, color=col, lw=2.6,
                solid_capstyle="round", zorder=4)
        flag = "  ⚠ band flips" if row.band_flips else ""
        ax.text(i + .35, row.yes_rate, f"{row.yes_rate*100:.0f}%\n{row.bands_seen}{flag}",
                fontsize=8.3, color=col, va="center", fontweight="semibold")

    allrow = sec_stats[sec_stats.section == "ALL"].iloc[0]
    ax.axhline(allrow.yes_rate, color=MUTED, lw=1.2, ls=(0, (2, 2)), zorder=1)
    ax.text(len(cfg.SECTIONS) - .55, allrow.yes_rate,
            f"all-section fidelity {allrow.yes_rate*100:.0f}% ", color=MUTED, fontsize=8,
            va="bottom", ha="right")

    ax.set_xticks(range(len(cfg.SECTIONS)))
    ax.set_xticklabels([f"{s}\n{FRAMEWORK[s]['title'][:22]}" for s in cfg.SECTIONS],
                       fontsize=8.5, color=INK2)
    ax.set_xlim(-.6, len(cfg.SECTIONS) - .18)
    ax.set_ylim(0, 1); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("fidelity score  (% of indicators answered YES)")
    style(ax, ygrid=False, spines=("left",))
    titles(ax, "Section fidelity — one dot per run, against the framework's own bands",
           sub=_ctx(res, cfg, meta))
    caption(fig, ax,
            "Binary scoring reproduces the framework's own arithmetic exactly: fidelity = actions "
            "observed ÷ actions prescribed, banded ≥85% High · 60–84% Medium · <60% Low. The "
            "question this chart answers is not what the percentage is but whether the BAND holds "
            "still: a section whose dots straddle a dashed line is one whose headline verdict is "
            "decided by which run you happened to report.", pts=58)
    save(fig, outdir, "05_section_fidelity")
    plt.close(fig)
    return fig


# ------------------------------------------------------------- 06 agreement
def chart_agreement(res, cfg, meta, outdir):
    rel = res["reliability"]
    labels = ["OVERALL"] + [s for s in cfg.SECTIONS]
    po = [rel.iloc[0].pairwise_exact_agreement] + [
        rel[rel.scope.str.startswith(f"Section {s}")].iloc[0].pairwise_exact_agreement
        for s in cfg.SECTIONS]
    ac = [rel.iloc[0].gwet_ac1] + [
        rel[rel.scope.str.startswith(f"Section {s}")].iloc[0].gwet_ac1
        for s in cfg.SECTIONS]
    al = [rel.iloc[0].kripp_alpha] + [
        rel[rel.scope.str.startswith(f"Section {s}")].iloc[0].kripp_alpha
        for s in cfg.SECTIONS]
    prev = [rel.iloc[0].yes_prevalence] + [
        rel[rel.scope.str.startswith(f"Section {s}")].iloc[0].yes_prevalence
        for s in cfg.SECTIONS]

    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.axhspan(0.80, 1.05, color="#f2f5f1", zorder=0)
    ax.axhline(0.80, color=BASELINE, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0.67, color=BASELINE, lw=1.0, ls=(0, (2, 3)), zorder=1)
    ax.text(len(labels) - .45, 0.805, "dependable ≥ .80", color=MUTED, fontsize=8,
            va="bottom", ha="right")

    for k, (vals, col, lab) in enumerate(
            ((po, ORDINAL[0], "raw pairwise agreement"),
             (ac, ORDINAL[2], "Gwet's AC1  (prevalence-robust)"),
             (al, ORDINAL[3], "Krippendorff's α  (prevalence-sensitive)"))):
        pos = x + (k - 1) * w
        ax.bar(pos, np.nan_to_num(vals, nan=0.0), width=w * 0.92, color=col, zorder=2,
               edgecolor=SURFACE, linewidth=1.5, label=lab)
        for xi, v in zip(pos, vals):
            ax.text(xi, (v if v == v else 0) + 0.015,
                    "—" if v != v else f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.6, color=INK2)

    # prevalence goes INSIDE the tick label — a separate annotation under the axis
    # collided with the labels at every figure width worth using
    ax.set_xticks(x)
    names = ["All sections"] + [SECTION_SHORT[s] for s in cfg.SECTIONS]
    heads = ["ALL"] + list(cfg.SECTIONS)
    ax.set_xticklabels(
        [f"{h}\n{textwrap.fill(nm, 15)}\n{p*100:.0f}% YES"
         for h, nm, p in zip(heads, names, prev)],
        fontsize=8.2, color=INK2, linespacing=1.5)
    # alpha and kappa can go negative; clamp the floor to the data so the bar stays visible
    _floor = float(np.nanmin(np.array(po + ac + al, float))) if np.isfinite(
        np.array(po + ac + al, float)).any() else 0.0
    ax.set_ylim(min(-0.05, _floor - 0.12), 1.05); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.set_ylabel("agreement between runs")
    style(ax, ygrid=True, spines=("left",))
    titles(ax, "Three agreement coefficients, because on a binary scale they disagree",
           sub=_ctx(res, cfg, meta))
    legend_below(fig, ax, pts=86, ncol=1)
    caption(fig, ax,
            "Read the GAP, not any single bar. Krippendorff's α and Fleiss' κ correct for chance "
            "using the observed YES/NO split, so on a section that is 90% NO they collapse toward "
            "zero even when the runs agree almost perfectly — the prevalence paradox, a property "
            "of the statistic rather than of the model. Gwet's AC1 corrects for chance without "
            "that failure mode. Where AC1 is high and α is low, trust AC1 and quote the raw "
            "agreement alongside it; where BOTH are low, the model genuinely is unreliable there. "
            "The YES share under each group is what drives the gap.", pts=138)
    save(fig, outdir, "06_agreement")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------- 07 drift
def chart_drift(res, cfg, meta, outdir):
    sec_run, overall_run, rel = res["sec_run"], res["overall_run"], res["reliability"]
    n_runs = res["matrix"].shape[1]
    high, med = getattr(cfg, "HIGH_BAND", 0.85), getattr(cfg, "MED_BAND", 0.60)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(1, n_runs + 1)
    for y in (med, high):
        ax.axhline(y, color=BASELINE, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.plot(x, overall_run.to_numpy(float), "-", lw=1.6, color=MUTED, alpha=.9,
            label="all sections", zorder=2)
    for s in cfg.SECTIONS:
        ys = sec_run.loc[s].to_numpy(float)
        ax.plot(x, ys, "-o", lw=2.0, ms=6, color=SERIES[s], mec=SURFACE, mew=1.8,
                label=f"Section {s}", zorder=3)
        ax.text(x[-1] + .14, ys[-1], s, color=SERIES[s], fontsize=9, va="center",
                fontweight="semibold")

    ax.set_xticks(x); ax.set_xlabel("iteration")
    ax.set_ylabel("share of indicators answered YES")
    ax.set_xlim(.7, n_runs + .6); ax.set_ylim(0, 1)
    ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    style(ax, ygrid=True, spines=("left", "bottom"))
    titles(ax, "Run-to-run drift in the YES rate", sub=_ctx(res, cfg, meta))
    legend_below(fig, ax, pts=48, ncol=5)
    q, p = rel.loc[0, "cochran_q"], rel.loc[0, "cochran_p"]
    verdict = ("Runs differ systematically — the noise has a direction, so a single pass is "
               "biased rather than merely imprecise, and majority-voting more passes converges "
               "on that bias instead of on the truth."
               if pd.notna(p) and p < cfg.ALPHA else
               "No significant systematic difference between runs: the noise is unbiased, so "
               "majority-voting across passes converges on the model's own stable verdict.")
    qtxt = "—" if pd.isna(q) else f"{q:.2f}"
    ptxt = "—" if pd.isna(p) else f"{p:.4f}"
    caption(fig, ax, f"Cochran's Q = {qtxt} (df={rel.loc[0, 'cochran_df']}), p={ptxt}. "
                     f"{verdict} Cochran's Q is Friedman's binary analogue: it blocks on "
                     f"indicator and asks whether the YES rate itself shifts across runs.",
            pts=76)
    save(fig, outdir, "07_drift")
    plt.close(fig)
    return fig


# ------------------------------------------------- 08 self-knowledge of the boundary
def chart_borderline(res, cfg, meta, outdir):
    """Does the model know which of its own verdicts are unstable? Only drawn when
    the run captured margins (EXPLAIN on)."""
    if not res.get("has_margin"):
        return None
    d = res["ind_stats"].dropna(subset=["borderline_rate", "single_pass_error"]).copy()
    if not len(d):
        return None

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.axvspan(-0.02, 0.001, color="#f4f3ef", zorder=0)
    ax.axhline(0, color=BASELINE, lw=1.0, zorder=1)
    # Many indicators land on exactly the same coordinates (0% error, 100% borderline is a
    # big pile), so label per COORDINATE rather than per point — individual annotations
    # overprint each other into unreadable mush.
    groups = {}
    for r in d.itertuples():
        ax.plot(r.borderline_rate, r.single_pass_error, "o", ms=9,
                color=SERIES[r.section], mec=SURFACE, mew=1.8, alpha=.75, zorder=3)
        groups.setdefault((round(r.borderline_rate, 3), round(r.single_pass_error, 3)),
                          []).append((r.code, r.section))
    for (bx, by), members in groups.items():
        if by <= 0:                       # the stable pile: count it, don't name 20 codes
            continue
        codes = ", ".join(c for c, _ in members)
        secs = {s for _, s in members}
        ax.annotate(codes, (bx, by), textcoords="offset points", xytext=(-10, 8),
                    fontsize=8.0, ha="right",           # mixed-section group gets neutral ink
                    color=SERIES[members[0][1]] if len(secs) == 1 else INK2,
                    fontweight="semibold")
    n_stable = sum(len(m) for (bx, by), m in groups.items() if by <= 0)
    if n_stable:                                        # above the line, clear of the ticks
        ax.annotate(f"{n_stable} indicators never moved", (0.5, 0.0),
                    textcoords="offset points", xytext=(0, 13), ha="center",
                    fontsize=8.2, color=MUTED)

    ax.set_xlabel("share of runs the model called this indicator BORDERLINE")
    ax.set_ylabel("single-pass verdict error\n(share of runs contradicting the majority)")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 0.55)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0\n(always CLEAR)", "25%", "50%", "75%", "100%"])
    ax.set_yticks([0, .1, .2, .3, .4, .5])
    ax.set_yticklabels(["0\n(never moves)", "10%", "20%", "30%", "40%", "50%"])
    style(ax, xgrid=True, ygrid=True, spines=("left", "bottom"))
    titles(ax, "Does the model know which of its verdicts are unstable?",
           sub=_ctx(res, cfg, meta))
    legend_below(fig, ax, pts=58, ncol=4,
                 handles=[Line2D([], [], marker="o", ls="none", ms=7, color=SERIES[s],
                                 mec=SURFACE, mew=1.6, label=f"Section {s}")
                          for s in cfg.SECTIONS])

    bt = res.get("boundary_test") or {}
    if bt and bt.get("fisher_p") == bt.get("fisher_p"):
        fb, fc, p = (bt["flip_rate_when_borderline"], bt["flip_rate_when_clear"],
                     bt["fisher_p"])
        n_flip = bt["borderline_and_flipped"] + bt["clear_but_flipped"]
        recall = bt["borderline_and_flipped"] / n_flip if n_flip else float("nan")
        if recall >= 0.999:
            verdict = ("Read it as a screen, not a predictor: it caught every mover "
                       "(100% recall) but only 31% of what it flags actually moves. That "
                       "asymmetry is still useful — a CLEAR call can be trusted from one "
                       "pass; a BORDERLINE call just means re-run it.")
        elif p < cfg.ALPHA:
            verdict = ("The flag is informative but leaks — some verdicts moved without "
                       "ever being flagged.")
        else:
            verdict = ("The flag misses movers, so it cannot tell you which indicators are "
                       "safe to quote from one pass.")
        tail = (f"Indicators ever flagged BORDERLINE flipped {fb*100:.0f}% of the time; "
                f"those always called CLEAR flipped {fc*100:.0f}%. Fisher exact p={p:.4f} "
                f"(underpowered at {n_flip} movers). {verdict}")
    else:
        tail = ("Not enough borderline calls to test whether the flag predicts instability.")
    caption(fig, ax,
            "Bottom-left is the ideal corner: the model was sure and it was right to be. "
            "Top-left is the dangerous quadrant — verdicts that move between runs while the "
            "model reports no doubt at all. " + tail, pts=86)
    save(fig, outdir, "08_borderline")
    plt.close(fig)
    return fig


# ------------------------------------------------------- 09 confidence calibration
def chart_calibration(res, cfg, meta, outdir):
    """Reliability diagram: stated confidence against the agreement it actually got."""
    if not res.get("has_conf") or not res.get("calib"):
        return None
    cb, bins = res["calib"], res["calib_bins"]
    bins = bins[bins.n_cells > 0]
    if not len(bins):
        return None

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    lo = min(0.45, float(bins.mean_stated.min()), float(bins.observed_agreement.min())) - .03
    ax.plot([lo, 1.0], [lo, 1.0], ls=(0, (5, 3)), lw=1.4, color=BASELINE, zorder=1)
    ax.annotate("perfect calibration", (lo + 0.06, lo + 0.06), textcoords="offset points",
                xytext=(10, -4), fontsize=8.2, color=MUTED)

    # OVERconfident = stated exceeds actual = BELOW the diagonal. Shade and label that half;
    # anything above the line is the model underselling itself.
    ax.fill_between([lo, 1.0], [lo, lo], [lo, 1.0], color="#faf1ee", zorder=0)
    ax.annotate("overconfident\n(stated > actual)", (1.0, lo), textcoords="offset points",
                xytext=(-8, 16), ha="right", fontsize=8.2, color=MUTED)
    ax.annotate("underconfident\n(actual > stated)", (lo, 1.0), textcoords="offset points",
                xytext=(10, -6), va="top", fontsize=8.2, color=MUTED)

    sizes = 40 + 320 * (bins.n_cells / max(1, int(bins.n_cells.max())))
    ax.plot(bins.mean_stated, bins.observed_agreement, "-", lw=1.6, color=ORDINAL[2],
            alpha=.55, zorder=2)
    ax.scatter(bins.mean_stated, bins.observed_agreement, s=sizes, color=ORDINAL[3],
               edgecolor=SURFACE, linewidth=1.8, zorder=3)
    for r in bins.itertuples():
        # high bins pile up in the top-right corner; drop their labels below the marker
        below = r.mean_stated > 0.80
        ax.annotate(f"{r.bin}\nn={int(r.n_cells)}",
                    (r.mean_stated, r.observed_agreement),
                    textcoords="offset points",
                    xytext=(0, -26) if below else (9, -4),
                    ha="center" if below else "left",
                    fontsize=7.8, color=INK2)

    ax.set_xlim(lo, 1.03); ax.set_ylim(lo, 1.03)
    ax.set_xlabel("confidence the model stated")
    ax.set_ylabel("agreement it actually got\n(share of other runs giving the same verdict)")
    style(ax, xgrid=True, ygrid=True, spines=("left", "bottom"))
    titles(ax, "Is the stated confidence worth anything?", sub=_ctx(res, cfg, meta))

    rho, p = cb["spearman_rho"], cb["spearman_p"]
    if rho == rho and p == p:
        signal = ("Higher stated confidence really does buy higher agreement "
                  f"(Spearman rho={rho:+.2f}, p={p:.4f}), so the number can be used to "
                  "decide which verdicts need a second pass."
                  if p < cfg.ALPHA and rho > 0 else
                  f"Stated confidence does not track actual agreement (Spearman "
                  f"rho={rho:+.2f}, p={p:.4f}), so it cannot be used to triage — a "
                  "confident call is no safer than a hesitant one.")
    else:
        signal = "Too little spread in the stated confidences to test the relationship."
    caption(fig, ax,
            f"Each point is a confidence band; area is how many calls fell in it. Points "
            f"below the diagonal are overconfident. Mean stated {cb['mean_confidence']:.2f} "
            f"vs mean actual {cb['mean_observed_agreement']:.2f} "
            f"({cb['overconfidence']:+.2f}); expected calibration error "
            f"{cb['expected_calibration_error']:.2f}. {signal} Agreement is computed "
            f"leave-one-out, so a call is never counted as agreeing with itself.", pts=64)
    save(fig, outdir, "09_calibration")
    plt.close(fig)
    return fig


# --------------------------------------------- 10 per-indicator claimed vs actual
def chart_confidence_by_indicator(res, cfg, meta, outdir):
    """One row per indicator: the confidence it claimed against the agreement it got."""
    if not res.get("has_conf"):
        return None
    d = res["ind_stats"].dropna(subset=["mean_confidence"]).copy()
    if not len(d):
        return None
    d["actual"] = 1 - d["single_pass_error"]
    d["ypos"] = np.arange(len(d))[::-1]

    fig, ax = plt.subplots(figsize=(10.0, 0.32 * len(d) + 2.2))
    for r in d.itertuples():
        col = SERIES[r.section]
        # the connector IS the calibration gap — length is the error, direction the sign
        ax.plot([r.mean_confidence, r.actual], [r.ypos, r.ypos], color=col, lw=1.6,
                alpha=.35, zorder=2, solid_capstyle="round")
        ax.plot(r.mean_confidence, r.ypos, "o", ms=7.5, color=col, mec=SURFACE, mew=1.8,
                zorder=4)
        ax.plot(r.actual, r.ypos, "D", ms=6.0, color=SURFACE, mec=col, mew=1.8, zorder=4)
        bits = []
        if r.conf_when_yes == r.conf_when_yes:
            bits.append(f"Y {r.conf_when_yes:.2f}")
        if r.conf_when_no == r.conf_when_no:
            bits.append(f"N {r.conf_when_no:.2f}")
        ax.text(1.035, r.ypos, "  ".join(bits), va="center", fontsize=7.4, color=INK2)

    ax.set_yticks(d.ypos)
    ax.set_yticklabels([f"{r.code}  {r.indicator[:38]}" for r in d.itertuples()], fontsize=8)
    for t, s in zip(ax.get_yticklabels(), d.section):
        t.set_color(SERIES[s])
    ax.set_ylim(-.8, len(d) - .2)
    ax.set_xlim(0.48, 1.02)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["0.50\n(toss-up)", "0.60", "0.70", "0.80", "0.90", "1.00\n(certain)"])
    ax.set_xlabel("confidence / agreement")
    style(ax, xgrid=True, spines=("bottom",))
    titles(ax, "Per indicator: the confidence it claimed vs the agreement it got",
           sub=_ctx(res, cfg, meta))
    legend_below(fig, ax, pts=54, ncol=2,
                 handles=[Line2D([], [], marker="o", ls="none", ms=7.5, color=MUTED,
                                 mec=SURFACE, mew=1.6,
                                 label="mean confidence the model stated"),
                          Line2D([], [], marker="D", ls="none", ms=6, color=SURFACE,
                                 mec=MUTED, mew=1.8,
                                 label="agreement it actually got across the runs")])
    caption(fig, ax,
            "The bar between the two markers is that indicator's calibration gap. Diamond to "
            "the RIGHT of the circle means the model undersold itself — it was more "
            "reproducible than it claimed. Diamond to the LEFT is the dangerous direction: "
            "it claimed more agreement than the runs delivered. Figures on the right are the "
            "mean confidence on the runs that said YES and the runs that said NO; an "
            "indicator showing both is one whose verdict moved.", pts=96)
    save(fig, outdir, "10_confidence_by_indicator")
    plt.close(fig)
    return fig


ALL_CHARTS = [chart_headline, chart_verdict_matrix, chart_yes_rate,
              chart_verdict_stability, chart_section_fidelity, chart_agreement,
              chart_drift, chart_borderline, chart_calibration,
              chart_confidence_by_indicator]


def render_all(res, cfg, meta, outdir):
    os.makedirs(outdir, exist_ok=True)
    made = []
    for fn in ALL_CHARTS:
        try:
            # a chart that returns None opted out (e.g. no margins captured this run) —
            # don't claim a file that was never written
            if fn(res, cfg, meta, outdir) is not None:
                made.append(fn.__name__)
        except Exception as exc:            # one bad chart must not kill the run
            print(f"  ! {fn.__name__} failed: {type(exc).__name__}: {exc}")
    return made
