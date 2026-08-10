"""Turns the long score frame into the wobble tables.

Same computations as the notebook (§8-§9), consolidated into one function so
the local runner and the notebook cannot drift apart.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from . import stats as S
from .framework import (ALL_CODES, CODE_ORDER, CODE2NAME, CODE2SECTION,
                        FRAMEWORK, SECTION_CODES)
from .stats import (boot_ci, t_ci, norm_entropy, p_vs_random, p_wobble_test, holm,
                    krippendorff_alpha, fleiss_kappa, icc21, pairwise_agreement,
                    grade_wobble, LEVELS)


def analyse(scores_long, cfg, run_meta=None):
    """-> dict of DataFrames + headline dict. Mirrors notebook cells 26/30/32/33/34."""
    S.init(cfg)
    stats = scipy_stats
    CFG = cfg
    EXPECTED_CODES = [c for c in ALL_CODES if CODE2SECTION[c] in cfg.SECTIONS]

    # NOTE: reindexed against EXPECTED_CODES on purpose - an indicator the model returned NA
    # for in *every* run must still appear as a row (na_rate = 1.0), not vanish from the report.
    piv = (scores_long.pivot_table(index="code", columns="iteration", values="score",
                                   observed=True, dropna=False)
           .reindex(index=EXPECTED_CODES, columns=range(CFG.N_ITERATIONS)))
    wide = piv.set_axis(pd.MultiIndex.from_arrays(
        [[CODE2SECTION[c] for c in piv.index], list(piv.index)], names=["section", "code"]))
    wide.columns = [f"run{c+1}" for c in wide.columns]
    wide.insert(0, "indicator", [CODE2NAME[c] for _, c in wide.index])
    wide["distinct"] = wide.filter(like="run").nunique(axis=1)
    wide["range"]    = wide.filter(like="run").max(axis=1) - wide.filter(like="run").min(axis=1)

    MATRIX = wide.filter(like="run").to_numpy(float)        # indicators x runs
    IND_CODES = [c for _, c in wide.index]

    # every row carries the full column set, so a run where nothing parsed still yields a
    # well-formed (all-NaN) table instead of a KeyError three cells later
    BLANK = dict.fromkeys(
        ["mean", "sd", "median", "mode", "modal_share", "min", "max", "range", "distinct",
         "ci_lo", "ci_hi", "ci_width", "t_lo", "t_hi", "entropy", "p_proficient", "flip_rate",
         "two_band_rate", "n_disagree", "p_wobble", "p_vs_random"], np.nan)

    rows = []
    for code_, xs in zip(IND_CODES, MATRIX):
        x = xs[~np.isnan(xs)]
        n_na = int(np.isnan(xs).sum())
        if len(x) == 0:
            rows.append(dict(section=CODE2SECTION[code_], code=code_, indicator=CODE2NAME[code_],
                             n=0, na_rate=1.0, grade="severe", **BLANK)); continue
        mode = int(stats.mode(x, keepdims=False).mode)
        modal_share = float((x == mode).mean())
        lo, hi = boot_ci(x)
        p_w, k_dis, n_dis = p_wobble_test(x)
        p_prof = float((x >= CFG.PROFICIENCY_CUT).mean())
        rng_ = float(x.max() - x.min())
        rows.append(dict(
            section=CODE2SECTION[code_], code=code_, indicator=CODE2NAME[code_],
            n=len(x), na_rate=n_na / len(xs),
            mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            median=float(np.median(x)), mode=mode, modal_share=modal_share,
            min=float(x.min()), max=float(x.max()), range=rng_, distinct=int(len(np.unique(x))),
            ci_lo=lo, ci_hi=hi, ci_width=hi - lo,
            t_lo=t_ci(x)[0], t_hi=t_ci(x)[1],
            entropy=norm_entropy(x),
            p_proficient=p_prof, flip_rate=2 * min(p_prof, 1 - p_prof),
            two_band_rate=float((np.abs(x - mode) >= 2).mean()),
            n_disagree=k_dis, p_wobble=p_w, p_vs_random=p_vs_random(x),
        ))

    ind_stats = pd.DataFrame(rows)
    if ind_stats.n.sum() == 0:
        raise RuntimeError(
            "No indicator produced a single parseable score. Check the raw model output in "
            "raw_store[0] — the model is probably refusing the JSON format. Try "
            "PROMPT_VARIANT='terse', a lower TEMPERATURE, or a different MODEL_KEY.")
    ind_stats["q_wobble"] = holm(ind_stats["p_wobble"].to_numpy())
    ind_stats["sig_wobble"] = ind_stats["q_wobble"] < CFG.ALPHA
    ind_stats["grade"] = [grade_wobble(r.modal_share, r.range, r.na_rate)
                          if r.n else "severe" for r in ind_stats.itertuples()]
    ind_stats["section"] = pd.Categorical(ind_stats.section, list(CFG.SECTIONS), ordered=True)
    ind_stats = ind_stats.sort_values(["section", "code"],
                                      key=lambda s: s.map(CODE_ORDER) if s.name == "code" else s)

    def reliability_block(mat, label):
        mat = np.asarray(mat, float)
        complete = mat[~np.isnan(mat).any(axis=1)]
        ic = icc21(mat)
        out = dict(scope=label, n_indicators=mat.shape[0], n_runs=mat.shape[1],
                   n_complete=len(complete),
                   kripp_alpha_ordinal=krippendorff_alpha(mat, "ordinal"),
                   kripp_alpha_nominal=krippendorff_alpha(mat, "nominal"),
                   fleiss_kappa=fleiss_kappa(mat),
                   icc21=ic.get("icc"), pairwise_exact_agreement=pairwise_agreement(mat),
                   p_units=ic.get("p_units"), p_raters_drift=ic.get("p_raters"))
        if len(complete) >= 3 and complete.shape[1] >= 3:
            fr = stats.friedmanchisquare(*complete.T)
            out.update(friedman_chi2=fr.statistic, friedman_p=fr.pvalue)
        else:
            out.update(friedman_chi2=np.nan, friedman_p=np.nan)
        return out

    rel_rows = [reliability_block(MATRIX, "OVERALL (all sections)")]
    for s in CFG.SECTIONS:
        m = wide.loc[s].filter(like="run").to_numpy(float)
        rel_rows.append(reliability_block(m, f"Section {s} — {FRAMEWORK[s]['title']}"))
    reliability = pd.DataFrame(rel_rows)

    # per-run section mean (a coaching report quotes these, so their wobble is what matters)
    # reindexed over every iteration so an all-NA iteration stays a column instead of
    # silently shortening the series (charts index these positionally against MATRIX)
    _ok = scores_long.dropna(subset=["score"])
    sec_run = (_ok.groupby(["section", "iteration"], observed=True)["score"].mean().unstack()
               .reindex(index=list(CFG.SECTIONS), columns=range(CFG.N_ITERATIONS)))
    overall_run = _ok.groupby("iteration")["score"].mean().reindex(range(CFG.N_ITERATIONS))

    sec_rows = []
    for s in CFG.SECTIONS:
        xs = sec_run.loc[s].to_numpy(float)
        lo, hi = boot_ci(xs)
        sub = ind_stats[ind_stats.section == s]
        sec_rows.append(dict(
            section=s, title=FRAMEWORK[s]["title"], n_indicators=len(sub),
            mean=float(np.nanmean(xs)), sd_across_runs=float(np.nanstd(xs, ddof=1)),
            cv=float(np.nanstd(xs, ddof=1) / np.nanmean(xs)),
            ci_lo=lo, ci_hi=hi, ci_width=hi - lo,
            min_run=float(np.nanmin(xs)), max_run=float(np.nanmax(xs)),
            mean_indicator_sd=float(sub.sd.mean()),
            pct_unstable=float((sub.grade != "stable").mean()),
            pct_sig_wobble=float(sub.sig_wobble.mean()),
            mean_flip_rate=float(sub.flip_rate.mean()),
            na_rate=float(sub.na_rate.mean()),
            kripp_alpha=reliability.loc[reliability.scope.str.startswith(f"Section {s}"),
                                        "kripp_alpha_ordinal"].iloc[0]))
    xs = overall_run.dropna().to_numpy(float); lo, hi = boot_ci(xs)
    sec_rows.append(dict(section="ALL", title="All four sections", n_indicators=len(ind_stats),
                         mean=float(xs.mean()), sd_across_runs=float(xs.std(ddof=1)),
                         cv=float(xs.std(ddof=1) / xs.mean()), ci_lo=lo, ci_hi=hi, ci_width=hi - lo,
                         min_run=float(xs.min()), max_run=float(xs.max()),
                         mean_indicator_sd=float(ind_stats.sd.mean()),
                         pct_unstable=float((ind_stats.grade != "stable").mean()),
                         pct_sig_wobble=float(ind_stats.sig_wobble.mean()),
                         mean_flip_rate=float(ind_stats.flip_rate.mean()),
                         na_rate=float(ind_stats.na_rate.mean()),
                         kripp_alpha=reliability.loc[0, "kripp_alpha_ordinal"]))

    sec_stats = pd.DataFrame(sec_rows)
    # Is one section significantly wobblier than another? Levene on within-indicator deviations.

    # spread comparison across sections (notebook cell 33 tail)
    dev = scores_long.dropna(subset=["score"]).copy()
    dev["abs_dev"] = (dev.score
                      - dev.groupby("code", observed=True).score.transform("mean")).abs()
    groups = [g.abs_dev.to_numpy() for _, g in dev.groupby("section", observed=True)
              if len(g) > 1]
    spread = {}
    if len(groups) >= 2:
        lev = stats.levene(*groups, center="median")
        kw = stats.kruskal(*groups)
        spread = dict(levene_W=float(lev.statistic), levene_p=float(lev.pvalue),
                      kruskal_H=float(kw.statistic), kruskal_p=float(kw.pvalue))

    n_sig   = int(ind_stats.sig_wobble.sum())
    n_tot   = int(ind_stats.p_wobble.notna().sum())
    flippy  = ind_stats[ind_stats.flip_rate > 0].sort_values("flip_rate", ascending=False)
    worst   = ind_stats.nlargest(5, "sd")[["code", "indicator", "mean", "sd", "ci_lo", "ci_hi", "grade"]]

    HEADLINE = {
        "model": cfg.MODEL, "backend": cfg.BACKEND,
        "iterations": CFG.N_ITERATIONS, "indicators": len(ind_stats),
        "overall_mean": round(float(overall_run.mean()), 3),
        "overall_ci": [round(v, 3) for v in boot_ci(overall_run.dropna().to_numpy(float))],
        "overall_sd_across_runs": round(float(overall_run.std(ddof=1)), 3),
        "mean_indicator_sd": round(float(ind_stats.sd.mean()), 3),
        "kripp_alpha_ordinal": round(float(reliability.loc[0, "kripp_alpha_ordinal"]), 3),
        "icc21": round(float(reliability.loc[0, "icc21"]), 3),
        "pairwise_exact_agreement": round(float(reliability.loc[0, "pairwise_exact_agreement"]), 3),
        "pct_indicators_fully_stable": round(float((ind_stats.grade == "stable").mean()), 3),
        "pct_indicators_sig_wobble": round(n_sig / max(n_tot, 1), 3),
        "n_indicators_flipping_proficiency": int((ind_stats.flip_rate > 0).sum()),
        "mean_ci_width": round(float(ind_stats.ci_width.mean()), 3),
        "na_rate": round(float(ind_stats.na_rate.mean()), 3),
        "parse_failure_rate": round(float((scores_long.parse == "none").mean()), 4),
    }

    HEADLINE.update(effort=cfg.EFFORT, thinking=cfg.THINKING,
                    scoring_mode=cfg.SCORING_MODE, prompt_variant=cfg.PROMPT_VARIANT)
    if run_meta:
        HEADLINE.update({k: v for k, v in run_meta.items() if k not in HEADLINE})

    return dict(wide=wide, matrix=MATRIX, ind_codes=IND_CODES, ind_stats=ind_stats,
                reliability=reliability, sec_stats=sec_stats, sec_run=sec_run,
                overall_run=overall_run, headline=HEADLINE, spread=spread,
                flippy=flippy, worst=worst)
