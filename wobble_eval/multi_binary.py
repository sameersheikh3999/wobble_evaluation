"""Pooled analysis across many transcripts for the binary (YES/NO) scorer.

WHY THIS EXISTS — the thing one transcript cannot tell you
----------------------------------------------------------
A single-session run has ONE unit per indicator. You can measure how much a
verdict wobbles on that lesson, but you cannot compute a reliability coefficient
for the indicator itself, and you cannot tell these two apart:

    "C6 is badly worded"                    (unstable on every lesson)
    "C6 was genuinely borderline HERE"      (unstable on this lesson only)

With L lessons x R runs each, every indicator gets its own L x R matrix, so it
gets its own Gwet AC1, its own alpha, and a real test of whether it discriminates
between lessons at all.

TWO ORTHOGONAL AXES — both must hold for an indicator to be usable
------------------------------------------------------------------
1. RELIABILITY  — does it give the same answer on re-run?
   Measured by AC1 / alpha over the lesson x run matrix, and by a pooled exact
   binomial of within-lesson disagreement against the negligible-noise floor.

2. DISCRIMINATION — does it distinguish one lesson from another?
   An indicator answered NO on all ten lessons is perfectly reliable and
   perfectly useless: it carries no coaching signal. Measured by a chi-square
   test of homogeneity of the per-lesson YES counts.

Crossing them gives four verdicts:

                     discriminates          does not discriminate
    reliable         HEALTHY                UNINFORMATIVE (always same answer)
    unreliable       NOISY (rubric fix)     BROKEN (noise, no signal)

Both p-values are Holm-corrected across the indicator set, because testing 37
indicators at once otherwise manufactures significance.

CLUSTERED TRANSCRIPTS
---------------------
Where two files are the same audio transcribed twice, the lessons are not
independent. `cluster` marks them; pooled figures are reported both with and
without the duplicates so the effect of the non-independence is visible rather
than assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .binary import (fidelity_band, gwet_ac1, grade_binary, votes_needed,
                     wilson_ci, yes_bar_text)
from .framework import (ALL_CODES, CODE2NAME, CODE2SECTION, CODE_ORDER,
                        FRAMEWORK)
from .stats import holm, krippendorff_alpha, pairwise_agreement

MIN_LESSONS_FOR_TEST = 3


# ------------------------------------------------------------------ helpers
def _matrix_for(df, code, sessions, n_runs):
    """lessons x runs matrix of 0/1/NaN for one indicator."""
    sub = df[df.code == code]
    piv = (sub.pivot_table(index="session_id", columns="iteration", values="score",
                           observed=True, dropna=False)
           .reindex(index=sessions, columns=range(n_runs)))
    return piv.to_numpy(float)


def _chi2_discrimination(mat):
    """H0: the YES rate is the same on every lesson. Significant = the indicator
    actually separates lessons. Rows with no usable verdicts are dropped."""
    rows = []
    for r in mat:
        r = r[~np.isnan(r)]
        if len(r):
            rows.append([int(r.sum()), int(len(r) - r.sum())])
    if len(rows) < 2:
        return np.nan, np.nan
    tab = np.array(rows, float)
    # a column of zeros (all-YES or all-NO everywhere) means nothing to test
    if tab[:, 0].sum() == 0 or tab[:, 1].sum() == 0:
        return np.nan, np.nan
    try:
        chi2, p, _, _ = scipy_stats.chi2_contingency(tab)
        return float(chi2), float(p)
    except Exception:
        return np.nan, np.nan


def _pooled_instability(mat, floor):
    """Pooled exact binomial: total within-lesson disagreements against the
    negligible-noise floor. Disagreement is measured per lesson against that
    lesson's own majority, then summed — so a genuine between-lesson difference
    is never counted as noise."""
    k = n = 0
    for r in mat:
        r = r[~np.isnan(r)]
        if len(r) < 2:
            continue
        maj = 1 if r.mean() > 0.5 else 0
        k += int((r != maj).sum())
        n += len(r)
    if n < 2:
        return np.nan, k, n
    return float(scipy_stats.binomtest(k, n, floor, alternative="greater").pvalue), k, n


def classify(reliable, discriminates):
    if reliable and discriminates:       return "HEALTHY"
    if reliable and not discriminates:   return "UNINFORMATIVE"
    if not reliable and discriminates:   return "NOISY"
    return "BROKEN"


# ---------------------------------------------------------------- the analysis
def analyse_multi(df, cfg, sessions_meta, clusters=None):
    """df: pooled long frame with a session_id column. -> dict of DataFrames."""
    floor = cfg.NEGLIGIBLE_DISAGREEMENT
    alpha = cfg.ALPHA
    n_runs = int(df.iteration.max()) + 1
    sessions = list(dict.fromkeys(df.session_id))
    _drop = set(getattr(cfg, "EXCLUDE_CODES", ()) or ())
    codes = [c for c in ALL_CODES
             if CODE2SECTION[c] in cfg.SECTIONS and c not in _drop]
    if _drop:
        df = df[~df["code"].isin(_drop)]
    clusters = clusters or {}
    # one representative per duplicate cluster, for the de-duplicated re-run
    primary = []
    seen_cl = set()
    for s in sessions:
        cl = clusters.get(s)
        if cl is None:
            primary.append(s)
        elif cl not in seen_cl:
            seen_cl.add(cl); primary.append(s)

    # ---------------- per lesson x indicator ----------------
    cell_rows = []
    for s in sessions:
        for c in codes:
            xs = df[(df.session_id == s) & (df.code == c)]["score"].to_numpy(float)
            x = xs[~np.isnan(xs)]
            if not len(x):
                cell_rows.append(dict(session_id=s, section=CODE2SECTION[c], code=c,
                                      indicator=CODE2NAME[c], n=0, n_yes=np.nan,
                                      p_yes=np.nan, verdict="NA", flip=False,
                                      single_pass_error=np.nan, grade="severe"))
                continue
            k, n = int(x.sum()), len(x)
            p = k / n
            cell_rows.append(dict(
                session_id=s, section=CODE2SECTION[c], code=c, indicator=CODE2NAME[c],
                n=n, n_yes=k, p_yes=p,
                verdict="YES" if p > .5 else "NO" if p < .5 else "TIED",
                flip=bool(0 < k < n), single_pass_error=1 - max(p, 1 - p),
                grade=grade_binary(max(p, 1 - p), float(np.isnan(xs).mean()))))
    cells = pd.DataFrame(cell_rows)

    # ---------------- per indicator, pooled over lessons ----------------
    conf = df["confidence"] if "confidence" in df else None
    rows = []
    for c in codes:
        mat = _matrix_for(df, c, sessions, n_runs)
        mat_p = _matrix_for(df, c, primary, n_runs)
        sub_cells = cells[(cells.code == c) & (cells.n > 0)]
        obs = mat[~np.isnan(mat)]
        chi2, p_disc = _chi2_discrimination(mat)
        p_unst, k_dis, n_dis = _pooled_instability(mat, floor)
        lesson_p = sub_cells.p_yes.to_numpy(float)
        d = df[df.code == c]
        cf = pd.to_numeric(d["confidence"], errors="coerce") if conf is not None else None
        ok = d["score"].notna()
        rows.append(dict(
            section=CODE2SECTION[c], code=c, indicator=CODE2NAME[c],
            yes_bar=yes_bar_text(c, getattr(cfg, "YES_AT", 3)),
            n_lessons=int(len(sub_cells)), n_cells=int(obs.size),
            na_rate=float(np.isnan(mat).mean()),
            pooled_p_yes=float(obs.mean()) if obs.size else np.nan,
            # ---- reliability
            ac1=gwet_ac1(mat), kripp_alpha=krippendorff_alpha(mat, "nominal"),
            pairwise_agreement=pairwise_agreement(mat),
            ac1_dedup=gwet_ac1(mat_p),
            mean_flip_rate=float(sub_cells.single_pass_error.mean())
            if len(sub_cells) else np.nan,
            n_lessons_flipped=int(sub_cells.flip.sum()),
            pct_lessons_flipped=float(sub_cells.flip.mean()) if len(sub_cells) else np.nan,
            n_disagreements=k_dis, n_verdicts=n_dis, p_unstable=p_unst,
            # ---- discrimination
            lesson_yes_sd=float(np.nanstd(lesson_p, ddof=1)) if len(lesson_p) > 1 else 0.0,
            lesson_yes_min=float(np.nanmin(lesson_p)) if len(lesson_p) else np.nan,
            lesson_yes_max=float(np.nanmax(lesson_p)) if len(lesson_p) else np.nan,
            n_lessons_yes=int((sub_cells.verdict == "YES").sum()),
            n_lessons_no=int((sub_cells.verdict == "NO").sum()),
            chi2_discrimination=chi2, p_discriminates=p_disc,
            # ---- confidence
            mean_confidence=float(cf[ok].mean()) if cf is not None and ok.any() else np.nan,
            conf_when_yes=float(cf[ok & (d.score == 1)].mean())
            if cf is not None and (ok & (d.score == 1)).any() else np.nan,
            conf_when_no=float(cf[ok & (d.score == 0)].mean())
            if cf is not None and (ok & (d.score == 0)).any() else np.nan,
            votes_needed=votes_needed(
                float(np.nanmean([max(p_, 1 - p_) for p_ in lesson_p]))
                if len(lesson_p) else np.nan, getattr(cfg, "VOTE_TARGET", .95)),
        ))
    ind = pd.DataFrame(rows)

    ind["q_unstable"] = holm(ind["p_unstable"].to_numpy(float))
    ind["q_discriminates"] = holm(ind["p_discriminates"].to_numpy(float))
    # "reliable" is the ABSENCE of significant instability; an indicator with too
    # little data to test is treated as untested, not as reliable
    ind["reliable"] = ~(ind["q_unstable"] < alpha)
    ind["discriminates"] = ind["q_discriminates"] < alpha
    enough = ind["n_lessons"] >= MIN_LESSONS_FOR_TEST
    ind["verdict_class"] = [
        classify(r.reliable, r.discriminates) if e else "UNTESTED"
        for r, e in zip(ind.itertuples(), enough)]
    ind["section"] = pd.Categorical(ind.section, list(cfg.SECTIONS), ordered=True)
    ind = ind.sort_values(["section", "code"],
                          key=lambda s: s.map(CODE_ORDER) if s.name == "code" else s)

    # ---------------- section fidelity bands, per lesson ----------------
    band_rows = []
    high, med = getattr(cfg, "HIGH_BAND", .85), getattr(cfg, "MED_BAND", .60)
    for s in sessions:
        d = df[(df.session_id == s)].dropna(subset=["score"])
        for sec in list(cfg.SECTIONS) + ["ALL"]:
            dd = d if sec == "ALL" else d[d.section == sec]
            if not len(dd):
                continue
            per_run = dd.groupby("iteration", observed=True)["score"].mean()
            xs = per_run.to_numpy(float)
            bands = [fidelity_band(v, high, med) for v in xs]
            uniq = sorted(set(bands), key=lambda b: {"Low": 0, "Medium": 1, "High": 2}[b])
            band_rows.append(dict(
                session_id=s, section=sec,
                yes_rate=float(xs.mean()), min_run=float(xs.min()),
                max_run=float(xs.max()), sd_across_runs=float(xs.std(ddof=1))
                if len(xs) > 1 else 0.0,
                band=fidelity_band(float(xs.mean()), high, med),
                bands_seen="/".join(uniq), band_flips=max(0, len(uniq) - 1),
                band_unstable=len(uniq) > 1))
    bands = pd.DataFrame(band_rows)

    band_summary = (bands.groupby("section", observed=True)
                    .agg(n_lessons=("session_id", "nunique"),
                         mean_yes_rate=("yes_rate", "mean"),
                         min_lesson=("yes_rate", "min"), max_lesson=("yes_rate", "max"),
                         mean_within_lesson_sd=("sd_across_runs", "mean"),
                         n_band_unstable=("band_unstable", "sum"),
                         pct_band_unstable=("band_unstable", "mean"))
                    .reset_index())

    # ---------------- duplicate-transcript robustness ----------------
    dup_rows = []
    for cl, members in _cluster_members(clusters).items():
        members = [m for m in members if m in sessions]
        if len(members) < 2:
            continue
        for c in codes:
            v = []
            for m in members:
                cc = cells[(cells.session_id == m) & (cells.code == c)]
                v.append(cc.verdict.iloc[0] if len(cc) else "NA")
            dup_rows.append(dict(cluster=cl, code=c, indicator=CODE2NAME[c],
                                 verdicts="/".join(v),
                                 agrees=len(set(v)) == 1,
                                 members=",".join(m[:8] for m in members)))
    dup = pd.DataFrame(dup_rows)

    HEAD = {
        "n_transcripts": len(sessions),
        "n_distinct_lessons": len(primary),
        "runs_per_transcript": n_runs,
        "n_indicators": len(ind),
        "total_verdict_cells": int(df["score"].notna().sum()),
        "pooled_yes_rate": round(float(df["score"].mean(skipna=True)), 3),
        "healthy": int((ind.verdict_class == "HEALTHY").sum()),
        "noisy": int((ind.verdict_class == "NOISY").sum()),
        "uninformative": int((ind.verdict_class == "UNINFORMATIVE").sum()),
        "broken": int((ind.verdict_class == "BROKEN").sum()),
        "untested": int((ind.verdict_class == "UNTESTED").sum()),
        "mean_ac1": round(float(ind.ac1.mean(skipna=True)), 3),
        "mean_ac1_dedup": round(float(ind.ac1_dedup.mean(skipna=True)), 3),
        "indicators_flipping_somewhere": int((ind.n_lessons_flipped > 0).sum()),
        "mean_flip_rate": round(float(ind.mean_flip_rate.mean(skipna=True)), 3),
        "parse_failure_rate": round(float((df.parse == "none").mean()), 4)
        if "parse" in df else None,
    }
    if len(dup):
        HEAD["duplicate_pair_verdict_agreement"] = round(float(dup.agrees.mean()), 3)

    return dict(cells=cells, ind=ind, bands=bands, band_summary=band_summary,
                dup=dup, headline=HEAD, sessions=sessions, primary=primary,
                n_runs=n_runs)


def _cluster_members(clusters):
    out = {}
    for sid, cl in (clusters or {}).items():
        out.setdefault(cl, []).append(sid)
    return out
