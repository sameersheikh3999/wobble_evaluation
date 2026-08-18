#!/usr/bin/env python
"""Run the wobble experiment across a DIRECTORY of session transcripts, then
pool the results.

    python run_multi.py --dir Transcripts --iterations 10 --out wobble_multi
    python run_multi.py --dir Transcripts --reuse wobble_out_local   # resume/ingest
    python run_multi.py --analyse-only --out wobble_multi            # re-pool, no calls

Everything about a single session is unchanged — this drives `run_wobble.py`'s
`run_experiment` once per file and writes the usual per-session CSVs and seven
charts into `<out>/<session8>/`. What it adds is the cross-session layer.

WHY THAT LAYER IS NOT JUST "MORE N"
    Several files in a transcript drop are often the SAME lesson transcribed
    more than once (repeat ASR passes over one recording). Pooling them as if
    they were independent lessons inflates n and understates uncertainty. So
    this script fingerprints the transcripts, groups files that share utterance
    text into a `lesson` group, and reports two variance components separately:

      within-transcript SD  — same input, N passes  -> SAMPLING wobble
                              (what the single-session run already measures)
      between-transcript SD — same lesson, different transcription, per-file
                              mean score          -> TRANSCRIPTION wobble

    A pipeline's real error is both. If between > within, the ASR step, not the
    scorer, is the thing to fix — and no amount of averaging model passes helps.

Resume-safe: a session whose scores_long.csv already has all N iterations is
skipped, so a crashed run can be restarted with the same command.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_wobble import report, run_experiment
from wobble_eval import analysis, exclusions
from wobble_eval.config import Config
from wobble_eval.framework import ALL_CODES, CODE2NAME, CODE2SECTION
from wobble_eval.session import load_session, session_meta, turn_stats

SHORT = lambda sid: str(sid)[:8]


# ------------------------------------------------------- lesson fingerprinting
def utterances(transcript):
    """Set of utterance texts, timestamps stripped — timestamps drift between
    ASR passes even when the words do not, so they must not enter the key."""
    return {re.sub(r"^\[\d\d:\d\d\]\s*", "", u).strip()
            for u in transcript.split("\n\n") if u.strip()}


def duration_s(session):
    return float(session.get("ffprobe_duration_seconds")
                 or session.get("db_duration_seconds") or 0)


def group_lessons(sessions, threshold=0.35, dur_tol=0.5):
    """Union-find over TWO fingerprints. Returns {session_id: lesson_label}.

    1. Jaccard(utterance sets) >= 0.35. Deliberately loose: two ASR passes over
       one recording share most utterances verbatim (0.65-0.98 on this drop),
       while different lessons share almost none (<0.05).
    2. |Δ ffprobe_duration| <= 0.5 s. Necessary because signal 1 is blind to a
       re-transcription into a DIFFERENT LANGUAGE — the same audio rendered once
       in Urdu script and once in English has ~0 text overlap but identical
       duration to the millisecond. Audio duration is the ground truth for
       "same recording"; the transcript is only evidence about it.
    """
    ids = [s["session_id"] for s in sessions]
    sets = {s["session_id"]: utterances(s["transcript"]) for s in sessions}
    durs = {s["session_id"]: duration_s(s) for s in sessions}
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            sa, sb = sets[a], sets[b]
            jac = len(sa & sb) / max(1, len(sa | sb))
            dd = abs(durs[a] - durs[b])
            same_audio = durs[a] > 0 and dd <= dur_tol
            why = ("text+duration" if jac >= threshold and same_audio else
                   "text" if jac >= threshold else
                   "duration" if same_audio else "")
            pairs.append((SHORT(a), SHORT(b), round(jac, 3), round(dd, 3), why))
            if why:
                parent[find(a)] = find(b)

    # label groups by size then by the earliest member, so labels are stable
    members = {}
    for i in ids:
        members.setdefault(find(i), []).append(i)
    order = sorted(members.values(), key=lambda m: (-len(m), sorted(m)[0]))
    labels = {}
    for n, group in enumerate(order, 1):
        for i in group:
            labels[i] = f"L{n}"
    return labels, pd.DataFrame(pairs, columns=["a", "b", "jaccard",
                                                "duration_delta_s", "merged_by"])


# --------------------------------------------------------------- session runs
def already_done(out_dir, n_iterations):
    """True if this session's scoring finished — used to make the run resumable."""
    csv = os.path.join(out_dir, "scores_long.csv")
    if not os.path.exists(csv):
        return False
    try:
        df = pd.read_csv(csv)
    except Exception:
        return False
    return int(df.iteration.max()) + 1 >= n_iterations and len(df) > 0


def load_done(out_dir, cfg):
    df = pd.read_csv(os.path.join(out_dir, "scores_long.csv"))
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    expected = [c for c in ALL_CODES if CODE2SECTION[c] in cfg.SECTIONS]
    df["section"] = pd.Categorical(df["section"], list(cfg.SECTIONS), ordered=True)
    df["code"] = pd.Categorical(df["code"], expected, ordered=True)
    meta_path = os.path.join(out_dir, "run_meta.json")
    run_meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    return df, run_meta


# ------------------------------------------------------------ cross-session
def pool(per_session, cfg, outdir):
    """Per-indicator table pooled over sessions + the variance decomposition."""
    long = pd.concat(per_session, ignore_index=True)
    long["score"] = pd.to_numeric(long["score"], errors="coerce")

    # ---- one row per (session, indicator): its mean and its within-session SD
    cell = (long.dropna(subset=["score"])
            .groupby(["lesson", "session", "section", "code"], observed=True)["score"]
            .agg(n="size", mean="mean", sd=lambda x: x.std(ddof=1) if len(x) > 1 else 0.0,
                 lo="min", hi="max")
            .reset_index())
    cell["indicator"] = cell.code.map(CODE2NAME)
    cell.to_csv(os.path.join(outdir, "cell_session_indicator.csv"), index=False)

    # ---- pooled per-indicator view
    rows = []
    for code_, g in cell.groupby("code", observed=True):
        per_lesson = g.groupby("lesson", observed=True)["mean"].mean()
        rows.append(dict(
            section=CODE2SECTION[code_], code=code_, indicator=CODE2NAME[code_],
            n_sessions=len(g), n_lessons=g.lesson.nunique(),
            grand_mean=float(g["mean"].mean()),
            within_sd=float(g["sd"].mean()),               # sampling wobble
            between_session_sd=float(g["mean"].std(ddof=1)) if len(g) > 1 else np.nan,
            between_lesson_sd=(float(per_lesson.std(ddof=1))
                               if len(per_lesson) > 1 else np.nan),
            min_session_mean=float(g["mean"].min()),
            max_session_mean=float(g["mean"].max()),
            na_rate=float(1 - g["n"].sum() / (len(g) * cfg.N_ITERATIONS)),
        ))
    pooled = pd.DataFrame(rows)
    # An indicator the model returned NA for in EVERY pass of EVERY session was dropped by
    # the dropna above. It must still appear, at na_rate 1.0 — "never scorable from a
    # transcript" is a finding about the framework, not an absence of data.
    missing = [c for c in ALL_CODES
               if CODE2SECTION[c] in cfg.SECTIONS and c not in set(pooled.code)]
    if missing:
        pooled = pd.concat([pooled, pd.DataFrame([
            dict(section=CODE2SECTION[c], code=c, indicator=CODE2NAME[c],
                 n_sessions=0, n_lessons=0, na_rate=1.0) for c in missing])],
            ignore_index=True)
    pooled["signal_to_noise"] = pooled.between_lesson_sd / pooled.within_sd.replace(0, np.nan)
    pooled["section"] = pd.Categorical(pooled.section, list(cfg.SECTIONS), ordered=True)
    pooled["code"] = pd.Categorical(pooled.code,
                                    [c for c in ALL_CODES if CODE2SECTION[c] in cfg.SECTIONS],
                                    ordered=True)
    pooled = pooled.sort_values(["section", "code"])
    pooled.to_csv(os.path.join(outdir, "pooled_indicator_wobble.csv"), index=False)

    # ---- within vs between TRANSCRIPT, inside each multi-transcript lesson
    #      This is the ASR-vs-sampling split; only lessons with >=2 files qualify.
    dec_rows = []
    for (lesson, code_), g in cell.groupby(["lesson", "code"], observed=True):
        if len(g) < 2:
            continue
        within = float(np.sqrt((g["sd"] ** 2).mean()))      # pooled SD across passes
        between = float(g["mean"].std(ddof=1))              # SD of per-transcript means
        dec_rows.append(dict(lesson=lesson, section=CODE2SECTION[code_], code=code_,
                             indicator=CODE2NAME[code_], n_transcripts=len(g),
                             within_transcript_sd=within, between_transcript_sd=between,
                             ratio=between / within if within > 0 else np.nan,
                             total_sd=float(np.sqrt(within ** 2 + between ** 2))))
    decomp = pd.DataFrame(dec_rows)
    if len(decomp):
        decomp["section"] = pd.Categorical(decomp.section, list(cfg.SECTIONS), ordered=True)
        decomp = decomp.sort_values(["lesson", "section", "code"])
        decomp.to_csv(os.path.join(outdir, "variance_decomposition.csv"), index=False)
    return long, cell, pooled, decomp


def cross_charts(headlines, cell, pooled, decomp, cfg, outdir):
    """Three figures that only make sense across sessions. Same palette as the
    per-session charts so the deck reads as one system."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from wobble_eval.charts import (BASELINE, INK, INK2, MUTED, ORDINAL, SERIES,
                                    SURFACE, caption, save, style, titles)

    made = []

    def emit(fig, name):
        save(fig, outdir, name)
        plt.close(fig)
        made.append(name)
    lesson_color = {}
    for i, l in enumerate(sorted(headlines.lesson.unique())):
        lesson_color[l] = list(SERIES.values())[i % len(SERIES)]

    # --- A: alpha per session ------------------------------------------------
    d = headlines.sort_values(["lesson", "session"])
    fig, ax = plt.subplots(figsize=(9.2, 0.42 * len(d) + 2.4))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d.kripp_alpha_ordinal, height=.62,
            color=[lesson_color[l] for l in d.lesson], zorder=3)
    for thr, lab in ((0.80, "0.80 dependable"), (0.67, "0.67 tentative")):
        ax.axvline(thr, color=BASELINE, lw=1.1, ls="--", zorder=2)
        ax.text(thr, len(d) - .2, lab, fontsize=7.5, color=MUTED, ha="center", va="bottom")
    for yy, r in zip(y, d.itertuples()):
        ax.text(r.kripp_alpha_ordinal + .008, yy, f"{r.kripp_alpha_ordinal:.3f}",
                va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.lesson} · {r.session}  ({r.duration_min:.0f} min)"
                        for r in d.itertuples()], fontsize=8.5)
    ax.set_xlim(0, max(1.0, d.kripp_alpha_ordinal.max() * 1.12))
    ax.set_xlabel("Krippendorff's α (ordinal)")
    style(ax, xgrid=True)
    titles(ax, "Reliability holds across transcripts?",
           f"{len(d)} transcripts · {headlines.lesson.nunique()} distinct lessons · "
           f"{cfg.N_ITERATIONS} passes each · {cfg.MODEL} effort={cfg.EFFORT}")
    caption(fig, ax, "Bars sharing a colour are re-transcriptions of the same lesson. "
                     "α is computed within a transcript, so a spread between same-colour "
                     "bars is transcription-driven, not sampling-driven.")
    emit(fig, "10_alpha_by_session")

    # --- B: within vs between transcript variance ----------------------------
    if len(decomp):
        fig, ax = plt.subplots(figsize=(7.6, 7.0))
        lim = max(decomp.within_transcript_sd.max(), decomp.between_transcript_sd.max()) * 1.15
        lim = max(lim, 0.1)
        ax.plot([0, lim], [0, lim], color=BASELINE, lw=1.1, ls="--", zorder=2)
        ax.text(lim * .97, lim * .93, "equal", fontsize=8, color=MUTED, ha="right")
        for sec, g in decomp.groupby("section", observed=True):
            ax.scatter(g.within_transcript_sd, g.between_transcript_sd, s=46,
                       color=SERIES.get(sec, INK2), alpha=.85, zorder=3,
                       edgecolor=SURFACE, linewidth=.8, label=f"Section {sec}")
        worst = decomp.nlargest(6, "between_transcript_sd")
        for r in worst.itertuples():
            ax.annotate(r.code, (r.within_transcript_sd, r.between_transcript_sd),
                        xytext=(5, 4), textcoords="offset points", fontsize=7.5, color=INK2)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("within-transcript SD  (sampling wobble, N passes on one transcript)")
        ax.set_ylabel("between-transcript SD  (transcription wobble, same lesson)")
        style(ax, xgrid=True, ygrid=True)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        titles(ax, "Which noise source dominates?",
               "one point per indicator, per multi-transcript lesson")
        caption(fig, ax, "Above the dashed line, re-transcribing the same lesson moves the "
                         "score more than re-running the model does — averaging more model "
                         "passes cannot fix those indicators.")
        emit(fig, "11_within_vs_between")

    # --- C: indicator mean by session ---------------------------------------
    piv = cell.pivot_table(index="code", columns="session", values="mean", observed=True)
    piv = piv.reindex([c for c in ALL_CODES if c in piv.index])
    fig, ax = plt.subplots(figsize=(max(7.5, 1.6 + .62 * piv.shape[1]), .30 * len(piv) + 2.2))
    cmap = LinearSegmentedColormap.from_list("wobble_seq", ORDINAL)   # CVD-safe single hue
    im = ax.imshow(piv.to_numpy(float), aspect="auto", cmap=cmap, vmin=1, vmax=4)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([f"{c} {CODE2NAME[c][:34]}" for c in piv.index], fontsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=.028, pad=.02)
    cb.set_label("mean score across passes", fontsize=8)
    cb.outline.set_visible(False)
    titles(ax, "Per-indicator mean score, every transcript",
           f"{piv.shape[1]} transcripts × {len(piv)} indicators · mean of "
           f"{cfg.N_ITERATIONS} passes")
    caption(fig, ax, "Columns from the same lesson should look alike. Where they do not, "
                     "the transcription — not the lesson — is driving the score.")
    emit(fig, "12_indicator_by_session")
    return made


# ----------------------------------------------------------------------- main
def main(argv=None):
    cfg0 = Config()
    p = argparse.ArgumentParser(
        description="Wobble evaluation across a directory of session transcripts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dir", default="Transcripts", help="directory of session JSON files")
    p.add_argument("--out", default="wobble_multi", help="root output directory")
    p.add_argument("--iterations", "-n", type=int, default=cfg0.N_ITERATIONS)
    p.add_argument("--model", default=cfg0.MODEL)
    p.add_argument("--effort", default=cfg0.EFFORT,
                   choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--sections", default=",".join(cfg0.SECTIONS))
    p.add_argument("--reuse", default="",
                   help="an existing finished run dir (e.g. wobble_out_local) to ingest "
                        "instead of re-scoring that session")
    p.add_argument("--per-session-charts", action="store_true", default=True)
    p.add_argument("--no-per-session-charts", dest="per_session_charts",
                   action="store_false")
    p.add_argument("--analyse-only", action="store_true",
                   help="skip all model calls; re-pool whatever is already in --out")
    p.add_argument("--exclude", default="",
                   help="indicators to drop from scoring AND analysis: a named "
                        "set (unreliable | wording | unobservable | none) or a "
                        "comma-separated code list")
    p.add_argument("--limit", type=int, default=0, help="score at most N sessions (smoke test)")
    a = p.parse_args(argv)
    _drop = tuple(exclusions.resolve(a.exclude))
    if _drop:
        print(f"EXCLUDED - {len(_drop)} indicator(s) dropped from scoring "
              f"and analysis:")
        print(exclusions.describe(_drop))

    sections = tuple(s.strip().upper() for s in a.sections.split(",") if s.strip())
    os.makedirs(a.out, exist_ok=True)

    # ---- discover + fingerprint -------------------------------------------
    paths = sorted(glob.glob(os.path.join(a.dir, "*.json")))
    if not paths:
        p.error(f"no JSON files in {a.dir}")
    sessions = []
    for path in paths:
        s = load_session(path)
        if not s.get("transcript"):
            print(f"  ! skipping {os.path.basename(path)} — no 'transcript' key")
            continue
        s["_path"] = path
        sessions.append(s)

    labels, pairs = group_lessons(sessions)
    pairs.to_csv(os.path.join(a.out, "transcript_similarity.csv"), index=False)
    n_lessons = len(set(labels.values()))

    print("=" * 78)
    print(f"{len(sessions)} transcripts in {a.dir}/  ->  {n_lessons} distinct lesson(s)")
    print("=" * 78)
    for lesson in sorted(set(labels.values())):
        mem = [s for s in sessions if labels[s["session_id"]] == lesson]
        note = ("  <- SAME RECORDING, re-transcribed" if len(mem) > 1 else "")
        print(f"  {lesson}: {len(mem)} transcript(s){note}")
        for s in mem:
            st = turn_stats(s["transcript"])
            print(f"      {SHORT(s['session_id'])}  {st['chars']:>6} ch  "
                  f"{st['n_turns']:>4} turns  {duration_s(s):>9.3f} s  "
                  f"lang={s.get('language', '?')}")
    if (pairs.merged_by == "duration").any():
        print("\n  merged on audio duration alone (different transcription language):")
        for r in pairs[pairs.merged_by == "duration"].itertuples():
            print(f"      {r.a} + {r.b}   text overlap {r.jaccard:.2f}, "
                  f"duration differs by {r.duration_delta_s:.3f}s")

    # ---- score each session ------------------------------------------------
    reuse_id = ""
    if a.reuse:
        rm = json.load(open(os.path.join(a.reuse, "run_meta.json")))
        reuse_id = rm.get("session_id", "")
        print(f"\nreusing {a.reuse}/ for session {SHORT(reuse_id)} "
              f"({rm.get('N_ITERATIONS')} iterations, effort={rm.get('EFFORT')})")

    per_session, headline_rows, failed = [], [], []
    todo = [s for s in sessions if s["session_id"] != reuse_id]
    if a.limit:
        todo = todo[:a.limit]
    t_all = time.time()

    for k, s in enumerate(sessions, 1):
        sid = s["session_id"]
        short = SHORT(sid)
        if a.limit and s not in todo and sid != reuse_id:
            continue
        out_dir = a.reuse if sid == reuse_id and a.reuse else os.path.join(a.out, short)
        cfg = Config(MODEL=a.model, EFFORT=a.effort, N_ITERATIONS=a.iterations,
                     SECTIONS=sections, MAX_CONCURRENCY=a.concurrency,
                     EXCLUDE_CODES=_drop,
                     OUT_DIR=out_dir, SESSION_PATH=s["_path"])
        meta = session_meta(s)

        if a.analyse_only or already_done(out_dir, a.iterations):
            if not os.path.exists(os.path.join(out_dir, "scores_long.csv")):
                print(f"\n[{k}/{len(sessions)}] {short} — nothing scored yet, skipping")
                continue
            print(f"\n[{k}/{len(sessions)}] {short} — already scored, re-using")
            scores, run_meta = load_done(out_dir, cfg)
            cfg.N_ITERATIONS = int(scores.iteration.max()) + 1
        else:
            print(f"\n[{k}/{len(sessions)}] {short} — scoring "
                  f"({a.iterations} iterations x {len(sections)} sections)")
            scores, run_meta, meta = run_experiment(cfg, s, label=f"{labels[sid]} {short}")

        # One transcript failing (rate limit, a refusal, a bad ASR dump) must not
        # discard the transcripts already scored. Skip it, keep its raw CSV on disk
        # for --resume, and carry on with the rest.
        try:
            res = analysis.analyse(scores, cfg, run_meta)
        except Exception as exc:
            failed.append(short)
            print(f"  ! {short} produced no analysable scores "
                  f"({type(exc).__name__}: {exc}).")
            print(f"    Skipping it and continuing; re-run with the same command to "
                  f"retry just this session.")
            continue
        if a.per_session_charts:
            report(res, cfg, meta, run_meta)
        else:
            res["ind_stats"].to_csv(os.path.join(out_dir, "indicator_wobble.csv"), index=False)
            res["sec_stats"].to_csv(os.path.join(out_dir, "section_wobble.csv"), index=False)
            with open(os.path.join(out_dir, "headline.json"), "w") as f:
                json.dump(res["headline"], f, indent=2)

        scores = scores.copy()
        scores["session"] = short
        scores["lesson"] = labels[sid]
        per_session.append(scores)
        st = turn_stats(s["transcript"])
        row = dict(lesson=labels[sid], session=short, session_id=sid, out_dir=out_dir,
                   chars=st["chars"], n_turns=st["n_turns"])
        # meta and headline overlap (session_id, duration_min, ...) — first writer wins,
        # otherwise dict(**a, **b) raises on the duplicate key.
        for source in (meta, res["headline"]):
            row.update({k2: v for k2, v in source.items()
                        if k2 not in row and not isinstance(v, (list, dict))})
        headline_rows.append(row)

    if not per_session:
        print("\nnothing to pool.")
        return 1

    if failed:
        print("")
        print(f"  ! {len(failed)} transcript(s) produced no usable scores "
              f"and were EXCLUDED from the pool: {', '.join(failed)}")
        print("    Every pooled figure below rests on the remaining "
              "transcripts only.")

    # ---- pool ---------------------------------------------------------------
    cfg = Config(MODEL=a.model, EFFORT=a.effort, N_ITERATIONS=a.iterations,
                 SECTIONS=sections, EXCLUDE_CODES=_drop, OUT_DIR=a.out)
    headlines = pd.DataFrame(headline_rows)
    headlines.to_csv(os.path.join(a.out, "sessions_headline.csv"), index=False)
    long, cell, pooled, decomp = pool(per_session, cfg, a.out)
    long.to_csv(os.path.join(a.out, "scores_long_all.csv"), index=False)

    made = cross_charts(headlines, cell, pooled, decomp, cfg, a.out)

    # ---- print --------------------------------------------------------------
    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.3f}"
    print("\n" + "=" * 78 + "\nPER-SESSION HEADLINE\n" + "=" * 78)
    cols = ["lesson", "session", "duration_min", "overall_mean", "mean_indicator_sd",
            "kripp_alpha_ordinal", "icc21", "pct_indicators_fully_stable",
            "n_indicators_flipping_proficiency", "na_rate"]
    print(headlines[[c for c in cols if c in headlines]].to_string(index=False,
                                                                   float_format=fmt))

    print("\n" + "=" * 78 + "\nRELIABILITY ACROSS TRANSCRIPTS\n" + "=" * 78)
    al = headlines.kripp_alpha_ordinal
    # alpha is undefined for a session with fewer than two usable runs; idxmin/idxmax
    # raise on an all-NA column, which would throw away a completed scoring run at the
    # very last step. Report what is computable and say how many were not.
    if al.notna().any():
        print(f"  alpha: mean {al.mean():.3f} | "
              f"min {al.min():.3f} ({headlines.loc[al.idxmin(), 'session']}) "
              f"| max {al.max():.3f} ({headlines.loc[al.idxmax(), 'session']})")
        if al.isna().any():
            print(f"  ! {int(al.isna().sum())} transcript(s) had too few usable runs to "
                  f"compute alpha and are excluded from that mean.")
    else:
        print("  alpha: not computable for any transcript (needs >= 2 usable runs each).")
    print(f"  transcripts at alpha >= 0.80 (indicator-level reportable): "
          f"{int((al >= .80).sum())}/{int(al.notna().sum())}")
    print(f"  transcripts at alpha >= 0.67 (section-level only)        : "
          f"{int((al >= .67).sum())}/{int(al.notna().sum())}")
    om = headlines.overall_mean
    if om.notna().any():
        print(f"  overall mean score across transcripts: {om.mean():.3f} "
              f"(SD {om.std(ddof=1):.3f}, range {om.min():.2f}-{om.max():.2f})")

    if len(decomp):
        w = decomp.within_transcript_sd
        b = decomp.between_transcript_sd
        print("\n" + "=" * 78 + "\nVARIANCE DECOMPOSITION (lessons with >1 transcript)\n" + "=" * 78)
        print(f"  mean within-transcript SD  (sampling wobble)     : {w.mean():.3f}")
        print(f"  mean between-transcript SD (transcription wobble): {b.mean():.3f}")
        print(f"  indicators where transcription > sampling        : "
              f"{int((b > w).sum())}/{len(decomp)} ({(b > w).mean():.0%})")
        print("\n  worst 10 by transcription wobble:")
        print(decomp.nlargest(10, "between_transcript_sd")[
            ["lesson", "code", "indicator", "n_transcripts", "within_transcript_sd",
             "between_transcript_sd", "ratio"]].to_string(index=False, float_format=fmt))

    print("\n" + "=" * 78 + "\nPOOLED PER-INDICATOR (all transcripts)\n" + "=" * 78)
    print(pooled[["section", "code", "indicator", "n_sessions", "grand_mean", "within_sd",
                  "between_session_sd", "between_lesson_sd", "signal_to_noise", "na_rate"]]
          .to_string(index=False, float_format=fmt))

    print(f"\nCross-session charts ({len(made)}): " + ", ".join(made))
    print(f"Everything in ./{a.out}/  (per-session subdirs + pooled CSVs)")
    print(f"Total wall time {(time.time() - t_all) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
