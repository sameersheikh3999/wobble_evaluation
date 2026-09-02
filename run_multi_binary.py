#!/usr/bin/env python
"""Binary (YES/NO) wobble across a DIRECTORY of transcripts, pooled.

    python run_multi_binary.py --dir Transcripts -n 10
    python run_multi_binary.py --dir Transcripts -n 10 --resume
    python run_multi_binary.py --analyse-only multi_binary_out/all_verdicts_long.csv

Scores every transcript N times with the same rubric and prompt, then answers the
question a single session cannot: which INDICATORS are problematic, as opposed to
which lessons were borderline.

Each indicator gets its own lessons x runs matrix, hence its own Gwet AC1, and two
Holm-corrected tests:

    RELIABILITY    pooled exact binomial of within-lesson disagreement against the
                   negligible-noise floor. Fail = the indicator does not reproduce.
    DISCRIMINATION chi-square of homogeneity of per-lesson YES counts. Fail = the
                   indicator gives every lesson the same answer, so it carries no
                   coaching signal however reliable it is.

    HEALTHY        reproduces AND separates lessons
    NOISY          separates lessons but does not reproduce  -> fix the wording
    UNINFORMATIVE  reproduces but answers identically everywhere -> carries no signal
    BROKEN         neither

Section fidelity bands (>=85% High / 60-84% Medium / <60% Low) are tracked per
lesson AND per run, so you can see how often the band a coaching report would
quote is decided by which pass happened to run.

Crash-safe: the pooled CSV is rewritten after every transcript, and --resume skips
transcripts already present in it.
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import sys
import textwrap
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_wobble_binary import run_iteration
from wobble_eval import binary, exclusions, multi_binary
from wobble_eval.backend import make_backend
from wobble_eval.binary import BinaryConfig
from wobble_eval.framework import ALL_CODES, CODE2SECTION, FRAMEWORK
from wobble_eval.session import load_session, session_meta, turn_stats


# ------------------------------------------------------------------ discovery
def find_transcripts(d):
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    out = []
    for f in files:
        try:
            j = json.load(open(f, encoding="utf-8"))
        except Exception as exc:
            print(f"  ! skipping {os.path.basename(f)}: {type(exc).__name__}: {exc}")
            continue
        if not j.get("transcript"):
            print(f"  ! skipping {os.path.basename(f)}: no 'transcript' field")
            continue
        out.append((f, j))
    return out


def detect_clusters(sessions, thresh=0.75):
    """Group transcripts that are the same audio transcribed twice.

    Duration matching to the millisecond is the strong signal; text similarity is
    the confirmation. Such lessons are NOT independent observations, and pooling
    them as if they were would overstate how well the findings generalise.
    """
    ids = [s["session_id"] for s in sessions]
    dur = {s["session_id"]: (s.get("ffprobe_duration_seconds")
                             or s.get("db_duration_seconds")) for s in sessions}
    txt = {s["session_id"]: s.get("transcript", "") for s in sessions}
    clusters, nxt = {}, 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            same_dur = (dur[a] is not None and dur[b] is not None
                        and abs(float(dur[a]) - float(dur[b])) < 0.01)
            if not same_dur:
                continue
            r = difflib.SequenceMatcher(None, txt[a][:6000], txt[b][:6000]).ratio()
            if r < thresh:
                continue
            cl = clusters.get(a, clusters.get(b))
            if cl is None:
                cl = f"dup{nxt}"; nxt += 1
            clusters[a] = clusters[b] = cl
    return clusters


# -------------------------------------------------------------------- scoring
def score_transcript(cfg, session, backend, label=""):
    meta = session_meta(session)
    ctx = session["transcript"]
    rows = []
    t0 = time.time()
    for it in range(cfg.N_ITERATIONS):
        r, _ = run_iteration(backend, cfg, it, ctx, meta)
        for row in r:
            row["session_id"] = meta["session_id"]
            row["language"] = meta["language"]
        rows += r
        done = it + 1
        el = time.time() - t0
        print(f"      [{done}/{cfg.N_ITERATIONS}] {el/done:.0f}s/run · "
              f"{el/60:.1f} min elapsed{label}")
    return rows


# --------------------------------------------------------------------- report
def report(res, cfg, out):
    def w(df, name, **kw):
        try:
            df.to_csv(os.path.join(out, name), **kw)
        except PermissionError:
            print(f"  ! could not write {name} — open in another program.")

    w(res["ind"], "indicator_reliability_pooled.csv", index=False)
    w(res["cells"], "lesson_x_indicator.csv", index=False)
    w(res["bands"], "section_bands_by_lesson.csv", index=False)
    w(res["band_summary"], "section_band_summary.csv", index=False)
    if len(res["dup"]):
        w(res["dup"], "duplicate_transcript_check.csv", index=False)
    try:
        with open(os.path.join(out, "headline.json"), "w") as f:
            json.dump(res["headline"], f, indent=2)
    except PermissionError:
        pass

    h = res["headline"]
    print("\n" + "=" * 78)
    print("POOLED HEADLINE — BINARY ACROSS TRANSCRIPTS")
    print("=" * 78)
    print(json.dumps(h, indent=2))

    ind = res["ind"]
    print("\n" + "=" * 78)
    print("INDICATOR CLASSIFICATION")
    print("=" * 78)
    print("  HEALTHY       reproduces AND separates lessons")
    print("  NOISY         separates lessons but does NOT reproduce -> fix the wording")
    print("  UNINFORMATIVE reproduces but gives every lesson the same answer")
    print("  BROKEN        neither")
    print("  UNTESTED      too few scored lessons to test\n")
    cols = ["section", "code", "indicator", "n_lessons", "pooled_p_yes", "ac1",
            "mean_flip_rate", "n_lessons_flipped", "lesson_yes_sd", "n_lessons_yes",
            "n_lessons_no", "q_unstable", "q_discriminates", "verdict_class"]
    with pd.option_context("display.width", 230, "display.max_columns", 40):
        print(ind[cols].to_string(index=False, na_rep="—",
              float_format=lambda v: f"{v:.3f}"))

    for cls, blurb in (("NOISY", "the rubric wording is the problem — these separate "
                                 "lessons but will not reproduce"),
                       ("BROKEN", "no reliable signal at all"),
                       ("UNINFORMATIVE", "same answer on every lesson — no coaching "
                                         "signal, however stable")):
        sub = ind[ind.verdict_class == cls]
        if not len(sub):
            continue
        print(f"\n{'=' * 78}\n{cls} — {blurb}\n{'=' * 78}")
        for r in sub.itertuples():
            bits = [f"AC1 {r.ac1:.2f}" if r.ac1 == r.ac1 else "AC1 —",
                    f"flip {r.mean_flip_rate*100:.0f}%",
                    f"moved on {int(r.n_lessons_flipped)}/{int(r.n_lessons)} lessons",
                    f"YES on {int(r.n_lessons_yes)} · NO on {int(r.n_lessons_no)}"]
            if r.conf_when_yes == r.conf_when_yes and r.conf_when_no == r.conf_when_no:
                bits.append(f"conf Y {r.conf_when_yes:.2f} / N {r.conf_when_no:.2f}")
            print(f"\n{r.code}  {r.indicator}\n   [{' · '.join(bits)}]")
            print("   THRESHOLD: " + "\n              ".join(
                textwrap.wrap(str(r.yes_bar), 70)))

    print("\n" + "=" * 78)
    print("SECTION FIDELITY BANDS")
    print("=" * 78)
    with pd.option_context("display.width", 200):
        print(res["band_summary"].to_string(index=False,
              float_format=lambda v: f"{v:.3f}"))
    print("\nPer lesson:")
    piv = res["bands"].pivot_table(index="session_id", columns="section", values="yes_rate",
                                   observed=True, aggfunc="first")
    bnd = res["bands"].pivot_table(index="session_id", columns="section", values="band",
                                   observed=True, aggfunc="first")
    with pd.option_context("display.width", 200):
        show = piv.round(3).astype(str) + " " + bnd.astype(str)
        print(show.to_string())
    unstable = res["bands"][res["bands"].band_unstable]
    if len(unstable):
        print(f"\n{len(unstable)} of {len(res['bands'])} lesson-section band readings are "
              f"NOT stable across runs:")
        print(unstable[["session_id", "section", "yes_rate", "min_run", "max_run",
                        "bands_seen"]].to_string(index=False,
                        float_format=lambda v: f"{v:.3f}"))

    if len(res["dup"]):
        d = res["dup"]
        print("\n" + "=" * 78)
        print("DUPLICATE-TRANSCRIPT ROBUSTNESS — same audio, transcribed twice")
        print("=" * 78)
        print(f"Verdicts agreed across the duplicate pair on "
              f"{d.agrees.sum()}/{len(d)} indicator comparisons "
              f"({d.agrees.mean()*100:.0f}%).")
        dis = d[~d.agrees]
        if len(dis):
            print("\nIndicators whose verdict changed with the TRANSCRIPTION alone — "
                  "instability\nthat has nothing to do with model sampling:")
            print(dis[["cluster", "code", "indicator", "verdicts"]].to_string(index=False))


# ----------------------------------------------------------------------- main
def main(argv=None):
    cfg0 = BinaryConfig()
    p = argparse.ArgumentParser(
        description="Binary YES/NO wobble pooled across a directory of transcripts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dir", default="Transcripts", help="directory of session JSON files")
    p.add_argument("--iterations", "-n", type=int, default=10)
    p.add_argument("--yes-at", type=int, default=cfg0.YES_AT, choices=[2, 3, 4])
    p.add_argument("--model", default=cfg0.MODEL)
    p.add_argument("--effort", default=cfg0.EFFORT,
                   choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--sections", default=",".join(cfg0.SECTIONS))
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--no-explain", action="store_true")
    p.add_argument("--exclude", default="",
                   help="indicators to drop from scoring AND analysis: a named set "
                        "(unreliable | wording | unobservable | none) or a "
                        "comma-separated code list, e.g. C4,B2")
    p.add_argument("--out", default="multi_binary_out")
    p.add_argument("--resume", action="store_true",
                   help="skip transcripts already scored in the pooled CSV")
    p.add_argument("--analyse-only", default="",
                   help="skip scoring; re-analyse an existing all_verdicts_long.csv")
    a = p.parse_args(argv)

    cfg = BinaryConfig(
        MODEL=a.model, EFFORT=a.effort, N_ITERATIONS=a.iterations,
        SECTIONS=tuple(s.strip().upper() for s in a.sections.split(",") if s.strip()),
        MAX_CONCURRENCY=a.concurrency, EXPLAIN=not a.no_explain,
        EXCLUDE_CODES=tuple(exclusions.resolve(a.exclude)),
        YES_AT=a.yes_at, OUT_DIR=a.out)
    for s in cfg.SECTIONS:
        if s not in FRAMEWORK:
            p.error(f"unknown section {s!r}")
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    pooled_csv = os.path.join(cfg.OUT_DIR, "all_verdicts_long.csv")

    found = find_transcripts(a.dir)
    if not found:
        p.error(f"no usable transcripts in {a.dir!r}")
    sessions = [j for _, j in found]
    clusters = detect_clusters(sessions)

    if a.analyse_only:
        df = pd.read_csv(a.analyse_only)
    else:
        if cfg.EXCLUDE_CODES:
            n_ex = len(cfg.EXCLUDE_CODES)
            print("")
            print(f"  EXCLUDED - {n_ex} indicator(s) dropped from scoring "
                  f"and analysis:")
            print(exclusions.describe(cfg.EXCLUDE_CODES))
        print(f"\n{'=' * 78}\nBINARY WOBBLE ACROSS {len(found)} TRANSCRIPTS\n{'=' * 78}")
        print(f"  {cfg.MODEL} · effort={cfg.EFFORT} · {cfg.N_ITERATIONS} runs each · "
              f"YES bar = level {cfg.YES_AT}+")
        n_cl = len(set(clusters.values()))
        if clusters:
            print(f"  ! {len(clusters)} transcripts form {n_cl} duplicate cluster(s) — "
                  f"same audio transcribed twice.\n"
                  f"    Kept, but marked: pooled figures are reported with AND without "
                  f"them.")
        for f, j in found:
            m = session_meta(j)
            tag = f"  [{clusters[m['session_id']]}]" if m["session_id"] in clusters else ""
            print(f"    {m['session_id'][:8]} · {m['language']:8s} · "
                  f"{m['duration_min']:5.1f} min · {turn_stats(j['transcript'])['n_turns']:4d} "
                  f"turns{tag}")
        total = len(found) * cfg.N_ITERATIONS * len(cfg.SECTIONS)
        print(f"\n  {total} model calls total. Ctrl-C is safe: the pooled CSV is "
              f"rewritten after every transcript,\n  and --resume picks up where it "
              f"stopped.")

        done_ids = set()
        records = []
        if a.resume and os.path.exists(pooled_csv):
            prev = pd.read_csv(pooled_csv)
            records = prev.to_dict("records")
            done_ids = set(prev.session_id.unique())
            print(f"\n  --resume: {len(done_ids)} transcript(s) already scored, skipping.")

        backend = make_backend(cfg)
        t_all = time.time()
        for i, (f, j) in enumerate(found, 1):
            m = session_meta(j)
            sid = m["session_id"]
            if sid in done_ids:
                print(f"\n[{i}/{len(found)}] {sid[:8]} — already done, skipped")
                continue
            print(f"\n[{i}/{len(found)}] {sid[:8]} · {m['language']} · "
                  f"{m['duration_min']} min")
            records += score_transcript(cfg, j, backend,
                                        label=f"  ({i}/{len(found)})")
            pd.DataFrame(records).to_csv(pooled_csv, index=False)   # crash-safe
            el = (time.time() - t_all) / 60
            print(f"    -> {sid[:8]} done · {el:.1f} min elapsed · "
                  f"{backend.calls} calls so far")
        df = pd.DataFrame(records)
        print(f"\nSCORING DONE in {(time.time() - t_all)/60:.1f} min · "
              f"{backend.calls} model calls")

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["section"] = pd.Categorical(df["section"], list(cfg.SECTIONS), ordered=True)
    expected = [c for c in ALL_CODES if CODE2SECTION[c] in cfg.SECTIONS]
    df["code"] = pd.Categorical(df["code"], expected, ordered=True)

    res = multi_binary.analyse_multi(df, cfg, sessions, clusters)
    report(res, cfg, cfg.OUT_DIR)
    print(f"\nAll output in ./{cfg.OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
