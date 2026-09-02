#!/usr/bin/env python
"""Head-to-head: 1-4 levels vs native YES/NO, on the same transcripts.

    python compare_scales.py --binary multi_binary_out --scale14 multi_scale14_out

WHY THE OBVIOUS COMPARISON WOULD BE WRONG
------------------------------------------
You cannot put Krippendorff's alpha from a 4-category scale next to alpha from a
2-category scale and declare a winner. Chance agreement depends on how many
categories there are, so the two coefficients have different baselines and the
binary scale wins almost by construction. That number would be meaningless.

The honest comparison is at the level of the DECISION, because both scales exist
to produce the same output: is this indicator met, yes or no.

    route A   ask for a level 1-4, then threshold it at >= 3
    route B   ask the threshold question directly

Collapse route A at the proficiency cut and both routes emit a binary verdict per
indicator per run. Now every statistic is computed on two-category data with the
same baseline, by the SAME code (multi_binary.analyse_multi), and the comparison
is genuinely like for like.

WHAT GETS COMPARED
    1. Verdict stability  - how often does a single pass contradict the majority?
    2. Per-indicator AC1  - reliability of each indicator under each route
    3. Band stability     - how often does a section's High/Medium/Low band move
                            between runs? This is what a coaching report quotes.
    4. Agreement          - where the two routes disagree on the verdict itself,
                            which one is the more stable of the two?
    5. Classification     - does an indicator come out HEALTHY / NOISY /
                            UNINFORMATIVE under both routes, or only one?

The native ordinal alpha of the 1-4 run is reported too, but clearly separated:
it describes a different measurement and must not be read as the comparison.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_multi_binary import detect_clusters, find_transcripts
from wobble_eval import exclusions, multi_binary
from wobble_eval.binary import BinaryConfig
from wobble_eval.framework import ALL_CODES, CODE2NAME, CODE2SECTION


def load_scale14(root, cut):
    """Pool every per-session scores_long.csv and collapse to binary at `cut`."""
    frames = []
    for f in sorted(glob.glob(os.path.join(root, "*", "scores_long.csv"))):
        d = pd.read_csv(f)
        if "session_id" not in d.columns:
            # run_multi.py names the per-session directory after the session id
            d["session_id"] = os.path.basename(os.path.dirname(f))
        frames.append(d)
    if not frames:
        raise SystemExit(f"no scores_long.csv under {root!r} — has the 1-4 run finished?")
    d = pd.concat(frames, ignore_index=True)
    d["level"] = pd.to_numeric(d["score"], errors="coerce")
    # THE COLLAPSE: level >= cut is a YES. NA stays NA — never silently a NO.
    d["score"] = np.where(d["level"].isna(), np.nan, (d["level"] >= cut).astype(float))
    if "parse" not in d:
        d["parse"] = "strict"
    return d


def load_binary(root):
    f = os.path.join(root, "all_verdicts_long.csv")
    if not os.path.exists(f):
        raise SystemExit(f"{f} not found — has the binary run finished?")
    d = pd.read_csv(f)
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    return d


def prep(d, cfg):
    d = d.copy()
    d["section"] = pd.Categorical(d["section"], list(cfg.SECTIONS), ordered=True)
    expected = [c for c in ALL_CODES if CODE2SECTION[c] in cfg.SECTIONS]
    d["code"] = pd.Categorical(d["code"], expected, ordered=True)
    # session ids are full uuids in one frame and may be directory names in the other
    d["session_id"] = d["session_id"].astype(str).str[:8]
    return d


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare 1-4 vs binary scoring head to head.")
    p.add_argument("--binary", default="multi_binary_out")
    p.add_argument("--scale14", default="multi_scale14_out")
    p.add_argument("--dir", default="Transcripts")
    p.add_argument("--cut", type=int, default=3, help="proficiency cut for the collapse")
    p.add_argument("--exclude", default="",
                   help="indicators to drop from scoring AND analysis: a named set "
                        "(unreliable | wording | unobservable | none) or a "
                        "comma-separated code list, e.g. C4,B2")
    p.add_argument("--out", default="scale_comparison")
    a = p.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    drop = exclusions.resolve(a.exclude)
    cfg = BinaryConfig(N_ITERATIONS=10, YES_AT=a.cut, EXCLUDE_CODES=tuple(drop))
    if drop:
        print(f"Excluding {len(drop)} indicator(s):")
        print(exclusions.describe(drop))
    sessions = [j for _, j in find_transcripts(a.dir)]
    clusters = {k[:8]: v for k, v in detect_clusters(sessions).items()}
    for s in sessions:                       # analyse_multi keys clusters by session_id
        s["session_id"] = s["session_id"][:8]

    b = prep(load_binary(a.binary), cfg)
    s14 = prep(load_scale14(a.scale14, a.cut), cfg)

    common = sorted(set(b.session_id) & set(s14.session_id))
    if not common:
        raise SystemExit("the two runs share no session ids — check --binary/--scale14")
    b = b[b.session_id.isin(common)]
    s14 = s14[s14.session_id.isin(common)]

    print(f"Comparing on {len(common)} transcripts · collapse at level >= {a.cut}")
    print(f"  binary   : {int(b.score.notna().sum()):5d} verdict cells")
    print(f"  1-4 coll.: {int(s14.score.notna().sum()):5d} verdict cells")

    res_b = multi_binary.analyse_multi(b, cfg, sessions, clusters)
    res_s = multi_binary.analyse_multi(s14, cfg, sessions, clusters)

    # ---------------------------------------------------------- headline
    def head(res, name):
        h = res["headline"]
        ind = res["ind"]
        return dict(
            route=name,
            yes_rate=h["pooled_yes_rate"],
            mean_ac1=h["mean_ac1"],
            mean_flip_rate=h["mean_flip_rate"],
            indicators_flipping_somewhere=h["indicators_flipping_somewhere"],
            healthy=h["healthy"], noisy=h["noisy"],
            uninformative=h["uninformative"], broken=h["broken"],
            mean_single_pass_error=round(float(
                res["cells"].single_pass_error.mean(skipna=True)), 4),
            pct_band_unstable=round(float(res["bands"].band_unstable.mean()), 4),
            n_band_unstable=int(res["bands"].band_unstable.sum()),
            n_band_readings=int(len(res["bands"])),
        )

    cmp_head = pd.DataFrame([head(res_s, "1-4 collapsed at >=%d" % a.cut),
                             head(res_b, "native binary")])
    cmp_head.to_csv(os.path.join(a.out, "headline_comparison.csv"), index=False)

    print("\n" + "=" * 78)
    print("HEAD TO HEAD — same decision, two routes")
    print("=" * 78)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(cmp_head.to_string(index=False))

    winner = lambda lo, hi, label: (
        f"  {label:34s} {'1-4' if lo < hi else 'binary' if hi < lo else 'tie':>8s}")
    print("\nLower is better on error/instability, higher is better on AC1:")
    hs, hb = cmp_head.iloc[0], cmp_head.iloc[1]
    print(f"  {'mean single-pass verdict error':34s} "
          f"1-4 {hs.mean_single_pass_error:.4f}  vs  binary {hb.mean_single_pass_error:.4f}"
          f"   -> {'1-4' if hs.mean_single_pass_error < hb.mean_single_pass_error else 'binary'}")
    print(f"  {'mean per-indicator AC1':34s} "
          f"1-4 {hs.mean_ac1:.4f}  vs  binary {hb.mean_ac1:.4f}"
          f"   -> {'1-4' if hs.mean_ac1 > hb.mean_ac1 else 'binary'}")
    print(f"  {'band readings unstable':34s} "
          f"1-4 {hs.n_band_unstable}/{hs.n_band_readings}  vs  "
          f"binary {hb.n_band_unstable}/{hb.n_band_readings}"
          f"   -> {'1-4' if hs.pct_band_unstable < hb.pct_band_unstable else 'binary'}")

    # ------------------------------------------------- per-indicator side by side
    keep = ["code", "indicator", "pooled_p_yes", "ac1", "mean_flip_rate",
            "n_lessons_flipped", "verdict_class"]
    m = (res_s["ind"][["section"] + keep].merge(
        res_b["ind"][keep], on=["code", "indicator"], suffixes=("_14", "_bin")))
    m["ac1_delta"] = m["ac1_bin"] - m["ac1_14"]
    m["flip_delta"] = m["mean_flip_rate_bin"] - m["mean_flip_rate_14"]
    m["class_changed"] = m["verdict_class_14"] != m["verdict_class_bin"]
    m.to_csv(os.path.join(a.out, "indicator_comparison.csv"), index=False)

    print("\n" + "=" * 78)
    print("PER-INDICATOR — AC1 under each route (positive delta favours binary)")
    print("=" * 78)
    show = ["section", "code", "indicator", "ac1_14", "ac1_bin", "ac1_delta",
            "mean_flip_rate_14", "mean_flip_rate_bin", "verdict_class_14",
            "verdict_class_bin"]
    with pd.option_context("display.width", 230, "display.max_columns", 30):
        print(m[show].to_string(index=False, na_rep="—",
              float_format=lambda v: f"{v:.3f}"))

    ch = m[m.class_changed]
    if len(ch):
        print(f"\n{len(ch)} indicator(s) change classification between routes:")
        print(ch[["code", "indicator", "verdict_class_14", "verdict_class_bin",
                  "ac1_14", "ac1_bin"]].to_string(index=False,
                  float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------------------------------ band comparison
    bb = res_b["bands"][["session_id", "section", "yes_rate", "band", "bands_seen",
                         "band_unstable"]]
    sb = res_s["bands"][["session_id", "section", "yes_rate", "band", "bands_seen",
                         "band_unstable"]]
    bands = sb.merge(bb, on=["session_id", "section"], suffixes=("_14", "_bin"))
    bands["band_agrees"] = bands["band_14"] == bands["band_bin"]
    bands.to_csv(os.path.join(a.out, "band_comparison.csv"), index=False)

    print("\n" + "=" * 78)
    print("BAND-WISE — does the reported High/Medium/Low hold, and do routes agree?")
    print("=" * 78)
    bs = (bands.groupby("section", observed=True)
          .agg(n=("session_id", "size"),
               unstable_14=("band_unstable_14", "sum"),
               unstable_bin=("band_unstable_bin", "sum"),
               routes_agree=("band_agrees", "sum")).reset_index())
    print(bs.to_string(index=False))
    print(f"\nThe two routes assign the SAME band on "
          f"{int(bands.band_agrees.sum())}/{len(bands)} lesson-section readings "
          f"({bands.band_agrees.mean()*100:.0f}%).")
    dis = bands[~bands.band_agrees]
    if len(dis):
        print("\nWhere they disagree:")
        print(dis[["session_id", "section", "yes_rate_14", "band_14",
                   "yes_rate_bin", "band_bin"]].to_string(index=False,
                   float_format=lambda v: f"{v:.3f}"))

    # -------------------------------------------------- verdict-level agreement
    j = (res_s["cells"][["session_id", "code", "verdict", "single_pass_error"]]
         .merge(res_b["cells"][["session_id", "code", "verdict", "single_pass_error"]],
                on=["session_id", "code"], suffixes=("_14", "_bin")))
    j = j[(j.verdict_14 != "NA") & (j.verdict_bin != "NA")]
    j["agree"] = j.verdict_14 == j.verdict_bin
    j.to_csv(os.path.join(a.out, "verdict_agreement.csv"), index=False)
    print("\n" + "=" * 78)
    print("VERDICT AGREEMENT BETWEEN ROUTES")
    print("=" * 78)
    print(f"Same verdict on {int(j.agree.sum())}/{len(j)} lesson-indicator cells "
          f"({j.agree.mean()*100:.0f}%).")
    d = j[~j.agree]
    if len(d):
        print(f"Where they disagree ({len(d)} cells), mean single-pass error was "
              f"{d.single_pass_error_14.mean():.3f} on the 1-4 route and "
              f"{d.single_pass_error_bin.mean():.3f} on binary "
              f"-> {'1-4' if d.single_pass_error_14.mean() < d.single_pass_error_bin.mean() else 'binary'} "
              f"was the steadier of the two where they parted company.")

    # ---------------------------------- native ordinal alpha, kept clearly apart
    hp = os.path.join(a.scale14, "pooled_headlines.csv")
    if os.path.exists(hp):
        try:
            hh = pd.read_csv(hp)
            if "kripp_alpha_ordinal" in hh:
                print("\n" + "-" * 78)
                print("FOR CONTEXT ONLY — the 1-4 run's NATIVE ordinal alpha "
                      f"(mean {hh.kripp_alpha_ordinal.mean():.3f}).")
                print("This is a 4-category coefficient and is NOT comparable with any "
                      "binary number\nabove; it describes how well the LEVEL reproduces, "
                      "not the verdict.")
        except Exception:
            pass

    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"comparison": cmp_head.to_dict("records"),
                   "band_agreement": float(bands.band_agrees.mean()),
                   "verdict_agreement": float(j.agree.mean()),
                   "n_transcripts": len(common), "cut": a.cut}, f, indent=2)
    print(f"\nAll output in ./{a.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
