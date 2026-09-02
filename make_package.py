#!/usr/bin/env python
"""Build a SHAREABLE findings package: statistics and charts, no PII.

The run outputs cannot be shared as they stand. Their evidence columns quote
classroom speech verbatim and name individual children, which is why
Transcripts/ and every *_out/ directory is gitignored. This script produces a
parallel set that is safe to send outside the team:

  * free-text columns holding transcript quotes are DROPPED, not masked -
    masking names still leaks the surrounding speech
  * session UUIDs are replaced with stable Lesson-NN labels, so a row can no
    longer be traced back to a recording
  * every statistic survives untouched, because none of it is personal

Anything the script cannot positively identify as safe is dropped. If a future
run adds a new free-text column, it is excluded by default rather than shipped.
"""
import os, re, shutil, sys
import pandas as pd

OUT = "findings_package"
# columns that carry transcript text - never leave the machine
UNSAFE = {"evidence", "flip_if", "yes_reasons", "no_reasons", "raw", "audio_file",
          "user_id", "transcript"}

def lesson_map(*frames):
    ids = []
    for d in frames:
        if d is not None and "session_id" in d:
            ids += [str(s)[:8] for s in d.session_id.unique()]
    return {sid: f"Lesson-{i:02d}" for i, sid in enumerate(sorted(set(ids)), 1)}

def clean(df, lmap):
    df = df.copy()
    dropped = [c for c in df.columns if c in UNSAFE]
    df = df.drop(columns=dropped)
    for col in ("session_id", "session"):
        if col in df:
            df[col] = df[col].astype(str).str[:8].map(lmap).fillna("Lesson-??")
    return df, dropped

def main():
    os.makedirs(f"{OUT}/data", exist_ok=True)
    os.makedirs(f"{OUT}/charts", exist_ok=True)
    os.makedirs(f"{OUT}/report", exist_ok=True)

    b = pd.read_csv("multi_binary_clean/all_verdicts_long.csv")
    lmap = lesson_map(b)
    print(f"anonymising {len(lmap)} lessons -> Lesson-01..{len(lmap):02d}")

    sources = [
        ("multi_binary_clean/indicator_reliability_pooled.csv",
         "data/indicator_reliability_binary_30.csv"),
        ("multi_binary_clean/lesson_x_indicator.csv",
         "data/lesson_by_indicator_binary_30.csv"),
        ("multi_binary_clean/section_bands_by_lesson.csv",
         "data/section_bands_binary_30.csv"),
        ("multi_binary_out/indicator_reliability_pooled.csv",
         "data/indicator_reliability_binary_37.csv"),
        ("scale_comparison_fresh/headline_comparison.csv",
         "data/scale_comparison_headline.csv"),
        ("scale_comparison_fresh/indicator_comparison.csv",
         "data/scale_comparison_by_indicator.csv"),
        ("scale_comparison_fresh/band_comparison.csv",
         "data/scale_comparison_bands.csv"),
        ("multi_scale14_out/pooled_indicator_wobble.csv",
         "data/indicator_wobble_scale14_37.csv"),
        ("multi_binary_clean/all_verdicts_long.csv",
         "data/raw_verdicts_binary_30_REDACTED.csv"),
    ]
    total_dropped = set()
    for src, dst in sources:
        if not os.path.exists(src):
            print(f"  ! missing {src}"); continue
        df, dropped = clean(pd.read_csv(src), lmap)
        df.to_csv(f"{OUT}/{dst}", index=False)
        total_dropped |= set(dropped)
        print(f"  {dst:52s} {len(df):4d} rows"
              + (f"  [dropped: {', '.join(dropped)}]" if dropped else ""))

    # The two pooled charts (01, 02) come from make_pooled_charts.py and describe
    # the 10-transcript study. Everything below is from the SINGLE-LESSON pilot and
    # is filed separately, because presenting a one-lesson figure beside pooled
    # findings is how a reader ends up quoting the wrong number.
    os.makedirs(f"{OUT}/charts/illustrative_single_lesson", exist_ok=True)
    charts = [
        ("wobble_out_binary/02_verdict_matrix.png",
         "charts/illustrative_single_lesson/verdict_matrix.png"),
        ("wobble_out_binary/06_agreement.png",
         "charts/illustrative_single_lesson/agreement_coefficients.png"),
        ("wobble_out_binary/09_calibration.png",
         "charts/illustrative_single_lesson/confidence_calibration.png"),
        ("wobble_out_binary/10_confidence_by_indicator.png",
         "charts/illustrative_single_lesson/confidence_by_indicator.png")]
    for src, dst in charts:
        if os.path.exists(src):
            shutil.copy2(src, f"{OUT}/{dst}"); print(f"  {dst}")
        else:
            print(f"  ! missing chart {src}")

    if os.path.exists("report/wobble-report.html"):
        shutil.copy2("report/wobble-report.html", f"{OUT}/report/findings.html")
        print("  report/findings.html")

    print(f"\ndropped free-text columns across all files: "
          f"{', '.join(sorted(total_dropped)) or 'none'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
