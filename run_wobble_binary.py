#!/usr/bin/env python
"""Run the Coaching Framework wobble experiment on a BINARY (YES / NO) scale,
scoring with Claude through the Claude Code subscription already logged in on
this machine.

    python run_wobble_binary.py -n 10
    python run_wobble_binary.py -n 10 --yes-at 2        # "observed at all" bar
    python run_wobble_binary.py -n 6  --effort xhigh --out wobble_bin_xhigh
    python run_wobble_binary.py --sweep-effort low,medium,high -n 4

Same 37 indicators, same transcript, same experiment hygiene as run_wobble.py —
the model just answers YES or NO instead of choosing a level 1-4.

    YES  the level-`--yes-at` descriptor (default 3, "Proficient / Effective")
         is CLEARLY met, or a stronger level is
    NO   anything below that bar
    NA   the indicator genuinely cannot apply (F5 MATH in a reading lesson)

WHY RUN THIS AS WELL AS THE 1-4 VERSION
    On the 1-4 scale most wobble is adjacent — a 3 becomes a 4 — and adjacent
    wobble rarely changes what a coach does. Binary strips that cushion out, so
    every disagreement measured here is a disagreement that changes the coaching
    decision. Expect the agreement coefficients to look better (two categories
    are easier to agree in) while the count of decision-relevant flips stays
    about the same. The pair of numbers is the finding; neither alone is.

    Read HEADLINE `gwet_ac1` before `kripp_alpha` — see the note that prints
    under the headline, and wobble_eval/binary.py for why.

There is no temperature knob on this path: Claude Code does not expose
temperature/top_p/top_k and Claude Opus 5 rejects them. The dial is `--effort`.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import sys
import textwrap
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wobble_eval import binary, charts_binary, exclusions, prompts
from wobble_eval.backend import make_backend
from wobble_eval.binary import BinaryConfig
from wobble_eval.config import CONTEXT_KIND
from wobble_eval.framework import (ALL_CODES, CODE2NAME, CODE2SECTION,
                                   FRAMEWORK, SECTION_CODES)
from wobble_eval.session import load_session, session_meta, turn_stats


# --------------------------------------------------------------------- scoring
def score_group(backend, cfg, section, codes, iteration, ctx, meta, group_idx):
    """One model call covering `codes`, with JSON re-asks on parse failure."""
    system, user = binary.build_binary_prompt(
        cfg, section, codes, ctx, CONTEXT_KIND, meta)
    got, det, method, raw = {}, {}, "none", ""
    for attempt in range(cfg.MAX_RETRIES + 1):
        u = user if attempt == 0 else (
            user + "\n\nIMPORTANT: your previous reply was not parseable. Reply with the raw "
                   'JSON object ONLY — no prose, no fences, no trailing commas, verdicts as '
                   'the strings "YES" or "NO".')
        try:
            raw = backend(system, u)
        except Exception as exc:
            print(f"      ! {section} call failed ({type(exc).__name__}: {exc}); "
                  f"attempt {attempt + 1}/{cfg.MAX_RETRIES + 1}")
            time.sleep(2 + 3 * attempt)
            continue
        got, det, method = binary.parse_verdicts(raw, codes)
        if any(v is not None for v in got.values()):
            break
    blank = dict(evidence="", margin="", flip_if="", confidence=float("nan"))
    cut = getattr(cfg, "CONF_BORDERLINE", 0.85)

    def _row(c):
        d = det.get(c, blank)
        conf = d.get("confidence", float("nan"))
        margin = d.get("margin", "")
        if conf == conf:                     # the number is the source of truth for the flag
            margin = "BORDERLINE" if conf < cut else "CLEAR"
        return dict(
            iteration=iteration, section=section, code=c, indicator=CODE2NAME[c],
            score=got.get(c),
            verdict=("YES" if got.get(c) == 1 else "NO" if got.get(c) == 0 else "NA"),
            yes_bar=binary.yes_bar_text(c, getattr(cfg, "YES_AT", 3)),
            confidence=conf, margin=margin,
            evidence=d.get("evidence", ""), flip_if=d.get("flip_if", ""),
            parse=method, na=(got.get(c) is None and method != "none"))

    return [_row(c) for c in codes], raw


def run_iteration(backend, cfg, iteration, ctx, meta):
    """All configured sections for one iteration (sections may run in parallel)."""
    jobs = []
    for section in cfg.SECTIONS:
        codes = prompts.order_codes(cfg, section, iteration)
        groups = [codes] if cfg.SCORING_MODE == "per_section" else [[c] for c in codes]
        for gi, group in enumerate(groups):
            jobs.append((section, group, gi))

    rows, raws = [], {}
    workers = max(1, min(cfg.MAX_CONCURRENCY, len(jobs)))
    if workers == 1:
        for section, group, gi in jobs:
            r, raw = score_group(backend, cfg, section, group, iteration, ctx, meta, gi)
            rows += r
            raws.setdefault(section, []).append(raw)
    else:
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(score_group, backend, cfg, s, g, iteration, ctx, meta, gi):
                    (s, gi) for s, g, gi in jobs}
            for fut in futures.as_completed(futs):
                section, _ = futs[fut]
                r, raw = fut.result()
                rows += r
                raws.setdefault(section, []).append(raw)

    if cfg.VERBOSE:
        for section in cfg.SECTIONS:
            got = sorted((r for r in rows if r["section"] == section),
                         key=lambda r: ALL_CODES.index(r["code"]))
            # lower-case = the model flagged that call BORDERLINE, i.e. sitting close
            # enough to the threshold that it could land the other way next run
            def _glyph(r):
                if r["score"] is None:
                    return "-"
                ch = "Y" if r["score"] == 1 else "N"
                return ch.lower() if r.get("margin") == "BORDERLINE" else ch
            shown = " ".join(_glyph(r) for r in got)
            n_ok = sum(r["score"] is not None for r in got)
            n_yes = sum(r["score"] == 1 for r in got)
            n_bl = sum(r.get("margin") == "BORDERLINE" and r["score"] is not None
                       for r in got)
            confs = [r["confidence"] for r in got
                     if r["score"] is not None and r["confidence"] == r["confidence"]]
            bl = f" · {n_bl} borderline" if n_bl else ""
            cf = f" · mean conf {sum(confs)/len(confs):.2f}" if confs else ""
            print(f"      {section}: {n_ok}/{len(got)} answered · {n_yes} YES{bl}{cf}"
                  f"  [{shown}]")
    return rows, raws


# --------------------------------------------------------------------- one arm
def run_experiment(cfg, session, label=""):
    meta = session_meta(session)
    ctx = session["transcript"]
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    backend = make_backend(cfg)
    tag = f" [{label}]" if label else ""
    print(f"\n=== BINARY · {cfg.MODEL} · effort={cfg.EFFORT} · thinking={cfg.THINKING} · "
          f"{cfg.N_ITERATIONS} iterations{tag} ===")
    print(f"    backend {backend.name}"
          + (" (subscription auth)" if backend.name == "agent_sdk" else ""))
    print(f"    session {meta['session_id']} · {meta['duration_min']} min · "
          f"{turn_stats(ctx)['n_turns']} turns · lang {meta['language']}")
    print(f"    sections {list(cfg.SECTIONS)} · "
          f"{sum(len(SECTION_CODES[s]) for s in cfg.SECTIONS)} indicators · "
          f"mode {cfg.SCORING_MODE} · concurrency {cfg.MAX_CONCURRENCY}")
    print(f"    YES bar = level {cfg.YES_AT}+ "
          f"({binary.LEVEL_LABEL[cfg.YES_AT]}), clearly met")
    if cfg.EXPLAIN and cfg.INCLUDE_EVIDENCE:
        print(f"    explain ON — each call also reports a confidence (0.50-1.00 = its own "
              f"estimate of\n              P(a second observer gives the same verdict)) and "
              f"what would flip it.\n              Lower-case y/n below = confidence < "
              f"{cfg.CONF_BORDERLINE:.2f}.")

    records, t_start = [], time.time()
    long_csv = os.path.join(cfg.OUT_DIR, "verdicts_long.csv")
    for it in range(cfg.N_ITERATIONS):
        t0 = time.time()
        print(f"  [iteration {it + 1}/{cfg.N_ITERATIONS}]")
        rows, _ = run_iteration(backend, cfg, it, ctx, meta)
        records += rows
        pd.DataFrame(records).to_csv(long_csv, index=False)     # crash-safe
        dt = time.time() - t0
        msg = f"      -> {dt:.0f}s (elapsed {(time.time() - t_start) / 60:.1f} min)"
        if it == 0:
            msg += f" | projected total {dt * cfg.N_ITERATIONS / 60:.0f} min"
        print(msg)

    scores_long = pd.DataFrame(records)
    scores_long["score"] = pd.to_numeric(scores_long["score"], errors="coerce")
    scores_long["section"] = pd.Categorical(scores_long["section"],
                                            list(cfg.SECTIONS), ordered=True)
    expected = [c for c in ALL_CODES if CODE2SECTION[c] in cfg.SECTIONS]
    scores_long["code"] = pd.Categorical(scores_long["code"], expected, ordered=True)

    wall = time.time() - t_start
    run_meta = dict(**meta, model=cfg.MODEL, backend=backend.name,
                    model_calls=backend.calls,
                    model_seconds=round(backend.total_seconds, 1),
                    wall_seconds=round(wall, 1), **cfg.as_dict())
    with open(os.path.join(cfg.OUT_DIR, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    n_yes = int((scores_long.score == 1).sum())
    n_no = int((scores_long.score == 0).sum())
    print(f"\n  DONE in {wall / 60:.1f} min — {len(scores_long)} verdict cells "
          f"({n_yes} YES, {n_no} NO, {int(scores_long.score.isna().sum())} NA/failed) "
          f"in {backend.calls} model calls")
    return scores_long, run_meta, meta


# ---------------------------------------------------------------------- report
def _write_csv(df, path, **kw):
    """A CSV open in Excel is locked on Windows. Losing one table must not abort the
    report and throw away the charts — warn and carry on."""
    try:
        df.to_csv(path, **kw)
        return True
    except PermissionError:
        print(f"  ! could not write {os.path.basename(path)} — the file is open in "
              f"another program. Close it and re-run with --analyse-only to regenerate.")
        return False


def report(res, cfg, meta, run_meta):
    out = cfg.OUT_DIR
    _write_csv(res["wide"], os.path.join(out, "verdicts_wide.csv"))
    _write_csv(res["ind_stats"], os.path.join(out, "indicator_wobble.csv"), index=False)
    _write_csv(res["reliability"], os.path.join(out, "reliability.csv"), index=False)
    _write_csv(res["sec_stats"], os.path.join(out, "section_fidelity.csv"), index=False)
    try:
        with open(os.path.join(out, "headline.json"), "w") as f:
            json.dump(res["headline"], f, indent=2)
    except PermissionError:
        print("  ! could not write headline.json — file is open elsewhere.")

    h = res["headline"]
    print("\n" + "=" * 78)
    print("HEADLINE — BINARY (YES / NO)")
    print("=" * 78)
    print(json.dumps(h, indent=2))

    ac1, alpha, po = h["gwet_ac1"], h["kripp_alpha"], h["pairwise_exact_agreement"]
    verdict = ("DEPENDABLE for indicator-level YES/NO reporting" if ac1 >= 0.80 else
               "TENTATIVE — report section fidelity %, and majority-of-N at indicator level"
               if ac1 >= 0.67 else
               "NOT decision-grade at indicator level")
    print(f"\nGwet's AC1 = {ac1:.3f}  ->  {verdict}")
    print(f"  (Krippendorff α = {alpha:.3f} · raw pairwise agreement = {po*100:.1f}% · "
          f"YES prevalence = {h['yes_prevalence']*100:.0f}%)")
    if h["prevalence_paradox"]:
        print("  ! PREVALENCE PARADOX: raw agreement is high but α is low. With a lopsided "
              "YES/NO\n    split, α and κ collapse by construction. Quote AC1 and raw "
              "agreement here, not α.")
    elif alpha < 0.67 and ac1 < 0.67:
        print("  Both coefficients are low — this is genuine unreliability, not a "
              "statistical artefact.")

    cq = res["reliability"].loc[0, "cochran_p"]
    cq_txt = "n/a" if pd.isna(cq) else f"{cq:.4f}"
    print(f"Run-to-run drift: Cochran's Q p={cq_txt}  ->  "
          + ("SIGNIFICANT systematic drift between runs (a single pass is biased, not just "
             "imprecise)" if pd.notna(cq) and cq < cfg.ALPHA else
             "no significant drift (noise is unbiased, so majority-voting converges)"))

    allrow = res["sec_stats"][res["sec_stats"].section == "ALL"].iloc[0]
    print(f"Overall fidelity {allrow.yes_rate*100:.1f}% -> band {allrow.band}"
          + (f"  ! band is NOT stable across runs: saw {allrow.bands_seen}"
             if allrow.band_flips else "  (band stable across all runs)"))

    cols = ["section", "code", "indicator", "n", "n_yes", "p_yes", "verdict", "modal_share",
            "ci_lo", "ci_hi", "single_pass_error", "votes_needed", "na_rate", "q_wobble",
            "grade"]
    print("\nPER-INDICATOR VERDICT WOBBLE\n" + "-" * 78)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(res["ind_stats"][cols].to_string(index=False,
              float_format=lambda v: f"{v:.2f}"))

    if res.get("has_conf"):
        d = res["ind_stats"].copy()
        d["actual_agreement"] = 1 - d["single_pass_error"]
        ccols = ["section", "code", "indicator", "verdict", "n_yes", "n",
                 "conf_when_yes", "conf_when_no", "mean_confidence", "min_confidence",
                 "actual_agreement", "calibration_gap", "grade"]
        print("\nPER-INDICATOR CONFIDENCE — what it claimed, and what it actually got")
        print("-" * 78)
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(d[ccols].to_string(index=False, na_rep="—",
                  float_format=lambda v: f"{v:.2f}"))
        print("  conf_when_yes / conf_when_no : mean stated confidence on the runs that "
              "answered YES / NO.\n"
              "  actual_agreement             : share of runs agreeing with the majority "
              "verdict.\n"
              "  calibration_gap              : mean_confidence - actual_agreement. "
              "Negative = it undersold itself;\n"
              "                                 positive = it claimed more than it earned.")

    print("\nSECTION FIDELITY ROLL-UP\n" + "-" * 78)
    keep = ["section", "title", "n_indicators", "yes_rate", "ci_lo", "ci_hi", "band",
            "bands_seen", "band_flips", "n_flipping", "pct_unanimous",
            "mean_single_pass_error", "gwet_ac1", "kripp_alpha", "na_rate"]
    # sections whose band is not settled are the headline risk — call them out by name
    wobbly = res["sec_stats"][res["sec_stats"].band_flips > 0]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(res["sec_stats"][keep].to_string(index=False,
              float_format=lambda v: f"{v:.3f}"))
    for r in wobbly.itertuples():
        print(f"  ! Section {r.section} fidelity band is NOT stable: individual runs landed "
              f"in {r.bands_seen} (reported figure {r.yes_rate*100:.1f}% = {r.band}). "
              f"Which run you quote decides the band.")

    flip = res["flippy"]
    if len(flip):
        print("\nVERDICT FLIPS — these indicators answered both YES and NO across runs, "
              "so no single pass can be quoted:")
        print(flip[["code", "indicator", "n_yes", "n", "p_yes", "single_pass_error",
                    "votes_needed", "p_coinflip"]].to_string(
                        index=False, float_format=lambda v: f"{v:.2f}"))
    else:
        print("\nNo verdict flips: every indicator returned the same YES/NO answer in every run.")

    # ---- the decision boundary: what threshold each verdict was measured against,
    # how close the model said it was, and what it said would flip it
    if res.get("has_conf") and res.get("calib"):
        cb = res["calib"]
        _write_csv(res["calib_bins"], os.path.join(out, "calibration.csv"), index=False)
        print("\n" + "=" * 78)
        print("CONFIDENCE CALIBRATION — was the stated confidence worth anything?")
        print("=" * 78)
        print(f"Mean stated confidence : {cb['mean_confidence']:.3f}   "
              f"(YES calls {cb['mean_confidence_yes']:.3f} · "
              f"NO calls {cb['mean_confidence_no']:.3f})")
        print(f"Mean ACTUAL agreement  : {cb['mean_observed_agreement']:.3f}   "
              f"(share of other runs that gave the same verdict)")
        gap = cb["overconfidence"]
        print(f"Overconfidence         : {gap:+.3f}  -> "
              + ("claims more agreement than it earns" if gap > 0.02 else
                 "claims less agreement than it earns" if gap < -0.02 else
                 "well matched on average"))
        print(f"Expected calib. error  : {cb['expected_calibration_error']:.3f}  "
              f"(mean |stated - actual| per call)")
        rho, rp = cb["spearman_rho"], cb["spearman_p"]
        if rho == rho:
            print(f"Rank correlation       : rho={rho:+.3f} (p={rp:.4f})  -> "
                  + ("higher stated confidence really does mean higher agreement — the "
                     "number carries usable signal" if rp < cfg.ALPHA and rho > 0 else
                     "stated confidence does NOT track actual agreement — the number is "
                     "not usable as a filter"))
        if len(res["calib_bins"]):
            print("\n  stated confidence     n   mean stated   actual agreement      gap")
            print("  " + "-" * 66)
            for r in res["calib_bins"].itertuples():
                if not r.n_cells:
                    continue
                print(f"  {str(r.bin):<18} {int(r.n_cells):>4}        "
                      f"{r.mean_stated:.3f}             {r.observed_agreement:.3f}   "
                      f"{r.gap:+.3f}")
            print("  A well-calibrated scorer has gap ~ 0 in every row. Positive gap = it "
                  "was\n  more sure than it deserved to be at that confidence level.")

    if res.get("has_margin"):
        _write_csv(res["boundaries"], os.path.join(out, "decision_boundaries.csv"),
                   index=False)
        bt = res.get("boundary_test") or {}
        print("\n" + "=" * 78)
        print("DECISION BOUNDARIES — what the YES/NO line actually was, and what moved it")
        print("=" * 78)
        if bt:
            fr_b, fr_c = bt["flip_rate_when_borderline"], bt["flip_rate_when_clear"]
            print(f"The model flagged {h.get('n_indicators_ever_borderline', 0)} of "
                  f"{len(res['ind_stats'])} indicators BORDERLINE at least once "
                  f"({h.get('borderline_rate', float('nan'))*100:.0f}% of all calls).")
            print(f"  flipped when ever-BORDERLINE : {bt['borderline_and_flipped']}/"
                  f"{bt['borderline_and_flipped'] + bt['borderline_not_flipped']} "
                  f"({fr_b*100:.0f}%)" if fr_b == fr_b else "  (no borderline calls)")
            print(f"  flipped when always CLEAR    : {bt['clear_but_flipped']}/"
                  f"{bt['clear_but_flipped'] + bt['clear_and_stable']} "
                  f"({fr_c*100:.0f}%)" if fr_c == fr_c else "")
            p = bt["fisher_p"]
            # Treat the flag as a SCREEN, not a predictor. Recall (did it catch every
            # verdict that moved?) and precision (how many flagged calls actually moved?)
            # answer different questions, and only recall licenses trusting a CLEAR call.
            n_flip = bt["borderline_and_flipped"] + bt["clear_but_flipped"]
            recall = bt["borderline_and_flipped"] / n_flip if n_flip else float("nan")
            prec = fr_b
            if p == p:
                print(f"  Fisher exact p={p:.4f}"
                      + ("  (significant)" if p < cfg.ALPHA else
                         "  (not significant at this sample size)"))
            if recall == recall:
                print(f"  As a screen: recall {recall*100:.0f}% "
                      f"({bt['borderline_and_flipped']}/{n_flip} moving verdicts were "
                      f"flagged), precision {prec*100:.0f}%.")
                if recall >= 0.999:
                    print("  -> Every verdict that moved was flagged BORDERLINE at least "
                          "once, and NO always-CLEAR indicator moved.\n"
                          "     So a CLEAR call is trustworthy from one pass; a BORDERLINE "
                          "call is not a prediction that it\n"
                          "     will move, only that it could. Re-run the flagged ones; "
                          "accept the rest.")
                elif p == p and p < cfg.ALPHA:
                    print("  -> The flag is informative but leaks: some verdicts moved "
                          "without ever being flagged.")
                else:
                    print("  -> The flag misses movers, so it cannot be used to decide "
                          "which indicators are safe from one pass.")

        b = res["boundaries"]
        n_moved = int((b.flip == True).sum())
        print(f"\nEVERY INDICATOR: threshold, verdict, confidence and reasoning "
              f"({n_moved} of {len(b)} moved)\n" + "-" * 78)

        def _field(label, text, width=70):
            if not text or str(text) == "nan":
                return
            pad = " " * 24
            print(f"   {label:<19}: " + f"\n{pad}".join(textwrap.wrap(str(text), width)))

        for r in b.itertuples():
            if not r.n:                       # never applicable (F5/F6 in this lesson)
                print(f"\n{r.code}  {r.indicator}   [NA in every run — indicator does not "
                      f"apply to this lesson]")
                continue
            bits = [f"{int(r.n_yes)} YES / {int(r.n - r.n_yes)} NO"]
            if r.conf_when_yes == r.conf_when_yes:
                bits.append(f"conf(YES) {r.conf_when_yes:.2f}")
            if r.conf_when_no == r.conf_when_no:
                bits.append(f"conf(NO) {r.conf_when_no:.2f}")
            if r.observed_agreement == r.observed_agreement:
                bits.append(f"agreed {r.observed_agreement*100:.0f}%")
            if r.calibration_gap == r.calibration_gap:
                bits.append(f"gap {r.calibration_gap:+.2f}")
            flag = "  ⚠ VERDICT MOVED" if r.flip else ""
            print(f"\n{r.code}  {r.indicator}   [{' · '.join(bits)}]{flag}")
            _field("THRESHOLD (YES bar)", r.yes_bar)
            _field("argued YES", r.yes_reasons)
            _field("argued NO", r.no_reasons)
            _field("flips if", r.flip_if)
        print("\n  Each reason is prefixed with [confidence] — the certainty stated on the "
              "run that gave it.\n  An indicator showing BOTH an 'argued YES' and an "
              "'argued NO' line is one whose verdict moved.")

    und = res["undecided"]
    if len(und):
        print("\nCOIN-FLIP INDICATORS — the YES rate is statistically indistinguishable from "
              "chance (p_coinflip > 0.20).\nMore passes will not settle these; the rubric "
              "wording is what needs fixing:")
        print(und[["code", "indicator", "n_yes", "n", "p_coinflip"]].to_string(
            index=False, float_format=lambda v: f"{v:.2f}"))

    made = charts_binary.render_all(res, cfg, meta, out)
    print(f"\nCharts written ({len(made)}): "
          + ", ".join(m.replace("chart_", "") for m in made))
    print(f"All output in ./{out}/")


# ------------------------------------------------------------------------ main
def main(argv=None):
    cfg0 = BinaryConfig()
    p = argparse.ArgumentParser(
        description="Binary (YES/NO) wobble evaluation of the Taleemabad Coaching Framework "
                    "using Claude via the Claude Code subscription.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--iterations", "-n", type=int, default=cfg0.N_ITERATIONS)
    p.add_argument("--yes-at", type=int, default=cfg0.YES_AT, choices=[2, 3, 4],
                   help="which level descriptor is the YES bar: 2 = observed at all, "
                        "3 = proficient (framework's own cut), 4 = highly effective")
    p.add_argument("--model", default=cfg0.MODEL)
    p.add_argument("--effort", default=cfg0.EFFORT,
                   choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--thinking", default=cfg0.THINKING, choices=["adaptive", "disabled"])
    p.add_argument("--backend", default=cfg0.BACKEND, choices=["agent_sdk", "api"])
    p.add_argument("--sections", default=",".join(cfg0.SECTIONS),
                   help="comma-separated subset, e.g. B,D")
    p.add_argument("--mode", default=cfg0.SCORING_MODE,
                   choices=["per_section", "per_indicator"])
    p.add_argument("--prompt-variant", default=cfg0.PROMPT_VARIANT,
                   choices=["standard", "terse", "cot"])
    p.add_argument("--indicator-order", default=cfg0.INDICATOR_ORDER,
                   choices=["fixed", "shuffled"])
    p.add_argument("--concurrency", type=int, default=cfg0.MAX_CONCURRENCY,
                   help="parallel model calls; 1 = serial")
    p.add_argument("--no-evidence", action="store_true",
                   help="don't ask for an evidence quote per verdict (faster)")
    p.add_argument("--no-na", action="store_true",
                   help="force YES/NO on every indicator, disallowing NA")
    p.add_argument("--no-explain", action="store_true",
                   help="don't ask how close to the threshold each call was, or what "
                        "would flip it (fewer output tokens per call)")
    p.add_argument("--exclude", default="",
                   help="indicators to drop from scoring AND analysis: a named set "
                        "(unreliable | wording | unobservable | none) or a "
                        "comma-separated code list, e.g. C4,B2")
    p.add_argument("--session", default="", help="path to a different session JSON")
    p.add_argument("--out", default=cfg0.OUT_DIR)
    p.add_argument("--sweep-effort", default="",
                   help="comma-separated effort levels to sweep, e.g. low,medium,high")
    p.add_argument("--analyse-only", default="",
                   help="skip scoring; re-analyse an existing verdicts_long.csv")
    a = p.parse_args(argv)

    cfg = BinaryConfig(
        MODEL=a.model, EFFORT=a.effort, THINKING=a.thinking, BACKEND=a.backend,
        N_ITERATIONS=a.iterations,
        SECTIONS=tuple(s.strip().upper() for s in a.sections.split(",") if s.strip()),
        SCORING_MODE=a.mode, PROMPT_VARIANT=a.prompt_variant,
        INDICATOR_ORDER=a.indicator_order, MAX_CONCURRENCY=a.concurrency,
        INCLUDE_EVIDENCE=not a.no_evidence, ALLOW_NA=not a.no_na,
        EXPLAIN=not a.no_explain, EXCLUDE_CODES=tuple(exclusions.resolve(a.exclude)),
        OUT_DIR=a.out, SESSION_PATH=a.session, YES_AT=a.yes_at)
    for s in cfg.SECTIONS:
        if s not in FRAMEWORK:
            p.error(f"unknown section {s!r}; choose from {list(FRAMEWORK)}")

    session = load_session(cfg.SESSION_PATH or None)
    meta = session_meta(session)

    if a.analyse_only:
        df = pd.read_csv(a.analyse_only)
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        if "parse" not in df:
            df["parse"] = "strict"
        cfg.N_ITERATIONS = int(df.iteration.max()) + 1
        df["section"] = pd.Categorical(df["section"], list(cfg.SECTIONS), ordered=True)
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        res = binary.analyse_binary(df, cfg)
        report(res, cfg, meta, {})
        return 0

    if a.sweep_effort:
        levels = [x.strip() for x in a.sweep_effort.split(",") if x.strip()]
        rows = []
        for lvl in levels:
            arm = BinaryConfig(**{**cfg.as_dict(), "EFFORT": lvl,
                                  "SECTIONS": tuple(cfg.SECTIONS),
                                  "OUT_DIR": f"{cfg.OUT_DIR}_effort_{lvl}"})
            scores, run_meta, _ = run_experiment(arm, session, label=f"effort={lvl}")
            res = binary.analyse_binary(scores, arm, run_meta)
            report(res, arm, meta, run_meta)
            rows.append(dict(effort=lvl, **res["headline"]))
        sweep = pd.DataFrame(rows)
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        sweep.to_csv(os.path.join(cfg.OUT_DIR, "sweep_effort.csv"), index=False)
        keep = ["effort", "overall_yes_rate", "gwet_ac1", "kripp_alpha",
                "pct_indicators_unanimous", "n_indicators_flipping_verdict",
                "mean_single_pass_verdict_error"]
        print("\n" + "=" * 78 + "\nEFFORT SWEEP (binary)\n" + "=" * 78)
        print(sweep[[c for c in keep if c in sweep]].to_string(index=False))
        return 0

    scores, run_meta, _ = run_experiment(cfg, session)
    res = binary.analyse_binary(scores, cfg, run_meta)
    report(res, cfg, meta, run_meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
