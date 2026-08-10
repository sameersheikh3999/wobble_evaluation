#!/usr/bin/env python
"""Run the Taleemabad Coaching Framework wobble experiment LOCALLY, scoring with
Claude Opus through the Claude Code subscription already logged in on this machine.

    python run_wobble.py --iterations 10
    python run_wobble.py --iterations 6 --effort xhigh --out wobble_xhigh
    python run_wobble.py --iterations 6 --model claude-sonnet-5   # model arm
    python run_wobble.py --sweep-effort low,medium,high --iterations 4

No API key is used or needed: the Claude Agent SDK spawns the local `claude`
CLI, which authenticates with ~/.claude/.credentials.json.

What it does
    1. scores all 37 indicators (Sections B, C, D, F) on the session transcript
    2. repeats the whole evaluation N times with an identical prompt
    3. measures the wobble: per-indicator SD + bootstrap CI, modal agreement,
       proficiency-flip rate, Krippendorff's alpha / ICC / Fleiss' kappa,
       Friedman drift test, and the section/overall roll-ups
    4. writes CSVs + 7 charts to the output directory

IMPORTANT — there is no temperature knob on this path. Claude Code does not
expose temperature/top_p/top_k and Claude Opus 5 rejects them. The dial here is
`--effort` (low..max). See wobble_eval/config.py for the full explanation.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wobble_eval import analysis, charts, prompts
from wobble_eval.backend import make_backend
from wobble_eval.config import CONTEXT_KIND, Config
from wobble_eval.framework import (ALL_CODES, CODE2NAME, CODE2SECTION,
                                   FRAMEWORK, SECTION_CODES)
from wobble_eval.session import load_session, session_meta, turn_stats


# --------------------------------------------------------------------- scoring
def score_group(backend, cfg, section, codes, iteration, ctx, meta, group_idx):
    """One model call covering `codes`, with JSON re-asks on parse failure."""
    system, user = prompts.build_scoring_prompt(
        cfg, section, codes, ctx, CONTEXT_KIND, meta)
    got, ev, method, raw = {}, {}, "none", ""
    for attempt in range(cfg.MAX_RETRIES + 1):
        u = user if attempt == 0 else (
            user + "\n\nIMPORTANT: your previous reply was not parseable. Reply with "
                   "the raw JSON object ONLY — no prose, no fences, no trailing commas.")
        try:
            raw = backend(system, u)
        except Exception as exc:
            print(f"      ! {section} call failed ({type(exc).__name__}: {exc}); "
                  f"attempt {attempt + 1}/{cfg.MAX_RETRIES + 1}")
            time.sleep(2 + 3 * attempt)
            continue
        got, ev, method = prompts.parse_scores(raw, codes)
        if any(v is not None for v in got.values()):
            break
    rows = [dict(iteration=iteration, section=section, code=c, indicator=CODE2NAME[c],
                 score=got.get(c), evidence=ev.get(c, ""), parse=method,
                 na=(got.get(c) is None and method != "none"))
            for c in codes]
    return rows, raw


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
            shown = " ".join(str(r["score"]) if r["score"] is not None else "-"
                             for r in got)
            n_ok = sum(r["score"] is not None for r in got)
            print(f"      {section}: {n_ok}/{len(got)} scored  [{shown}]")
    return rows, raws


# ------------------------------------------------------------------- one arm
def run_experiment(cfg, session, label=""):
    meta = session_meta(session)
    ctx = session["transcript"]
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    backend = make_backend(cfg)
    tag = f" [{label}]" if label else ""
    print(f"\n=== {cfg.MODEL} · effort={cfg.EFFORT} · thinking={cfg.THINKING} · "
          f"{cfg.N_ITERATIONS} iterations{tag} ===")
    print(f"    backend {backend.name} (subscription auth)" if backend.name == "agent_sdk"
          else f"    backend {backend.name}")
    print(f"    session {meta['session_id']} · {meta['duration_min']} min · "
          f"{turn_stats(ctx)['n_turns']} turns · lang {meta['language']}")
    print(f"    sections {list(cfg.SECTIONS)} · {sum(len(SECTION_CODES[s]) for s in cfg.SECTIONS)} "
          f"indicators · mode {cfg.SCORING_MODE} · concurrency {cfg.MAX_CONCURRENCY}")

    records, t_start = [], time.time()
    long_csv = os.path.join(cfg.OUT_DIR, "scores_long.csv")
    for it in range(cfg.N_ITERATIONS):
        t0 = time.time()
        print(f"  [iteration {it + 1}/{cfg.N_ITERATIONS}]")
        rows, _ = run_iteration(backend, cfg, it, ctx, meta)
        records += rows
        pd.DataFrame(records).to_csv(long_csv, index=False)   # crash-safe
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

    print(f"\n  DONE in {wall / 60:.1f} min — {len(scores_long)} score cells "
          f"({int(scores_long.score.notna().sum())} numeric, "
          f"{int(scores_long.score.isna().sum())} NA/failed) "
          f"in {backend.calls} model calls")
    return scores_long, run_meta, meta


def report(res, cfg, meta, run_meta):
    out = cfg.OUT_DIR
    res["wide"].to_csv(os.path.join(out, "scores_wide.csv"))
    res["ind_stats"].to_csv(os.path.join(out, "indicator_wobble.csv"), index=False)
    res["reliability"].to_csv(os.path.join(out, "reliability.csv"), index=False)
    res["sec_stats"].to_csv(os.path.join(out, "section_wobble.csv"), index=False)
    with open(os.path.join(out, "headline.json"), "w") as f:
        json.dump(res["headline"], f, indent=2)

    h = res["headline"]
    a = h["kripp_alpha_ordinal"]
    verdict = ("DEPENDABLE for indicator-level reporting" if a >= 0.80 else
               "TENTATIVE — section means only, not single-pass indicator scores"
               if a >= 0.67 else
               "NOT decision-grade at indicator level")
    print("\n" + "=" * 78)
    print("HEADLINE")
    print("=" * 78)
    print(json.dumps(h, indent=2))
    print(f"\nKrippendorff alpha (ordinal) = {a:.3f}  ->  {verdict}")

    fr = res["reliability"].loc[0, "friedman_p"]
    print(f"Run-to-run drift: Friedman p={fr:.4f}  ->  "
          + ("SIGNIFICANT systematic drift between runs"
             if pd.notna(fr) and fr < cfg.ALPHA else
             "no significant drift (noise is unbiased, so averaging converges)"))
    if res["spread"]:
        s = res["spread"]
        print(f"Wobble differs across sections? Levene p={s['levene_p']:.4f} "
              f"({'yes' if s['levene_p'] < cfg.ALPHA else 'no'})")

    cols = ["section", "code", "indicator", "n", "mean", "sd", "mode", "modal_share",
            "range", "ci_lo", "ci_hi", "flip_rate", "na_rate", "q_wobble", "grade"]
    print("\nPER-INDICATOR WOBBLE\n" + "-" * 78)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(res["ind_stats"][cols].to_string(index=False,
              float_format=lambda v: f"{v:.2f}"))
    print("\nSECTION ROLL-UP\n" + "-" * 78)
    with pd.option_context("display.width", 200):
        print(res["sec_stats"].to_string(index=False,
              float_format=lambda v: f"{v:.3f}"))

    flip = res["flippy"]
    if len(flip):
        print(f"\nProficiency verdict FLIPS between runs — not reportable from one pass:")
        print(flip[["code", "indicator", "mode", "mean", "p_proficient",
                    "flip_rate"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    made = charts.render_all(res, cfg, meta, out)
    print(f"\nCharts written ({len(made)}): " + ", ".join(m.replace("chart_", "")
                                                          for m in made))
    print(f"All output in ./{out}/")


# ---------------------------------------------------------------------- main
def main(argv=None):
    cfg0 = Config()
    p = argparse.ArgumentParser(
        description="Local wobble evaluation of the Coaching Framework using "
                    "Claude Opus via the Claude Code subscription.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--iterations", "-n", type=int, default=cfg0.N_ITERATIONS)
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
                   help="don't ask for an evidence quote per score (faster)")
    p.add_argument("--no-na", action="store_true",
                   help="force 1-4 on every indicator, disallowing NA")
    p.add_argument("--session", default="", help="path to a different session JSON")
    p.add_argument("--out", default=cfg0.OUT_DIR)
    p.add_argument("--sweep-effort", default="",
                   help="comma-separated effort levels to sweep, e.g. low,medium,high")
    p.add_argument("--analyse-only", default="",
                   help="skip scoring; re-analyse an existing scores_long.csv")
    a = p.parse_args(argv)

    cfg = Config(MODEL=a.model, EFFORT=a.effort, THINKING=a.thinking, BACKEND=a.backend,
                 N_ITERATIONS=a.iterations, SECTIONS=tuple(s.strip().upper()
                                                           for s in a.sections.split(",")
                                                           if s.strip()),
                 SCORING_MODE=a.mode, PROMPT_VARIANT=a.prompt_variant,
                 INDICATOR_ORDER=a.indicator_order, MAX_CONCURRENCY=a.concurrency,
                 INCLUDE_EVIDENCE=not a.no_evidence, ALLOW_NA=not a.no_na,
                 OUT_DIR=a.out, SESSION_PATH=a.session)
    for s in cfg.SECTIONS:
        if s not in FRAMEWORK:
            p.error(f"unknown section {s!r}; choose from {list(FRAMEWORK)}")

    session = load_session(cfg.SESSION_PATH or None)
    meta = session_meta(session)

    if a.analyse_only:
        df = pd.read_csv(a.analyse_only)
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        cfg.N_ITERATIONS = int(df.iteration.max()) + 1
        res = analysis.analyse(df, cfg)
        report(res, cfg, meta, {})
        return 0

    if a.sweep_effort:
        levels = [x.strip() for x in a.sweep_effort.split(",") if x.strip()]
        rows = []
        for lvl in levels:
            arm = Config(**{**cfg.as_dict(), "EFFORT": lvl,
                            "SECTIONS": tuple(cfg.SECTIONS),
                            "OUT_DIR": f"{cfg.OUT_DIR}_effort_{lvl}"})
            scores, run_meta, _ = run_experiment(arm, session, label=f"effort={lvl}")
            res = analysis.analyse(scores, arm, run_meta)
            report(res, arm, meta, run_meta)
            rows.append(dict(effort=lvl, **res["headline"]))
        sweep = pd.DataFrame(rows)
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        sweep.to_csv(os.path.join(cfg.OUT_DIR, "sweep_effort.csv"), index=False)
        keep = ["effort", "overall_mean", "mean_indicator_sd", "kripp_alpha_ordinal",
                "pct_indicators_fully_stable", "n_indicators_flipping_proficiency"]
        print("\n" + "=" * 78 + "\nEFFORT SWEEP\n" + "=" * 78)
        print(sweep[[c for c in keep if c in sweep]].to_string(index=False))
        return 0

    scores, run_meta, _ = run_experiment(cfg, session)
    res = analysis.analyse(scores, cfg, run_meta)
    report(res, cfg, meta, run_meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
