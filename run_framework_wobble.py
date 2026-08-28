#!/usr/bin/env python
"""Measure the wobble and inter-rater reliability of ANY coaching framework.

    # 1. import a framework from the spreadsheets it arrived as
    python run_framework_wobble.py import --csv-dir tanzania_csvs \
           --out frameworks/tanzania.yaml

    # 2. check the spec, free, before spending anything
    python run_framework_wobble.py check --framework frameworks/tanzania.yaml

    # 3. estimate the cost of a run
    python run_framework_wobble.py plan --framework frameworks/tanzania.yaml \
           --dir Transcripts -n 10

    # 4. run it
    python run_framework_wobble.py run --framework frameworks/tanzania.yaml \
           --dir Transcripts -n 10 --out tanzania_out

    # 5. re-analyse without re-scoring (free)
    python run_framework_wobble.py analyse --framework frameworks/tanzania.yaml \
           --out tanzania_out

WHAT THIS MEASURES, AND WHY IT NEEDS REPEATS
--------------------------------------------
Score the same lesson twice with the same rubric and the same prompt. Any
difference cannot come from the lesson, the rubric or the question - only from
the scorer. That difference is the wobble, and it sets a ceiling on what the
framework can tell anyone: an indicator that will not reproduce cannot support a
coaching conversation however sensible it looks on paper.

TWO AXES, BOTH REQUIRED
-----------------------
  RELIABILITY    does the indicator give the same answer on re-run?
                 Pooled exact binomial of within-lesson disagreement against a
                 negligible-noise floor, plus Gwet's AC1 per indicator.
  DISCRIMINATION does it tell lessons apart at all?
                 Chi-square of homogeneity across lessons. An indicator scored
                 the same on every lesson is perfectly reliable and perfectly
                 useless - it carries no coaching signal.

                     discriminates        does not
    reliable         HEALTHY              UNINFORMATIVE
    unreliable       NOISY                BROKEN

Both p-values are Holm-corrected across the indicator set, because testing many
indicators at once otherwise manufactures significance.

WHY AC1 AND NOT KAPPA
---------------------
When an indicator is nearly always "not met" - and in practice many are - the
chance-agreement term in Cohen's kappa and Krippendorff's alpha grows until the
coefficient collapses toward zero even at near-perfect agreement. That is the
prevalence paradox: a property of the statistic, not of the scorer. Gwet's AC1
corrects for chance without that failure mode. All three are reported so the
divergence stays visible rather than being hidden by the choice of one.

YOU CANNOT SKIP THE RUN
-----------------------
`check` reviews the wording, but wording does not predict wobble: on the one
framework where both were measured, no textual feature correlated with the
observed flip rate (all |rho| < 0.16, p > 0.36). Reviewing is cheap and worth
doing; it is not a substitute for measuring.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import glob
import json
import math
import os
import re
import sys
import textwrap
import time

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wobble_eval import spec as specmod
from wobble_eval.backend import make_backend
from wobble_eval.binary import gwet_ac1, votes_needed, wilson_ci
from wobble_eval.stats import holm, krippendorff_alpha, pairwise_agreement

MIN_LESSONS_FOR_TEST = 3
NEGLIGIBLE = 0.05          # the noise floor reliability is tested against
ALPHA = 0.05


# ----------------------------------------------------------------- transcripts
def find_transcripts(d, text_field="transcript"):
    """Any JSON with a text field. `id` falls back to the filename, so a drop of
    plain {"transcript": "..."} files works with no other metadata."""
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            j = json.load(open(f, encoding="utf-8"))
        except Exception as exc:
            print(f"  ! {os.path.basename(f)}: {type(exc).__name__}, skipped")
            continue
        text = j.get(text_field) or j.get("text") or j.get("content")
        if not text:
            print(f"  ! {os.path.basename(f)}: no {text_field!r} field, skipped")
            continue
        sid = str(j.get("session_id") or j.get("id")
                  or os.path.splitext(os.path.basename(f))[0])[:8]
        out.append(dict(id=sid, text=text, path=f,
                        language=j.get("language", "?"),
                        meta={k: v for k, v in j.items() if k != text_field}))
    return out


# --------------------------------------------------------------------- prompts
SYSTEM = (
    "You are a trained classroom observer scoring a lesson against the rubric you "
    "are given. Judge only what the material actually evidences.\n"
    "- Absence of evidence is the lowest level, never a generous guess.\n"
    "- Do not reward effort, warmth or busyness when the descriptor asks for "
    "something else.\n"
    "- Output valid JSON only. No preamble, no markdown fences, no commentary.")


def build_prompt(fw, section_code, codes, text, binary, yes_at, allow_na,
                 want_evidence=True):
    rubric = fw.render_section(section_code, codes=codes, binary=binary, yes_at=yes_at)
    na = ('\nIf an indicator genuinely cannot apply to this lesson, use the string '
          '"NA". A practice that could have happened but did not is NOT "NA".'
          if allow_na else "")
    if binary:
        shape = ('{"%s": {"verdict": "YES"|"NO", "confidence": <0.50-1.00>, '
                 '"evidence": "<max 20 words from the material>"}, ...}' % codes[0])
        task = ("For EVERY indicator below, decide whether its YES bar is clearly met "
                "by the evidence in this lesson. YES or NO, nothing in between.")
    else:
        shape = ('{"%s": {"score": <1-%d>, "confidence": <0.50-1.00>, '
                 '"evidence": "<max 20 words from the material>"}, ...}'
                 % (codes[0], fw.n_levels))
        task = (f"For EVERY indicator below, choose the level 1-{fw.n_levels} whose "
                f"descriptor best matches what this lesson evidences.")
    if not want_evidence:
        shape = shape.replace(', "evidence": "<max 20 words from the material>"', "")
    conf = ('\n"confidence" is the probability a second equally careful observer would '
            'give the same answer: 0.50 is a toss-up, 1.00 is certain. Never below '
            '0.50. Be honest - this is checked against what actually happens.')
    user = (f"## MATERIAL TO SCORE\n\"\"\"\n{text}\n\"\"\"\n\n"
            f"## RUBRIC\n{rubric}\n\n"
            f"## TASK\n{task}{na}{conf}\n\n"
            f"Return exactly one JSON object with these {len(codes)} keys and nothing "
            f"else: {', '.join(codes)}\nShape: {shape}")
    return SYSTEM, user


# --------------------------------------------------------------------- parsing
def _brace_slice(text):
    for s in (i for i, c in enumerate(text) if c == "{"):
        depth, in_str, esc = 0, False, False
        for i in range(s, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[s:i + 1]
    return None


_TRUE = {"yes", "y", "true", "met", "1"}
_FALSE = {"no", "n", "false", "not met", "0"}


def _coerce(val, binary, n_levels):
    if isinstance(val, dict):
        for k in ("verdict", "score", "level", "answer", "value", "rating"):
            if k in val:
                return _coerce(val[k], binary, n_levels)
        return None
    if isinstance(val, bool):
        return (1 if val else 0) if binary else None
    if isinstance(val, (int, float)):
        if isinstance(val, float) and math.isnan(val):
            return None
        v = int(round(val))
        if binary:
            return v if v in (0, 1) else None
        return v if 1 <= v <= n_levels else None
    if isinstance(val, str):
        s = val.strip().strip(".!").lower()
        if s in ("na", "n/a", "none", "null", ""):
            return None
        if binary:
            if s in _TRUE:
                return 1
            if s in _FALSE:
                return 0
            if re.search(r"\b(not met|no|false)\b", s):
                return 0
            if re.search(r"\b(yes|true|met)\b", s):
                return 1
            return None
        m = re.search(r"\d+", s)
        if m:
            v = int(m.group())
            return v if 1 <= v <= n_levels else None
    return None


def _sub(val, keys):
    if isinstance(val, dict):
        for k in keys:
            v = val.get(k)
            if isinstance(v, str) and v.strip():
                return v[:300]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
    return "" if keys[0] == "evidence" else np.nan


def parse(text, codes, binary, n_levels):
    out = {c: None for c in codes}
    det = {c: dict(evidence="", confidence=np.nan) for c in codes}
    payload, method = None, "regex"
    body = re.sub(r"^\s*```(?:json)?|```\s*$", " ", text.strip(), flags=re.M)
    for cand, name in ((body.strip(), "strict"), (_brace_slice(body), "brace")):
        if not cand:
            continue
        for attempt in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
            try:
                payload, method = json.loads(attempt), name
                break
            except Exception:
                payload = None
        if payload is not None:
            break
    if isinstance(payload, dict):
        if (len(payload) == 1 and isinstance(next(iter(payload.values())), dict)
                and not set(payload) & set(codes)):
            payload = next(iter(payload.values()))
        upper = {str(k).strip().upper(): v for k, v in payload.items()}
        for c in codes:
            if c in upper:
                out[c] = _coerce(upper[c], binary, n_levels)
                det[c] = dict(evidence=_sub(upper[c], ["evidence", "reason"]),
                              confidence=_sub(upper[c], ["confidence", "conf"]))
    if all(v is None for v in out.values()) and payload is None:
        method = "none"
    return out, det, method


# --------------------------------------------------------------------- scoring
def score_section(backend, fw, section, codes, text, binary, yes_at, allow_na,
                  retries=2):
    system, user = build_prompt(fw, section, codes, text, binary, yes_at, allow_na)
    got, det, method, raw = {}, {}, "none", ""
    for attempt in range(retries + 1):
        u = user if attempt == 0 else (
            user + "\n\nIMPORTANT: your previous reply was not parseable. Reply with "
                   "the raw JSON object ONLY.")
        try:
            raw = backend(system, u)
        except Exception as exc:
            print(f"      ! {section} call failed ({type(exc).__name__}); "
                  f"attempt {attempt + 1}/{retries + 1}")
            time.sleep(2 + 3 * attempt)
            continue
        got, det, method = parse(raw, codes, binary, fw.n_levels)
        if any(v is not None for v in got.values()):
            break
    blank = dict(evidence="", confidence=np.nan)
    return [dict(section=section, code=c, indicator=fw.code2name[c],
                 score=got.get(c), parse=method,
                 evidence=det.get(c, blank)["evidence"],
                 confidence=det.get(c, blank)["confidence"]) for c in codes]


def score_transcript(backend, fw, t, n_runs, binary, yes_at, allow_na, workers):
    rows = []
    t0 = time.time()
    for it in range(n_runs):
        jobs = [(s.code, [i.code for i in s.indicators]) for s in fw.sections
                if s.indicators]
        got = []
        if workers <= 1:
            for sc, codes in jobs:
                got += score_section(backend, fw, sc, codes, t["text"], binary,
                                     yes_at, allow_na)
        else:
            with futures.ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
                futs = [ex.submit(score_section, backend, fw, sc, codes, t["text"],
                                  binary, yes_at, allow_na) for sc, codes in jobs]
                for f in futures.as_completed(futs):
                    got += f.result()
        for r in got:
            r.update(session_id=t["id"], iteration=it, language=t["language"])
        rows += got
        n_ok = sum(r["score"] is not None for r in got)
        print(f"      [{it + 1}/{n_runs}] {n_ok}/{len(got)} scored · "
              f"{(time.time() - t0) / (it + 1):.0f}s/run")
    return rows


# -------------------------------------------------------------------- analysis
def analyse(df, fw, binary, yes_at):
    """-> per-indicator reliability + discrimination, per-lesson cells, bands."""
    df = df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    # A level scale is collapsed at the proficiency cut so every framework is
    # judged on the same thing: the decision the rubric exists to produce.
    df["met"] = (df["score"] if binary
                 else np.where(df["score"].isna(), np.nan,
                               (df["score"] >= yes_at).astype(float)))
    sessions = list(dict.fromkeys(df.session_id))
    n_runs = int(df.iteration.max()) + 1 if len(df) else 0
    codes = fw.all_codes

    cells = []
    for s in sessions:
        for c in codes:
            xs = df[(df.session_id == s) & (df.code == c)]["met"].to_numpy(float)
            x = xs[~np.isnan(xs)]
            if not len(x):
                cells.append(dict(session_id=s, code=c, n=0, n_met=np.nan,
                                  p_met=np.nan, verdict="NA", flip=False,
                                  single_pass_error=np.nan))
                continue
            k, n = int(x.sum()), len(x)
            p = k / n
            cells.append(dict(session_id=s, code=c, n=n, n_met=k, p_met=p,
                              verdict="MET" if p > .5 else "NOT" if p < .5 else "TIED",
                              flip=bool(0 < k < n),
                              single_pass_error=1 - max(p, 1 - p)))
    cells = pd.DataFrame(cells)
    cells["section"] = cells.code.map(fw.code2section)
    cells["indicator"] = cells.code.map(fw.code2name)

    rows = []
    for c in codes:
        sub = df[df.code == c]
        piv = (sub.pivot_table(index="session_id", columns="iteration", values="met",
                               dropna=False)
               .reindex(index=sessions, columns=range(n_runs)))
        mat = piv.to_numpy(float)
        cc = cells[(cells.code == c) & (cells.n > 0)]
        obs = mat[~np.isnan(mat)]

        # reliability: pooled disagreement against the negligible-noise floor
        k = n = 0
        for r in mat:
            r = r[~np.isnan(r)]
            if len(r) < 2:
                continue
            maj = 1 if r.mean() > 0.5 else 0
            k += int((r != maj).sum())
            n += len(r)
        p_unstable = (float(scipy_stats.binomtest(k, n, NEGLIGIBLE,
                                                  alternative="greater").pvalue)
                      if n >= 2 else np.nan)

        # discrimination: does the met-rate differ across lessons?
        tab = [[int(r[~np.isnan(r)].sum()), int((~np.isnan(r)).sum()
                                                - r[~np.isnan(r)].sum())]
               for r in mat if (~np.isnan(r)).any()]
        p_disc = np.nan
        if len(tab) >= 2:
            a = np.array(tab, float)
            if a[:, 0].sum() > 0 and a[:, 1].sum() > 0:
                try:
                    p_disc = float(scipy_stats.chi2_contingency(a)[1])
                except Exception:
                    pass

        conf = pd.to_numeric(sub["confidence"], errors="coerce")
        lo, hi = (wilson_ci(int(obs.sum()), int(obs.size))
                  if obs.size else (np.nan, np.nan))
        rows.append(dict(
            section=fw.code2section[c], code=c, indicator=fw.code2name[c],
            yes_bar=fw.yes_bar(c, yes_at),
            n_lessons=int(len(cc)), n_cells=int(obs.size),
            na_rate=float(np.isnan(mat).mean()),
            pooled_p_met=float(obs.mean()) if obs.size else np.nan,
            ci_lo=lo, ci_hi=hi,
            ac1=gwet_ac1(mat), kripp_alpha=krippendorff_alpha(mat, "nominal"),
            pairwise_agreement=pairwise_agreement(mat),
            mean_flip_rate=float(cc.single_pass_error.mean()) if len(cc) else np.nan,
            n_lessons_flipped=int(cc.flip.sum()),
            n_lessons_met=int((cc.verdict == "MET").sum()),
            n_lessons_not=int((cc.verdict == "NOT").sum()),
            lesson_spread=float(np.nanstd(cc.p_met.to_numpy(float), ddof=1))
            if len(cc) > 1 else 0.0,
            mean_confidence=float(conf.mean()) if conf.notna().any() else np.nan,
            votes_needed=votes_needed(
                float(np.nanmean([max(p, 1 - p) for p in cc.p_met])) if len(cc)
                else np.nan),
            p_unstable=p_unstable, p_discriminates=p_disc))
    ind = pd.DataFrame(rows)
    ind["q_unstable"] = holm(ind["p_unstable"].to_numpy(float))
    ind["q_discriminates"] = holm(ind["p_discriminates"].to_numpy(float))
    ind["reliable"] = ~(ind["q_unstable"] < ALPHA)
    ind["discriminates"] = ind["q_discriminates"] < ALPHA
    ind["verdict_class"] = [
        ("UNTESTED" if r.n_lessons < MIN_LESSONS_FOR_TEST else
         "HEALTHY" if r.reliable and r.discriminates else
         "UNINFORMATIVE" if r.reliable else
         "NOISY" if r.discriminates else "BROKEN")
        for r in ind.itertuples()]

    # section bands, per lesson and per run
    bands = []
    ok = df.dropna(subset=["met"])
    for s in sessions:
        d = ok[ok.session_id == s]
        for sec in [x.code for x in fw.sections] + ["ALL"]:
            dd = d if sec == "ALL" else d[d.section == sec]
            if not len(dd):
                continue
            per_run = dd.groupby("iteration")["met"].mean().to_numpy(float)
            seen = [fw.band_for(v) for v in per_run]
            uniq = list(dict.fromkeys(seen))
            bands.append(dict(session_id=s, section=sec,
                              met_rate=float(per_run.mean()),
                              min_run=float(per_run.min()),
                              max_run=float(per_run.max()),
                              band=fw.band_for(float(per_run.mean())),
                              bands_seen="/".join(sorted(set(uniq))),
                              band_unstable=len(set(uniq)) > 1))
    bands = pd.DataFrame(bands)

    complete = np.vstack([
        df[df.code == c].pivot_table(index="session_id", columns="iteration",
                                     values="met", dropna=False)
          .reindex(index=sessions, columns=range(n_runs)).to_numpy(float)
        for c in codes]) if codes else np.zeros((0, 0))
    head = dict(
        framework=fw.name, scale="binary" if binary else f"1-{fw.n_levels}",
        cut=yes_at, n_lessons=len(sessions), runs_per_lesson=n_runs,
        n_indicators=len(ind), scored_cells=int(df["met"].notna().sum()),
        pooled_met_rate=round(float(df["met"].mean(skipna=True)), 3),
        overall_ac1=round(float(gwet_ac1(complete)), 3),
        overall_kripp_alpha=round(float(krippendorff_alpha(complete, "nominal")), 3),
        overall_pairwise_agreement=round(float(pairwise_agreement(complete)), 3),
        mean_indicator_ac1=round(float(ind.ac1.mean(skipna=True)), 3),
        mean_flip_rate=round(float(ind.mean_flip_rate.mean(skipna=True)), 4),
        healthy=int((ind.verdict_class == "HEALTHY").sum()),
        noisy=int((ind.verdict_class == "NOISY").sum()),
        uninformative=int((ind.verdict_class == "UNINFORMATIVE").sum()),
        broken=int((ind.verdict_class == "BROKEN").sum()),
        untested=int((ind.verdict_class == "UNTESTED").sum()),
        band_readings_unstable=int(bands.band_unstable.sum()) if len(bands) else 0,
        band_readings=int(len(bands)),
        parse_failure_rate=round(float((df.parse == "none").mean()), 4)
        if "parse" in df else None)
    return dict(ind=ind, cells=cells, bands=bands, headline=head)


# ---------------------------------------------------------------------- report
def write_report(res, out, fw):
    os.makedirs(out, exist_ok=True)

    def w(d, n, **kw):
        try:
            d.to_csv(os.path.join(out, n), **kw)
        except PermissionError:
            print(f"  ! could not write {n} - open elsewhere")
    w(res["ind"], "indicator_reliability.csv", index=False)
    w(res["cells"], "lesson_by_indicator.csv", index=False)
    w(res["bands"], "section_bands.csv", index=False)
    with open(os.path.join(out, "headline.json"), "w") as f:
        json.dump(res["headline"], f, indent=2)

    h = res["headline"]
    ind = res["ind"]
    print("\n" + "=" * 78)
    print(f"WOBBLE + IRR - {h['framework']}")
    print("=" * 78)
    print(json.dumps(h, indent=2))

    a = h["overall_ac1"]
    verdict = ("DEPENDABLE at indicator level" if a >= .80 else
               "TENTATIVE - report section/overall roll-ups, not single indicators"
               if a >= .67 else "NOT decision-grade")
    print(f"\nGwet's AC1 = {a:.3f}  ->  {verdict}")
    print(f"  (Krippendorff alpha = {h['overall_kripp_alpha']:.3f} · raw agreement "
          f"= {h['overall_pairwise_agreement']*100:.1f}%)")
    if h["overall_pairwise_agreement"] >= .80 and h["overall_kripp_alpha"] < .50:
        print("  ! PREVALENCE PARADOX: raw agreement high, alpha low. With a lopsided\n"
              "    met/not-met split, alpha collapses by construction. Quote AC1.")

    print("\n" + "=" * 78)
    print("INDICATOR CLASSIFICATION")
    print("=" * 78)
    print("  HEALTHY        reproduces AND separates lessons -> usable")
    print("  NOISY          separates lessons but will not reproduce -> fix wording")
    print("  UNINFORMATIVE  reproduces but same answer everywhere -> no signal")
    print("  BROKEN         neither")
    print("  UNTESTED       too few scored lessons\n")
    cols = ["section", "code", "indicator", "n_lessons", "pooled_p_met", "ac1",
            "mean_flip_rate", "n_lessons_flipped", "n_lessons_met", "n_lessons_not",
            "q_unstable", "q_discriminates", "verdict_class"]
    with pd.option_context("display.width", 230, "display.max_columns", 40):
        print(ind[cols].to_string(index=False, na_rep="-",
                                  float_format=lambda v: f"{v:.3f}"))

    for cls, blurb in (("NOISY", "will not reproduce - rewrite before using"),
                       ("BROKEN", "no reliable signal at all"),
                       ("UNINFORMATIVE", "same answer on every lesson - carries no "
                                         "coaching signal, however stable")):
        sub = ind[ind.verdict_class == cls]
        if not len(sub):
            continue
        print(f"\n{'=' * 78}\n{cls} ({len(sub)}) - {blurb}\n{'=' * 78}")
        for r in sub.itertuples():
            print(f"\n{r.code}  {r.indicator}")
            print(f"     AC1 {r.ac1:.2f} · flip {r.mean_flip_rate*100:.0f}% · "
                  f"met on {r.n_lessons_met}/{r.n_lessons} lessons")
            if isinstance(r.yes_bar, str) and r.yes_bar:
                print("     bar: " + "\n          ".join(textwrap.wrap(r.yes_bar, 66)))

    b = res["bands"]
    if len(b):
        print("\n" + "=" * 78 + "\nSECTION BANDS\n" + "=" * 78)
        g = (b.groupby("section").agg(lessons=("session_id", "size"),
                                      mean_rate=("met_rate", "mean"),
                                      unstable=("band_unstable", "sum")).reset_index())
        print(g.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\n'unstable' = the High/Medium/Low band changed between identical "
              "reruns on that\nmany lessons. A section that is unstable cannot be "
              "quoted from a single pass.")
    print(f"\nAll output in ./{out}/")


# ----------------------------------------------------------------------- modes
def cmd_import(a):
    fw = specmod.from_csv_dir(a.csv_dir)
    errs = specmod.validate(fw)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    specmod.to_yaml(fw, a.out)
    print(f"imported: {fw.summary()}")
    print(f"wrote {a.out}")
    if errs:
        print("\n! the spec has problems - fix them in the YAML before running:")
        for e in errs:
            print("   -", e)
    else:
        print("spec validates clean")
    return 0


def cmd_check(a):
    fw, errs = specmod.load(a.framework, strict=False)
    print(fw.summary())
    print(f"  sections: " + ", ".join(f"{s.code}({len(s.indicators)})"
                                      for s in fw.sections))
    print(f"  bands: " + ", ".join(f"{b['name']}>={b['min']:.0%}" for b in fw.bands))
    if errs:
        print(f"\nVALIDATION FAILED ({len(errs)}):")
        for e in errs:
            print("   -", e)
    else:
        print("\nvalidation: clean")

    r = specmod.review(fw)
    defects = [x for x in r if "IDENTICAL_LEVELS" in x["flags"]
               or "TOO_SHORT" in x["flags"]]
    advisory = [x for x in r if x not in defects]
    print(f"\nSPEC DEFECTS ({len(defects)}) - wrong regardless of how they would score:")
    for x in defects:
        print(f"   {x['code']:6s} {x['flags']:28s} {x['why']}")
    if not defects:
        print("   none")
    print(f"\nADVISORY - worth a human read ({len(advisory)}). NOT a prediction of "
          f"reliability:")
    for x in advisory[:12]:
        print(f"   {x['code']:6s} {x['flags']}")
    if len(advisory) > 12:
        print(f"   ... and {len(advisory) - 12} more (see --verbose)")
    if a.verbose:
        for x in advisory[12:]:
            print(f"   {x['code']:6s} {x['flags']}")
    print("\nWording does not predict wobble - on the one framework where both were\n"
          "measured, no textual feature correlated with the observed flip rate\n"
          "(all |rho| < 0.16, p > 0.36). Review is cheap; it is not a substitute\n"
          "for measuring. Run `plan` next.")
    return 1 if errs else 0


def cmd_plan(a):
    fw, _ = specmod.load(a.framework, strict=False)
    ts = find_transcripts(a.dir)
    n_sec = len([s for s in fw.sections if s.indicators])
    calls = len(ts) * a.iterations * n_sec
    print(f"{fw.summary()}")
    print(f"  transcripts : {len(ts)}")
    print(f"  runs each   : {a.iterations}")
    print(f"  sections    : {n_sec}  (one model call per section per run)")
    print(f"\n  MODEL CALLS : {calls}")
    print(f"  judgements  : {calls and len(ts) * a.iterations * len(fw.all_codes)}")
    print(f"  est. time   : {calls * 60 / 4 / 60:.0f}-{calls * 90 / 4 / 60:.0f} min "
          f"at concurrency 4")
    print(f"\n  statistical power")
    print(f"    {len(ts)} lessons x {a.iterations} runs = "
          f"{len(ts) * a.iterations} verdicts per indicator")
    if len(ts) < MIN_LESSONS_FOR_TEST:
        print(f"    ! fewer than {MIN_LESSONS_FOR_TEST} lessons - discrimination "
              f"cannot be tested, every indicator will come back UNTESTED")
    if a.iterations < 5:
        print(f"    ! {a.iterations} runs detects only indicators that flip often; "
              f"10 is the tested default")
    if len(ts) >= MIN_LESSONS_FOR_TEST and a.iterations >= 5:
        print(f"    adequate for both reliability and discrimination")
    return 0


def cmd_run(a):
    fw, errs = specmod.load(a.framework, strict=True)
    ts = find_transcripts(a.dir)
    if not ts:
        raise SystemExit(f"no usable transcripts in {a.dir!r}")
    yes_at = a.cut or fw.proficiency_cut
    os.makedirs(a.out, exist_ok=True)
    pooled = os.path.join(a.out, "all_scores_long.csv")

    records, done = [], set()
    if a.resume and os.path.exists(pooled):
        prev = pd.read_csv(pooled)
        prev["score"] = pd.to_numeric(prev["score"], errors="coerce")
        for sid, g in prev.groupby("session_id"):
            if g.iteration.nunique() >= a.iterations and g.score.notna().any():
                done.add(sid)
        records = prev.to_dict("records")
        print(f"--resume: {len(done)} transcript(s) already scored")

    from wobble_eval.config import Config
    cfg = Config(MODEL=a.model, EFFORT=a.effort, MAX_CONCURRENCY=a.concurrency)
    backend = make_backend(cfg)
    print(f"\n{fw.summary()}")
    print(f"  {a.model} · effort={a.effort} · {a.iterations} runs · "
          f"{'binary at level ' + str(yes_at) if a.binary else f'1-{fw.n_levels} scale'}")
    print(f"  {len(ts)} transcripts, {len(fw.all_codes)} indicators")

    t0 = time.time()
    for i, t in enumerate(ts, 1):
        if t["id"] in done:
            print(f"\n[{i}/{len(ts)}] {t['id']} - done, skipped")
            continue
        print(f"\n[{i}/{len(ts)}] {t['id']} · {t['language']} · {len(t['text'])} chars")
        try:
            records += score_transcript(backend, fw, t, a.iterations, a.binary,
                                        yes_at, not a.no_na, a.concurrency)
        except Exception as exc:
            print(f"  ! {t['id']} failed ({type(exc).__name__}: {exc}); continuing")
            continue
        pd.DataFrame(records).to_csv(pooled, index=False)
        print(f"    -> {(time.time() - t0) / 60:.1f} min elapsed, "
              f"{backend.calls} calls")
    print(f"\nSCORING DONE in {(time.time() - t0)/60:.1f} min · "
          f"{backend.calls} model calls")

    df = pd.DataFrame(records)
    res = analyse(df, fw, a.binary, yes_at)
    write_report(res, a.out, fw)
    return 0


def cmd_analyse(a):
    fw, _ = specmod.load(a.framework, strict=False)
    f = os.path.join(a.out, "all_scores_long.csv")
    if not os.path.exists(f):
        raise SystemExit(f"{f} not found - nothing to analyse")
    df = pd.read_csv(f)
    binary = a.binary or set(pd.to_numeric(df.score, errors="coerce")
                             .dropna().unique()) <= {0.0, 1.0}
    res = analyse(df, fw, binary, a.cut or fw.proficiency_cut)
    write_report(res, a.out, fw)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("import", help="spreadsheet folder -> framework YAML")
    q.add_argument("--csv-dir", required=True)
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_import)

    q = sub.add_parser("check", help="validate + review a spec (free)")
    q.add_argument("--framework", required=True)
    q.add_argument("--verbose", action="store_true")
    q.set_defaults(fn=cmd_check)

    q = sub.add_parser("plan", help="cost and power of a run (free)")
    q.add_argument("--framework", required=True)
    q.add_argument("--dir", default="Transcripts")
    q.add_argument("--iterations", "-n", type=int, default=10)
    q.set_defaults(fn=cmd_plan)

    q = sub.add_parser("run", help="score and analyse")
    q.add_argument("--framework", required=True)
    q.add_argument("--dir", default="Transcripts")
    q.add_argument("--out", default="framework_out")
    q.add_argument("--iterations", "-n", type=int, default=10)
    q.add_argument("--binary", action="store_true",
                   help="ask met/not-met directly instead of a level")
    q.add_argument("--cut", type=int, default=0,
                   help="proficiency cut (default: the spec's)")
    q.add_argument("--model", default="claude-opus-5")
    q.add_argument("--effort", default="high",
                   choices=["low", "medium", "high", "xhigh", "max"])
    q.add_argument("--concurrency", type=int, default=4)
    q.add_argument("--no-na", action="store_true")
    q.add_argument("--resume", action="store_true")
    q.set_defaults(fn=cmd_run)

    q = sub.add_parser("analyse", help="re-analyse existing scores (free)")
    q.add_argument("--framework", required=True)
    q.add_argument("--out", default="framework_out")
    q.add_argument("--binary", action="store_true")
    q.add_argument("--cut", type=int, default=0)
    q.set_defaults(fn=cmd_analyse)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
