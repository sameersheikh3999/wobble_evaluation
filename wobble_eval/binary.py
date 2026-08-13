"""Binary (YES / NO) variant of the Coaching Framework wobble experiment.

Same framework, same 37 indicators, same transcript, same experiment hygiene —
but the model returns a **verdict**, not a level. Each indicator is collapsed to
a single bar:

    YES  the descriptor at level YES_AT (default 3, "Proficient / Effective")
         is CLEARLY met, or a stronger level is met
    NO   anything below that bar
    NA   the indicator genuinely cannot apply (e.g. F5 MATH in a reading lesson)

WHY THIS IS A DIFFERENT MEASUREMENT, NOT A SIMPLER ONE
------------------------------------------------------
On the 1-4 scale most wobble is *adjacent* — a 3 becomes a 4 — and adjacent
wobble usually does not change what a coach does. Binary removes that cushion.
Every disagreement is a verdict flip, so the wobble you measure here is exactly
the wobble that changes the coaching decision. Expect the headline agreement
numbers to look *better* (fewer categories to disagree in) while the number of
decision-relevant flips stays roughly the same. Read them together.

THE STATISTICS CHANGE TOO — three deliberate substitutions
----------------------------------------------------------
1. **Gwet's AC1 alongside Krippendorff's alpha.** With two categories and a
   skewed split (most indicators NO on a weak lesson, most YES on a strong one)
   kappa/alpha collapse toward 0 even at 95% raw agreement. That is the
   prevalence paradox, not unreliability. AC1 is chance-corrected but
   prevalence-robust, so where alpha and AC1 diverge the split is skewed, and
   AC1 is the honest number. Both are reported, plus raw agreement, so the
   divergence is visible rather than hidden.
2. **Cochran's Q replaces Friedman** for run-to-run drift — Friedman's binary
   analogue, testing whether the YES rate differs systematically across runs.
3. **Wilson score intervals replace bootstrap CIs** on the per-indicator YES
   rate. A bootstrap of 10 Bernoulli draws is degenerate at the ends (it returns
   [1.0, 1.0] for a unanimous indicator, implying certainty that does not exist);
   Wilson stays honest at p = 0 and p = 1.

ICC(2,1) is dropped: it is a variance-components model for interval data and
means nothing on a 0/1 scale.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from . import stats as S
from .config import Config
from .framework import (ALL_CODES, CODE2NAME, CODE2SECTION, CODE_ORDER,
                        FRAMEWORK)
from .stats import (boot_ci, holm, krippendorff_alpha, p_wobble_test,
                    pairwise_agreement)

YES, NO = 1, 0
BINARY_LEVELS = np.array([0, 1])
LEVEL_LABEL = {1: "Not Observed / Emerging", 2: "Developing",
               3: "Proficient / Effective", 4: "Highly Effective"}


# --------------------------------------------------------------------- config
@dataclass
class BinaryConfig(Config):
    """Config plus the binary-only knobs. Everything else is inherited."""
    YES_AT: int = 3          # 2 = "observed at all", 3 = "proficient", 4 = "highly effective"
    HIGH_BAND: float = 0.85  # Section B's own fidelity banding: >=85% High
    MED_BAND: float = 0.60   #                                   60-84% Medium, <60% Low
    VOTE_TARGET: float = 0.95   # reproducibility target for `votes_needed`
    EXPLAIN: bool = True        # ask for confidence + flip_if alongside every verdict
    CONF_BORDERLINE: float = 0.85   # stated confidence below this is treated as BORDERLINE
    OUT_DIR: str = "wobble_out_binary"
    SCALE: str = "binary"


# ------------------------------------------------------------------- rendering
def _descriptors(ind):
    return [ind["l1"], ind["l2"], ind["l3"], ind["l4"]]


def render_binary_indicator(ind, yes_at=3, terse=False):
    """One indicator as a YES/NO decision rule built from its own level descriptors.

    Nothing is hand-authored here: the YES bar *is* the level-`yes_at` descriptor,
    so the binary run and the 1-4 run are asking about the same evidence.
    """
    d = _descriptors(ind)
    yes_bar = d[yes_at - 1]
    stronger = d[yes_at:]
    below = d[:yes_at - 1]

    if terse:
        lines = [f"{ind['code']} — {ind['name']}",
                 f"  YES: {yes_bar}",
                 f"  NO: {' | '.join(below) if below else '(anything weaker)'}"]
        if ind.get("subject"):
            lines.append(f"  [only for {ind['subject']} lessons]")
        return "\n".join(lines)

    lines = [f"### {ind['code']} — {ind['name']}",
             f"  YES — answer YES only if this is clearly true of the lesson:",
             f"      {yes_bar}"]
    if stronger:
        lines.append("  (a stronger version of the same thing also counts as YES: "
                     + " ".join(stronger) + ")")
    lines.append("  NO — answer NO if what you see is any of these instead, or nothing at all:")
    for k, txt in enumerate(below):
        lines.append(f"      ({LEVEL_LABEL[k + 1]}) {txt}")
    if not below:
        lines.append("      (anything short of the YES bar)")
    if ind.get("subject"):
        lines.append(f"  [Applies only to {ind['subject']} lessons]")
    return "\n".join(lines)


def render_binary_section(section_code, codes=None, yes_at=3, terse=False):
    s = FRAMEWORK[section_code]
    inds = [i for i in s["indicators"] if codes is None or i["code"] in codes]
    if codes is not None:                       # honour caller's order (shuffling)
        inds = sorted(inds, key=lambda i: codes.index(i["code"]))
    head = (f"SECTION {s['code']} — {s['title'].upper()}\n"
            f"Section guidance: {s['note']}\n"
            f"The YES bar for every indicator in this section is the "
            f"\"{LEVEL_LABEL[yes_at]}\" standard, clearly met.")
    return head + "\n\n" + "\n\n".join(render_binary_indicator(i, yes_at, terse) for i in inds)


# --------------------------------------------------------------------- prompts
BINARY_SYSTEM = (
    "You are a trained classroom observer for Taleemabad, scoring a lesson against the "
    "Taleemabad Coaching Framework. Every indicator receives ONE binary verdict: YES or NO.\n"
    "- YES means the YES descriptor is CLEARLY met on the evidence in the material provided. "
    "Partial, promising, one-off or almost-there evidence is NO.\n"
    "- Absence of evidence is NO, never a generous guess.\n"
    "- Do not reward effort, warmth or busyness when the descriptor asks for something else.\n"
    "- There is no middle category and no partial credit. Do not hedge, qualify, or invent "
    "'partially'. Commit to YES or NO.\n"
    "- Report your distance from the bar honestly. BORDERLINE is not a hedge and does not "
    "soften your verdict — you still commit to YES or NO — it records that the evidence sat "
    "close enough to the bar that a second careful observer could land the other way. Marking "
    "a genuinely close call CLEAR is an error, and so is marking an obvious call BORDERLINE.\n"
    "- Output valid JSON only. No preamble, no markdown fences, no commentary.")

_NA_CLAUSE = ('If an indicator genuinely cannot apply to this lesson (e.g. a MATH-specific '
              'indicator in a language lesson), use the string "NA" instead of YES or NO. '
              'A practice that could have happened but did not is NO, not NA.')

_SHAPE_EV = ('{{"{first}": {{"verdict": "YES"|"NO", "evidence": "<max 20 words quoted or '
             'paraphrased from the material>"}}, ...}}')
_SHAPE_NO = '{{"{first}": "YES"|"NO", ...}}'
# The explain shape asks for the decision boundary itself, not just the verdict:
# how close to the bar the call was, and what specific change would flip it.
_SHAPE_EXPLAIN = (
    '{{"{first}": {{"verdict": "YES"|"NO", "confidence": <number 0.50-1.00>, '
    '"evidence": "<max 20 words quoted or paraphrased from the material>", '
    '"flip_if": "<max 15 words: the specific thing that would have to change to flip '
    'this verdict>"}}, ...}}')

# The confidence definition is deliberately operational — a probability of AGREEMENT,
# not a feeling. That makes it directly checkable against how often the other runs
# actually did agree, which is what the calibration analysis does.
_EXPLAIN_CLAUSE = (
    'For each indicator also report:\n'
    '  "confidence" — a number from 0.50 to 1.00: the probability that a second, equally '
    'careful observer, given this same material and this same YES bar, would return the '
    'SAME verdict you just gave. 1.00 means certain agreement; 0.50 means a genuine '
    'toss-up. Never go below 0.50 — if you would, your verdict should have been the other '
    'one. Use the full range and be precise; do not default to round numbers, and do not '
    'inflate. This number will be checked against how often that agreement actually '
    'happens.\n'
    '  "flip_if" — the single most specific thing that would have to be different in this '
    'lesson for your verdict to flip. For a NO, what was missing; for a YES, what would '
    'have to be absent.')


def build_binary_prompt(cfg, section_code, codes, context_text, context_kind,
                        session_meta, variant=None):
    variant = variant or cfg.PROMPT_VARIANT
    yes_at = getattr(cfg, "YES_AT", 3)
    rubric = render_binary_section(section_code, codes=codes, yes_at=yes_at,
                                   terse=(variant == "terse"))
    explain = getattr(cfg, "EXPLAIN", False) and cfg.INCLUDE_EVIDENCE
    shape = (_SHAPE_EXPLAIN if explain else
             _SHAPE_EV if cfg.INCLUDE_EVIDENCE else _SHAPE_NO).format(first=codes[0])
    na = ("\n" + _NA_CLAUSE) if cfg.ALLOW_NA else ""
    ex = ("\n\n" + _EXPLAIN_CLAUSE) if explain else ""
    keys = ", ".join(codes)

    task = {
        "standard": (
            f"For EVERY indicator listed below, decide whether its YES bar is clearly met by "
            f"the evidence in this lesson. Answer YES or NO. Nothing in between.{na}{ex}\n\n"
            f"Return exactly one JSON object with these {len(codes)} keys and nothing else: "
            f"{keys}\nShape: {shape}"),
        "terse": (
            f"YES or NO for each indicator against its YES bar.{na}{ex}\n"
            f"JSON only, keys: {keys}\nShape: {shape}"),
        "cot": (
            f"For each indicator: in one sentence weigh the strongest evidence for YES against "
            f"the strongest evidence for NO, then commit to YES or NO.{na}{ex}\n\n"
            f"Write your reasoning inside a single <reasoning>...</reasoning> block (under 25 "
            f"words per indicator), then output the JSON object with keys {keys} after the "
            f"closing tag.\nShape: {shape}"),
    }[variant]

    user = (f"## MATERIAL TO SCORE\n"
            f"Source: {context_kind} of a lesson observation.\n"
            f"Session: {session_meta['session_id']} | language {session_meta['language']} | "
            f"{session_meta['duration_min']} minutes.\n\n"
            f'"""\n{context_text}\n"""\n\n'
            f"## RUBRIC\n{rubric}\n\n"
            f"## TASK\n{task}")
    return BINARY_SYSTEM, user


# --------------------------------------------------------------------- parsing
_TRUE = {"yes", "y", "true", "t", "met", "present", "observed", "pass", "1"}
_FALSE = {"no", "n", "false", "f", "not met", "notmet", "absent", "not observed",
          "fail", "0"}
_NA = {"na", "n/a", "none", "null", "not applicable", "n.a.", ""}


def _coerce_binary(val):
    """-> 1 (YES) / 0 (NO) / None (NA or unparseable)."""
    if isinstance(val, dict):
        for k in ("verdict", "answer", "value", "score", "rating", "result", "met"):
            if k in val:
                return _coerce_binary(val[k])
        return None
    if isinstance(val, bool):
        return YES if val else NO
    if isinstance(val, (int, float)):
        if isinstance(val, float) and math.isnan(val):
            return None
        v = int(round(val))
        return v if v in (0, 1) else None
    if isinstance(val, str):
        s = val.strip().strip(".!").lower()
        if s in _NA:
            return None
        if s in _TRUE:
            return YES
        if s in _FALSE:
            return NO
        # embedded in a sentence — check the negatives first so "not met" never reads as "met"
        if re.search(r"\b(not met|no\b|false|absent|not observed)", s):
            return NO
        if re.search(r"\b(yes|true|met|observed|present)\b", s):
            return YES
        return None
    return None


def _evidence(val):
    if isinstance(val, dict):
        for k in ("evidence", "justification", "reason", "note"):
            if isinstance(val.get(k), str):
                return val[k][:300]
    return ""


def _margin(val):
    """-> 'BORDERLINE' | 'CLEAR' | '' (not reported)."""
    if not isinstance(val, dict):
        return ""
    for k in ("margin", "confidence", "closeness", "distance"):
        v = val.get(k)
        if isinstance(v, str):
            s = v.strip().lower()
            if "border" in s or "close" in s or "marginal" in s:
                return "BORDERLINE"
            if "clear" in s or "obvious" in s or "high" in s:
                return "CLEAR"
    return ""


def _confidence(val):
    """-> float in [0,1], or nan. Accepts 0.85, '0.85', '85%', 85."""
    if not isinstance(val, dict):
        return np.nan
    for k in ("confidence", "conf", "p_agree", "probability", "certainty", "p"):
        v = val.get(k)
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, str):
            m = re.search(r"-?\d*\.?\d+", v)
            if not m:
                continue
            try:
                v = float(m.group())
            except ValueError:
                continue
            if "%" in val[k]:
                v /= 100.0
        if isinstance(v, (int, float)):
            v = float(v)
            if v != v:
                continue
            if 1.0 < v <= 100.0:          # tolerate "85" meaning 85%
                v /= 100.0
            return float(min(max(v, 0.0), 1.0))
    return np.nan


def _flip_if(val):
    if isinstance(val, dict):
        for k in ("flip_if", "flipif", "would_flip", "flip", "what_would_flip"):
            if isinstance(val.get(k), str):
                return val[k][:300]
    return ""


def yes_bar_text(code, yes_at=3):
    """The exact threshold sentence separating YES from NO for one indicator —
    the level-`yes_at` descriptor, verbatim from the framework."""
    for s in FRAMEWORK.values():
        for ind in s["indicators"]:
            if ind["code"] == code:
                return _descriptors(ind)[yes_at - 1]
    return ""


def _brace_slice(text):
    """Longest balanced {...} span in the text."""
    for s in (i for i, c in enumerate(text) if c == "{"):
        depth, in_str, esc = 0, False, False
        for i in range(s, len(text)):
            c = text[i]
            if in_str:
                if esc:         esc = False
                elif c == "\\": esc = True
                elif c == '"':  in_str = False
                continue
            if c == '"':   in_str = True
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[s:i + 1]
    return None


# NA / N/A first so they never get eaten by the N alternative; YES before Y, NO before N.
_RE_VERDICT = (r'["\']?\b{code}\b["\']?\s*[:\-=]\s*'
               r'(?:\{{[^}}]*?["\'](?:verdict|answer|value|score|result)["\']\s*:\s*)?'
               r'["\']?(N/A|NA|YES|NO|TRUE|FALSE|1|0|Y|N)\b')


def parse_verdicts(text, codes):
    """-> (verdicts {code: 1|0|None}, details {code: {evidence, margin, flip_if}}, method).

    Three tiers: strict JSON, longest balanced brace slice, then per-key regex gap-fill.
    `details` is empty-string-filled when the model was not asked to explain itself, so
    callers never have to branch on whether EXPLAIN was on.
    """
    out = {c: None for c in codes}
    det = {c: dict(evidence="", margin="", flip_if="", confidence=np.nan) for c in codes}
    payload, method = None, "regex"

    body = re.sub(r"<reasoning>.*?</reasoning>", " ", text, flags=re.S | re.I)
    body = re.sub(r"^\s*```(?:json)?|```\s*$", " ", body.strip(), flags=re.M)

    for cand, name in ((body.strip(), "strict"), (_brace_slice(body), "brace")):
        if not cand:
            continue
        for attempt in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):     # drop trailing commas
            try:
                payload, method = json.loads(attempt), name
                break
            except Exception:
                payload = None
        if payload is not None:
            break

    if isinstance(payload, dict):
        # tolerate {"verdicts": {...}} and {"B1": {...}} alike
        if len(payload) == 1 and isinstance(next(iter(payload.values())), dict) \
           and not set(payload) & set(codes):
            payload = next(iter(payload.values()))
        upper = {str(k).strip().upper(): v for k, v in payload.items()}
        for c in codes:
            if c in upper:
                out[c] = _coerce_binary(upper[c])
                det[c] = dict(evidence=_evidence(upper[c]), margin=_margin(upper[c]),
                              flip_if=_flip_if(upper[c]),
                              confidence=_confidence(upper[c]))

    if any(v is None for v in out.values()):
        for c in codes:
            if out[c] is not None:
                continue
            m = re.search(_RE_VERDICT.format(code=c), text, flags=re.I)
            if m:
                out[c] = _coerce_binary(m.group(1))
    if all(v is None for v in out.values()) and payload is None:
        method = "none"
    return out, det, method


# ------------------------------------------------------------ binary statistics
def wilson_ci(k, n, level=0.95):
    """Wilson score interval for a proportion. Honest at k=0 and k=n, where the
    bootstrap of a Bernoulli sample degenerates to a zero-width interval."""
    if n == 0:
        return (np.nan, np.nan)
    z = scipy_stats.norm.ppf(0.5 + level / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def gwet_ac1(data, categories=(0, 1)):
    """Gwet's AC1 for multiple raters, tolerant of missing cells.

    data: units x raters, NaN = missing. Units with <2 ratings are dropped.
    Chance agreement is estimated as the probability two raters agree *by
    accident*, which — unlike kappa's — does not blow up when one category
    dominates. This is the coefficient to quote when a binary indicator is
    almost always NO (or almost always YES) yet the runs agree with each other.
    """
    data = np.asarray(data, float)
    rows = []
    for row in data:
        r = row[~np.isnan(row)]
        if len(r) >= 2:
            rows.append(r)
    if not rows:
        return np.nan
    q = len(categories)
    pa_terms, pi = [], np.zeros(q)
    for r in rows:
        ri = len(r)
        cnt = np.array([(r == c).sum() for c in categories], float)
        pa_terms.append(float((cnt * (cnt - 1)).sum() / (ri * (ri - 1))))
        pi += cnt / ri
    pa = float(np.mean(pa_terms))
    pi = pi / len(rows)
    pe = float((pi * (1 - pi)).sum() / (q - 1))
    return float((pa - pe) / (1 - pe)) if pe < 1 else np.nan


def pabak(po):
    """Prevalence-and-bias-adjusted kappa = 2*Po - 1. A crude second opinion on
    the same paradox: it depends only on raw agreement."""
    return float(2 * po - 1) if po == po else np.nan


def cochran_q(data):
    """Cochran's Q — Friedman's binary analogue. H0: the YES rate is the same in
    every run. data: units(indicators) x raters(runs), complete rows only."""
    Y = np.asarray(data, float)
    Y = Y[~np.isnan(Y).any(axis=1)]
    if Y.size == 0:
        return dict(Q=np.nan, df=np.nan, p=np.nan, n_units=0)
    n, k = Y.shape
    if n < 2 or k < 2:
        return dict(Q=np.nan, df=np.nan, p=np.nan, n_units=n)
    R = Y.sum(axis=1)                       # per-indicator YES count across runs
    C = Y.sum(axis=0)                       # per-run YES count across indicators
    denom = k * R.sum() - (R ** 2).sum()    # all-YES and all-NO rows contribute 0
    if denom <= 0:
        return dict(Q=np.nan, df=k - 1, p=np.nan, n_units=n)
    Q = (k - 1) * (k * (C ** 2).sum() - C.sum() ** 2) / denom
    return dict(Q=float(Q), df=k - 1,
                p=float(scipy_stats.chi2.sf(Q, k - 1)), n_units=n)


def fleiss_kappa_binary(data):
    """Fleiss' kappa on {0,1}, complete rows only. Kept for continuity with the
    1-4 report — read it next to AC1, not instead of it."""
    data = np.asarray(data, float)
    data = data[~np.isnan(data).any(axis=1)]
    if data.size == 0:
        return np.nan
    N, k = data.shape
    if N == 0 or k < 2:
        return np.nan
    C = np.array([[(row == c).sum() for c in (0, 1)] for row in data], float)
    Pi = ((C ** 2).sum(axis=1) - k) / (k * (k - 1))
    Pbar = Pi.mean()
    pj = C.sum(axis=0) / (N * k)
    Pe = (pj ** 2).sum()
    return float((Pbar - Pe) / (1 - Pe)) if Pe < 1 else np.nan


def binary_entropy(p):
    """Shannon entropy of a Bernoulli(p), normalised to [0,1]. 1 = coin flip."""
    if not (0 <= p <= 1) or p != p:
        return np.nan
    if p in (0.0, 1.0):
        return 0.0
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


def p_vs_coin(k, n):
    """Two-sided exact binomial against p=0.5. A LARGE p-value is the bad news:
    it means this indicator's verdict is statistically indistinguishable from
    tossing a coin."""
    if n < 2:
        return np.nan
    return float(scipy_stats.binomtest(int(k), int(n), 0.5).pvalue)


def votes_needed(p_hat, target=0.95, max_n=99):
    """Smallest ODD number of independent passes whose majority vote reproduces
    the observed majority verdict with probability >= target.

    1 means one pass is enough. Anything above ~7 means the indicator is not
    worth voting on — fix the rubric instead of buying more passes.
    """
    if p_hat != p_hat:
        return np.nan
    p = max(p_hat, 1 - p_hat)
    if p >= 1.0:
        return 1
    if p <= 0.5:
        return np.nan                        # a coin flip never converges
    for n in range(1, max_n + 1, 2):
        if scipy_stats.binom.sf((n - 1) // 2, n, p) >= target:
            return n
    return np.nan


def grade_binary(modal_share, na_rate):
    """Verdict stability grade. Tighter than the 1-4 grades on purpose: there is
    no 'adjacent band' to hide in, so 70% modal agreement is already bad."""
    if na_rate >= 0.5:              return "severe"
    if modal_share >= 0.999:        return "stable"
    if modal_share >= 0.90:         return "minor"
    if modal_share >= 0.70:         return "material"
    return "severe"


def fidelity_band(pct, high=0.85, med=0.60):
    """Section B's own banding, applied to the YES rate."""
    if pct != pct:   return "—"
    if pct >= high:  return "High"
    if pct >= med:   return "Medium"
    return "Low"


# ---------------------------------------------------------------- the analysis
def analyse_binary(scores_long, cfg, run_meta=None):
    """-> dict of DataFrames + headline dict, the binary counterpart of
    analysis.analyse. `scores_long.score` holds 1 (YES) / 0 (NO) / NaN (NA)."""
    S.init(cfg)
    CFG = cfg
    yes_at = getattr(cfg, "YES_AT", 3)
    high, med = getattr(cfg, "HIGH_BAND", 0.85), getattr(cfg, "MED_BAND", 0.60)
    target = getattr(cfg, "VOTE_TARGET", 0.95)
    EXPECTED = [c for c in ALL_CODES if CODE2SECTION[c] in cfg.SECTIONS]

    # reindexed against EXPECTED on purpose — an indicator answered NA in every run must
    # still appear as a row (na_rate = 1.0), not silently vanish from the report
    piv = (scores_long.pivot_table(index="code", columns="iteration", values="score",
                                   observed=True, dropna=False)
           .reindex(index=EXPECTED, columns=range(CFG.N_ITERATIONS)))
    wide = piv.set_axis(pd.MultiIndex.from_arrays(
        [[CODE2SECTION[c] for c in piv.index], list(piv.index)], names=["section", "code"]))
    wide.columns = [f"run{c + 1}" for c in wide.columns]
    wide.insert(0, "indicator", [CODE2NAME[c] for _, c in wide.index])
    runcols = wide.filter(like="run")
    wide["n_yes"] = runcols.sum(axis=1, min_count=1)
    wide["flips"] = runcols.nunique(axis=1) > 1

    MATRIX = wide.filter(like="run").to_numpy(float)          # indicators x runs
    IND_CODES = [c for _, c in wide.index]

    BLANK = dict.fromkeys(
        ["n_yes", "p_yes", "verdict", "modal_share", "sd", "ci_lo", "ci_hi", "ci_width",
         "entropy", "flip", "disagree_rate", "single_pass_error", "n_disagree",
         "p_wobble", "p_coinflip", "votes_needed"], np.nan)

    rows = []
    for code_, xs in zip(IND_CODES, MATRIX):
        x = xs[~np.isnan(xs)]
        n_na = int(np.isnan(xs).sum())
        if len(x) == 0:
            rows.append(dict(section=CODE2SECTION[code_], code=code_,
                             indicator=CODE2NAME[code_], n=0, na_rate=1.0,
                             grade="severe", **{**BLANK, "verdict": "—", "flip": False}))
            continue
        k = int(x.sum())
        n = len(x)
        p = k / n
        modal_share = max(p, 1 - p)
        lo, hi = wilson_ci(k, n, CFG.CI_LEVEL)
        p_w, k_dis, _ = p_wobble_test(x)
        rows.append(dict(
            section=CODE2SECTION[code_], code=code_, indicator=CODE2NAME[code_],
            n=n, na_rate=n_na / len(xs),
            n_yes=k, p_yes=p,
            verdict=("YES" if p > 0.5 else "NO" if p < 0.5 else "TIED"),
            modal_share=modal_share,
            sd=float(math.sqrt(p * (1 - p))),          # Bernoulli SD, population form
            ci_lo=lo, ci_hi=hi, ci_width=hi - lo,
            entropy=binary_entropy(p),
            flip=bool(0 < k < n),
            disagree_rate=float(2 * p * (1 - p)),      # chance two runs contradict each other
            single_pass_error=float(1 - modal_share),  # chance one run contradicts the majority
            n_disagree=k_dis, p_wobble=p_w,
            p_coinflip=p_vs_coin(k, n),
            votes_needed=votes_needed(p, target),
        ))

    ind_stats = pd.DataFrame(rows)
    if int(ind_stats.n.sum()) == 0:
        raise RuntimeError(
            "No indicator produced a single parseable verdict. The model is probably refusing "
            "the JSON format — try --prompt-variant terse, or a different --model.")
    ind_stats["q_wobble"] = holm(ind_stats["p_wobble"].to_numpy(float))
    ind_stats["sig_wobble"] = ind_stats["q_wobble"] < CFG.ALPHA
    ind_stats["grade"] = [grade_binary(r.modal_share, r.na_rate) if r.n else "severe"
                          for r in ind_stats.itertuples()]
    ind_stats["section"] = pd.Categorical(ind_stats.section, list(CFG.SECTIONS), ordered=True)
    ind_stats = ind_stats.sort_values(
        ["section", "code"], key=lambda s: s.map(CODE_ORDER) if s.name == "code" else s)

    def reliability_block(mat, label):
        mat = np.asarray(mat, float)
        complete = mat[~np.isnan(mat).any(axis=1)]
        po = pairwise_agreement(mat)
        cq = cochran_q(mat)
        obs = mat[~np.isnan(mat)]
        prev = float(obs.mean()) if obs.size else np.nan
        alpha = krippendorff_alpha(mat, "nominal")   # 2 categories: ordinal == nominal
        ac1 = gwet_ac1(mat)
        return dict(
            scope=label, n_indicators=mat.shape[0], n_runs=mat.shape[1],
            n_complete=len(complete), yes_prevalence=prev,
            pairwise_exact_agreement=po,
            kripp_alpha=alpha, gwet_ac1=ac1, pabak=pabak(po),
            fleiss_kappa=fleiss_kappa_binary(mat),
            # the tell-tale: high raw agreement + low alpha = prevalence paradox, not noise
            prevalence_paradox=bool(po == po and alpha == alpha and po >= 0.80
                                    and alpha < 0.50),
            cochran_q=cq["Q"], cochran_df=cq["df"], cochran_p=cq["p"])

    rel_rows = [reliability_block(MATRIX, "OVERALL (all sections)")]
    rel_at = {"ALL": 0}                       # section -> row position in `reliability`
    for s in CFG.SECTIONS:
        rel_at[s] = len(rel_rows)
        rel_rows.append(reliability_block(wide.loc[s].filter(like="run").to_numpy(float),
                                          f"Section {s} — {FRAMEWORK[s]['title']}"))
    reliability = pd.DataFrame(rel_rows)

    # ---- per-run section YES rate = the fidelity percentage a coaching report quotes
    _ok = scores_long.dropna(subset=["score"])
    sec_run = (_ok.groupby(["section", "iteration"], observed=True)["score"].mean().unstack()
               .reindex(index=list(CFG.SECTIONS), columns=range(CFG.N_ITERATIONS)))
    overall_run = _ok.groupby("iteration")["score"].mean().reindex(range(CFG.N_ITERATIONS))

    _BAND_ORDER = {"Low": 0, "Medium": 1, "High": 2, "—": 3}

    def band_row(xs):
        """Bands of the individual runs, worst-first. `modal` is the band most runs
        landed in, which is NOT necessarily the band of the section's mean — when
        those two disagree the section is sitting on a boundary, which is exactly
        what band_flips is there to flag."""
        bands = [fidelity_band(v, high, med) for v in xs if v == v]
        uniq = sorted(set(bands), key=lambda b: _BAND_ORDER[b])
        modal = max(uniq, key=bands.count) if bands else "—"
        return bands, uniq, modal

    sec_rows = []
    for s in list(CFG.SECTIONS) + ["ALL"]:
        xs = (overall_run if s == "ALL" else sec_run.loc[s]).to_numpy(float)
        sub = ind_stats if s == "ALL" else ind_stats[ind_stats.section == s]
        valid = xs[~np.isnan(xs)]
        bands, uniq, modal = band_row(xs)
        # Bootstrap over the PER-RUN section rates, not Wilson over the pooled cells. The
        # cells inside a section are clustered by indicator and are nowhere near independent
        # Bernoulli draws, so a pooled Wilson interval here would be far too narrow and would
        # claim precision the design does not have. Resampling runs measures the thing this
        # report is actually about: how much the section's fidelity % moves between passes.
        lo, hi = boot_ci(xs) if valid.size else (np.nan, np.nan)
        sec_rows.append(dict(
            section=s,
            title="All sections" if s == "ALL" else FRAMEWORK[s]["title"],
            n_indicators=len(sub),
            yes_rate=float(np.nanmean(xs)) if valid.size else np.nan,
            sd_across_runs=float(np.nanstd(xs, ddof=1)) if valid.size > 1 else 0.0,
            min_run=float(np.nanmin(xs)) if valid.size else np.nan,
            max_run=float(np.nanmax(xs)) if valid.size else np.nan,
            ci_lo=lo, ci_hi=hi, ci_width=hi - lo,
            # `band` is the band of the reported figure — the one a coaching report would
            # quote. `modal_run_band` is where most individual runs landed. They diverge
            # when the section sits on a boundary; band_flips counts how far it ranged.
            band=fidelity_band(float(np.nanmean(xs)) if valid.size else np.nan, high, med),
            modal_run_band=modal, bands_seen="/".join(uniq),
            band_flips=max(0, len(uniq) - 1),
            pct_unanimous=float((sub.grade == "stable").mean()),
            n_flipping=int(sub.flip.fillna(False).sum()),
            pct_flipping=float(sub.flip.fillna(False).mean()),
            mean_single_pass_error=float(sub.single_pass_error.mean(skipna=True)),
            median_votes_needed=float(sub.votes_needed.median(skipna=True)),
            na_rate=float(sub.na_rate.mean()),
            kripp_alpha=float(reliability.iloc[rel_at[s]]["kripp_alpha"]),
            gwet_ac1=float(reliability.iloc[rel_at[s]]["gwet_ac1"]),
        ))
    sec_stats = pd.DataFrame(sec_rows)

    # ---- calibration: was the stated confidence any good?
    #
    # The prompt defines confidence operationally — "P(a second equally careful observer
    # returns the same verdict)" — so it has a directly observable counterpart. For a cell
    # whose indicator scored k YES out of n runs, the leave-one-out probability that
    # ANOTHER run agrees with this cell is (k-1)/(n-1) for a YES, (n-k-1)/(n-1) for a NO.
    # Leave-one-out matters: including the cell in its own reference inflates agreement.
    has_conf = "confidence" in scores_long.columns
    calib, calib_bins = {}, pd.DataFrame()
    if has_conf:
        cal = scores_long.dropna(subset=["score"]).copy()
        cal["confidence"] = pd.to_numeric(cal["confidence"], errors="coerce")
        kmap = ind_stats.set_index("code")["n_yes"].to_dict()
        nmap = ind_stats.set_index("code")["n"].to_dict()

        def _loo(row):
            k, n_ = kmap.get(row["code"], np.nan), nmap.get(row["code"], np.nan)
            if n_ != n_ or n_ < 2:
                return np.nan
            same = (k - 1) if row["score"] == 1 else (n_ - k - 1)
            return float(same) / (n_ - 1)

        cal["observed_agreement"] = cal.apply(_loo, axis=1)
        cal = cal.dropna(subset=["confidence", "observed_agreement"])
        if len(cal):
            err = cal["confidence"] - cal["observed_agreement"]
            calib = dict(
                n_cells=int(len(cal)),
                mean_confidence=float(cal["confidence"].mean()),
                mean_observed_agreement=float(cal["observed_agreement"].mean()),
                overconfidence=float(err.mean()),          # >0 = claims more than it earns
                expected_calibration_error=float(err.abs().mean()),
                calibration_mse=float((err ** 2).mean()),
                pct_cells_below_half=float((cal["confidence"] < 0.5).mean()),
                mean_confidence_yes=float(cal.loc[cal.score == 1, "confidence"].mean())
                if (cal.score == 1).any() else np.nan,
                mean_confidence_no=float(cal.loc[cal.score == 0, "confidence"].mean())
                if (cal.score == 0).any() else np.nan,
                spearman_rho=float(scipy_stats.spearmanr(
                    cal["confidence"], cal["observed_agreement"]).statistic)
                if cal["confidence"].nunique() > 1 else np.nan,
                spearman_p=float(scipy_stats.spearmanr(
                    cal["confidence"], cal["observed_agreement"]).pvalue)
                if cal["confidence"].nunique() > 1 else np.nan)

            edges = [0.0, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0001]
            lbl = ["<0.60", "0.60-0.69", "0.70-0.79", "0.80-0.89", "0.90-0.94", "0.95-1.00"]
            cal["bin"] = pd.cut(cal["confidence"], bins=edges, labels=lbl, right=False)
            calib_bins = (cal.groupby("bin", observed=False)
                          .agg(n_cells=("confidence", "size"),
                               mean_stated=("confidence", "mean"),
                               observed_agreement=("observed_agreement", "mean"))
                          .reset_index())
            calib_bins["gap"] = calib_bins["mean_stated"] - calib_bins["observed_agreement"]

        # per-indicator confidence summary
        g = cal.groupby("code", observed=True)
        ind_stats["mean_confidence"] = ind_stats["code"].map(g["confidence"].mean())
        ind_stats["min_confidence"] = ind_stats["code"].map(g["confidence"].min())
        yy = cal[cal.score == 1].groupby("code", observed=True)["confidence"].mean()
        nn = cal[cal.score == 0].groupby("code", observed=True)["confidence"].mean()
        ind_stats["conf_when_yes"] = ind_stats["code"].map(yy)
        ind_stats["conf_when_no"] = ind_stats["code"].map(nn)
        ind_stats["calibration_gap"] = (ind_stats["mean_confidence"]
                                        - (1 - ind_stats["single_pass_error"]))
    else:
        for c_ in ("mean_confidence", "min_confidence", "conf_when_yes", "conf_when_no",
                   "calibration_gap"):
            ind_stats[c_] = np.nan

    # ---- decision boundaries: the threshold each verdict was measured against, how
    # close to it the model said it was, and what it said would flip the call
    has_margin = "margin" in scores_long.columns or has_conf
    if has_margin:
        _m = scores_long.dropna(subset=["score"]).copy()
        if has_conf:
            # derive the flag from the number so both views stay consistent
            _c = pd.to_numeric(_m["confidence"], errors="coerce")
            _m["borderline"] = _c < getattr(cfg, "CONF_BORDERLINE", 0.85)
        else:
            _m["margin"] = _m["margin"].fillna("").astype(str).str.upper()
            _m["borderline"] = _m["margin"].eq("BORDERLINE")
        bl = _m.groupby("code", observed=True)["borderline"].mean()
        ind_stats["borderline_rate"] = ind_stats["code"].map(bl)
        ind_stats["ever_borderline"] = ind_stats["borderline_rate"].fillna(0) > 0
    else:
        ind_stats["borderline_rate"] = np.nan
        ind_stats["ever_borderline"] = False

    def _join(series, limit=3):
        seen, out_ = [], []
        for v in series:
            v = ("" if v is None else str(v)).strip()
            if v and v.lower() not in seen:
                seen.append(v.lower()); out_.append(v)
            if len(out_) >= limit:
                break
        return " | ".join(out_)

    def _join_conf(sub, limit=3):
        """Distinct reasons, each tagged with the confidence stated on that run, so a
        reason and the certainty behind it are never read apart."""
        if sub is None or not len(sub):
            return ""
        seen, out_ = [], []
        confs = (sub["confidence"] if "confidence" in sub
                 else pd.Series([np.nan] * len(sub), index=sub.index))
        for ev, cf in zip(sub["evidence"], confs):
            ev = ("" if ev is None else str(ev)).strip()
            if not ev or ev.lower() in seen:
                continue
            seen.append(ev.lower())
            out_.append((f"[{cf:.2f}] " if cf == cf else "") + ev)
            if len(out_) >= limit:
                break
        return " | ".join(out_)

    src = scores_long.dropna(subset=["score"]).copy() if has_margin else None
    if src is not None and "confidence" in src:
        src["confidence"] = pd.to_numeric(src["confidence"], errors="coerce")
    bound_rows = []
    for r in ind_stats.itertuples():
        sub = src[src.code == r.code] if src is not None else None
        yes_side = sub[sub.score == 1] if sub is not None else None
        no_side = sub[sub.score == 0] if sub is not None else None
        bound_rows.append(dict(
            section=r.section, code=r.code, indicator=r.indicator,
            yes_bar=yes_bar_text(r.code, yes_at),
            verdict=r.verdict, n_yes=r.n_yes, n=r.n, p_yes=r.p_yes,
            flip=r.flip, borderline_rate=r.borderline_rate,
            mean_confidence=r.mean_confidence, conf_when_yes=r.conf_when_yes,
            conf_when_no=r.conf_when_no,
            observed_agreement=(1 - r.single_pass_error)
            if r.single_pass_error == r.single_pass_error else np.nan,
            calibration_gap=r.calibration_gap,
            yes_reasons=_join_conf(yes_side),
            no_reasons=_join_conf(no_side),
            flip_if=_join(sub["flip_if"]) if sub is not None and "flip_if" in sub else ""))
    boundaries = pd.DataFrame(bound_rows)

    # Carry the threshold and the reasoning onto the per-indicator stats table too, so
    # indicator_wobble.csv is self-contained: the number, the bar it was judged against,
    # and why each side was argued — without cross-referencing a second file.
    if len(boundaries):
        _b = boundaries.set_index("code")
        for col in ("yes_bar", "yes_reasons", "no_reasons", "flip_if"):
            ind_stats[col] = ind_stats["code"].map(_b[col])

    # Does the model's own BORDERLINE flag predict which verdicts actually move?
    # If it does, the flag is a usable triage signal; if not, the instability is
    # invisible to the scorer and only repeated runs can find it.
    boundary_test = {}
    if has_margin and ind_stats["borderline_rate"].notna().any():
        d = ind_stats[ind_stats.n > 0]
        a = int(((d.ever_borderline) & (d.flip == True)).sum())
        b = int(((d.ever_borderline) & (d.flip != True)).sum())
        c_ = int(((~d.ever_borderline) & (d.flip == True)).sum())
        dd = int(((~d.ever_borderline) & (d.flip != True)).sum())
        try:
            orr, pv = scipy_stats.fisher_exact([[a, b], [c_, dd]])
        except Exception:
            orr, pv = np.nan, np.nan
        boundary_test = dict(
            borderline_and_flipped=a, borderline_not_flipped=b,
            clear_but_flipped=c_, clear_and_stable=dd,
            flip_rate_when_borderline=(a / (a + b)) if (a + b) else np.nan,
            flip_rate_when_clear=(c_ / (c_ + dd)) if (c_ + dd) else np.nan,
            odds_ratio=float(orr) if orr == orr else np.nan,
            fisher_p=float(pv) if pv == pv else np.nan)

    flippy = ind_stats[ind_stats.flip == True].sort_values(
        ["single_pass_error", "code"], ascending=[False, True])
    undecided = ind_stats[(ind_stats.p_coinflip > 0.20) & (ind_stats.flip == True)]

    allrow = sec_stats[sec_stats.section == "ALL"].iloc[0]
    r0 = reliability.iloc[0]
    HEADLINE = {
        "scale": f"binary YES/NO at level >= {yes_at} ({LEVEL_LABEL[yes_at]})",
        "model": cfg.MODEL, "backend": cfg.BACKEND,
        "iterations": CFG.N_ITERATIONS, "indicators": len(ind_stats),
        "overall_yes_rate": round(float(allrow.yes_rate), 3),
        "overall_yes_ci": [round(float(allrow.ci_lo), 3), round(float(allrow.ci_hi), 3)],
        "overall_yes_rate_sd_across_runs": round(float(allrow.sd_across_runs), 4),
        "fidelity_band": allrow.band, "fidelity_bands_seen": allrow.bands_seen,
        "kripp_alpha": round(float(r0.kripp_alpha), 3),
        "gwet_ac1": round(float(r0.gwet_ac1), 3),
        "pairwise_exact_agreement": round(float(r0.pairwise_exact_agreement), 3),
        "pabak": round(float(r0.pabak), 3),
        "yes_prevalence": round(float(r0.yes_prevalence), 3),
        "prevalence_paradox": bool(r0.prevalence_paradox),
        "pct_indicators_unanimous": round(float((ind_stats.grade == "stable").mean()), 3),
        "n_indicators_flipping_verdict": int(ind_stats.flip.fillna(False).sum()),
        "pct_indicators_flipping_verdict": round(float(ind_stats.flip.fillna(False).mean()), 3),
        "n_indicators_coinflip": int(len(undecided)),
        "mean_single_pass_verdict_error": round(float(ind_stats.single_pass_error.mean()), 3),
        "median_votes_needed": (None if ind_stats.votes_needed.isna().all()
                                else float(ind_stats.votes_needed.median(skipna=True))),
        "n_indicators_never_reproducible": int(ind_stats.votes_needed.isna().sum()
                                               - (ind_stats.n == 0).sum()),
        "pct_indicators_sig_wobble": round(float(ind_stats.sig_wobble.mean()), 3),
        "cochran_q_p": (None if r0.cochran_p != r0.cochran_p else round(float(r0.cochran_p), 4)),
        "mean_ci_width": round(float(ind_stats.ci_width.mean(skipna=True)), 3),
        "na_rate": round(float(ind_stats.na_rate.mean()), 3),
        "parse_failure_rate": round(float((scores_long.parse == "none").mean()), 4),
        "effort": cfg.EFFORT, "thinking": cfg.THINKING,
        "scoring_mode": cfg.SCORING_MODE, "prompt_variant": cfg.PROMPT_VARIANT,
    }
    if run_meta:
        HEADLINE.update({k: v for k, v in run_meta.items() if k not in HEADLINE})

    if has_margin:
        HEADLINE["borderline_rate"] = round(
            float(ind_stats.borderline_rate.mean(skipna=True)), 3)
        HEADLINE["n_indicators_ever_borderline"] = int(ind_stats.ever_borderline.sum())
        if boundary_test:
            HEADLINE["flip_rate_when_borderline"] = (
                None if boundary_test["flip_rate_when_borderline"] !=
                boundary_test["flip_rate_when_borderline"]
                else round(boundary_test["flip_rate_when_borderline"], 3))
            HEADLINE["flip_rate_when_clear"] = (
                None if boundary_test["flip_rate_when_clear"] !=
                boundary_test["flip_rate_when_clear"]
                else round(boundary_test["flip_rate_when_clear"], 3))
            HEADLINE["borderline_predicts_flip_p"] = (
                None if boundary_test["fisher_p"] != boundary_test["fisher_p"]
                else round(boundary_test["fisher_p"], 4))

    if calib:
        HEADLINE.update(
            mean_confidence=round(calib["mean_confidence"], 3),
            mean_confidence_yes=round(calib["mean_confidence_yes"], 3),
            mean_confidence_no=round(calib["mean_confidence_no"], 3),
            mean_observed_agreement=round(calib["mean_observed_agreement"], 3),
            overconfidence=round(calib["overconfidence"], 3),
            expected_calibration_error=round(calib["expected_calibration_error"], 3),
            confidence_rank_correlation=(None if calib["spearman_rho"] != calib["spearman_rho"]
                                         else round(calib["spearman_rho"], 3)))

    return dict(wide=wide, matrix=MATRIX, ind_codes=IND_CODES, ind_stats=ind_stats,
                reliability=reliability, sec_stats=sec_stats, sec_run=sec_run,
                overall_run=overall_run, headline=HEADLINE, flippy=flippy,
                undecided=undecided, yes_at=yes_at, boundaries=boundaries,
                boundary_test=boundary_test, has_margin=has_margin,
                has_conf=has_conf, calib=calib, calib_bins=calib_bins)
