"""Scoring prompts and JSON parsing — same prompts and same three-tier parser
as the Colab notebook, with the context passed in explicitly instead of read
from notebook globals."""
import json, math, random, re

from .framework import (FRAMEWORK, SECTION_CODES, CODE2NAME, CODE2SECTION,
                        ALL_CODES, render_section_rubric)

SCORING_SYSTEM = (
    "You are a trained classroom observer for Taleemabad, scoring a lesson against the "
    "Taleemabad Coaching Framework. You are strict, evidence-bound and calibrated:\n"
    "- Score ONLY on evidence present in the material provided. Absence of evidence is "
    "score 1, never a generous guess.\n"
    "- A score of 3 requires the level-3 descriptor to be clearly met, not partially.\n"
    "- Do not reward effort, warmth or busyness when the descriptor asks for something else.\n"
    "- Output valid JSON only. No preamble, no markdown fences, no commentary.")

_NA_CLAUSE = ('If an indicator genuinely cannot apply to this lesson (e.g. a MATH-specific '
              'indicator in a language lesson), use the string "NA" instead of a number.')

_JSON_SHAPE_EV = ('{{"{first}": {{"score": <1|2|3|4>, "evidence": "<max 20 words quoted or '
                  'paraphrased from the material>"}}, ...}}')
_JSON_SHAPE_NO = '{{"{first}": <1|2|3|4>, ...}}'

def build_scoring_prompt(cfg, section_code, codes, context_text, context_kind,
                         session_meta, variant=None):
    variant = variant or cfg.PROMPT_VARIANT
    rubric  = render_section_rubric(section_code, codes=codes, terse=(variant == "terse"))
    shape   = (_JSON_SHAPE_EV if cfg.INCLUDE_EVIDENCE else _JSON_SHAPE_NO).format(first=codes[0])
    na      = ("\n" + _NA_CLAUSE) if cfg.ALLOW_NA else ""
    keys    = ", ".join(codes)

    task = {
        "standard": (
            f"Score EVERY indicator listed below on the 1-4 scale, using the level descriptors "
            f"as the definition of each score. Whole numbers only.{na}\n\n"
            f"Return exactly one JSON object with these {len(codes)} keys and nothing else: {keys}\n"
            f"Shape: {shape}"),
        "terse": (
            f"Score each indicator 1-4. Whole numbers.{na}\n"
            f"JSON only, keys: {keys}\nShape: {shape}"),
        "cot": (
            f"For each indicator: first weigh the evidence for and against each level in one "
            f"sentence, then commit to a whole-number score 1-4.{na}\n\n"
            f"Write your reasoning inside a single <reasoning>...</reasoning> block "
            f"(keep it under 25 words per indicator), then output the JSON object with keys "
            f"{keys} after the closing tag.\nShape: {shape}"),
    }[variant]

    user = (f"## MATERIAL TO SCORE\n"
            f"Source: {context_kind} of a lesson observation.\n"
            f"Session: {session_meta['session_id']} | language {session_meta['language']} | "
            f"{session_meta['duration_min']} minutes.\n\n"
            f'"""\n{context_text}\n"""\n\n'
            f"## RUBRIC\n{rubric}\n\n"
            f"## TASK\n{task}")
    return SCORING_SYSTEM, user

def _brace_slice(text):
    """Longest balanced {...} span in the text."""
    starts = [i for i, c in enumerate(text) if c == "{"]
    for s in starts:
        depth, in_str, esc = 0, False, False
        for i in range(s, len(text)):
            c = text[i]
            if in_str:
                if esc:            esc = False
                elif c == "\\":    esc = True
                elif c == '"':     in_str = False
                continue
            if c == '"':   in_str = True
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[s:i + 1]
    return None

def _coerce(val):
    """-> int 1..4, or None for NA / unparseable."""
    if isinstance(val, dict):
        val = val.get("score", val.get("value", val.get("rating")))
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        if isinstance(val, float) and math.isnan(val):
            return None
        v = int(round(val))
        return v if 1 <= v <= 4 else None
    if isinstance(val, str):
        s = val.strip()
        if s.upper() in ("NA", "N/A", "NONE", "NULL", "NOT APPLICABLE", ""):
            return None
        m = re.search(r"[1-4]", s)
        return int(m.group()) if m else None
    return None

def _evidence(val):
    if isinstance(val, dict):
        for k in ("evidence", "justification", "reason", "note"):
            if isinstance(val.get(k), str):
                return val[k][:300]
    return ""

def parse_scores(text, codes):
    """-> (scores {code: int|None}, evidence {code: str}, method str)"""
    scores  = {c: None for c in codes}
    ev      = {c: "" for c in codes}
    payload, method = None, "regex"

    body = re.sub(r"<reasoning>.*?</reasoning>", " ", text, flags=re.S | re.I)
    body = re.sub(r"^\s*```(?:json)?|```\s*$", " ", body.strip(), flags=re.M)

    for cand, name in ((body.strip(), "strict"), (_brace_slice(body), "brace")):
        if not cand:
            continue
        for attempt in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):   # drop trailing commas
            try:
                payload, method = json.loads(attempt), name
                break
            except Exception:
                payload = None
        if payload is not None:
            break

    if isinstance(payload, dict):
        # tolerate {"scores": {...}} and {"B1": {...}} alike
        if len(payload) == 1 and isinstance(next(iter(payload.values())), dict) \
           and not set(payload) & set(codes):
            payload = next(iter(payload.values()))
        upper = {str(k).strip().upper(): v for k, v in payload.items()}
        for c in codes:
            if c in upper:
                scores[c] = _coerce(upper[c]); ev[c] = _evidence(upper[c])

    # fallback / gap-fill by regex
    if any(v is None for v in scores.values()):
        for c in codes:
            if scores[c] is not None:
                continue
            m = re.search(rf'["\']?\b{c}\b["\']?\s*[:\-=]\s*(?:\{{[^}}]*?["\']score["\']\s*:\s*)?'
                          rf'["\']?(NA|N/A|[1-4])', text, flags=re.I)
            if m:
                scores[c] = _coerce(m.group(1))
    return scores, ev, method


def order_codes(cfg, section_code, iteration):
    """Indicator order for one section in one iteration."""
    codes = list(SECTION_CODES[section_code])
    # Excluded indicators are dropped before the prompt is built, so they cost no
    # tokens and cannot influence the model's reading of the ones that remain.
    drop = set(getattr(cfg, "EXCLUDE_CODES", ()) or ())
    if drop:
        codes = [c for c in codes if c not in drop]
    if cfg.INDICATOR_ORDER == "shuffled":
        random.Random(cfg.BASE_SEED + 977 * iteration
                      + sum(map(ord, section_code))).shuffle(codes)
    return codes
