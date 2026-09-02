"""Framework-agnostic rubric spec: load, validate, lint, render.

The original pipeline hard-coded one framework in framework.py. This replaces
that with a spec any coaching framework can be expressed in - Tanzania, a
regional variant, TEACH, a home-grown rubric - without touching the scoring or
statistics code, which never cared about the content anyway.

A spec is YAML (hand-editable) or JSON, or is imported from the one-column-per-
level spreadsheets frameworks usually arrive as.

WHAT THE REVIEW PASS IS - AND WHAT IT IS NOT
--------------------------------------------
`review()` surfaces descriptors a human should read before spending money on a
run. It is NOT a predictor of reliability, and must never be presented as one.

That is a measured claim, not a caution. On the Taleemabad framework - 37
indicators with flip rates measured over 800 scoring passes - every textual
feature tested was uncorrelated with the wobble that was actually observed:

    words in the YES bar      Spearman rho = -0.155   p = 0.37
    sentences in the YES bar  Spearman rho = -0.145   p = 0.40
    clauses in the YES bar    Spearman rho = -0.057   p = 0.74

Rule-based flags did no better. "Two or more sentences" caught 4 of the 7
indicators that genuinely failed, at 21% precision; the literal word "and"
caught NONE of them, because their compound structure is punctuation-joined.

So the compound-descriptor story that came out of that study explains the
failures after the fact; it does not identify them in advance. THE EMPIRICAL
RUN IS NOT OPTIONAL. You cannot inspect a rubric and know which indicators
will hold - you have to score it repeatedly and measure.

What the flags are still good for: a reviewer's reading list, and catching
outright spec defects (identical adjacent levels, one-word descriptors) that
are wrong regardless of how they would have scored.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field

# Words that only exist for the eye. An audio transcript cannot evidence any of
# them, so an indicator resting on one is unscoreable from transcripts however
# well it is worded.
VISUAL_ONLY = [
    "seat", "seated", "seating", "arranged", "arrangement", "displayed", "display",
    "posture", "eye contact", "gaze", "body language", "written on the board",
    "wall", "chart paper", "poster", "raise their hands", "raised hands",
    "circulates", "circulating", "walks around", "layout", "tidy", "visible",
    "gestures", "facial",
]
# A descriptor that states two requirements gives the scorer no rule for the
# common case where a lesson meets one and fails the other.
COMPOUND_JOINERS = [r"\band\b", r";", r"\+", r"\bas well as\b",
                    r"\bwhile also\b", r"\bplus\b"]


@dataclass
class Indicator:
    code: str
    name: str
    levels: dict                                       # {1: "...", 2: "...", ...}
    section: str = ""
    applies_when: dict = field(default_factory=dict)   # e.g. {"subject": "MATH"}
    # Extra documents this indicator cannot be judged without. The observation
    # transcript is always supplied; anything listed here is an ADDITIONAL input,
    # e.g. ["lesson_plan"] for an indicator whose bar is "followed the plan".
    # Scoring such an indicator without its reference document does not produce a
    # low score - it produces a meaningless one.
    requires: list = field(default_factory=list)
    note: str = ""

    def descriptor(self, level):
        return self.levels.get(level, "")


@dataclass
class Section:
    code: str
    title: str
    note: str = ""
    indicators: list = field(default_factory=list)


@dataclass
class Framework:
    name: str
    n_levels: int
    level_labels: list
    proficiency_cut: int
    bands: list                      # [{"name": "High", "min": 0.85}, ...] desc
    sections: list
    version: str = ""
    source: str = ""

    # ---------- derived tables (the names the pipeline already expects) ----------
    @property
    def all_codes(self):
        return [i.code for s in self.sections for i in s.indicators]

    @property
    def code2name(self):
        return {i.code: i.name for s in self.sections for i in s.indicators}

    @property
    def code2section(self):
        return {i.code: s.code for s in self.sections for i in s.indicators}

    @property
    def section_codes(self):
        return {s.code: [i.code for i in s.indicators] for s in self.sections}

    @property
    def indicator_by_code(self):
        return {i.code: i for s in self.sections for i in s.indicators}

    def band_for(self, pct):
        """Percentage -> band name. Bands are sorted high to low on load."""
        if pct != pct:
            return "-"
        for b in self.bands:
            if pct >= b["min"]:
                return b["name"]
        return self.bands[-1]["name"] if self.bands else "-"

    def yes_bar(self, code, at=None):
        """The exact descriptor separating YES from NO for one indicator."""
        at = at or self.proficiency_cut
        return self.indicator_by_code[code].descriptor(at)

    # ---------- prompt rendering ----------
    def _label(self, lv):
        return (self.level_labels[lv - 1]
                if lv - 1 < len(self.level_labels) else f"Level {lv}")

    def render_indicator(self, ind, terse=False):
        if terse:
            body = "\n".join(f"  {lv}={ind.descriptor(lv)}"
                             for lv in range(1, self.n_levels + 1))
            return f"{ind.code} - {ind.name}\n{body}"
        lines = [f"### {ind.code} - {ind.name}"]
        for lv in range(1, self.n_levels + 1):
            lines.append(f"  {lv} ({self._label(lv)}): {ind.descriptor(lv)}")
        if ind.applies_when:
            cond = ", ".join(f"{k}={v}" for k, v in ind.applies_when.items())
            lines.append(f"  [Applies only when {cond}]")
        if ind.requires:
            docs = ", ".join(x.replace("_", " ") for x in ind.requires)
            lines.append(f"  [Judge this against the supplied {docs}, not against "
                         f"the observation alone]")
        return "\n".join(lines)

    def render_binary_indicator(self, ind, yes_at=None, terse=False):
        yes_at = yes_at or self.proficiency_cut
        bar = ind.descriptor(yes_at)
        stronger = [ind.descriptor(l) for l in range(yes_at + 1, self.n_levels + 1)]
        below = [ind.descriptor(l) for l in range(1, yes_at)]
        if terse:
            return (f"{ind.code} - {ind.name}\n  YES: {bar}\n"
                    f"  NO: {' | '.join(below) if below else '(anything weaker)'}")
        lines = [f"### {ind.code} - {ind.name}",
                 "  YES - answer YES only if this is clearly true of the lesson:",
                 f"      {bar}"]
        if stronger:
            lines.append("  (a stronger version of the same thing also counts as YES: "
                         + " ".join(stronger) + ")")
        lines.append("  NO - answer NO if what you see is any of these instead, "
                     "or nothing at all:")
        for lv in range(1, yes_at):
            lines.append(f"      ({self._label(lv)}) {ind.descriptor(lv)}")
        if not below:
            lines.append("      (anything short of the YES bar)")
        if ind.applies_when:
            cond = ", ".join(f"{k}={v}" for k, v in ind.applies_when.items())
            lines.append(f"  [Applies only when {cond}]")
        if ind.requires:
            docs = ", ".join(x.replace("_", " ") for x in ind.requires)
            lines.append(f"  [Judge this against the supplied {docs}, not against "
                         f"the observation alone]")
        return "\n".join(lines)

    def render_section(self, section_code, codes=None, binary=False,
                       yes_at=None, terse=False):
        s = next(x for x in self.sections if x.code == section_code)
        inds = [i for i in s.indicators if codes is None or i.code in codes]
        if codes is not None:
            inds = sorted(inds, key=lambda i: codes.index(i.code))
        head = f"SECTION {s.code} - {s.title.upper()}"
        if s.note:
            head += f"\nSection guidance: {s.note}"
        body = "\n\n".join(
            self.render_binary_indicator(i, yes_at, terse) if binary
            else self.render_indicator(i, terse) for i in inds)
        return head + "\n\n" + body

    def unassessable(self, available):
        """Indicators whose required reference documents are not in `available`.

        Scoring these anyway does not yield a low score - it yields a
        meaningless one, because the model is asked to compare the observation
        against a document it was never shown.
        """
        have = set(available or [])
        return {i.code: [r for r in i.requires if r not in have]
                for s in self.sections for i in s.indicators
                if i.requires and not set(i.requires) <= have}

    def summary(self):
        return (f"{self.name}" + (f" v{self.version}" if self.version else "")
                + f" - {len(self.sections)} sections, {len(self.all_codes)} indicators, "
                + f"{self.n_levels}-level scale, proficiency cut {self.proficiency_cut}")


# ------------------------------------------------------------------ validation
class SpecError(ValueError):
    pass


def validate(fw):
    """Hard errors only - things that make the spec unusable. -> list of strings."""
    errs = []
    if not fw.sections:
        errs.append("no sections defined")
    if fw.n_levels < 2:
        errs.append(f"n_levels must be >= 2, got {fw.n_levels}")
    if not (1 <= fw.proficiency_cut <= fw.n_levels):
        errs.append(f"proficiency_cut {fw.proficiency_cut} outside 1..{fw.n_levels}")
    if len(fw.level_labels) not in (0, fw.n_levels):
        errs.append(f"{len(fw.level_labels)} level_labels for {fw.n_levels} levels")

    seen = {}
    for s in fw.sections:
        if not s.indicators:
            errs.append(f"section {s.code!r} has no indicators")
        for i in s.indicators:
            if not i.code:
                errs.append(f"indicator with no code in section {s.code!r}")
            if i.code in seen:
                errs.append(f"duplicate indicator code {i.code!r} "
                            f"(sections {seen[i.code]} and {s.code})")
            seen[i.code] = s.code
            missing = [lv for lv in range(1, fw.n_levels + 1)
                       if not str(i.descriptor(lv)).strip()]
            if missing:
                errs.append(f"{i.code}: missing descriptor for level(s) {missing}")

    if fw.bands:
        mins = [b["min"] for b in fw.bands]
        if mins != sorted(mins, reverse=True):
            errs.append("bands must be ordered from highest min to lowest")
        if any(not (0 <= m <= 1) for m in mins):
            errs.append("band 'min' values must be fractions between 0 and 1")
    return errs


# ---------------------------------------------------------------------- lint
def _sentences(text):
    return [p.strip() for p in re.split(r"[.;]\s+", str(text)) if p.strip()]


def review(fw, yes_at=None):
    """A reviewer's reading list - NOT a reliability predictor (see module
    docstring for the calibration that establishes this). -> list of dicts.

    MULTI_REQUIREMENT and VISUAL_EVIDENCE are advisory: they mark descriptors
    worth a human read. IDENTICAL_LEVELS and TOO_SHORT are genuine spec defects
    and should be fixed whatever the reliability run later says.
    """
    yes_at = yes_at or fw.proficiency_cut
    out = []
    for s in fw.sections:
        for i in s.indicators:
            bar = str(i.descriptor(yes_at))
            flags, why = [], []

            parts = _sentences(bar)
            joiners = [j for j in COMPOUND_JOINERS if re.search(j, bar, re.I)]
            if len(parts) > 1 or joiners:
                flags.append("MULTI_REQUIREMENT")
                why.append(f"the YES bar appears to state {max(len(parts), 2)} "
                           f"requirements - check that it says which one decides when a "
                           f"lesson meets one and fails the other (advisory only)")

            hits = sorted({w for w in VISUAL_ONLY
                           if re.search(r"\b" + re.escape(w), bar, re.I)})
            if hits:
                flags.append("VISUAL_EVIDENCE")
                why.append("may depend on something an audio transcript cannot "
                           "evidence: " + ", ".join(hits))

            for lv in range(1, fw.n_levels):
                a = str(i.descriptor(lv)).strip().lower()
                b = str(i.descriptor(lv + 1)).strip().lower()
                if a and a == b:
                    flags.append("IDENTICAL_LEVELS")
                    why.append(f"levels {lv} and {lv + 1} have identical wording")
                    break

            if len(bar.split()) < 5:
                flags.append("TOO_SHORT")
                why.append(f"YES bar is only {len(bar.split())} words - likely too "
                           f"vague to judge consistently (spec defect)")

            if flags:
                out.append(dict(section=s.code, code=i.code, indicator=i.name,
                                flags="+".join(sorted(set(flags))),
                                yes_bar=bar, why="; ".join(why),
                                severity=len(set(flags))))
    return sorted(out, key=lambda r: (-r["severity"], r["code"]))


# --------------------------------------------------------------------- loading
def _from_dict(d):
    sections = []
    for sd in d.get("sections", []):
        inds = []
        for idd in sd.get("indicators", []):
            lv = {int(k): v for k, v in (idd.get("levels") or {}).items()}
            inds.append(Indicator(code=str(idd["code"]).strip(),
                                  name=str(idd.get("name", "")).strip(),
                                  levels=lv, section=str(sd["code"]).strip(),
                                  applies_when=idd.get("applies_when") or {},
                                  requires=list(idd.get("requires") or []),
                                  note=idd.get("note", "")))
        sections.append(Section(code=str(sd["code"]).strip(),
                                title=sd.get("title", ""), note=sd.get("note", ""),
                                indicators=inds))
    scale = d.get("scale", {})
    n_levels = int(scale.get("levels", 4))
    bands = list(d.get("bands", []))
    bands.sort(key=lambda b: -b["min"])
    return Framework(name=d.get("name", "unnamed framework"),
                     version=str(d.get("version", "")),
                     source=d.get("source", ""),
                     n_levels=n_levels,
                     level_labels=list(scale.get("labels", [])),
                     proficiency_cut=int(scale.get("proficiency_cut",
                                                   max(1, n_levels - 1))),
                     bands=bands, sections=sections)


def load(path, strict=True):
    """Load from .yaml/.yml/.json, or a directory of level-column CSVs."""
    if os.path.isdir(path):
        fw = from_csv_dir(path)
    else:
        with open(path, encoding="utf-8") as f:
            if path.lower().endswith((".yaml", ".yml")):
                import yaml
                d = yaml.safe_load(f)
            else:
                d = json.load(f)
        fw = _from_dict(d)
    errs = validate(fw)
    if errs and strict:
        raise SpecError(f"{path}: invalid framework spec:\n  - " + "\n  - ".join(errs))
    return fw, errs


def from_csv_dir(d, level_pattern=r"^\s*(\d+)\s*="):
    """Import the spreadsheet shape frameworks usually arrive in: one file per
    section, one COLUMN per level, headed like `3 = Proficient`.

    The section banner (row 0) and the header row are auto-detected, so a sheet
    exported straight from Google Sheets works untouched.
    """
    import pandas as pd
    sections, labels, n_levels = [], [], 0
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        raw = pd.read_csv(f, header=None, dtype=str, encoding="utf-8").fillna("")
        hdr = None
        for r in range(min(6, len(raw))):
            row = [str(x) for x in raw.iloc[r]]
            if (any(re.match(level_pattern, c) for c in row)
                    and any(c.strip().lower() == "code" for c in row)):
                hdr = r
                break
        if hdr is None:
            print(f"  ! {os.path.basename(f)}: no header row with a 'Code' column "
                  f"and level columns, skipped")
            continue
        banner = " ".join(str(x) for x in raw.iloc[0] if str(x).strip())
        cols = [str(x).strip() for x in raw.iloc[hdr]]
        body = raw.iloc[hdr + 1:]
        ci = [k for k, c in enumerate(cols) if c.lower() == "code"][0]
        ni = ci + 1
        lvcols = {}
        for j, c in enumerate(cols):
            m = re.match(level_pattern, c)
            if m:
                lv = int(m.group(1))
                lvcols[lv] = j
                if lv > len(labels):
                    labels.append(c.split("=", 1)[1].strip())
        if not lvcols:
            continue
        n_levels = max(n_levels, max(lvcols))
        stem = os.path.splitext(os.path.basename(f))[0]
        m = re.search(r"-\s*([A-Za-z0-9]+)\.", stem)
        scode = m.group(1) if m else stem[:2].strip()
        inds = []
        for _, row in body.iterrows():
            code = str(row.iloc[ci]).strip()
            if not code or code.lower() == "code":
                continue
            inds.append(Indicator(
                code=code, name=str(row.iloc[ni]).strip(),
                levels={lv: str(row.iloc[j]).strip() for lv, j in lvcols.items()},
                section=scode))
        title = re.sub(r"^.*?-\s*[A-Za-z0-9]+\.\s*", "", stem).strip() or stem
        sections.append(Section(code=scode, title=title, note=banner[:400],
                                indicators=inds))
    return _from_dict(dict(
        name=os.path.basename(os.path.abspath(d)),
        scale=dict(levels=n_levels or 4, labels=labels,
                   proficiency_cut=max(1, (n_levels or 4) - 1)),
        bands=[{"name": "High", "min": 0.85}, {"name": "Medium", "min": 0.60},
               {"name": "Low", "min": 0.0}],
        sections=[dict(code=s.code, title=s.title, note=s.note,
                       indicators=[dict(code=i.code, name=i.name, levels=i.levels)
                                   for i in s.indicators])
                  for s in sections]))


def to_yaml(fw, path):
    import yaml
    d = dict(name=fw.name, version=fw.version, source=fw.source,
             scale=dict(levels=fw.n_levels, labels=fw.level_labels,
                        proficiency_cut=fw.proficiency_cut),
             bands=fw.bands,
             sections=[dict(code=s.code, title=s.title, note=s.note,
                            indicators=[dict(code=i.code, name=i.name, levels=i.levels,
                                             **({"applies_when": i.applies_when}
                                                if i.applies_when else {}),
                                             **({"requires": i.requires}
                                                if i.requires else {}))
                                        for i in s.indicators])
                       for s in fw.sections])
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False, allow_unicode=True, width=100)
    return path


# Back-compat alias. `lint` implied a predictive check the calibration does not
# support; `review` is the honest name.
lint = review
