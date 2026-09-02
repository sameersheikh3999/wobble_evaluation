"""Convert the HOTS spreadsheet and the TEACH observation sheet into framework specs.

Both arrive in shapes the generic CSV importer does not cover:

  HOTS   level columns are headed "Emerging (1)" not "1 = Emerging", and the
         domains are rows inside the sheet rather than one file per section.
  TEACH  is a PDF tick-sheet. It carries the behaviour statements and the L/M/H
         boxes but NOT the L/M/H descriptors - those live in the TEACH manual.
         See the note in the generated YAML: this is a material limitation on
         any comparison against a framework that ships full descriptors.
"""
import io, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import yaml

OUT = os.path.dirname(os.path.abspath(__file__))


def build_hots(src="Coaching Framework - HOTS Framework.csv"):
    d = pd.read_csv(src, header=None, dtype=str).fillna("")
    hdr = next(i for i in range(len(d))
               if str(d.iloc[i, 0]).strip().lower() == "code")
    labels, lvcols = [], {}
    for j, c in enumerate(d.iloc[hdr]):
        m = re.search(r"\((\d+)\)\s*$", str(c).strip())
        if m:
            lvcols[int(m.group(1))] = j
            labels.append(re.sub(r"\s*\(\d+\)\s*$", "", str(c)).strip())
    sections, cur = [], None
    for i in range(hdr + 1, len(d)):
        code = str(d.iloc[i, 0]).strip()
        name = str(d.iloc[i, 1]).strip()
        if not code:
            continue
        # a row with a code but no indicator name is a domain banner
        if not name:
            m = re.match(r"^(DOMAIN\s+(\d+)|MULTIGRADE ONLY)\s*[-\u2014]?\s*(.*)$",
                         code, re.I)
            if m:
                sc = m.group(2) or "MG"
                cur = dict(code=f"D{sc}", title=(m.group(3) or code).strip(),
                           note="", indicators=[])
                sections.append(cur)
            continue
        if cur is None or not re.match(r"^[\dM]", code):
            continue
        cur["indicators"].append(dict(
            code=code, name=name,
            levels={lv: str(d.iloc[i, j]).strip() for lv, j in lvcols.items()}))
    sections = [s for s in sections if s["indicators"]]
    spec = dict(
        name="HOTS Observation Framework",
        version="", source=src,
        scale=dict(levels=max(lvcols), labels=labels, proficiency_cut=max(lvcols)),
        # HOTS ships no banding of its own; use the same thresholds as FICO so the
        # band-stability figures are comparable across frameworks.
        bands=[{"name": "High", "min": 0.85}, {"name": "Medium", "min": 0.60},
               {"name": "Low", "min": 0.0}],
        sections=sections)
    return spec


# TEACH scores every behaviour Low / Medium / High. The observation sheet gives
# no per-behaviour definition of those points, so the generic TEACH convention is
# used. This is weaker than a rubric that ships descriptors, and the difference
# must be reported alongside any comparison.
TEACH_L = ("The behaviour is not evidenced in the segment, or the evidence runs "
           "counter to it.")
TEACH_M = ("The behaviour is evidenced but only partially, inconsistently, or "
           "with some students and not others.")
TEACH_H = ("The behaviour is evidenced fully and consistently across the segment.")


def build_teach(src="Teach_Tool.pdf"):
    from pypdf import PdfReader
    txt = PdfReader(src).pages[2].extract_text() or ""
    areas, cur, elem = [], None, ""
    for raw in txt.split("\n"):
        line = raw.strip()
        m_area = re.match(r"^([A-C])\.\s+(.+)$", line)
        m_elem = re.match(r"^(\d+)\.\s+([A-Z][A-Z &]+?)\s{2,}1 2 3 4 5", line)
        m_beh = re.match(r"^(\d+\.\d+)\s+(.+)$", line)
        if m_area:
            cur = dict(code=m_area.group(1), title=m_area.group(2).strip(),
                       note="", indicators=[])
            areas.append(cur)
        elif m_elem:
            elem = m_elem.group(2).strip().title()
        elif m_beh and cur is not None:
            body = re.sub(r"\s*(N/A\s*)?L\s*M\s*H.*$", "", m_beh.group(2)).strip()
            if not body or len(body) < 12:
                continue
            cur["indicators"].append(dict(
                code=m_beh.group(1), name=f"{elem}: {body}"[:110],
                levels={1: TEACH_L, 2: TEACH_M, 3: TEACH_H},
                note=f"TEACH behaviour - judge: {body}"))
    areas = [a for a in areas if a["indicators"]]
    return dict(
        name="TEACH Classroom Observation Tool",
        version="", source=src,
        scale=dict(levels=3, labels=["Low", "Medium", "High"], proficiency_cut=3),
        bands=[{"name": "High", "min": 0.85}, {"name": "Medium", "min": 0.60},
               {"name": "Low", "min": 0.0}],
        sections=areas)


if __name__ == "__main__":
    for name, fn in (("hots", build_hots), ("teach", build_teach)):
        spec = fn()
        p = os.path.join(OUT, f"{name}.yaml")
        with io.open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True, width=100)
        n = sum(len(s["indicators"]) for s in spec["sections"])
        print(f"{name:6s} {n:3d} indicators in {len(spec['sections'])} sections, "
              f"{spec['scale']['levels']}-level -> {p}")
