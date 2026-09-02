"""Indicator exclusion sets — which indicators to stop scoring, and why.

An indicator earns a place here by failing the reliability test in the pooled
run (10 transcripts x 10 passes, both scales), not by looking awkward. Each set
below records the evidence so the decision can be re-argued later rather than
inherited blindly.

The two reasons are NOT interchangeable, and the distinction decides what to do
about them:

  WORDING     the standard joins two conditions that real lessons satisfy
              separately, so the scorer has no rule for which one decides.
              Fixable by rewriting the descriptor — excluding these is a
              stopgap, not a verdict on the construct.

  UNOBSERVABLE the evidence is not in an audio transcript at all. No rewrite
              helps; this needs video, a live observer, or dropping the
              indicator. Excluding these is the permanent answer unless the
              data source changes.

Figures are mean single-pass flip rate averaged across the 1-4 and binary
routes; AC1 is the lower of the two.
"""

WORDING = {
    "C4": ("Equitable Participation", 0.182, 0.40,
           "'deliberate strategies' AND 'diverse students included' — teacher names "
           "students, but volunteers dominate and many stay silent"),
    "B2": ("Lesson Structure & Sequence", 0.133, 0.55,
           "'clear I Do/We Do/You Do' AND 'logical flow with transitions' — phases "
           "present in substance but not explicitly framed"),
    "C6": ("Classroom Management & Routines", 0.115, 0.72,
           "'clear routines' AND 'minimal disruptions' — routines exist, execution "
           "is chaotic"),
    "B9": ("Time on Task / Time on Learning", 0.110, 0.68,
           "'70-85% on task' AND 'efficient transitions' — mostly on task, long "
           "messy regrouping"),
}

UNOBSERVABLE = {
    "D5": ("On-Task Behavior During Independent Work", 0.158, 0.55,
           "sustained focus during silent work is not audible"),
    "D4": ("Student Confidence & Risk-Taking", 0.115, 0.66,
           "willingness to attempt is inferred from who speaks; silence is ambiguous"),
    "D6": ("Student Use of Learning Materials", 0.102, 0.69,
           "whether students handle materials is largely inaudible"),
}

# The seven that failed the pooled reliability screen. Flip rate >= 10% on the
# average of both scales; every other indicator sits below 9%.
UNRELIABLE = {**WORDING, **UNOBSERVABLE}

SETS = {
    "unreliable":   sorted(UNRELIABLE),
    "wording":      sorted(WORDING),
    "unobservable": sorted(UNOBSERVABLE),
    "none":         [],
}


def resolve(spec):
    """'unreliable' | 'C4,B2' | '' -> list of codes. Unknown names raise."""
    if not spec:
        return []
    spec = spec.strip()
    if spec in SETS:
        return list(SETS[spec])
    codes = [c.strip().upper() for c in spec.split(",") if c.strip()]
    from .framework import ALL_CODES
    bad = [c for c in codes if c not in ALL_CODES]
    if bad:
        raise ValueError(f"unknown indicator code(s): {bad}. "
                         f"Named sets: {sorted(SETS)}")
    return codes


def describe(codes):
    """One line per excluded indicator, for the run header."""
    out = []
    for c in codes:
        if c in UNRELIABLE:
            name, flip, ac1, why = UNRELIABLE[c]
            kind = "wording" if c in WORDING else "unobservable"
            out.append(f"    {c:4s} {name[:42]:44s} flip {flip*100:.0f}%  AC1 {ac1:.2f}  {kind}")
        else:
            out.append(f"    {c:4s} (excluded by request)")
    return "\n".join(out)
