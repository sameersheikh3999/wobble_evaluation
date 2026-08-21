# Can an AI reliably score classroom teaching?

Reliability study of the Taleemabad Coaching Framework, August 2026.
**This package is safe to share outside the team.** See *Privacy* at the end.

---

## The verdict in one paragraph

An AI scorer reading lesson transcripts reproduces its own judgements well enough
to use: mean reliability 0.87 across the framework's 37 indicators, and 0.92
across the 30 that survive screening. Reliability is not what blocks deployment.
What blocks it is the rubric. **Nine of 37 indicators returned the same answer on
every lesson**, carrying no coaching signal at all, and **seven more were too
unstable to quote** — those seven were 19% of the framework but produced 50% of
its total disagreement. Fixing the wording of four of them, and accepting that
three cannot be judged from audio, is worth more than any change to the model.

---

## What was measured

The same ten lesson transcripts were scored **ten times each**, with the
transcript, rubric and instructions identical every time. Any difference between
those ten answers cannot come from the lesson — only from the scorer.

The whole experiment ran four times over: on a **1-4 scale** and a **binary
yes/no** scale, each **before and after** removing the seven unstable indicators.

| | |
|---|---|
| Lessons | 10 transcripts = **8 distinct lessons** (2 pairs are the same audio transcribed twice) |
| Languages | Urdu, Swahili, English, 10-32 minutes |
| Indicators | 37 across 4 sections (B, C, D, F) |
| Repetitions | 10 identical passes per transcript, per configuration |
| Scale of study | 1,600 model calls, 400 scoring passes, **12,432 individual judgements** |
| Scorer | Claude Opus 5, no tools, transcript only |

Reliability is reported as **Gwet's AC1** rather than Cohen's kappa. When an
indicator is nearly always "no" — as many here are — kappa collapses toward zero
even at near-perfect agreement. That is a known flaw in the statistic, not a fact
about the scorer, and it would have made good indicators look broken.
Krippendorff's alpha and raw agreement are reported alongside so the divergence
stays visible.

---

## Findings

### 1. The scorer is consistent. Reliability is not the blocker

Mean AC1 **0.87** over 37 indicators, **0.92** over the screened 30. A single
pass contradicts the majority about **5%** of the time, falling to **3.5%**.
After screening, the **overall fidelity band is stable on all ten lessons**, on
both scales. One AI pass can be quoted for a lesson's overall score.

### 2. A quarter of the framework carries no signal. This is the biggest finding

Nine indicators answered **NO on every lesson, on both scales, in every pass**:
B6, B7, C1, C3, C5, C10, C11, C12, F8. Six sit in Section C, so half of the
section meant to capture high-leverage teaching practice distinguished between
none of the eight lessons. Perfectly reliable and perfectly uninformative.

This is invisible in a single-lesson study, where "always NO" looks like a
well-behaved indicator.

### 3. Seven indicators cannot be quoted

19% of the framework, 50% of its wobble. Worst is **C4 Equitable Participation**
(AC1 0.40, changed verdict on 8 of 10 lessons).

Two distinct causes, needing different remedies:

| Cause | Indicators | Remedy |
|---|---|---|
| **Compound wording** — the standard welds together two requirements that real lessons satisfy separately | C4, B2, C6, B9 | Rewrite: split it, or state which clause decides |
| **Not observable in audio** | D4, D5, D6 | No rewrite helps. Needs video, or drop |

C4 reads *"Deliberate strategies: cold call, pair-share, name sticks. Diverse
students included."* Passes answering YES cited the teacher calling on students
by name; passes answering NO cited that those were mostly volunteers while many
children stayed silent. Both readings are correct. The rubric never says which
governs — and the same split recurred across classrooms and languages.

### 4. Scale choice does not matter

1-4 and binary are **indistinguishable**: 94% verdict agreement, AC1 within
0.008, identical 3.5% flip rate, band stability differing by one reading in fifty
(McNemar p = 1.000). The ranking even reverses between the full and screened
sets.

Choose on information retention instead. 1-4 keeps more detail, but loses more
signal at the threshold: **13 uninformative indicators against 9** on binary.

### 5. Screening cuts reported wobble by a third, but improves nothing

Flip rate fell **5.4% to 3.5%** (binary) and **5.2% to 3.6%** (1-4). This is
**selection, not repair**: the 30 surviving indicators reproduce exactly as well
as they always did (per-indicator correlation r = 0.979 between runs). Bad
measurements were removed; measurement did not improve.

### 6. Section D is structurally broken, and screening made it worse

Its band moved on **5-6 of 10 lessons after** screening, against 4-6 before.
Removing D4/D5/D6 left only four indicators, so a single flip shifts the section
25 points. Section D needs rebuilding around observable evidence, not trimming.

### 7. The scorer's own confidence is usable for triage

Asked to state the probability a second observer would agree, it was
**under**-confident (claimed 0.73, earned 0.92) but correctly ordered
(Spearman rho = +0.58). **Judgements it rated 0.70 or above proved 99.7%
reproducible.** Quote the confident calls from one pass; rerun only the hesitant
ones.

---

## Recommendations

1. **Report the overall lesson score from a single pass. Do not report Sections B
   or D from one pass.**
2. **Rewrite the four compound standards** — C4, B2, C6, B9. Highest-leverage
   change available.
3. **Review the nine silent indicators.** Is the bar above these classrooms, or
   is the evidence absent from a transcript? Different problems, different fixes.
4. **Rebuild Section D** around what audio can capture, or accept it needs video.
5. **Keep the 1-4 scale** unless cost dominates. It matched binary on every
   reliability measure and preserves more information.
6. **Use the confidence score to triage** rather than rerunning everything.

---

## What this study does NOT show

**This measures whether the scorer agrees with itself, not whether it is right.**
An indicator returning the same wrong answer ten times scores perfectly here.
Nothing was compared against a trained human observer.

If a reader hears "92% reliable" and concludes "92% accurate", the study has been
misread. **The necessary next study is a comparison against expert human ratings
on the same lessons.**

Three narrower limits:

- **Eight lessons, not ten.** Two pairs are the same audio transcribed twice.
  Kept deliberately: they measure transcription sensitivity. The pairs agreed on
  98% of verdicts after screening (92% before), and one indicator, F3, still
  changes verdict from re-transcription alone, which better scoring cannot fix.
- **Transcripts only.** The scorer never sees the room. Seating, body language
  and who is actually writing are invisible, which likely explains part of
  Finding 2.
- **Rate limiting cost some data.** One transcript's 1-4 pass failed entirely and
  was re-run; another lost 10% of its cells. Affected figures are noted in the
  data files.

---

## Contents

```
report/findings.html     the written report, start here

charts/
  01_indicator_wobble_ranked.png     every indicator's flip rate, both scales
  02_band_stability_grid.png         where a lesson's band is decided by luck
  illustrative_single_lesson/        explanatory figures from a ONE-LESSON
                                     pilot. Illustrative only. Do not quote
                                     these as findings.

data/
  indicator_reliability_binary_30.csv    per-indicator reliability, screened set
  indicator_reliability_binary_37.csv    per-indicator reliability, full set
  indicator_wobble_scale14_37.csv        1-4 scale, per indicator
  lesson_by_indicator_binary_30.csv      every lesson x indicator verdict
  section_bands_binary_30.csv            band per lesson per section
  scale_comparison_headline.csv          1-4 vs binary, head to head
  scale_comparison_by_indicator.csv      the same, per indicator
  scale_comparison_bands.csv             the same, per band reading
  raw_verdicts_binary_30_REDACTED.csv    all 3,000 raw cells (see Privacy)
```

## Privacy

The transcripts are recordings of real children in Pakistani and Tanzanian
classrooms. Nothing in this package contains them.

- **Free-text columns were dropped, not masked.** `evidence` and `flip_if` quoted
  classroom speech verbatim and named individual pupils. Masking the names would
  still have leaked the surrounding speech.
- **Session identifiers were replaced** with `Lesson-01` to `Lesson-10`, so no row
  can be traced back to a recording, a pupil or a teacher.
- **Every statistic is unchanged.** None of it is personal.

The raw transcripts and the unredacted run outputs stay on the analysis machine
and are excluded from version control.
