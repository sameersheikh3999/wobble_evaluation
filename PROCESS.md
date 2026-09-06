# Coaching Framework Reliability Evaluation — full process record

Context file for anyone (or any agent) picking this project up cold. Covers what
was asked, what was built, what was found, what was got wrong and corrected, and
what remains undone.

Repo: `sameersheikh3999/wobble_evaluation`, branch `main`.
Local: `C:\Users\HP\Music\AI Projects\wobble_evaluation`

---

## 1. The question

Taleemabad coaches teachers against a rubric. A trained observer watches a lesson
and rates it on 37 indicators; those ratings become the coaching conversation.
Doing that by hand for thousands of classrooms is slow, so the proposal is to have
an AI read a transcript and produce the ratings.

Before "is the AI right?" comes a cheaper, prior question: **does the AI give the
same answer twice?** If the identical transcript yields "good feedback" on Monday
and "poor feedback" on Tuesday, the measurement is unusable regardless of how
sensible any single answer looks.

That movement is what this project measures. It is called **wobble** throughout.

**The finding, in one line:** the scorer is consistent; the instability that
remains is concentrated in a minority of indicators, and it is a property of how
those indicators are written (or of what a transcript can evidence), not of the
model.

---

## 2. Unit of assessment — get this right

The unit is a **coaching observation**: an audio recording of one classroom
session, transcribed. That transcript is what the framework is applied to.

A **lesson plan** is a separate, optional document the teacher may attach. Only
indicators that compare the observation *against* it need it — in FICO that is
Section B's job, not the observation's identity. Do not refer to the whole
observation as a lesson plan.

This distinction is load-bearing. See correction #6 in §8.

---

## 3. Data

**Observations** — `Transcripts/` (local only, gitignored: contains PII)

- 10 JSON files = **8 distinct lessons**. Two pairs are the same audio transcribed
  twice (`01cbd387`/`4d0535e8` and `01ecfbfe`/`5bfafc87`, identical durations to
  the millisecond). Kept deliberately: they measure transcription sensitivity.
- Languages: Urdu, Swahili, English. Durations 10–32 minutes.
- Schema: `session_id`, `language`, `transcript`, `ffprobe_duration_seconds`,
  `audio_file`, `user_id`, `created_at`. **No scores of any kind.**
- The single-lesson pilot (`8938cc17`, embedded in `wobble_eval/session.py`) is an
  **eleventh** lesson, not one of the ten. Do not mix its figures with the pooled
  ones.

**Frameworks** — `frameworks/*.yaml`

| Framework | Indicators | Sections | Scale | Source |
|---|---|---|---|---|
| FICO (Taleemabad) | 37 | 4 (B, C, D, F) | 1–4 | `Coaching Framework - *.csv` |
| HOTS | 17 | 6 domains | 1–3 | `Coaching Framework - HOTS Framework.csv` |
| TEACH | 28 of 30 | 3 areas | L/M/H | `Teach_Tool.pdf` |

TEACH caveats: the PDF is the **observation tick-sheet only** — behaviour
statements and L/M/H boxes, but *not* the behaviour-level descriptors, which live
in the TEACH manual. Generic anchors were substituted. Its items `0.1`/`0.2` (Time
on Learning) were **dropped from the spec** because they are timed snapshots at
4/9/14 min that a whole-transcript scorer cannot reproduce. TEACH results are a
floor, not a fair estimate.

---

## 4. Method

Score the same observation N times with the same transcript, rubric and prompt.
Nothing varies but the scorer, so any difference *is* the scorer.

With L observations × R runs, every indicator gets its own **observations × runs
grid**, read two ways:

- **Reliability** — *across a row.* Observation fixed, only the run changes, so
  disagreement can only be scorer noise. Gwet's AC1, flip rate, and a pooled exact
  binomial against a 5% negligible-noise floor.
- **Discrimination** — *down a column.* Do different observations get different
  answers at all? Chi-square of homogeneity. An indicator scored the same on every
  observation is perfectly reliable and perfectly useless.

Both p-values Holm-corrected across the indicator set.

|  | discriminates | does not |
|---|---|---|
| **reliable** | HEALTHY | UNINFORMATIVE |
| **unreliable** | NOISY | BROKEN |

Plus `NOT_ASSESSABLE` (a required reference document was never supplied) and
`UNTESTED` (too few scored observations).

### Why Gwet's AC1 and not kappa

When an indicator is nearly always "not met" — and many are — the chance-agreement
term in Cohen's kappa and Krippendorff's alpha grows until the coefficient
collapses toward zero **even at 95% raw agreement**. That is the prevalence
paradox: a property of the statistic, not of the scorer. AC1 corrects for chance
without that failure mode. All three are reported so the divergence stays visible.

### Other statistics used

- **Wilson score intervals** on per-indicator rates (a bootstrap of 10 Bernoulli
  draws returns [1.0, 1.0] for a unanimous indicator — false certainty).
- **Cochran's Q** for run-to-run drift (Friedman's binary analogue).
- **McNemar exact** for paired band-stability comparisons between scales.
- **`votes_needed`** — smallest odd number of passes whose majority reproduces a
  verdict 95% of the time. Blank = too near a coin flip for voting to converge.

---

## 5. Runs actually performed

| Study | Framework | Design | Calls |
|---|---|---|---|
| Pilot | FICO | 1 lesson × 10, binary, 3 arms | ~120 |
| Main | FICO | 10 obs × 10, binary, 37 ind | 400 |
| Main | FICO | 10 obs × 10, 1–4 scale, 37 ind | 400 |
| Screened | FICO | 10 obs × 10, both scales, 30 ind | 800 |
| Effort sweep | FICO | 3 obs, `low` arm only — **stopped early** | ~90 |
| 3-framework | FICO/HOTS/TEACH | 3 obs × 10 each | 390 |

Model: `claude-opus-5`, effort `high`, thinking adaptive, via the Claude Agent SDK
(subscription auth, no API key).

**There is no temperature knob on this path.** Verified two ways:
`ClaudeAgentOptions` has no such field (all 45 enumerated), and sampling
parameters were *removed* from the Claude 5 family — `temperature`/`top_p`/`top_k`
return HTTP 400 on Opus 5. So the measured wobble *is* the production sampling
configuration; there was never a dial set wrongly. `effort` is the only
sampling-adjacent knob, and the sweep testing it was stopped before completion.

---

## 6. Findings

### 6.1 The scorer is consistent — reliability is not the blocker

FICO, 10 observations × 10 runs, binary:

| | 37 indicators | 30 (screened) |
|---|---|---|
| Mean AC1 | 0.866 | **0.916** |
| Raw pairwise agreement | 91.4% | — |
| Mean flip rate | 5.4% | **3.5%** |
| Unanimous judgements | 78.6% of cells | 85.4% |
| Parse failures | 4.0% | **0.0%** |

Overall fidelity band held on 9 of 10 observations before screening, **10 of 10**
after. If you report one number per lesson, that number is solid.

### 6.2 Seven indicators produce half the disagreement

19% of the framework, **48–50% of all wobble**. Everything else sits under 9%.

| Code | Indicator | Flip | AC1 | Cause |
|---|---|---|---|---|
| **C4** | Equitable Participation | **18.2%** | **0.40** | wording |
| **D5** | On-Task Behavior in Independent Work | 15.8% | 0.55 | unobservable |
| **B2** | Lesson Structure & Sequence | 13.3% | 0.55 | wording |
| **D4** | Student Confidence & Risk-Taking | 11.5% | 0.66 | unobservable |
| **C6** | Classroom Management & Routines | 11.5% | 0.72 | wording |
| **B9** | Time on Task | 11.0% | 0.68 | wording |
| **D6** | Student Use of Learning Materials | 10.2% | 0.69 | unobservable |

**Two causes, two different fixes:**

- **Compound wording** (C4, B2, C6, B9) — the standard joins two requirements a
  real lesson can satisfy separately, with no rule for which decides. C4 reads
  *"Deliberate strategies: cold call, pair-share, name sticks. Diverse students
  included."* Runs answering YES cited the teacher naming students; runs answering
  NO cited that those were mostly volunteers while many stayed silent. Both
  correct. **Rewritable.**
- **Not in the audio** (D4, D5, D6) — sustained focus during silent work,
  willingness to attempt, whether pupils handle materials. **No rewrite helps.**

Four are also ASR-sensitive (D5, B2, C6, D6): their verdict changed from
*re-transcribing the same audio*, independent of the model.

### 6.3 A quarter of the framework carries no signal

Nine indicators answered **"not met" on every observation, on both scales, in
every run**: `B6 B7 C1 C3 C5 C10 C11 C12 F8`. Six are in Section C — half of the
section meant to capture high-leverage practice discriminated between none of the
eight lessons. Perfectly reliable, perfectly uninformative.

Invisible in a single-lesson study, where "always no" looks well-behaved.

*(B7 was later reclassified — see correction #6.)*

### 6.4 Screening reduces reported wobble but improves nothing

Flip rate fell 5.4% → 3.5% (−35%) and AC1 rose 0.866 → 0.916. But the **30
surviving indicators reproduce exactly as well as they always did** —
per-indicator correlation r = 0.979 between the 37-run and the 30-run, 16
unchanged / 6 improved / 7 worsened, i.e. random scatter.

**This is selection, not repair.** You removed the measurements that weren't worth
making; you did not make the scorer better. Say so before someone else notices.

Two side effects *were* causal improvements: parse failures 4.0% → 0.0% (shorter
prompt), and duplicate-transcript agreement 92% → 98.3% (five of the six
ASR-sensitive indicators were among the seven).

### 6.5 Scale choice does not matter

1–4 collapsed at ≥3 versus native binary, same 10 observations:

| | 1–4 | binary |
|---|---|---|
| Mean AC1 | 0.908 | 0.916 |
| Flip rate | 3.6% | 3.5% |
| Verdict agreement between routes | **94%** | |
| Band readings unstable | 10/50 | 9/50 (McNemar **p = 1.000**) |

A dead heat. The ranking even reverses between the full and screened sets. **Scale
choice is not a reliability decision.** Choose on information retention: 1–4 keeps
more detail but loses more signal at the threshold (13 uninformative vs 9).

This **overturned the design assumption** the binary scorer was built on.

### 6.6 Band stability is driven by section size, not indicator quality

| Section | Indicators | Band unstable |
|---|---|---|
| ALL | 37 → 30 | 1/10 → **0/10** |
| C | 12 → 10 | 0/10 |
| B | 10 → 8 | 4–5/10 → 2–4/10 |
| F | 8 | 1/10 |
| **D** | 7 → **4** | 4–6/10 → **5–6/10** (worse) |

Section D got **worse** after screening. With four indicators, one flip moves the
section 25 points — enough to cross a band line alone. **Trimming a small section
makes it less stable, not more.** Below ~6 indicators, rebuild rather than trim.

### 6.7 The scorer's confidence is usable for triage

Asked for the probability a second careful observer would agree (checked against
leave-one-out actual agreement across 3,360 judgements):

- Stated **0.73** vs earned **0.92** — *under*-confident by 0.19, the safe direction
- Monotonic: Spearman **ρ = +0.58**, p < 0.0001
- **Everything stated ≥ 0.70 proved 99.7% reproducible** (1,854 calls); below 0.60,
  71%

Practical rule: quote the ≥0.70 calls from one pass, re-run only the hesitant ones.

Cost: eliciting a precise number made scoring slightly *less* consistent
(AC1 0.844 → 0.773 on the single-session arms). Trade-off, not free.

### 6.8 Wording does not predict wobble — measured, not assumed

The "compound descriptor" story explains the seven failures *after the fact*. It
does **not** identify them in advance. Across all 37 indicators, no textual feature
correlated with the flip rate observed:

| Feature | Spearman ρ | p |
|---|---|---|
| Words in the YES bar | −0.155 | 0.37 |
| Sentences in the YES bar | −0.145 | 0.40 |
| Clauses in the YES bar | −0.057 | 0.74 |

Rule-based flags did no better: "two or more sentences" caught 4 of 7 at 21%
precision, and the literal word "and" caught **none** (their compounds are
punctuation-joined).

**Therefore the empirical run is mandatory.** You cannot inspect a rubric and know
which indicators will hold. This is why `spec.lint` was renamed `spec.review` and
relabelled advisory.

### 6.9 Three frameworks compared (3 observations each)

At each framework's own proficiency bar:

| | Indicators | Met | AC1 | Carrying signal |
|---|---|---|---|---|
| FICO | 37 | 30.5% | 0.879 | 18/35 — 51% |
| **HOTS** | 17 | **0.0%** | 1.000* | 0/17 — **0%** |
| TEACH | 28 | 19.4% | 0.893 | 10/28 — 36% |

**HOTS awarded its top level exactly zero times in 510 judgements.** Its 3-point
scale tops out at "Proficient", so a proficiency verdict is unreachable in these
classrooms; the AC1 of 1.000 is the arithmetic of a constant.

At a comparable bar (above the bottom level) the picture inverts — HOTS becomes
the most efficient at **53%** carrying signal vs FICO's 29%. Its problem is
entirely where the bar sits, not what it asks.

TEACH fails in the opposite direction: 11 of 28 behaviours are met on *every*
observation at the lower bar.

**Reliability barely differed across the three** (AC1 0.89–0.92, flip 3.3–4.7%,
zero parse failures). The frameworks are distinguished by what their scales can
*detect*, not by how consistently an AI applies them.

Two blocks work fully and agree with each other: **HOTS Domain 4** (Student
Engagement) and **TEACH Element 6** (Critical Thinking).

*Power caveat:* three observations is the minimum. All three frameworks carry 5–6
indicators at flip ≥10%, and **HOTS has the worst mean flip rate (4.7%)** — but
only FICO's C2 cleared the Holm-corrected test. **"0 NOISY" means "not proven
unreliable", not "proven reliable."** See `framework_comparison/wobblers.csv`.

---

## 7. The standing limit on everything above

**This measures whether the scorer agrees with itself, not whether it is right.**
An indicator returning the same wrong answer ten times scores perfectly here.
Nothing has been compared against a trained human observer.

If a reader hears "92% reliable" and concludes "92% accurate", the study has been
misread. Reliability also **bounds** validity: a scorer cannot agree with a human
better than it agrees with itself, so AC1 0.87 is the ceiling on any future
human–AI figure.

---

## 8. Corrections made along the way

Kept because the reasoning matters more than the conclusions, and because several
were caught only by checking rather than assuming.

1. **"No Python on this machine"** — wrong. A recursive filesystem scan died early
   and I treated its truncated output as conclusive. The registry found Python
   3.14.7 immediately.
2. **`backend.py` could not find the `claude` CLI.** It lives at `~/.local/bin`
   (native installer default), which was not in the search list — so *no* scoring
   run could have started. Fixed.
3. **Section `band` reported the modal per-run band, not the band of the reported
   figure** — printing `yes_rate 0.686` beside `band Low`. Now `band` is the band
   of the number you'd quote; `modal_run_band` is separate.
4. **A chart labelled the region above the diagonal "overconfident."** Above the
   line is *under*-confident. Corrected.
5. **The linter claimed to predict wobble.** Calibration showed it does not (§6.8).
   Renamed to `review`, output relabelled advisory, calibration in the docstring.
6. **B7 was classified UNINFORMATIVE when the lesson plan was never supplied.**
   Its bar is *"Plan followed with minor contextual adaptations"* — a scorer cannot
   judge adherence to a document it was never shown. That verdict measured a gap in
   the inputs, not a defect in the rubric. Now `NOT_ASSESSABLE`, a distinct class,
   with `Indicator.requires` making it structural. B8 has the same problem.
7. **TEACH items `0.1`/`0.2` were dropped from the spec without disclosure**, and
   several TEACH behaviours need visual evidence (`1.4` bias by gender/disability,
   `4.2` monitoring, `7.3` volunteering, `9.1` collaboration) that the automated
   check could not see, because the criterion sits in the behaviour's *name* while
   the L/M/H anchors are generic. Three of those four scored "met" on 100% of
   observations — what you'd expect when a scorer cannot see counter-evidence.
8. **"0 NOISY" for HOTS/TEACH read as a clean bill of health.** It is a power
   artefact at n=3. Corrected in the report with an asterisk on every affected row.
9. **Numeric indicator codes coerced to float on re-analysis.** TEACH's `1.1`…`9.3`
   became floats via `read_csv`, every spec lookup missed silently, and it surfaced
   as all 28 indicators `UNTESTED` rather than as an error. Now read as `str`.
10. **`already_done()` accepted an all-unparseable session as complete.** A
    rate-limit storm scored 300 cells as NA; the session counted as finished and
    would have been reused forever. Now requires ≥1 parseable score.
11. **`run_multi.py` aborted the whole run when one transcript failed**, discarding
    seven already-scored transcripts. Now records, skips, and continues — naming
    skipped sessions above the pooled tables.
12. **`run_clean_sweep.sh` had no `set -e`**, so a comparison ran over
    partially-scored data and looked complete. Fixed.
13. **One-pager chart 2 put a 5% rate and an 87-point score on one axis**,
    flattening the bars that carried the point. Redrawn as a single metric.

---

## 9. Repository map

### Documents
```
report/wobble-report.html      the study and its findings
report/protocol.html           the reusable method + diagrams
report/three-frameworks.html   FICO vs HOTS vs TEACH
report/workflow.html           stage-by-stage process diagram
PROCESS.md                     this file
findings_package/README.md      the verdict, standalone
findings_package/Wobble_Evaluation_One_Pager.docx (+ .pdf)
```

### The answer: which indicators fail
```
findings_package/data/problematic_indicators.csv   FICO — all 18, categorised
framework_comparison/wobblers.csv                  17 wobblers across 3 frameworks
framework_comparison/all_indicators.csv            every indicator, both bars
wobble_eval/exclusions.py                          the seven, AS CODE, with evidence
```

### Pipeline
```
run_framework_wobble.py   generic: import / check / plan / run / analyse
run_wobble_binary.py      binary, one observation
run_multi_binary.py       binary, a directory
run_multi.py              1–4 scale, a directory
compare_scales.py         1–4 vs binary, head to head
wobble_eval/spec.py       load/validate/review ANY framework
wobble_eval/binary.py     binary rubric, prompts, parser, AC1/Cochran/Wilson
wobble_eval/multi_binary.py   reliability + discrimination + classification
wobble_eval/stats.py      alpha, ICC, kappa, Holm, bootstrap
frameworks/*.yaml         FICO, HOTS, TEACH
make_package.py           builds the redacted shareable bundle
```

### Local only — NOT in git, contains PII
```
Transcripts/              raw observations: classroom speech, pupil names
multi_binary_out/  multi_scale14_out/  multi_*_clean/
cmp_taleemabad/  cmp_hots/  cmp_teach/
scale_comparison*/  wobble_out_binary*/  effort_*_out/
```
These exist **only on this machine**. Losing it costs the raw judgements —
reproducing them is ~1,300 model calls. The code, findings and conclusions are all
in git. If the raw evidence matters, it needs a private encrypted store; it must
not go in the repo.

---

## 10. How to run it

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

# any framework, end to end
python run_framework_wobble.py import  --csv-dir <folder> --out frameworks/x.yaml
python run_framework_wobble.py check   --framework frameworks/x.yaml      # free
python run_framework_wobble.py plan    --framework frameworks/x.yaml --dir Transcripts -n 10
python run_framework_wobble.py run     --framework frameworks/x.yaml --dir Transcripts -n 10 --out x_out --resume
python run_framework_wobble.py analyse --framework frameworks/x.yaml --out x_out --cut 2   # free

# drop the known-unreliable seven from scoring AND analysis
python run_multi_binary.py --dir Transcripts -n 10 --exclude unreliable
```

Only `run` costs anything. `analyse` is free and repeatable — re-cut the
threshold or exclude indicators without re-scoring. Everything is crash-safe:
scores are written after each observation and `--resume` skips completed work.

Requires the `claude` CLI on PATH (`~/.local/bin/claude.exe` on this machine) —
authenticates via the Claude subscription, no API key.

---

## 11. What is not done

1. **Human IRR — the big one.** No human-scored observations exist anywhere in the
   study. `report/workflow.html` specifies the stage in full: same observations
   scored by 2+ trained raters, ~20–30 of them, with quadratic-weighted kappa
   (ordinal), AC1 (binary), exact + adjacent %, ICC(2,1), and Bland–Altman for
   systematic leniency. No code exists yet because there has been nothing to run it
   against. **Exclude the seven unreliable indicators from that comparison** — you
   would be measuring noise, not disagreement.
2. **HOTS and TEACH on 10 observations** (600 and 300 calls). Would give the
   reliability test the power FICO's study had and rule on the 11 unresolved
   `healthy*` candidates.
3. **The TEACH manual descriptors.** Would remove the one confound that could not
   be designed around.
4. **Lesson plans.** Would settle FICO's B7/B8 and HOTS Domain 2 — the place where
   missing inputs are most likely masquerading as missing signal.
5. **The effort sweep** was stopped mid-`low` arm. Never completed, so whether
   thinking depth changes wobble is unanswered.
6. **The generic pipeline has not been cross-validated** against the original
   FICO results. Running `frameworks/taleemabad.yaml` over the 10 observations
   should reproduce `multi_binary_out` — worth confirming before trusting it on an
   unfamiliar framework.

---

## 12. Recommendations as they stand

1. **Rewrite the four compound standards** — C4, B2, C6, B9. Highest leverage,
   costs only editing time.
2. **Decide about D4, D5, D6** — accept they need video, or reword around what is
   audible.
3. **Review the nine silent indicators.** Is the bar above these classrooms, or is
   the evidence absent from a transcript? Different problems, different fixes.
4. **Rebuild Section D**, do not trim it further — it is already too small to band
   reliably.
5. **Report the overall lesson score from a single pass. Do not report Section B or
   D from one pass.**
6. **Keep the 1–4 scale** unless cost dominates; it matched binary on every
   reliability measure and preserves more information.
7. **Use the confidence score to triage** rather than re-running everything.
8. **Run the human comparison.** Everything above is about consistency. None of it
   establishes correctness.

---

*Note: if you want this auto-loaded as context by Claude Code, copy it to
`CLAUDE.md` — that filename is read at session start. It is kept as `PROCESS.md`
here because `CLAUDE.md` is conventionally agent instructions rather than a
project record, and `/init` may overwrite it.*
