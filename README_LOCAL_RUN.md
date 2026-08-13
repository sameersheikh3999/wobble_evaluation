# Local wobble run — Claude Opus via your Claude Code subscription

Runs the same experiment as the Colab notebook, but scoring with **Claude Opus 5**
on this machine, authenticated by the Claude subscription you're already logged
into. No API key, no GPU, no Colab.

```bash
python run_wobble.py -n 10                    # the main run: 10 iterations
```

## How the auth works

`run_wobble.py` → Claude Agent SDK → local `claude` CLI → `~/.claude/.credentials.json`.

That's your subscription, so the run draws on your Claude usage limits rather than
API credit. Nothing reads `ANTHROPIC_API_KEY`. If the CLI is missing:

```bash
npm install -g @anthropic-ai/claude-code
```

It picks up the existing credentials — no `/login` needed.

## ⚠️ The one real difference from the notebook: no temperature

Claude Code does not expose `temperature`, `top_p`, or `top_k`, and Claude Opus 5
**rejects those parameters outright**. So the notebook's primary wobble dial does
not exist here. What you get instead:

| Knob | Flag | Values |
|---|---|---|
| Model | `--model` | `claude-opus-5` (default), `claude-sonnet-5`, `claude-haiku-4-5` |
| **Effort** | `--effort` | `low`, `medium`, `high` (default), `xhigh`, `max` — the real intelligence dial |
| Thinking | `--thinking` | `adaptive` (default), `disabled` |
| Iterations | `-n` | how precisely you measure wobble (≥10 for a usable CI) |
| Scoring granularity | `--mode` | `per_section` (4 calls/iter) or `per_indicator` (37 calls/iter) |
| Prompt shape | `--prompt-variant` | `standard`, `terse`, `cot` |
| Order sensitivity | `--indicator-order` | `fixed`, `shuffled` |
| Sections | `--sections` | e.g. `B,D` |
| Parallelism | `--concurrency` | parallel model calls (default 3) |

**This is not a downgrade.** The wobble you measure here is the wobble at the
*production* sampling configuration — the noise a deployed Claude-scored pipeline
would actually experience. The notebook's temperature sweep answers "how does
wobble respond to a dial I control"; this answers "how much wobble do I have."

## Commands

```bash
# Main run
python run_wobble.py -n 10

# Effort sweep — the analogue of the notebook's temperature sweep
python run_wobble.py --sweep-effort low,medium,high,xhigh -n 5

# Is per-section prompting letting indicators anchor on each other?
python run_wobble.py -n 6 --mode per_indicator

# Order sensitivity on top of sampling noise
python run_wobble.py -n 8 --indicator-order shuffled --out wobble_shuffled

# Model arm at matched settings
python run_wobble.py -n 8 --model claude-sonnet-5 --out wobble_sonnet

# Score a different session
python run_wobble.py -n 10 --session path/to/other_session.json

# Re-analyse an existing run without spending any quota
python run_wobble.py --analyse-only wobble_out_local/scores_long.csv
```

## Output (per run directory)

| File | Contents |
|---|---|
| `scores_long.csv` | one row per indicator × iteration (written after every iteration, so a crash keeps what finished) |
| `scores_wide.csv` | indicators × runs matrix |
| `indicator_wobble.csv` | SD, bootstrap CI, mode, modal share, entropy, flip rate, NA rate, `p_wobble`/`q_wobble`, grade |
| `reliability.csv` | Krippendorff's α (ordinal + nominal), ICC(2,1), Fleiss' κ, pairwise agreement, Friedman drift — overall and per section |
| `section_wobble.csv` | section means, CIs, and roll-ups |
| `headline.json` | the numbers to quote |
| `run_meta.json` | full config + model-call count + wall time (reproducibility record) |
| `01..07_*.png` | the seven charts |

## Many transcripts at once — `run_multi.py`

```bash
python run_multi.py --dir Transcripts -n 10 --out wobble_multi --reuse wobble_out_local
python run_multi.py --dir Transcripts --out wobble_multi --analyse-only   # re-pool, 0 calls
```

Runs the single-session experiment once per JSON in `--dir`, writing the usual
per-session directory under `<out>/<session8>/`, then adds a pooled layer.
Resume-safe: a session whose `scores_long.csv` already has all N iterations is
skipped, so a crashed run restarts with the same command. `--reuse DIR` ingests
an already-finished run instead of paying for it again.

### It de-duplicates lessons first, and that matters

A transcript drop is often not what it looks like. On the `Transcripts/` drop,
**11 files are 4 distinct recordings** — seven of them are the same 27-minute
lesson transcribed seven times under seven session IDs. Pooling those as
independent lessons would inflate n and understate every confidence interval.

Two fingerprints, union-found:

| Signal | Rule | Catches |
|---|---|---|
| utterance-set Jaccard | ≥ 0.35 | repeat ASR passes (observed 0.65–0.98 within a group; < 0.05 between) |
| audio duration | \|Δ ffprobe\| ≤ 0.5 s | re-transcription into a **different language** — 0.04 text overlap, 0.044 s duration gap |

The duration signal is not redundant. Two of these files are one recording
rendered once in Urdu script and once in English; on text alone they look like
unrelated lessons.

### What the pooled layer answers that one session cannot

Splitting the noise into two components:

- **within-transcript SD** — same input, N passes. Sampling wobble. This is what
  the single-session run measures.
- **between-transcript SD** — same recording, different transcription, per-file
  mean. Transcription wobble.

If between > within on an indicator, the ASR step is the thing to fix and
averaging more model passes will not help: every pass is reading a different
account of the same lesson.

| File | Contents |
|---|---|
| `sessions_headline.csv` | one row per transcript: α, ICC, mean, stability, flips |
| `cell_session_indicator.csv` | per (session × indicator) mean and within-session SD |
| `pooled_indicator_wobble.csv` | grand mean, within SD, between-session SD, between-lesson SD, signal-to-noise |
| `variance_decomposition.csv` | within- vs between-transcript SD per indicator, for multi-transcript lessons |
| `transcript_similarity.csv` | every pair: Jaccard, duration delta, and which signal merged them |
| `scores_long_all.csv` | every score cell from every session, tagged with lesson + session |
| `10..12_*.png` | α by transcript · within-vs-between scatter · indicator × transcript heatmap |

`signal_to_noise` = between-lesson SD ÷ within-transcript SD. Below ~1, an
indicator cannot separate two different lessons more reliably than it separates
two runs on the *same* lesson — it is not measuring the classroom.

## Binary mode — `run_wobble_binary.py`

Same 37 indicators, same transcript, same hygiene; the model answers **YES or NO**
instead of picking a level.

```bash
python run_wobble_binary.py -n 10                 # YES bar = level 3 (Proficient)
python run_wobble_binary.py -n 10 --yes-at 2      # YES bar = level 2 (observed at all)
python run_wobble_binary.py -n 10 --yes-at 4      # YES bar = level 4 (highly effective)
python run_wobble_binary.py --sweep-effort low,medium,high -n 4
python run_wobble_binary.py --analyse-only wobble_out_binary/verdicts_long.csv
```

Every other flag is identical to `run_wobble.py`.

### Where the YES bar comes from

Nothing is hand-authored. The YES bar **is** the level-`--yes-at` descriptor from
the framework CSVs, so the two runs interrogate the same evidence:

> **YES** — the level-3 descriptor is *clearly* met (or a stronger level is)
> **NO** — anything below that
> **NA** — the indicator genuinely cannot apply (F5 MATH in a reading lesson)

That default is the framework's own proficiency cut, so binary YES-rate and the
1–4 run's `p_proficient` are directly comparable.

### Why run it as well as the 1–4 version

On the 1–4 scale most wobble is **adjacent** — a 3 becomes a 4 — and adjacent
wobble rarely changes what a coach does. Binary strips that cushion out, so every
disagreement measured here is one that changes the coaching decision. Expect the
agreement coefficients to look *better* (two categories are easier to agree in)
while the count of decision-relevant flips stays about the same. **The pair of
numbers is the finding; neither alone is.**

It also reproduces the framework's own arithmetic exactly: fidelity = actions
observed ÷ actions prescribed, banded ≥85% High · 60–84% Medium · <60% Low. The
report tracks whether that **band** holds still across runs, which is the number a
coaching report actually quotes.

### Three deliberate statistical substitutions

| 1–4 run | Binary run | Why |
|---|---|---|
| Krippendorff α (ordinal) | **Gwet's AC1**, α (nominal) and raw agreement, side by side | With two categories and a skewed split, α and κ collapse toward 0 even at 95% agreement — the *prevalence paradox*, a property of the statistic, not of the model. AC1 is chance-corrected but prevalence-robust. Chart 06 shows all three so the gap is visible; the runner prints a warning when it detects the paradox. |
| Friedman | **Cochran's Q** | Friedman's binary analogue for run-to-run drift. |
| Bootstrap CI | **Wilson score interval** | A bootstrap of 10 Bernoulli draws returns [1.0, 1.0] for a unanimous indicator — certainty that does not exist. Wilson stays honest at p = 0 and p = 1. |

ICC(2,1) is dropped: it is a variance-components model for interval data and means
nothing on a 0/1 scale.

**Read `gwet_ac1` before `kripp_alpha`.** Where AC1 is high and α is low, trust AC1
and quote raw agreement next to it. Where *both* are low, the model genuinely is
unreliable there.

### Binary-only columns worth knowing

| Column | Meaning |
|---|---|
| `single_pass_error` | chance one run contradicts the majority verdict — the practical error rate of quoting a single pass |
| `disagree_rate` | chance two runs contradict each other |
| `votes_needed` | smallest **odd** number of passes whose majority vote reproduces this verdict 95% of the time. `1` = one pass is enough. Blank = too close to a coin flip for voting to ever converge — fix the rubric, not the sample size |
| `p_coinflip` | two-sided binomial vs p=0.5. A **large** value is the bad news: the verdict is indistinguishable from chance |
| `band` / `bands_seen` / `band_flips` | the section's High/Medium/Low fidelity band, and whether it moved between runs |

### Output (per run directory)

| File | Contents |
|---|---|
| `verdicts_long.csv` | one row per indicator × iteration, with the YES/NO string and evidence (written after every iteration) |
| `verdicts_wide.csv` | indicators × runs matrix |
| `indicator_wobble.csv` | YES rate, Wilson CI, modal share, single-pass error, votes needed, `p_coinflip`, `p_wobble`/`q_wobble`, grade |
| `reliability.csv` | α, **AC1**, PABAK, Fleiss' κ, pairwise agreement, YES prevalence, paradox flag, Cochran's Q — overall and per section |
| `section_fidelity.csv` | section YES rate, CI, band, band flips, flip counts |
| `headline.json` | the numbers to quote |
| `run_meta.json` | full config + model-call count + wall time |
| `01..07_*.png` | headline · verdict matrix · YES rate · verdict stability · section fidelity · **agreement coefficients** · drift |

Grades are tighter than the 1–4 run's on purpose — there is no adjacent band to
hide in, so 70% modal agreement is already `material`:

| Grade | Rule |
|---|---|
| stable | identical verdict in every run |
| minor | ≥90% agree with the majority |
| material | ≥70% agree |
| severe | below 70%, or mostly NA |

`run_multi.py` has no binary counterpart yet — pooling across transcripts is still
1–4 only.

## Reading the result

Look at **overall Krippendorff's α (ordinal)** first:

| α | What it licenses |
|---|---|
| ≥ 0.80 | report individual indicator scores from a single pass |
| 0.67–0.80 | report section means and mode-of-N indicator scores only |
| < 0.67 | not decision-grade at indicator level — raise N and take the mode, raise effort, or report sections |

Then **flip rate**: SD is the statistician's number, but flip rate is the coaching
one. An indicator that only ever moves between 3 and 4 never changes the verdict;
one moving between 2 and 3 changes it every other run.

Then the **Friedman p**: non-significant means the noise is unbiased, so averaging
N runs converges at roughly √N. Significant means individual passes are *biased*,
and averaging converges on that bias instead.

## Experiment hygiene (deliberate choices)

- **No tools.** The model gets zero tool access, so it can only score the
  transcript in the prompt — it cannot read files or search the web.
- **`setting_sources=None`.** Your `CLAUDE.md` and project settings never enter
  the scoring prompt, so the experiment isn't contaminated by local context.
- **`max_turns=1`.** One response per call, no agentic loop.
- **Identical prompt every iteration.** The only thing that varies is the model's
  sampling draw, so every score change is measurement noise.

## Two things this does *not* measure

1. **Accuracy.** Everything here is precision — agreement with itself. A model can
   be perfectly stable and perfectly wrong. For accuracy you need human-scored
   sessions and quadratic-weighted κ against the human score.
2. **What audio can't carry.** C12 (space/seating), D1/D7 (visible engagement,
   gender) and B8/C10 (physical resources, tech) depend on things a transcript
   doesn't contain. Persistently low scores there are a method limit, not a
   teacher finding.

Also note **F5 (Math) and F6 (Science) do not apply** to this English reading
lesson — a high NA rate there is the model being right. `--no-na` forces a 1–4 on
every indicator if you need it, but that manufactures wobble on indicators with
nothing to score.

## Layout

```
run_wobble.py           CLI entry point — one session, 1-4 scale
run_wobble_binary.py    CLI entry point — one session, YES/NO scale
run_multi.py            CLI entry point — a directory of sessions, pooled (1-4 only)
wobble_eval/
  config.py             hyperparameters (+ the temperature explanation)
  framework.py          all 37 indicators, 4 level descriptors each
  session.py            embedded transcript + loader
  prompts.py            1-4 scoring prompts + three-tier JSON parser
  backend.py            Agent SDK (subscription) / API backends
  stats.py              bootstrap CI, α, ICC, κ, Holm, Monte-Carlo tests
  analysis.py           long scores -> wobble tables
  charts.py             the seven 1-4 figures
  binary.py             YES/NO rubric + prompts + parser + AC1/Cochran/Wilson + analysis
  charts_binary.py      the seven binary figures (shares charts.py's palette)
```

`binary.py` and `charts_binary.py` sit alongside the 1–4 path rather than replacing
it: both scales share `framework.py`, `session.py`, `backend.py` and the reliability
primitives in `stats.py`, so the two reports cannot drift apart.

`framework.py`, `stats.py`, `analysis.py` and `charts.py` are extracted from the
same sources as the Colab notebook, so the two paths cannot drift apart.
