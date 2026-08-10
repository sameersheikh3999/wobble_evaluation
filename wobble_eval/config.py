"""Hyperparameters for the local (subscription / Claude Opus) run.

WHAT IS AND IS NOT TUNABLE HERE — read this before comparing against the
Colab notebook. The Claude Agent SDK drives Claude Code, which does not expose
`temperature`, `top_p`, or `top_k`; Claude Opus 5 rejects those parameters
outright. So the notebook's primary wobble dial does not exist on this path.
What you get instead are the knobs Claude Code actually has:

    MODEL     claude-opus-5 | claude-sonnet-5 | claude-haiku-4-5
    EFFORT    low | medium | high | xhigh | max     <- the real intelligence dial
    THINKING  adaptive | disabled

Residual run-to-run variation therefore reflects the *production* sampling
configuration rather than a temperature you chose. That is arguably the more
decision-relevant number: it is the wobble a deployed Claude-scored pipeline
would actually experience.
"""
from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    # ---------- model / backend ----------
    MODEL: str            = "claude-opus-5"
    EFFORT: str           = "high"       # low | medium | high | xhigh | max
    THINKING: str         = "adaptive"   # adaptive | disabled
    BACKEND: str          = "agent_sdk"  # agent_sdk (subscription) | api (ANTHROPIC_API_KEY)
    MAX_CONCURRENCY: int  = 3            # parallel section calls; 1 = strictly serial
    CALL_TIMEOUT_S: int   = 600

    # ---------- experiment design ----------
    N_ITERATIONS: int     = 10
    SECTIONS: tuple       = ("B", "C", "D", "F")
    SCORING_MODE: str     = "per_section"   # per_section | per_indicator
    PROMPT_VARIANT: str   = "standard"      # standard | terse | cot
    INDICATOR_ORDER: str  = "fixed"         # fixed | shuffled
    INCLUDE_EVIDENCE: bool = True
    ALLOW_NA: bool        = True
    MAX_RETRIES: int      = 2
    BASE_SEED: int        = 1234            # labels/shuffling only - the model is not seedable

    # ---------- statistics (identical to the notebook) ----------
    N_BOOTSTRAP: int      = 5000
    CI_LEVEL: float       = 0.95
    NEGLIGIBLE_DISAGREEMENT: float = 0.05
    MC_SIMS: int          = 20000
    PROFICIENCY_CUT: int  = 3
    ALPHA: float          = 0.05
    STATS_SEED: int       = 7

    # ---------- io ----------
    OUT_DIR: str          = "wobble_out_local"
    SESSION_PATH: str     = ""      # "" = use the embedded session
    VERBOSE: bool         = True

    def as_dict(self):
        d = asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


# The transcript is short enough for a 1M-token context window, so there is no
# digest/truncate strategy here - the model always sees the verbatim transcript.
CONTEXT_KIND = "verbatim transcript"
