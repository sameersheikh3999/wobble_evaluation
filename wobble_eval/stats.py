"""Wobble statistics — extracted verbatim from the validated notebook.

`CFG` is injected by the runner before use (see config.Config).
"""
import numpy as np
import pandas as pd
from scipy import stats

CFG = None          # set by the runner: wobble_eval.stats.CFG = cfg
RNG = None
LEVELS = np.array([1, 2, 3, 4])


def init(cfg):
    """Bind the config and seed the shared RNG."""
    global CFG, RNG
    CFG = cfg
    RNG = np.random.default_rng(cfg.STATS_SEED)
    _MC_CACHE.clear()

# ---------- basic ----------
def boot_ci(x, stat=np.mean, n_boot=None, level=None, rng=None):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:   return (np.nan, np.nan)
    if len(x) == 1:   return (float(x[0]), float(x[0]))
    n_boot = n_boot or CFG.N_BOOTSTRAP
    level  = level or CFG.CI_LEVEL
    rng    = rng or RNG
    draws  = rng.choice(x, size=(n_boot, len(x)), replace=True)
    vals   = stat(draws, axis=1)
    a      = (1 - level) / 2
    return tuple(float(v) for v in np.quantile(vals, [a, 1 - a]))

def t_ci(x, level=None):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    level = level or CFG.CI_LEVEL
    if len(x) < 2: return (np.nan, np.nan)
    se = stats.sem(x)
    if se == 0:    return (float(x.mean()), float(x.mean()))
    h = se * stats.t.ppf(0.5 + level / 2, len(x) - 1)
    return (float(x.mean() - h), float(x.mean() + h))

def norm_entropy(x, k=4):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return np.nan
    p = np.array([(x == lv).mean() for lv in LEVELS])
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(k))

_MC_CACHE = {}
def p_vs_random(x, sims=None):
    """H0: scores ~ Uniform{1,2,3,4}. p = P(SD_sim <= SD_obs)."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 2: return np.nan
    sims = sims or CFG.MC_SIMS
    key = (n, sims)
    if key not in _MC_CACHE:
        draws = np.random.default_rng(CFG.STATS_SEED + n).integers(1, 5, size=(sims, n))
        _MC_CACHE[key] = draws.std(axis=1, ddof=1)
    return float(((_MC_CACHE[key] <= x.std(ddof=1)) .sum() + 1) / (sims + 1))

def p_wobble_test(x, h0=None):
    """H0: P(run disagrees with the modal score) <= h0. One-sided exact binomial."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return np.nan, 0, 0
    h0 = CFG.NEGLIGIBLE_DISAGREEMENT if h0 is None else h0
    mode = stats.mode(x, keepdims=False).mode
    k, n = int((x != mode).sum()), len(x)
    return float(stats.binomtest(k, n, h0, alternative="greater").pvalue), k, n

def holm(pvals):
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m, running = len(idx), 0.0
    for i, j in enumerate(idx):
        running = max(running, min(1.0, (m - i) * p[j]))
        out[j] = running
    return out

# ---------- reliability ----------
def _delta2_ordinal(marg):
    """Ordinal difference function from coincidence marginals."""
    V = len(marg)
    cum = np.concatenate([[0.0], np.cumsum(marg)])
    d = np.zeros((V, V))
    for c in range(V):
        for k in range(V):
            lo, hi = min(c, k), max(c, k)
            s = cum[hi + 1] - cum[lo]                  # sum of n_g, g = lo..hi
            d[c, k] = (s - (marg[c] + marg[k]) / 2.0) ** 2
    return d

def krippendorff_alpha(data, level="ordinal"):
    """data: units x raters, NaN = missing. Units with <2 values are dropped."""
    data = np.asarray(data, float)
    obs = data[~np.isnan(data)]
    if obs.size == 0: return np.nan
    vals = np.unique(obs)
    V = len(vals)
    if V < 2: return 1.0                               # no disagreement possible
    ix = {v: i for i, v in enumerate(vals)}
    O = np.zeros((V, V))
    for row in data:
        r = row[~np.isnan(row)]
        m = len(r)
        if m < 2: continue
        cnt = np.zeros(V)
        for v in r: cnt[ix[v]] += 1
        for c in range(V):
            for k in range(V):
                O[c, k] += (cnt[c] * (cnt[k] - (1 if c == k else 0))) / (m - 1)
    marg = O.sum(axis=1)
    n = marg.sum()
    if n < 2: return np.nan
    d2 = _delta2_ordinal(marg) if level == "ordinal" else (1 - np.eye(V))
    Do = (O * d2).sum()
    E  = np.outer(marg, marg).astype(float)
    np.fill_diagonal(E, 0.0)
    De = (E * d2).sum() / (n - 1)
    return float(1 - Do / De) if De > 0 else np.nan

def fleiss_kappa(data, categories=LEVELS):
    """data: units x raters (complete rows only)."""
    data = np.asarray(data, float)
    data = data[~np.isnan(data).any(axis=1)]
    N, k = data.shape
    if N == 0 or k < 2: return np.nan
    C = np.array([[(row == c).sum() for c in categories] for row in data], float)
    Pi = ((C ** 2).sum(axis=1) - k) / (k * (k - 1))
    Pbar = Pi.mean()
    pj = C.sum(axis=0) / (N * k)
    Pe = (pj ** 2).sum()
    return float((Pbar - Pe) / (1 - Pe)) if Pe < 1 else np.nan

def icc21(data):
    """ICC(2,1) two-way random, single measure + F tests for units and for raters."""
    Y = np.asarray(data, float)
    Y = Y[~np.isnan(Y).any(axis=1)]
    n, k = Y.shape
    if n < 2 or k < 2: return dict(icc=np.nan)
    gm = Y.mean()
    SSR = k * ((Y.mean(axis=1) - gm) ** 2).sum()
    SSC = n * ((Y.mean(axis=0) - gm) ** 2).sum()
    SST = ((Y - gm) ** 2).sum()
    SSE = SST - SSR - SSC
    MSR = SSR / (n - 1); MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    denom = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    icc = (MSR - MSE) / denom if denom != 0 else np.nan
    out = dict(icc=float(icc), n_units=n, k_raters=k, MSR=MSR, MSC=MSC, MSE=MSE)
    if MSE > 0:
        out["F_units"]  = MSR / MSE
        out["p_units"]  = float(stats.f.sf(MSR / MSE, n - 1, (n - 1) * (k - 1)))
        out["F_raters"] = MSC / MSE
        out["p_raters"] = float(stats.f.sf(MSC / MSE, k - 1, (n - 1) * (k - 1)))
    else:                                              # perfectly consistent
        out.update(F_units=np.inf, p_units=0.0, F_raters=np.nan, p_raters=np.nan)
    return out

def pairwise_agreement(data):
    """Mean share of run-pairs giving the identical score, over units."""
    data = np.asarray(data, float)
    vals = []
    for row in data:
        r = row[~np.isnan(row)]
        if len(r) < 2: continue
        same = sum(1 for i in range(len(r)) for j in range(i + 1, len(r)) if r[i] == r[j])
        vals.append(same / (len(r) * (len(r) - 1) / 2))
    return float(np.mean(vals)) if vals else np.nan

def grade_wobble(modal_share, rng_, na_rate):
    if na_rate >= 0.5:                             return "severe"
    if modal_share >= 0.999 and rng_ == 0:         return "stable"
    if modal_share >= 0.80 and rng_ <= 1:          return "minor"
    if modal_share >= 0.60 and rng_ <= 2:          return "material"
    return "severe"
