"""Statistical inference on backtest results.

A backtest gives one number per strategy. Whether the gap between two of them
means anything depends on how noisy that number is, which a single path cannot
tell you. This module answers "is strategy A different from strategy B?" with
an interval and a p-value rather than a point estimate.

The comparison that matters most in this app is a strategy against its own
equilibrium prior. The two are highly correlated -- they hold nearly the same
portfolio -- and that correlation is exactly what makes a small gap measurable,
so every test here works on the PAIRED returns, never on two separate summaries.
"""
import math
import numpy as np


# =========================================================
# Stationary block bootstrap (Politis & Romano, 1994)
# =========================================================
def default_block_length(T):
    """Mean block length ~ T^(1/3), the rate-optimal order for variance
    estimation. About 13 for eight years of daily data."""
    return max(2, int(round(T ** (1.0 / 3.0))))


def stationary_bootstrap_indices(T, mean_block, n_boot, rng):
    """(n_boot, T) array of resampled time indices.

    Blocks start at a uniformly random point and have geometric length with
    the given mean, wrapping around the end of the sample. Keeping runs of
    consecutive days together preserves volatility clustering, which an i.i.d.
    bootstrap would destroy -- understating the uncertainty of anything
    computed from returns.
    """
    p = 1.0 / float(mean_block)
    idx = np.empty((n_boot, T), dtype=np.int64)
    starts = rng.integers(0, T, size=(n_boot, T))
    new_block = rng.random((n_boot, T)) < p
    new_block[:, 0] = True
    cur = starts[:, 0].copy()
    for t in range(T):
        cur = np.where(new_block[:, t], starts[:, t], (cur + 1) % T)
        idx[:, t] = cur
    return idx


def _ann_mean(x, fpy):
    return np.mean(x, axis=-1) * fpy


def _ann_sharpe(x, fpy):
    sd = np.std(x, axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, np.mean(x, axis=-1) / sd * math.sqrt(fpy), np.nan)


def paired_bootstrap(a, b, freq_per_year, rf=0.0, n_boot=2000, mean_block=None,
                     alpha=0.05, seed=0):
    """Bootstrap the difference between two strategies run over the same dates.

    a, b : per-period simple returns of the two strategies (same length, same dates)
    rf   : annual risk-free rate, subtracted per period before the Sharpe ratio

    Returns, for the annualised arithmetic mean return and for the Sharpe ratio:
    the point estimate of (a - b), a (1-alpha) percentile interval, and a
    two-sided p-value for "no difference". The p-value centres the bootstrap
    distribution on zero (the null) and asks how often a gap as large as the
    observed one appears -- the standard construction (Hall & Wilson, 1991).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    T = len(a)
    if T < 30:
        return None
    L = mean_block or default_block_length(T)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(T, L, n_boot, rng)
    rfp = rf / freq_per_year
    xa, xb = a - rfp, b - rfp

    out = {"n": T, "mean_block": L, "n_boot": n_boot, "alpha": alpha}
    for name, fn in (("ann_return", _ann_mean), ("sharpe", _ann_sharpe)):
        point = float(fn(xa, freq_per_year) - fn(xb, freq_per_year))
        boot = fn(xa[idx], freq_per_year) - fn(xb[idx], freq_per_year)
        boot = boot[np.isfinite(boot)]
        if not np.isfinite(point) or len(boot) < 0.9 * n_boot:
            # e.g. a Sharpe ratio when one strategy has zero volatility
            out[name] = {"diff": point, "lo": np.nan, "hi": np.nan, "p": np.nan, "se": np.nan}
            continue
        lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
        p = float(np.mean(np.abs(boot - point) >= abs(point)))
        out[name] = {"diff": point, "lo": float(lo), "hi": float(hi), "p": p,
                     "se": float(np.std(boot, ddof=1))}
    return out


# =========================================================
# Ledoit & Wolf (2008): robust test for a difference in Sharpe ratios
# =========================================================
def newey_west_lag(T):
    """Newey-West (1994) plug-in bandwidth, floor(4 (T/100)^(2/9))."""
    return int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def hac_cov(Y, lag=None):
    """Newey-West long-run covariance of the columns of Y (already demeaned)."""
    Y = np.asarray(Y, dtype=float)
    T = Y.shape[0]
    L = newey_west_lag(T) if lag is None else int(lag)
    S = Y.T @ Y / T
    for j in range(1, L + 1):
        w = 1.0 - j / (L + 1.0)                     # Bartlett kernel
        G = Y[j:].T @ Y[:-j] / T
        S += w * (G + G.T)
    return S


def sharpe_diff_test(a, b, freq_per_year, rf=0.0, lag=None):
    """HAC delta-method test of SR(a) - SR(b) = 0.

    With v = (mu_a, mu_b, g_a, g_b), g = E[r^2], each Sharpe ratio is
    f(v) = mu / sqrt(g - mu^2). The delta method gives
        Var(SR_a - SR_b) ~ grad' Psi grad / T,
    where Psi is the long-run covariance of (r_a - mu_a, r_b - mu_b,
    r_a^2 - g_a, r_b^2 - g_b). Using a HAC estimate of Psi keeps the test valid
    under fat tails and serial dependence, where Jobson-Korkie is not.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    rfp = rf / freq_per_year
    ra, rb = a[ok] - rfp, b[ok] - rfp
    T = len(ra)
    if T < 30:
        return None
    mu_a, mu_b = ra.mean(), rb.mean()
    g_a, g_b = (ra ** 2).mean(), (rb ** 2).mean()
    va, vb = g_a - mu_a ** 2, g_b - mu_b ** 2
    if va <= 0 or vb <= 0:
        return None
    sr_a, sr_b = mu_a / math.sqrt(va), mu_b / math.sqrt(vb)
    grad = np.array([g_a / va ** 1.5, -g_b / vb ** 1.5,
                     -mu_a / (2 * va ** 1.5), mu_b / (2 * vb ** 1.5)])
    Y = np.column_stack([ra - mu_a, rb - mu_b, ra ** 2 - g_a, rb ** 2 - g_b])
    var = float(grad @ hac_cov(Y, lag) @ grad) / T
    se = math.sqrt(max(var, 0.0))
    diff = sr_a - sr_b
    z = diff / se if se > 0 else np.nan
    p = float(math.erfc(abs(z) / math.sqrt(2))) if np.isfinite(z) else np.nan
    k = math.sqrt(freq_per_year)
    return {"sharpe_a": sr_a * k, "sharpe_b": sr_b * k, "diff": diff * k, "se": se * k,
            "z": z, "p": p, "lag": newey_west_lag(T) if lag is None else int(lag), "n": T}


# =========================================================
# Trading costs: where does the ranking flip?
# =========================================================
def breakeven_cost_bps(bt_a, bt_b):
    """One-way cost (bps) at which strategy A's gross edge over B is used up.

    Costs reduce annual return by about cost x annual turnover, so A stays
    ahead while  gap > cost x (turnover_A - turnover_B). Solving for cost:

        breakeven = (ann_ret_A - ann_ret_B) / (turnover_A - turnover_B)

    First-order: it ignores the compounding of costs and the overlay's scaling,
    which is fine for a ranking and is labelled as approximate on screen.
    Returns None when A trades no more than B (costs cannot flip the ranking
    in B's favour) or when A is not ahead gross of costs.
    """
    if not bt_a or not bt_b:
        return None
    gap = bt_a["strat"]["ann_ret"] - bt_b["strat"]["ann_ret"]
    dturn = bt_a.get("ann_turnover", 0.0) - bt_b.get("ann_turnover", 0.0)
    if dturn <= 1e-9 or gap <= 0:
        return None
    return float(gap / dturn * 1e4)


# =========================================================
# Is a rule's hit rate better than chance?
# =========================================================
def hit_rate_pvalue(hit_rate, n_independent):
    """One-sided p-value for hit rate > 50%, normal approximation to the
    binomial on the number of independent checks."""
    if hit_rate is None or not n_independent:
        return None
    z = (float(hit_rate) - 0.5) / math.sqrt(0.25 / float(n_independent))
    return 0.5 * math.erfc(z / math.sqrt(2))
