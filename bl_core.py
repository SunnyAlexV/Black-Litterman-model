"""The numeric core: returns, covariance, Black-Litterman, view engines, optimiser.

Pure functions of arrays and DataFrames. No network, no UI.
"""
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from bl_config import *  # noqa: F401,F403

# =========================================================
# RETURNS
# =========================================================
def to_returns(prices, freq, use_log):
    rule = FREQ_RULE[freq]
    px = prices if rule is None else prices.resample(rule).last()
    px = px.dropna(how="all")
    r = np.log(px).diff() if use_log else px.pct_change()
    return r.dropna()

# =========================================================
# COVARIANCE ESTIMATORS  (return ANNUALISED matrices)
# =========================================================
def nearest_psd(matrix, eps=1e-10):
    matrix = (matrix + matrix.T) / 2.0
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(vals) @ vecs.T


def ledoit_wolf_identity(X):
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward a scaled
    identity. X is a T x N matrix of (periodic) returns. Returns periodic cov."""
    X = np.asarray(X, dtype=float)
    t, n = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / t
    m = np.trace(S) / n
    d2 = np.sum((S - m * np.eye(n)) ** 2) / n

    # The textbook form of b_bar2 loops over every observation, building an
    # n x n outer product each time. With 15 years of daily data that is ~3,800
    # Python iterations per call, and a walk-forward backtest calls this once
    # per rebalance on an expanding window — twelve seconds of a fifteen-second
    # backtest went here. Expanding the Frobenius norm removes the loop:
    #
    #   sum_i ||x_i x_i' - S||_F^2
    #     = sum_i (x_i'x_i)^2  -  2 sum_i x_i' S x_i  +  t ||S||_F^2
    #
    # which is three vectorised reductions. Algebraically identical, no
    # approximation — verified against the loop to machine precision by a test.
    sq = np.einsum("ij,ij->i", Xc, Xc)              # ||x_i||^2 for each i
    term1 = float(np.dot(sq, sq))                   # sum_i (x_i'x_i)^2
    term2 = float(np.sum((Xc @ S) * Xc))            # sum_i x_i' S x_i
    term3 = t * float(np.sum(S * S))                # t ||S||_F^2
    b_bar2 = (term1 - 2.0 * term2 + term3) / (t ** 2) / n
    b2 = min(b_bar2, d2)
    a2 = d2 - b2
    shrink = b2 / d2 if d2 > 0 else 1.0
    sigma = shrink * m * np.eye(n) + (a2 / d2 if d2 > 0 else 0.0) * S
    return sigma


def _ewma_weights(t, halflife):
    lam = 0.5 ** (1.0 / max(halflife, 1e-9))
    w = lam ** np.arange(t - 1, -1, -1)
    return w / w.sum()


def horizon_returns(returns, h):
    """Overlapping h-period compounded returns from a period-return panel.

    Row k is the return between the price at k and the price at k + h, i.e. it
    compounds periods k+1 through k+h. The output therefore has len(R) - h rows,
    one per window that fully fits inside the sample.
    """
    R = np.asarray(returns, dtype=float)
    h = max(1, int(h))
    if h == 1 or len(R) <= h:
        return R
    g = np.cumprod(1.0 + R, axis=0)
    return g[h:] / g[:-h] - 1.0


def variance_ratio(returns, h, shrink_c=10.0, lo=0.25, hi=4.0):
    """Per-asset variance ratio VR(h) = Var(h-period) / (h x Var(1-period)).

    The whole reason a holding period should change a portfolio. Scaling daily
    variance by h assumes returns are independent from one day to the next. They
    are not: an asset that mean-reverts is LESS risky over a year than its daily
    volatility implies (VR < 1) and deserves more weight in a long hold, while a
    trending one is MORE risky (VR > 1) and deserves less. Under strict i.i.d.
    returns VR = 1 at every horizon and the holding period genuinely does not
    matter -- which is why the app used to give the same answer for a week and a
    year.

    Overlapping windows mean the estimate is far noisier than the raw sample
    size suggests: 3,780 daily observations contain only ~15 independent annual
    ones. So VR is shrunk toward 1 by the effective sample count, which leaves a
    weekly horizon essentially untouched and pulls a yearly one most of the way
    back unless the evidence is strong. Clipped to [lo, hi] so a single freak
    window cannot rescale a book.
    """
    R = np.asarray(returns, dtype=float)
    h = max(1, int(h))
    if R.ndim == 1:
        R = R[:, None]
    n = R.shape[1]
    if h == 1 or len(R) <= 2 * h:
        return np.ones(n)
    var1 = np.var(R, axis=0, ddof=1)
    Rh = horizon_returns(R, h)
    if len(Rh) < 3:
        return np.ones(n)
    varh = np.var(Rh, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vr = np.where(var1 > 1e-18, varh / (h * var1), 1.0)
    vr = np.where(np.isfinite(vr), vr, 1.0)
    # Effective independent observations at this horizon, not the overlapping count.
    n_eff = len(R) / float(h)
    k = n_eff / (n_eff + float(shrink_c))
    vr = 1.0 + (vr - 1.0) * k
    return np.clip(vr, lo, hi)


def apply_variance_ratio(cov_annual, vr):
    """Rescale each asset's variance by its variance ratio, keeping correlations.

    Correlations are estimated well from daily data and badly from a handful of
    overlapping annual windows, so only the volatilities are re-scaled here.
    That is a deliberate compromise and the reason it is stated in the app.
    """
    S = np.asarray(cov_annual, dtype=float)
    d = np.sqrt(np.clip(np.asarray(vr, dtype=float), 1e-12, None))
    return nearest_psd(S * np.outer(d, d))


def estimate_cov(returns, method, freq_per_year):
    R = returns.values
    t = R.shape[0]
    if method == "Ledoit-Wolf shrinkage":
        cov_p = ledoit_wolf_identity(R)
    elif method == "Exponentially weighted":
        hl = max(t / 3.0, 5)
        w = _ewma_weights(t, hl)
        mu = np.average(R, axis=0, weights=w)
        Xc = R - mu
        cov_p = (Xc * w[:, None]).T @ Xc
    else:  # Sample
        cov_p = np.cov(R, rowvar=False)
    cov_annual = nearest_psd(np.atleast_2d(cov_p) * freq_per_year)
    return pd.DataFrame(cov_annual, index=returns.columns, columns=returns.columns)

# =========================================================
# EXPECTED-RETURN ESTIMATORS  (return ANNUALISED Series)
# =========================================================
def estimate_mu(returns, method, freq_per_year, rf, cov_annual=None, market_caps=None):
    cols = list(returns.columns)
    R = returns.values
    t, n = R.shape
    hist = returns.mean().values * freq_per_year  # annual historical mean

    if method == "Exponentially weighted":
        w = _ewma_weights(t, max(t / 3.0, 5))
        mu = np.average(R, axis=0, weights=w) * freq_per_year
    elif method == "James-Stein shrinkage":
        grand = float(np.mean(hist))
        denom = float(np.sum((hist - grand) ** 2))
        if cov_annual is not None:
            avg_var = float(np.mean(np.diag(np.asarray(cov_annual)))) / max(t, 1)
        else:
            avg_var = float(np.var(hist)) / max(t, 1)
        c = 0.0 if denom <= 0 else min(1.0, max(0.0, (n - 2) * avg_var / denom))
        mu = grand + (1.0 - c) * (hist - grand)
    elif method == "CAPM equilibrium":
        Sigma = np.asarray(cov_annual, dtype=float)
        if market_caps is not None:
            caps = np.array([market_caps.get(t_, np.nan) for t_ in cols], dtype=float)
        else:
            caps = np.full(n, np.nan)
        if np.all(np.isfinite(caps)) and np.nansum(caps) > 0:
            w_mkt = caps / np.nansum(caps)
        else:
            w_mkt = np.repeat(1.0 / n, n)
        mkt_ret = float(w_mkt @ hist)
        mkt_var = float(w_mkt @ Sigma @ w_mkt)
        delta = (mkt_ret - rf) / mkt_var if mkt_var > 0 else 2.5
        if not np.isfinite(delta) or delta <= 0:
            delta = 2.5
        mu = rf + delta * (Sigma @ w_mkt)
    else:  # Historical mean
        mu = hist
    return pd.Series(mu, index=cols)

def _scalar(x):
    """float() on a 1-element array is an error in NumPy 2.x — this keeps the
    matrix algebra below readable without sprinkling .item() everywhere."""
    return float(np.asarray(x).reshape(-1)[0])



def market_weights(tickers, caps, fallback_weights=None):
    """The market portfolio: cap-weighted where caps are available.

    Yahoo does not return a market cap for every symbol (and never for an ETF
    or an index). Rather than silently falling back to equal weight — which
    would quietly destroy the whole point of the equilibrium prior — we fill a
    small number of gaps with the median cap, and only abandon cap weighting
    when most of the universe is missing. The note we return is shown in the UI
    so the user always knows which prior they actually got.
    """
    caps = caps or {}
    n = len(tickers)
    v = np.array([caps.get(t, np.nan) for t in tickers], dtype=float)
    v = np.where(np.isfinite(v) & (v > 0), v, np.nan)
    ok = np.isfinite(v)

    if ok.sum() >= max(2, int(math.ceil(0.6 * n))) and np.nansum(v) > 0:
        n_missing = int((~ok).sum())
        v = np.where(ok, v, np.nanmedian(v[ok]))
        w = v / v.sum()
        note = "market capitalisation"
        if n_missing:
            note += f" ({n_missing} missing cap{'s' if n_missing > 1 else ''} filled with the median)"
        return w, note

    if fallback_weights is not None:
        fb = np.asarray(fallback_weights, dtype=float)
        if np.all(np.isfinite(fb)) and fb.sum() > 0:
            # Last resort, and a poor one — volume is not size. Flagged loudly
            # in the UI because a prior built this way is not an equilibrium.
            return fb / fb.sum(), "traded value (sizes unavailable — see warning)"

    return np.repeat(1.0 / n, n), "equal weight (market caps unavailable)"


def backcast_caps(caps_today, prices, asof):
    """Market caps as they stood at `asof`, rebuilt from today's caps and prices.

        cap_i(t) ~= cap_i(today) * P_i(t) / P_i(today)

    Why this exists: Yahoo only gives TODAY's market cap. Building a 2022
    prior from 2026 caps hands the 2022 portfolio the weights of whichever
    names went on to win -- look-ahead bias inside the prior itself, and in
    the implied delta that is computed from the same weights.

    The approximation holds the share count fixed, so it ignores buybacks and
    issuance. Because `prices` are dividend-adjusted, the backcast also
    understates past caps of high-yield names by roughly their cumulative
    dividend yield. For a fund, "cap" is AUM, which also contains flows that
    no price path can remove. All three are stated in the README; none of them
    use information from after `asof`, which is the property that matters.

    Only prices up to and including `asof` are read, and today's price enters
    purely as the normalising constant for today's cap.
    """
    caps_today = caps_today or {}
    px = prices.loc[:asof]
    if px.empty:
        return dict(caps_today)
    p_then = px.ffill().iloc[-1]
    p_now = prices.ffill().iloc[-1]
    out = {}
    for t, c in caps_today.items():
        try:
            c = float(c)
            ratio = float(p_then[t]) / float(p_now[t])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            out[t] = np.nan
            continue
        out[t] = c * ratio if (np.isfinite(c) and np.isfinite(ratio) and ratio > 0) else np.nan
    return out


def implied_risk_aversion(w_mkt, Sigma, mkt_excess_return):
    """delta = (E[r_mkt] - rf) / var(r_mkt) — the market's price of risk.

    Clamped to [0.5, 10]: a market whose realised excess return over the sample
    was near zero or negative produces a delta that is meaningless (or negative,
    which flips the sign of every equilibrium return).
    """
    Sigma = np.asarray(Sigma, dtype=float)
    w_mkt = np.asarray(w_mkt, dtype=float)
    var = float(w_mkt @ Sigma @ w_mkt)
    if var <= 0 or not np.isfinite(mkt_excess_return):
        return 2.5, "default 2.5 — market variance unavailable"
    d = float(mkt_excess_return) / var
    if not np.isfinite(d) or d <= 0:
        return 2.5, ("default 2.5 — the market portfolio's realised excess return over this "
                     "window was not positive, so the implied value is unusable")
    if d < 0.5 or d > 10.0:
        cl = float(np.clip(d, 0.5, 10.0))
        return cl, f"implied value {d:,.2f} clamped to {cl:,.2f}"
    return d, "implied from the cap-weighted market portfolio"


def equilibrium_returns(delta, Sigma, w_mkt):
    """Reverse optimisation: pi = delta * Sigma * w_mkt.

    This is the whole trick of Black-Litterman. Instead of asking 'what returns
    do I forecast?' (noisy, and the optimiser will amplify the noise), we ask
    'what returns would make today's market portfolio optimal?' and treat that
    as the prior. Returned as EXCESS returns.
    """
    return float(delta) * (np.asarray(Sigma, dtype=float) @ np.asarray(w_mkt, dtype=float))


def implied_confidence_weights(mu_excess, Sigma, delta):
    """Unconstrained optimal weights for a given excess-return vector:
    w = (delta * Sigma)^-1 * mu.  Note w = w_mkt exactly when mu = pi."""
    Sigma = np.asarray(Sigma, dtype=float)
    try:
        return np.linalg.solve(delta * Sigma, np.asarray(mu_excess, dtype=float))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(delta * Sigma) @ np.asarray(mu_excess, dtype=float)


# ---------------------------------------------------------
# Views  ->  P (picking matrix), Q (view returns), confidences
# ---------------------------------------------------------
def blank_views(tickers, n_rows=3):
    """Empty views grid, sized to the loaded universe."""
    first = tickers[0] if tickers else ""
    second = tickers[1] if len(tickers) > 1 else first
    return pd.DataFrame({
        "Use": [False] * n_rows,
        "Type": ["Absolute"] * n_rows,
        "Asset": [first] * n_rows,
        "Versus": [second] * n_rows,
        "Return % p.a.": [0.0] * n_rows,
        "Confidence %": [DEFAULT_VIEW_CONFIDENCE] * n_rows,
    })


def build_pq(views_df, tickers, rf):
    """Turn the editable views table into (P, Q, confidences, labels, errors).

    Q is returned in EXCESS space:
      - an ABSOLUTE view ('AAPL will return 12%') is a total return, so we
        subtract rf to compare it with pi;
      - a RELATIVE view ('AAPL beats MSFT by 3%') is a difference, so the
        risk-free rate cancels and nothing is subtracted.
    """
    idx = {t: i for i, t in enumerate(tickers)}
    n = len(tickers)
    P, Q, conf, labels, errors = [], [], [], [], []

    if views_df is None or len(views_df) == 0:
        return (np.zeros((0, n)), np.zeros(0), np.zeros(0), [], [])

    for pos, (_, row) in enumerate(views_df.iterrows(), start=1):
        try:
            use = bool(row.get("Use", False))
        except Exception:
            use = False
        if not use:
            continue

        vtype = str(row.get("Type", "Absolute") or "Absolute").strip()
        asset = str(row.get("Asset", "") or "").strip()
        versus = str(row.get("Versus", "") or "").strip()

        raw_q = row.get("Return % p.a.", None)
        raw_c = row.get("Confidence %", None)
        # A blank cell in st.data_editor arrives as NaN, not None, so float()
        # succeeds and silently produces a NaN view. Check explicitly.
        try:
            q = float(raw_q) / 100.0
        except (TypeError, ValueError):
            q = np.nan
        if not np.isfinite(q):
            errors.append(f"View {pos}: '{raw_q}' is not a valid expected return — row skipped.")
            continue
        try:
            c = float(raw_c) / 100.0
        except (TypeError, ValueError):
            c = np.nan
        if not np.isfinite(c):
            errors.append(f"View {pos}: '{raw_c}' is not a valid confidence — row skipped.")
            continue

        if asset not in idx:
            errors.append(f"View {pos}: '{asset}' is not in the loaded universe — row skipped.")
            continue
        if c <= 0:
            errors.append(f"View {pos}: confidence is 0%, which means 'ignore this view' — row skipped.")
            continue
        c = min(c, 0.9999)

        p = np.zeros(n)
        if vtype == "Relative":
            if versus not in idx:
                errors.append(f"View {pos}: '{versus}' is not in the loaded universe — row skipped.")
                continue
            if versus == asset:
                errors.append(f"View {pos}: a relative view against itself is empty — row skipped.")
                continue
            p[idx[asset]] = 1.0
            p[idx[versus]] = -1.0
            labels.append(f"{asset} outperforms {versus} by {q*100:,.2f}%")
            Q.append(q)                    # rf cancels in a difference
        else:
            p[idx[asset]] = 1.0
            labels.append(f"{asset} returns {q*100:,.2f}% p.a.")
            Q.append(q - rf)               # total -> excess

        P.append(p)
        conf.append(c)

    if not P:
        return (np.zeros((0, n)), np.zeros(0), np.zeros(0), [], errors)
    return (np.array(P), np.array(Q), np.array(conf), labels, errors)


def _engine_scores(prices, engine, freq_per_year):
    """Per-asset attractiveness score from price history alone. Higher = more attractive.

    `prices` must contain ONLY data the model is allowed to see at this point.
    Returns (scores, ok) where ok is False when there is not enough history.
    """
    px = prices.dropna(how="all")
    n_obs = len(px)
    per_month = max(1, int(round(freq_per_year / 12)))
    if n_obs < 3 * per_month:
        return None, False

    if engine == "momentum":
        look = min(n_obs - 1, 12 * per_month)
        skip = per_month
        if look - skip < per_month:
            return None, False
        s = px.iloc[-1 - skip] / px.iloc[-look] - 1.0
    elif engine == "reversal":
        look = min(n_obs - 1, per_month)
        s = -(px.iloc[-1] / px.iloc[-1 - look] - 1.0)          # negative: losers score high
    elif engine == "trend":
        win = min(n_obs, max(20, int(round(freq_per_year * 200 / 252))))
        ma = px.iloc[-win:].mean()
        s = px.iloc[-1] / ma - 1.0
    elif engine == "lowvol":
        win = min(n_obs - 1, 6 * per_month)
        r = px.iloc[-(win + 1):].pct_change().dropna()
        if len(r) < 5:
            return None, False
        s = -(r.std(ddof=1) * math.sqrt(freq_per_year))         # negative: calm scores high
    else:
        return None, False

    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return None, False
    return s, True


def systematic_views(prices, tickers, engine, freq_per_year, n_pairs=2,
                     damping=0.30, max_spread=0.20, conf=None):
    """Build (P, Q, conf, labels) from a rule, using only `prices`.

    Cross-sectional engines (momentum, reversal, lowvol) produce RELATIVE views
    pairing the best-ranked asset against the worst. `trend` produces ABSOLUTE
    views on the strongest and weakest names, because it is a time-series signal
    and can legitimately be negative on everything at once.

    Q is damped: the raw spread between winners and losers is a backward-looking
    number and taking it at face value would be absurdly aggressive. `damping`
    scales it down and `max_spread` caps it.
    """
    n = len(tickers)
    empty = (np.zeros((0, n)), np.zeros(0), np.zeros(0), [])
    if engine is None:
        return empty

    scores, ok = _engine_scores(prices[tickers], engine, freq_per_year)
    if not ok:
        return empty

    order = scores.sort_values(ascending=False)
    names = list(order.index)
    idx = {t: i for i, t in enumerate(tickers)}
    k = min(int(n_pairs), len(names) // 2)
    if k < 1:
        return empty

    P, Q, labels = [], [], []
    for j in range(k):
        win, lose = names[j], names[-(j + 1)]
        if win == lose:
            continue
        raw = float(order.iloc[j] - order.iloc[-(j + 1)])
        q = float(np.clip(raw * damping, -max_spread, max_spread))
        if abs(q) < 1e-6:
            continue
        if engine == "trend":
            # absolute views: the strongest gets a positive tilt, the weakest a
            # negative one, both relative to the risk-free rate (excess space).
            p_up = np.zeros(n); p_up[idx[win]] = 1.0
            P.append(p_up); Q.append(abs(q) * 0.5)
            labels.append(f"{win} above its 200d average — expect +{abs(q)*50:,.2f}% excess")
            p_dn = np.zeros(n); p_dn[idx[lose]] = 1.0
            P.append(p_dn); Q.append(-abs(q) * 0.5)
            labels.append(f"{lose} below its 200d average — expect {-abs(q)*50:,.2f}% excess")
        else:
            p = np.zeros(n); p[idx[win]] = 1.0; p[idx[lose]] = -1.0
            P.append(p); Q.append(q)
            labels.append(f"{win} outperforms {lose} by {q*100:,.2f}% [{engine}]")

    if not P:
        return empty
    c = 0.5 if conf is None else float(conf)
    return (np.array(P), np.array(Q), np.full(len(P), c), labels)


def engine_hit_rate(prices, engine, freq_per_year, holding_periods,
                    n_pairs=2, max_checks=60):
    """How often has this rule been directionally right, out of sample?

    Walks forward through the history: at each step, form the view using only
    prices up to that point, then check the SIGN of what actually happened over
    the following `holding_periods`. Never looks beyond the point being tested.

    Returns (hit_rate, n_independent). A rule with no information lands at 0.5.

    The hit rate averages every pair scored, but the sample size returned is
    the number of distinct DATES checked. Pairs scored on the same date share
    the same market move, so with two pairs per date "74 checks" is really ~37
    independent observations. Dates themselves never overlap: they are spaced
    at least one holding period apart, so each forward window is disjoint.
    """
    px = prices.dropna(how="all")
    T = len(px)
    hp = max(1, int(holding_periods))
    per_month = max(1, int(round(freq_per_year / 12)))
    warm = 13 * per_month                      # momentum needs 12m + 1m skip
    if T < warm + hp + 5:
        return None, 0

    step = max(hp, (T - warm - hp) // max_checks + 1)
    hits = total = dates = 0
    for t in range(warm, T - hp, step):
        scored_here = False
        hist = px.iloc[:t]
        scores, ok = _engine_scores(hist, engine, freq_per_year)
        if not ok:
            continue
        order = scores.sort_values(ascending=False)
        names = list(order.index)
        k = min(int(n_pairs), len(names) // 2)
        for j in range(k):
            win, lose = names[j], names[-(j + 1)]
            if win == lose:
                continue
            try:
                fwd_w = px[win].iloc[t + hp] / px[win].iloc[t] - 1.0
                fwd_l = px[lose].iloc[t + hp] / px[lose].iloc[t] - 1.0
            except Exception:
                continue
            if not (np.isfinite(fwd_w) and np.isfinite(fwd_l)):
                continue
            predicted_up = True if engine != "trend" else (float(order.iloc[j]) > 0)
            realised_up = (fwd_w - fwd_l) > 0
            hits += int(realised_up == predicted_up)
            total += 1
            scored_here = True
        dates += int(scored_here)
    if total == 0:
        return None, 0
    return hits / total, dates


def confidence_from_hit_rate(hit_rate, n_checks, shrink_k=25.0, cap=0.90):
    """Turn a realised hit rate into an Idzorek confidence.

    A rule that is right half the time carries no information, so it should get
    ZERO confidence and be ignored by the posterior. Hence:

        confidence = 2 * (hit_rate - 0.5)

    A 50% hit rate gives 0%; 60% gives 20%; 75% gives 50%. Then shrink toward
    zero when the sample is small — twelve observations of a 70% hit rate is
    not evidence — and cap below 100% because no rule deserves certainty.
    """
    if hit_rate is None or n_checks <= 0:
        return 0.0, "no history — view ignored"
    raw = 2.0 * (float(hit_rate) - 0.5)
    if raw <= 0:
        return 0.0, (f"hit rate {hit_rate*100:,.0f}% over {n_checks} checks — no better than "
                     f"chance, so the view is ignored")
    shrunk = raw * (n_checks / (n_checks + shrink_k))
    c = float(np.clip(shrunk, 0.0, cap))
    return c, (f"hit rate {hit_rate*100:,.0f}% over {n_checks} checks → confidence "
               f"{c*100:,.0f}% (shrunk for sample size)")


def omega_he_litterman(P, Sigma, tau):
    """Omega = diag(P (tau Sigma) P').  The classic choice: a view is assumed to
    be exactly as uncertain as the prior is about that same combination of
    assets. Needs no user input at all."""
    P = np.asarray(P, dtype=float)
    if P.size == 0:
        return np.zeros((0, 0))
    M = P @ (tau * np.asarray(Sigma, dtype=float)) @ P.T
    d = np.clip(np.diag(M).astype(float), 1e-12, None)
    return np.diag(d)


def omega_idzorek(P, Q, Sigma, tau, delta, w_mkt, pi, conf):
    """Idzorek (2005): back Omega out of a 0-100% confidence per view.

    For each view k, on its own:
      1. compute the posterior you would get at 100% confidence (Omega -> 0);
      2. that posterior implies unconstrained weights w_100; the departure from
         the market portfolio is  D = w_100 - w_mkt;
      3. the user's confidence c means they want a tilt of  c * D, i.e. target
         weights  w_target = w_mkt + c * D;
      4. solve numerically for the scalar omega that reproduces w_target.

    Step 4 is a well-behaved 1-D problem, so a bounded scalar search over
    log10(omega) is both faster and far more robust than Idzorek's original
    closed-form approximation. Note this whole construction is UNCONSTRAINED —
    Omega is calibrated on the analytic weights, and any box / gross-exposure
    constraints are applied afterwards by the optimiser. That is standard
    practice, but it does mean a heavily constrained portfolio will not tilt by
    exactly the confidence the user asked for.
    """
    P = np.asarray(P, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    pi = np.asarray(pi, dtype=float)
    w_mkt = np.asarray(w_mkt, dtype=float)
    k = P.shape[0]
    if k == 0:
        return np.zeros((0, 0))

    tauS = tau * Sigma
    omegas = np.zeros(k)

    for i in range(k):
        Pk = P[i:i + 1, :]
        Qk = np.array([Q[i]], dtype=float)
        c = float(np.clip(conf[i], 1e-4, 0.9999))

        denom = _scalar(Pk @ tauS @ Pk.T)
        if denom <= 0 or not np.isfinite(denom):
            omegas[i] = 1e6          # degenerate view -> effectively ignored
            continue

        # 100%-confidence posterior for this view alone
        mu_100 = pi + (tauS @ Pk.T).ravel() * ((Qk[0] - _scalar(Pk @ pi)) / denom)
        w_100 = implied_confidence_weights(mu_100, Sigma, delta)
        w_target = w_mkt + c * (w_100 - w_mkt)

        def _resid(log_om, Pk=Pk, Qk=Qk, w_target=w_target):
            om = np.array([[10.0 ** float(log_om)]])
            mu_k = posterior_mu(pi, Sigma, Pk, Qk, om, tau)
            w_k = implied_confidence_weights(mu_k, Sigma, delta)
            return float(np.sum((w_k - w_target) ** 2))

        try:
            res = minimize_scalar(_resid, bounds=(-14.0, 8.0), method="bounded",
                                  options={"xatol": 1e-8})
            omegas[i] = 10.0 ** float(res.x)
        except Exception:
            omegas[i] = denom        # fall back to He-Litterman for this view

    return np.diag(np.clip(omegas, 1e-14, None))


# ---------------------------------------------------------
# The posterior
# ---------------------------------------------------------
def posterior_mu(pi, Sigma, P, Q, Omega, tau):
    """Black-Litterman posterior expected (excess) returns.

    Uses the Sherman-Morrison-Woodbury form
        mu = pi + tau*Sigma*P' (P tau*Sigma P' + Omega)^-1 (Q - P pi)
    rather than the textbook
        mu = [(tau Sigma)^-1 + P' Omega^-1 P]^-1 [(tau Sigma)^-1 pi + P' Omega^-1 Q]
    They are algebraically identical, but the first inverts a k x k matrix
    (k = number of views, usually 1-5) instead of an n x n one, and never
    inverts Omega — which matters because a high-confidence view drives Omega
    toward zero and makes Omega^-1 blow up.
    """
    pi = np.asarray(pi, dtype=float)
    if P is None or np.asarray(P).size == 0:
        return pi.copy()
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    tauS = tau * np.asarray(Sigma, dtype=float)

    A = P @ tauS @ P.T + Omega
    resid = Q - P @ pi
    try:
        adj = tauS @ P.T @ np.linalg.solve(A, resid)
    except np.linalg.LinAlgError:
        adj = tauS @ P.T @ (np.linalg.pinv(A) @ resid)
    return pi + adj


def posterior_sigma(Sigma, P, Omega, tau):
    """Posterior covariance of returns: Sigma_p = Sigma + M, where M is the
    posterior variance of the mean. Using this instead of Sigma is the
    theoretically correct thing to do — it says 'I am not certain about the
    expected returns either' — and it pushes the optimiser toward slightly more
    diversified portfolios. Off by default because most practitioners, and most
    textbook worked examples, just use Sigma."""
    Sigma = np.asarray(Sigma, dtype=float)
    tauS = tau * Sigma
    if P is None or np.asarray(P).size == 0:
        return nearest_psd(Sigma + tauS)
    P = np.asarray(P, dtype=float)
    A = P @ tauS @ P.T + np.asarray(Omega, dtype=float)
    try:
        M = tauS - tauS @ P.T @ np.linalg.solve(A, P @ tauS)
    except np.linalg.LinAlgError:
        M = tauS - tauS @ P.T @ np.linalg.pinv(A) @ P @ tauS
    return nearest_psd(Sigma + M)


def black_litterman(Sigma, w_mkt, rf, tau, delta, P, Q, conf,
                    omega_method="Idzorek confidence", use_posterior_cov=False):
    """Full pipeline: equilibrium -> views -> posterior.

    Returns a dict with everything the UI needs to explain what happened,
    with mu_* reported as TOTAL annual returns (rf added back on).
    """
    Sigma = np.asarray(Sigma, dtype=float)
    w_mkt = np.asarray(w_mkt, dtype=float)
    pi = equilibrium_returns(delta, Sigma, w_mkt)          # excess

    has_views = P is not None and np.asarray(P).size > 0
    if has_views:
        if omega_method == "He-Litterman proportional":
            Omega = omega_he_litterman(P, Sigma, tau)
        else:
            Omega = omega_idzorek(P, Q, Sigma, tau, delta, w_mkt, pi, conf)
        mu_ex = posterior_mu(pi, Sigma, P, Q, Omega, tau)
        Sigma_used = posterior_sigma(Sigma, P, Omega, tau) if use_posterior_cov else Sigma
        view_err = np.asarray(Q, dtype=float) - np.asarray(P, dtype=float) @ pi
    else:
        Omega = np.zeros((0, 0))
        mu_ex = pi.copy()
        Sigma_used = nearest_psd(Sigma + tau * Sigma) if use_posterior_cov else Sigma
        view_err = np.zeros(0)

    return {
        "pi_excess": pi,
        "mu_excess": mu_ex,
        "pi_total": pi + rf,
        "mu_total": mu_ex + rf,
        "Sigma_used": Sigma_used,
        "Omega": Omega,
        "omega_diag": np.diag(Omega) if Omega.size else np.zeros(0),
        "view_surprise": view_err,          # how far each view sits from equilibrium
        "has_views": bool(has_views),
        "w_mkt": w_mkt,
        "delta": float(delta),
        "tau": float(tau),
    }

# =========================================================
# OPTIMIZATION
# =========================================================
def portfolio_stats(w, mu, cov, rf):
    w = np.asarray(w, dtype=float)
    ret = float(w @ mu)
    vol = math.sqrt(max(float(w @ cov @ w), 0.0))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def _downside_dev(w, R, target_period, freq_per_year):
    p = R @ w
    neg = np.minimum(p - target_period, 0.0)
    dd = math.sqrt(max(np.mean(neg ** 2), 1e-16))
    return dd * math.sqrt(freq_per_year)


def _cvar(w, R, alpha):
    p = R @ w
    losses = -p
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    return float(tail.mean()) if len(tail) else float(var)


def _stance_bounds(stance, cap, n):
    if stance == "Long only":
        return 0.0, cap, 1.0
    if stance == "Short only":
        return -cap, 0.0, -1.0
    return -cap, cap, 1.0  # long-short


def _solve(obj_fun, n, lb, ub, net, gross, seed_w=None, extra_starts=None):
    """Minimise obj_fun over weights subject to the stance/cap/gross constraints."""
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - net}]
    if gross is not None:
        cons.append({"type": "ineq", "fun": lambda w, g=gross: g - np.sum(np.abs(w))})
    bounds = tuple((lb, ub) for _ in range(n))
    starts = [np.clip(np.repeat(net / n, n), lb, ub)]
    for s in (extra_starts or []):
        if s is not None:
            starts.append(np.clip(np.asarray(s, dtype=float), lb, ub))
    if seed_w is not None:
        starts.append(np.clip(seed_w, lb, ub))
    # NB: only SOLVER OUTPUTS are candidates here, never the raw start points.
    # Starts are clipped into the bounds, which breaks the budget equality --
    # the analytic pinv seed clips to a book summing to 1.30 on a 30% cap --
    # and an unconstrained score on an infeasible point beats every legitimate
    # one. The penalised path needs start points as candidates and so screens
    # them through feasible() first; this path does not need them at all.
    best = None
    for x0 in starts:
        try:
            res = minimize(obj_fun, x0, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})
        except Exception:
            continue
        if res.x is None:
            continue
        val = obj_fun(res.x)
        if np.isfinite(val) and (best is None or val < best[0]):
            best = (val, res.x)
    if best is None:
        raise ValueError("Optimisation failed to converge with the chosen constraints.")
    return best[1]


def _optimize_once(objective, mu, cov, R, rf, stance, cap, gross, alpha, freq_per_year,
                   delta=None, R_ppy=None):
    """Solve for the optimal weights under the chosen objective and constraints.

    `R_ppy` is how many observations of `R` there are in a year. It exists so
    the loss-based objectives can be handed HOLDING-PERIOD returns rather than
    daily ones: the 5% worst day and the 5% worst year are different questions,
    and a portfolio held for a year should be judged on the second.
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)
    lb, ub, net = _stance_bounds(stance, cap, n)
    ppy = float(R_ppy) if (R_ppy and R_ppy > 0) else float(freq_per_year)
    target_period = rf / ppy

    if objective == "Min variance":
        obj = lambda w: float(w @ cov @ w)
    elif objective == "Max Sortino":
        obj = lambda w: -((w @ mu) - rf) / max(
            _downside_dev(w, R, target_period, ppy), 1e-9)
    elif objective == "Min CVaR":
        obj = lambda w: _cvar(w, R, alpha)
    else:  # Max Sharpe
        obj = lambda w: -((w @ mu) - rf) / math.sqrt(max(w @ cov @ w, 1e-16))

    seed = None
    if objective in ("Max Sharpe", "Min variance"):
        try:
            inv = np.linalg.pinv(cov)
            raw = inv @ (mu - rf * np.ones(n)) if objective == "Max Sharpe" else inv @ np.ones(n)
            if abs(raw.sum()) > 1e-12:
                seed = raw / raw.sum() * net
        except Exception:
            seed = None

    return _solve(obj, n, lb, ub, net, gross, seed_w=seed)


def resample_stack(objective, mu, cov, rf, stance, max_weight, gross_limit,
                   alpha, freq_per_year, resample_n, seed=7, delta=None):
    """Michaud resampling: return an array (n_draws x n_assets) of optimal weights
    from simulated histories. Its mean is the robust portfolio; its column std
    measures how stable each weight is."""
    cap = max_weight if max_weight is not None else 1.0
    mu_p = np.asarray(mu, dtype=float) / freq_per_year
    cov_p = nearest_psd(np.asarray(cov, dtype=float) / freq_per_year)
    t = 250
    acc = []
    for i in range(resample_n):
        rng = np.random.default_rng(seed + i)
        sim = rng.multivariate_normal(mu_p, cov_p, size=t)
        mu_hat = sim.mean(axis=0) * freq_per_year
        cov_hat = nearest_psd(np.cov(sim, rowvar=False) * freq_per_year)
        try:
            acc.append(_optimize_once(objective, mu_hat, cov_hat, sim, rf, stance,
                                      cap, gross_limit, alpha, freq_per_year, delta=delta))
        except Exception:
            continue
    if not acc:
        raise ValueError("Resampled optimisation failed on all draws.")
    return np.array(acc)


def optimize_portfolio(objective, mu, cov, R, rf, stance, max_weight, gross_limit,
                       alpha, freq_per_year, resample_n=0, seed=7, delta=None,
                       R_ppy=None):
    cap = max_weight if max_weight is not None else 1.0
    if resample_n and resample_n > 0:
        return resample_stack(objective, mu, cov, rf, stance, max_weight, gross_limit,
                              alpha, freq_per_year, resample_n, seed,
                              delta=delta).mean(axis=0)
    return _optimize_once(objective, mu, cov, R, rf, stance, cap, gross_limit, alpha,
                          freq_per_year, delta=delta, R_ppy=R_ppy)


def _min_var_at_target(mu, cov, lb, ub, net, gross, target):
    """Minimum-variance portfolio hitting a target expected return; None if infeasible."""
    n = len(mu)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - net},
            {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t}]
    if gross is not None:
        cons.append({"type": "ineq", "fun": lambda w, g=gross: g - np.sum(np.abs(w))})
    bounds = tuple((lb, ub) for _ in range(n))
    starts = [np.clip(np.repeat(net / n, n), lb, ub)]
    best = None
    for x0 in starts:
        try:
            res = minimize(lambda w: float(w @ cov @ w), x0, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 800, "ftol": 1e-11})
        except Exception:
            continue
        if res.x is None or not res.success:
            continue
        if abs(float(res.x @ mu) - target) > 1e-4 or abs(float(res.x.sum()) - net) > 1e-4:
            continue
        v = float(res.x @ cov @ res.x)
        if best is None or v < best[0]:
            best = (v, res.x)
    return None if best is None else best[1]


def efficient_frontier(mu, cov, rf, stance, max_weight, gross_limit, n_points=40):
    """Return dict of arrays describing the efficient frontier under the stance/constraints."""
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)
    cap = max_weight if max_weight is not None else 1.0
    lb, ub, net = _stance_bounds(stance, cap, n)

    def _extreme(sign):
        try:
            return _solve(lambda w: -sign * float(w @ mu), n, lb, ub, net, gross_limit)
        except Exception:
            return None

    w_hi, w_lo = _extreme(1), _extreme(-1)
    r_hi = float(w_hi @ mu) if w_hi is not None else float(np.max(mu))
    r_lo = float(w_lo @ mu) if w_lo is not None else float(np.min(mu))
    lo, hi = min(r_lo, r_hi), max(r_lo, r_hi)
    if hi - lo < 1e-9:
        hi = lo + 1e-3

    rets, vols, sharpes, wts = [], [], [], []
    for tgt in np.linspace(lo, hi, n_points):
        w = _min_var_at_target(mu, cov, lb, ub, net, gross_limit, tgt)
        if w is None:
            continue
        r = float(w @ mu)
        v = math.sqrt(max(float(w @ cov @ w), 0.0))
        rets.append(r); vols.append(v)
        sharpes.append((r - rf) / v if v > 0 else np.nan)
        wts.append(w)
    return {"ret": np.array(rets), "vol": np.array(vols),
            "sharpe": np.array(sharpes), "weights": wts}


def monte_carlo_cloud(mu, cov, rf, stance, n_points=3000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(mu)
    rets, vols, sharpes = [], [], []
    for _ in range(n_points):
        if stance == "Long only":
            w = rng.random(n); w = w / w.sum()
        elif stance == "Short only":
            w = rng.random(n); w = -(w / w.sum())
        else:
            w = rng.normal(size=n); s = w.sum()
            if abs(s) < 1e-9:
                continue
            w = w / s
        r, v, sh = portfolio_stats(w, mu, cov, rf)
        rets.append(r); vols.append(v); sharpes.append(sh)
    return np.array(vols), np.array(rets), np.array(sharpes)
