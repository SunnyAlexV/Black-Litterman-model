"""Walk-forward backtesting and performance measurement."""
import math
import numpy as np
import pandas as pd

from bl_config import *  # noqa: F401,F403
from bl_core import *  # noqa: F401,F403

# =========================================================
# BACKTEST
# =========================================================
def perf_metrics(period_returns, freq_per_year, rf, years=None):
    """Realised performance. `years` is the actual CALENDAR time the returns span.

    Without it, annualising divides the period count by a nominal 252 trading
    days. India trades about 247 days a year, so 2,035 observations get treated
    as 8.075 years when they really span 8.225 -- and every 'per year' figure
    comes out about 0.25pp too high. That is small, but it is a free
    overstatement in the app's favour, and 'per year' has to mean a year on the
    calendar if anyone is going to compare it against a deposit rate.
    """
    pr = np.asarray(period_returns, dtype=float)
    if len(pr) == 0:
        return None
    # Periods per year implied by the data itself, falling back to the nominal
    # figure when the caller cannot supply a span.
    ppy = (len(pr) / float(years)) if (years and float(years) > 0) else float(freq_per_year)
    growth = np.prod(1.0 + pr)
    ann_ret = growth ** (ppy / len(pr)) - 1.0 if growth > 0 else -1.0
    ann_vol = pr.std(ddof=1) * math.sqrt(ppy) if len(pr) > 1 else 0.0
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    eq = np.cumprod(1.0 + pr)
    peak = np.maximum.accumulate(eq)
    max_dd = float((eq / peak - 1.0).min())
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
            "max_dd": max_dd, "equity": eq}


def _period_returns_of(series, freq, index):
    """Period returns of a price Series at `freq`, aligned to `index` (0 where missing)."""
    if series is None or len(series) == 0:
        return np.zeros(len(index))
    rule = FREQ_RULE[freq]
    ps = series if rule is None else series.resample(rule).last()
    r = ps.pct_change().dropna()
    return r.reindex(index).fillna(0.0).values




def vol_target_scale(past_returns, target_vol, freq_per_year, lookback,
                     max_leverage=1.0, min_scale=0.0):
    """Exposure multiplier k = target_vol / realised_vol(past_returns).

    `past_returns` must contain ONLY periods that have already happened — the
    caller is responsible for that, and it is the whole ballgame: peeking one
    period ahead turns volatility targeting from a real effect into a fantasy.

    Why this works at all: volatility is strongly autocorrelated (a turbulent
    week is followed by a turbulent week) while returns essentially are not. So
    yesterday's volatility forecasts tomorrow's, and you can size against it,
    whereas yesterday's return tells you nothing about tomorrow's. Cutting
    exposure when risk is high therefore avoids losses more often than it
    misses gains, which is why the Sharpe ratio improves rather than just the
    volatility falling.

    max_leverage=1.0 means the strategy can only ever DE-RISK into cash — it
    never borrows. That is the conservative default here.
    """
    pr = np.asarray(past_returns, dtype=float)
    pr = pr[np.isfinite(pr)]
    if len(pr) < max(5, lookback // 4):
        return 1.0                       # not enough history yet: stay fully invested
    window = pr[-int(lookback):] if len(pr) > lookback else pr
    realised = float(np.std(window, ddof=1)) * math.sqrt(freq_per_year)
    if not np.isfinite(realised) or realised <= 1e-8:
        return float(max_leverage)
    k = float(target_vol) / realised
    return float(np.clip(k, min_scale, max_leverage))


def buy_and_hold_returns(R, w):
    """Period returns of a basket bought at weights `w` and then LEFT ALONE.

    The obvious-looking `R @ w` is not this. Applying a fixed weight vector to
    every period silently rebalances the portfolio back to `w` every single
    period — so a 'yearly rebalanced' strategy computed that way is really a
    daily-rebalanced one that happens to re-optimise once a year, and every
    rebalance frequency collapses to nearly the same result.

    Here the holdings drift: each asset's value compounds at its own return, and
    the portfolio return is the change in the total.
    """
    R = np.asarray(R, dtype=float)
    w = np.asarray(w, dtype=float)
    if R.size == 0:
        return np.zeros(0), w.copy()
    growth = np.cumprod(1.0 + R, axis=0)          # per-asset value path
    V = growth @ w                                # portfolio value path, V(0-)=1
    prev = np.concatenate(([float(np.sum(w))], V[:-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(prev) > 1e-12, V / prev - 1.0, 0.0)
    # weights the portfolio has DRIFTED to by the end — what the next rebalance
    # actually trades away from, and therefore what turnover must be measured
    # against.
    end_val = w * growth[-1]
    tot = float(np.sum(end_val))
    w_end = end_val / tot if abs(tot) > 1e-12 else w.copy()
    return r, w_end


def run_backtest(prices, freq, use_log, mu_builder, cov_method, freq_per_year, rf,
                 stance, max_weight, gross_limit, objective, alpha,
                 train_frac, resample_n,
                 rebalance_periods=21, rebal_label="periodic",
                 bench_prices=None, fx_prices=None, bench_label="Index",
                 caps_weights=None,
                 vol_target=None, vol_lookback=None, vol_max_leverage=1.0,
                 vol_target_frac=None, delta=None, capital=None,
                 horizon_aware=True, cost_bps=0.0):
    """Walk-forward backtest: starting after an initial training window, re-build
    the expected returns every `rebalance_periods` periods using all data up to
    that point (expanding window) and hold those weights until the next
    rebalance. Trading costs, if `cost_bps` > 0, are charged at each rebalance
    as cost_bps/1e4 x one-way turnover, taken out of the portfolio's value at the
    moment of trading. All returns are converted into the reporting currency using the
    FX path (fx_prices), and an index buy-and-hold (bench_prices) is included as
    a 'beat the market' benchmark.

    The only structural difference from the Markowitz version is `mu_builder`:
    a callable (train_returns, cov_annual) -> (mu_annual_total, Sigma_to_use).
    That lets the same engine drive either a plain historical/shrinkage mu or a
    Black-Litterman posterior, and lets the BL prior be re-derived at every
    rebalance from data available at that time — which is what keeps the test
    genuinely out-of-sample.
    """
    # Everything the holding period changes about the model is derived here from
    # training data only, and re-derived at every rebalance.
    R_est = to_returns(prices, freq, use_log)           # for estimation
    R_simple = to_returns(prices, freq, False)          # native realised P&L
    R_simple = R_simple.reindex(R_est.index).dropna()
    R_est = R_est.reindex(R_simple.index).dropna()
    T = len(R_est)
    if T < 24:
        return None
    start_i = int(T * train_frac)
    if start_i < 12 or (T - start_i) < 6:
        return None
    rebalance_periods = max(1, int(rebalance_periods))

    # FX path (report per native) and index path, aligned to R_simple.
    fx_ret = _period_returns_of(fx_prices, freq, R_simple.index)
    idx_ret = _period_returns_of(bench_prices, freq, R_simple.index)
    fx_applied = fx_prices is not None

    def to_report(native_arr, i0, i1):
        return (1.0 + np.asarray(native_arr)) * (1.0 + fx_ret[i0:i1]) - 1.0

    n = R_est.shape[1]
    prev_w = np.zeros(n)
    pnl, dates = [], []
    n_rebalances = 0
    turnover_sum = 0.0
    cost_rate = max(0.0, float(cost_bps or 0.0)) / 1e4
    cost_paid = []
    port_value = float(capital) if (capital and capital > 0) else 1.0
    # Where each holding period starts and stops inside `pnl`. The walk-forward
    # loop already buys a portfolio and holds it for exactly one horizon before
    # touching it again -- so every one of these segments IS a complete
    # "buy it and hold it for a week / a month / a year" experiment. Recording
    # the boundaries costs nothing and turns one 8-year path into a few hundred
    # independent answers to the question people actually ask.
    seg_bounds = []

    # Volatility targeting state. `hist` accumulates the fully-invested (k=1)
    # portfolio return of each period as it happens, so the scale applied at
    # period j is computed strictly from periods < j.
    # Two ways to set the target:
    #   ABSOLUTE  — a fixed number (12%), which is right for a diversified book
    #               and badly wrong for a concentrated one. A 12% target on a
    #               24% portfolio parks it permanently at half exposure.
    #   RELATIVE  — a fraction of the portfolio's OWN volatility, recomputed at
    #               each rebalance from the training window only. This scales
    #               with whatever you happen to be holding and never needs
    #               hand-tuning per market.
    # The relative target must come from data available at the rebalance, not
    # from the full sample — using the realised volatility of the whole backtest
    # to set the target would be look-ahead bias dressed up as convenience.
    relative_vt = vol_target_frac is not None and vol_target_frac > 0
    do_vt = relative_vt or (vol_target is not None and vol_target > 0)
    targets_used = []
    if vol_lookback is None:
        vol_lookback = max(10, int(freq_per_year / 4))     # ~3 months
    hist_unscaled = []
    scales = []
    pnl_unscaled = []
    prev_k = 1.0
    rf_period = rf / freq_per_year

    i = start_i
    while i < T:
        tr = R_est.iloc[:i]                              # expanding window
        try:
            cov_tr = estimate_cov(tr, cov_method, freq_per_year)
            # The holding period reaches the model here: rescale each asset's
            # variance by how it actually behaves over a hold of this length,
            # measured on data available at this point and nowhere else.
            if horizon_aware and rebalance_periods > 1:
                vr_tr = variance_ratio(tr.values, rebalance_periods)
                cov_use = apply_variance_ratio(cov_tr.values, vr_tr)
                R_obj = horizon_returns(tr.values, rebalance_periods)
                ppy_obj = freq_per_year / float(rebalance_periods)
            else:
                cov_use = cov_tr.values
                R_obj = tr.values
                ppy_obj = freq_per_year
            mu_tr, sigma_tr = mu_builder(tr, cov_use)
            w = optimize_portfolio(objective, np.asarray(mu_tr, dtype=float), sigma_tr,
                                   R_obj, rf, stance, max_weight, gross_limit,
                                   alpha, freq_per_year, resample_n=resample_n,
                                   delta=delta, R_ppy=ppy_obj)
        except Exception:
            w = prev_w if n_rebalances > 0 else np.repeat(1.0 / n, n)

        turnover = float(np.sum(np.abs(w - prev_w)))
        # One-way cost on everything traded, including the initial purchase
        # from cash (turnover 1.0 for a long-only book).
        cost_frac = cost_rate * turnover
        turnover_sum += turnover
        n_rebalances += 1

        end_j = min(i + rebalance_periods, T)
        # Buy the target weights, then LET THEM DRIFT until the next rebalance.
        # `w_drift` is where the portfolio actually ends up, and is what the next
        # rebalance trades away from.
        seg, w_drift = buy_and_hold_returns(R_simple.iloc[i:end_j].values, w)
        seg = seg.astype(float)
        seg_report = to_report(seg, i, end_j)                       # -> reporting ccy
        if cost_frac > 0 and len(seg_report):
            # Paid at the moment of trading: the first period of the hold starts
            # from a value already reduced by the cost. Under the volatility
            # overlay this return is then scaled by k, which correctly charges
            # only the fraction actually invested.
            seg_report = seg_report.copy()
            seg_report[0] = (1.0 + seg_report[0]) * (1.0 - cost_frac) - 1.0
            cost_paid.append(cost_frac)

        if do_vt:
            # Resolve THIS rebalance's target. In relative mode it is a fraction
            # of the portfolio's own volatility measured on the training window
            # (data strictly before period i), so it adapts to a concentrated
            # tech book and a diversified one alike without any hand-tuning.
            if relative_vt:
                train_p = R_simple.iloc[:i].values @ w
                train_p = train_p[np.isfinite(train_p)]
                if len(train_p) > 5:
                    train_vol = float(np.std(train_p, ddof=1)) * math.sqrt(freq_per_year)
                    target_now = float(vol_target_frac) * train_vol
                else:
                    target_now = vol_target if vol_target else 0.12
            else:
                target_now = vol_target
            targets_used.append(target_now)

            # Seed the volatility history with the training period the first
            # time through, so period one is already sized sensibly rather than
            # defaulting to fully invested.
            if not hist_unscaled:
                warm = (R_simple.iloc[max(0, start_i - 4 * vol_lookback):start_i].values @ w)
                hist_unscaled.extend(np.asarray(warm, dtype=float).tolist())
            scaled = np.empty_like(seg_report)
            for t, r_t in enumerate(seg_report):
                k = vol_target_scale(hist_unscaled, target_now, freq_per_year,
                                     vol_lookback, vol_max_leverage)
                # the un-invested fraction earns the risk-free rate
                scaled[t] = k * r_t + (1.0 - k) * rf_period
                scales.append(k)
                prev_k = k
                port_value *= (1.0 + float(scaled[t]))
                hist_unscaled.append(float(r_t))       # only past data feeds the next k
            pnl.extend(scaled.tolist())
        else:
            pnl.extend(seg_report.tolist())
            for r_t in seg_report:
                port_value *= (1.0 + float(r_t))

        seg_bounds.append((len(pnl) - (end_j - i), len(pnl), i, end_j))
        pnl_unscaled.extend(seg_report.tolist())        # same strategy, always fully invested
        dates.extend(list(R_simple.index[i:end_j]))
        prev_w = w_drift                                # what we actually hold now
        i = end_j

    if not pnl:
        return None

    m = len(pnl)
    # The real elapsed time of the out-of-sample window, from the dates
    # themselves rather than from a nominal trading-day count.
    _d = pd.DatetimeIndex(dates)
    test_years = ((_d[-1] - _d[0]).days / 365.25) if len(_d) > 1 else None
    if not test_years or test_years <= 0:
        test_years = m / float(freq_per_year)
    wb = np.repeat(1.0 / n, n)
    pb_native, _ = buy_and_hold_returns(R_simple.iloc[start_i:start_i + m].values, wb)
    pb = to_report(pb_native, start_i, start_i + m)                  # equal-weight, reporting ccy
    if cost_rate > 0 and len(pb):
        pb = pb.copy()                                  # buy-and-hold pays to buy once
        pb[0] = (1.0 + pb[0]) * (1.0 - cost_rate) - 1.0

    # ---- one holding period at a time ----
    # Compounded return of each individual hold, plus what the index did over
    # exactly the same days. This is the honest answer to "if I buy this and
    # hold it for one week, what happens?" -- not one path, but every such
    # period the out-of-sample window contains.
    pnl_arr = np.asarray(pnl, dtype=float)
    idx_full = (to_report(idx_ret[start_i:start_i + m], start_i, start_i + m)
                if (bench_prices is not None and np.any(idx_ret[start_i:start_i + m] != 0))
                else None)
    hold = []
    for (a, b, i0, i1) in seg_bounds:
        if b <= a or b > len(pnl_arr):
            continue
        r_s = float(np.prod(1.0 + pnl_arr[a:b]) - 1.0)
        r_i = (float(np.prod(1.0 + idx_full[a:b]) - 1.0)
               if idx_full is not None and b <= len(idx_full) else np.nan)
        hold.append({"start": R_simple.index[i0], "end": R_simple.index[i1 - 1],
                     "periods": int(b - a), "strat": r_s, "index": r_i})

    strat = perf_metrics(np.array(pnl), freq_per_year, rf, years=test_years)
    bench = perf_metrics(pb, freq_per_year, rf, years=test_years)

    # Buy-and-hold the CAP-WEIGHTED market portfolio of the same names. This is
    # the benchmark that actually matters for Black-Litterman: with no views the
    # model is anchored on exactly this portfolio, so it is the honest
    # 'did the machinery add anything?' line.
    # `caps_weights` may be a callable date -> weights. That is the honest form:
    # the benchmark buys the cap weights as they stood on the day the test
    # starts, then drifts. Passing a fixed array (today's weights) is kept for
    # backwards compatibility, but it imports the future into the benchmark.
    market_perf = None
    if caps_weights is not None:
        if callable(caps_weights):
            wm = np.asarray(caps_weights(R_simple.index[start_i - 1]), dtype=float)
        else:
            wm = np.asarray(caps_weights, dtype=float)
        if wm.shape[0] == n and np.isfinite(wm).all():
            pm_native, _ = buy_and_hold_returns(
                R_simple.iloc[start_i:start_i + m].values, wm)
            pm = to_report(pm_native, start_i, start_i + m)
            if cost_rate > 0 and len(pm):
                pm = pm.copy()
                pm[0] = (1.0 + pm[0]) * (1.0 - cost_rate * float(np.abs(wm).sum())) - 1.0
            market_perf = perf_metrics(pm,
                                       freq_per_year, rf, years=test_years)

    index_perf = None
    if bench_prices is not None and np.any(idx_ret[start_i:start_i + m] != 0):
        idx_report = to_report(idx_ret[start_i:start_i + m], start_i, start_i + m)
        index_perf = perf_metrics(idx_report, freq_per_year, rf, years=test_years)

    # The same strategy without the volatility overlay — the honest like-for-like
    # comparison that shows whether the overlay earned its keep.
    unscaled_perf = perf_metrics(np.array(pnl_unscaled), freq_per_year, rf,
                                 years=test_years) if do_vt else None

    return {"strat": strat, "bench": bench, "index": index_perf, "index_label": bench_label,
            "market": market_perf,
            "vt_applied": bool(do_vt), "unscaled": unscaled_perf,
            "vt_relative": bool(relative_vt), "vt_frac": vol_target_frac,
            "vt_target": (float(np.mean(targets_used)) if targets_used else vol_target),
            "vt_lookback": vol_lookback,
            "vt_max_leverage": vol_max_leverage,
            "avg_scale": float(np.mean(scales)) if scales else 1.0,
            "min_scale": float(np.min(scales)) if scales else 1.0,
            "max_scale": float(np.max(scales)) if scales else 1.0,
            "scales": np.array(scales) if scales else None,
            "dates": pd.DatetimeIndex(dates), "fx_applied": fx_applied,
            "train_start": R_est.index[0], "split_date": R_est.index[start_i],
            "n_test": m, "n_train": start_i, "test_years": float(test_years),
            "holds": hold,
            "n_rebalances": n_rebalances, "rebal_label": rebal_label,
            "avg_turnover": turnover_sum / max(n_rebalances, 1),
            # Turnover per year of test window, INCLUDING the initial purchase.
            # Comparing two strategies, (gross return gap) / (turnover gap) is the
            # one-way cost at which their ranking flips -- see breakeven_cost_bps.
            "ann_turnover": turnover_sum / float(test_years),
            "cost_bps": cost_rate * 1e4,
            "cost_drag_ann": float(np.sum(cost_paid)) / float(test_years),
            "pnl": np.asarray(pnl, dtype=float)}
