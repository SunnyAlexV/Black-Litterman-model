"""The model pipeline, with no user interface.

Everything the app does between "the user picked some settings" and "here are
the weights and the backtest" lives here, as plain functions of a `Settings`
object. The Streamlit app calls these functions; so can a script, a notebook or
a test. That is the point: a number on screen and a number in a research note
come from the same code path, not from two copies that can drift apart.

    from bl_pipeline import Settings, load_universe, run_backtests
    s = Settings(market="Global multi-asset (ETFs)", horizon="Yearly — long-term")
    uni = load_universe(s)
    bt, bt_eq = run_backtests(uni, s)
"""
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bl_config import *  # noqa: F401,F403
from bl_data import *  # noqa: F401,F403
from bl_execution import *  # noqa: F401,F403
from bl_core import *  # noqa: F401,F403
from bl_backtest import *  # noqa: F401,F403


class PipelineError(Exception):
    """A condition the model cannot proceed past. The message is user-facing."""


@dataclass
class Settings:
    """Every choice that affects the numbers. Defaults are the app's Simple mode."""
    market: str
    horizon: str = "Monthly — medium-term"
    stance: str = "Long only"
    capital: float = 100_000.0
    base_currency: str = None          # None -> the market's own currency
    rf: float = None                   # None -> the market's recommended rate
    n_holdings: int = None             # None -> AUTO_HOLDINGS, capped by universe size
    start: pd.Timestamp = None         # None -> derived from the horizon
    end: pd.Timestamp = None
    ret_freq: str = None               # None -> daily, or weekly across time zones
    use_log: bool = True
    cov_method: str = "Ledoit-Wolf shrinkage"
    objective: str = "Max Sharpe"
    cvar_alpha: float = 0.95
    resample_n: int = 0
    max_weight_pct: int = None         # None -> stock or asset-class default
    gross_pct: int = None              # None -> 200 for long-short, else no limit
    tau: float = 0.05
    delta_mode: str = "Imply it from the market"
    delta_manual: float = 2.5
    omega_method: str = "Idzorek confidence"
    use_posterior_cov: bool = False
    use_vol_target: bool = True
    vol_target_relative: bool = True
    vol_target_pct: float = DEFAULT_VOL_TARGET_PCT
    vol_target_frac_pct: float = DEFAULT_VOL_TARGET_FRAC_PCT
    vol_max_lev: float = 1.0
    vol_lookback: int = None
    train_frac: float = DEFAULT_TRAIN_FRAC
    cost_bps: float = 0.0              # one-way cost per unit of turnover
    auto_trim: bool = True             # Simple mode trims names the capital can't hold
    today: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.today().normalize())

    # ---- everything below is derived, in exactly one place ----
    @property
    def hcfg(self):
        return HORIZON_CFG[self.horizon]

    @property
    def native_ccy(self):
        return MARKETS[self.market]["currency"]

    @property
    def currency(self):
        return self.base_currency or self.native_ccy

    @property
    def rf_annual(self):
        return float(market_default(self.market, "rf") if self.rf is None else self.rf)

    @property
    def is_asset_class(self):
        return MARKETS[self.market].get("kind") == "asset-class"

    @property
    def freq(self):
        # Return frequency is a property of the ASSETS: daily within one
        # exchange, weekly when the universe spans time zones (Epps effect).
        if self.ret_freq:
            return self.ret_freq
        return "Weekly" if MARKETS[self.market].get("cross_tz") else self.hcfg["freq"]

    @property
    def freq_per_year(self):
        return FREQ_PER_YEAR[self.freq]

    @property
    def reb_periods(self):
        # hcfg["rebal"] counts periods of hcfg["freq"]; rescale to the actual
        # return frequency so "every 5" never silently means five weeks.
        return max(1, int(round(self.hcfg["rebal"] * FREQ_PER_YEAR[self.freq]
                                / FREQ_PER_YEAR[self.hcfg["freq"]])))

    @property
    def weight_cap_pct(self):
        if self.max_weight_pct is not None:
            return self.max_weight_pct
        return DEFAULT_ASSET_CLASS_CAP_PCT if self.is_asset_class else DEFAULT_MAX_WEIGHT_PCT

    @property
    def max_weight(self):
        return None if self.weight_cap_pct >= 100 else self.weight_cap_pct / 100.0

    @property
    def gross_limit(self):
        g = self.gross_pct if self.gross_pct is not None else (
            200 if self.stance == "Long-short (both)" else None)
        return g / 100.0 if g is not None else None

    @property
    def top_n(self):
        n_uni = len(MARKETS[self.market]["tickers"])
        return min(self.n_holdings or AUTO_HOLDINGS, n_uni)

    @property
    def window(self):
        start = self.start if self.start is not None else (
            self.today - pd.DateOffset(years=self.hcfg["years"] + 1))
        end = self.end if self.end is not None else self.today
        return pd.Timestamp(start), pd.Timestamp(end)


# =========================================================
# STAGE A: the universe and its equilibrium inputs
# =========================================================
def load_universe(s: Settings):
    """Download, screen and estimate. Returns the dict the app caches as `uni`.

    `uni["warnings"]` holds anything the user should see but that does not stop
    the model; conditions that do stop it raise PipelineError.
    """
    start, end = s.window
    if start >= end:
        raise PipelineError("History start must be earlier than history end.")
    warnings = []
    seed = MARKETS[s.market]["tickers"]
    close, volume = download_prices_volume(
        seed, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
        interval=("1wk" if s.freq == "Weekly" else "1d"))
    if close.empty:
        raise PipelineError(
            "No price data returned. Common causes:\n\n"
            "1. **Corrupted yfinance cache** (`DatabaseError: database disk image is malformed`). "
            "Close the app and run in PowerShell: "
            "`Remove-Item -Recurse -Force \"$env:LOCALAPPDATA\\py-yfinance\"`, then relaunch.\n\n"
            "2. **Rate-limited by Yahoo** (`Too Many Requests`). Wait a few minutes.\n\n"
            "3. **Outdated library**: `python -m pip install --upgrade yfinance curl_cffi`.")

    top_n = s.top_n
    top_tickers = rank_by_liquidity(close, volume, top_n)

    # Capital-aware trim: can the money express this many positions in whole lots?
    cand_px = [float(close[t].dropna().iloc[-1]) for t in top_tickers
               if t in close.columns and not close[t].dropna().empty]
    lots = [lot_size_for(t) for t in top_tickers if t in close.columns]
    fit_n = affordable_holdings([p * l for p, l in zip(cand_px, lots)], s.capital,
                                len(top_tickers))
    auto_trimmed = None
    if fit_n < len(top_tickers):
        if s.auto_trim:
            auto_trimmed = (len(top_tickers), fit_n)
            top_tickers = top_tickers[:fit_n]
            top_n = fit_n
        else:
            warnings.append(
                f"**{s.capital:,.0f} {s.currency} may not stretch to {len(top_tickers)} "
                f"positions.** At these share prices roughly **{fit_n} names** can be held "
                f"with enough granularity to express their weights; beyond that, positions "
                f"round down hard and money sits in cash. Lower *Number of stocks to hold*, "
                f"or accept the rounding — the 'What to buy' tab reports exactly how much "
                f"stays idle.")

    # The window is chosen from the data: the longest one that still supports a
    # full-sized universe, backfilling young names from long-lived candidates.
    ranked = rank_by_liquidity(close, volume, min(len(close.columns), 4 * top_n))
    win_start, keep, win_note = choose_history_window(close, ranked, top_n, s.hcfg,
                                                      s.freq_per_year)
    cand = keep[:top_n] if len(keep) >= top_n else keep
    prices = close.loc[close.index >= win_start, cand].dropna()
    if prices.shape[1] < 2 or prices.empty:
        raise PipelineError("Could not assemble at least 2 liquid tickers with overlapping "
                            "history. Widen the date range or increase the number of stocks.")

    usable = list(prices.columns)
    n_assets = len(usable)
    returns = to_returns(prices, s.freq, s.use_log)
    if returns.shape[0] < 10:
        raise PipelineError(f"Not enough {s.freq.lower()} observations ({returns.shape[0]}) "
                            f"to estimate risk/return. Widen the date range or use a higher "
                            f"frequency.")
    if s.max_weight is not None and s.max_weight * n_assets < 1.0 - 1e-9:
        raise PipelineError(
            f"Max weight per stock ({s.weight_cap_pct}%) is too low for {n_assets} stocks. "
            f"Raise it to at least {math.ceil(100 / n_assets)}% or add more stocks.")

    caps = get_market_caps(usable)
    cov = estimate_cov(returns, s.cov_method, s.freq_per_year)

    # Traded value, the last-resort prior when caps are missing. Stored on the
    # universe so the backtest falls back to the SAME thing the live screen
    # does, rather than to equal weight.
    liq = None
    if volume is not None and not volume.empty:
        common = [t for t in usable if t in volume.columns]
        if len(common) == n_assets:
            dv = (prices[common] * volume[common]).median(axis=0, skipna=True)
            if np.isfinite(dv.values).all() and dv.values.sum() > 0:
                liq = dv.values
    w_mkt, w_mkt_note = market_weights(usable, caps, fallback_weights=liq)

    hist_annual = returns.mean().values * s.freq_per_year
    n_available = int(sum(1 for c in close.columns if not close[c].dropna().empty))
    n_universe = len(seed)
    note = (win_note +
            (f"**Trimmed from {auto_trimmed[0]} to {auto_trimmed[1]} names to fit "
             f"{s.capital:,.0f} {s.currency}.** At these share prices, {auto_trimmed[0]} "
             f"positions would round down hard and leave a large slice of your money "
             f"sitting in cash — a drag bigger than anything the optimiser is deciding. "
             f"Override in Advanced settings. " if auto_trimmed else "") +
            f"Selected the {n_assets} most liquid of {n_available} stocks that downloaded "
            f"successfully from {s.market} ({n_universe} in the index). " +
            ("" if n_available >= n_universe else
             f"{n_universe - n_available} symbol(s) were unavailable on Yahoo and skipped."))

    return {
        "signature": f"{s.market}|{s.top_n}|{s.freq}|{start.date()}|{end.date()}|{s.cov_method}",
        "market_name": s.market, "usable": usable, "n_assets": n_assets,
        "prices": prices, "returns": returns, "cov": cov,
        "sigma": np.sqrt(np.diag(cov.values)),
        "caps": caps, "liq": liq, "w_mkt": w_mkt, "w_mkt_note": w_mkt_note,
        "mkt_hist_return": float(w_mkt @ hist_annual),
        "hist_annual": hist_annual,
        "freq_per_year": s.freq_per_year, "ret_freq": s.freq, "use_log": s.use_log,
        "native_ccy": s.native_ccy,
        "start_str": start.strftime("%Y-%m-%d"), "end_str": end.strftime("%Y-%m-%d"),
        "universe_note": note, "warnings": warnings,
    }


def equilibrium(uni, s: Settings, rf=None):
    """delta, its explanation, and pi for the universe at today's market weights."""
    rf = s.rf_annual if rf is None else float(rf)
    Sigma = uni["cov"].values
    w_mkt = uni["w_mkt"]
    if s.delta_mode == "Set it myself":
        return float(s.delta_manual), "set manually", equilibrium_returns(
            s.delta_manual, Sigma, w_mkt)
    delta, note = implied_risk_aversion(w_mkt, Sigma, uni["mkt_hist_return"] - rf)
    return delta, note, equilibrium_returns(delta, Sigma, w_mkt)


# =========================================================
# STAGE B: the walk-forward backtest
# =========================================================
def market_weights_at(uni):
    """date -> cap weights as they stood on that date (back-cast, no look-ahead)."""
    usable, caps, prices, liq = uni["usable"], uni["caps"], uni["prices"], uni.get("liq")

    def _at(asof):
        w, _ = market_weights(usable, backcast_caps(caps, prices, asof), fallback_weights=liq)
        return w
    return _at


def make_builder(uni, s: Settings, rf=None, P=None, Q=None, conf=None,
                 engine=None, n_pairs=2):
    """The backtest's mu_builder: rebuild the prior (and any rule views) at each rebalance.

    engine set   -> views regenerated from the rule, confidence re-earned from its
                    hit rate, all from data available at that rebalance.
    P/Q given    -> fixed typed views (look-ahead by nature; the UI says so).
    neither      -> pure equilibrium.
    """
    rf = s.rf_annual if rf is None else float(rf)
    usable, prices = uni["usable"], uni["prices"]
    fpy = uni["freq_per_year"]
    n = len(usable)
    w_at = market_weights_at(uni)
    hr_cache = {}

    def _hit_rate(px_now):
        # Re-walking the history costs ~90ms; recompute only when the training
        # window has grown 5%. Still strictly backward-looking.
        bucket = int(math.log(max(len(px_now), 2)) / math.log(1.05))
        key = (engine, s.reb_periods, bucket)
        if key not in hr_cache:
            hr_cache[key] = engine_hit_rate(px_now, engine, fpy, holding_periods=s.reb_periods)
        return hr_cache[key]

    def _builder(train_returns, cov_annual):
        w_m = w_at(train_returns.index[-1])
        if s.delta_mode == "Set it myself":
            d = float(s.delta_manual)
        else:
            hist = train_returns.mean().values * fpy
            d, _ = implied_risk_aversion(w_m, cov_annual, float(w_m @ hist) - rf)
        if engine is not None:
            px_now = prices.loc[:train_returns.index[-1]]
            hr, nc = _hit_rate(px_now)
            c, _ = confidence_from_hit_rate(hr, nc)
            Pv, Qv, cv, _lab = systematic_views(px_now, usable, engine, fpy,
                                                n_pairs=n_pairs, conf=c)
        elif P is not None and len(P):
            Pv, Qv, cv = P, Q, conf
        else:
            Pv, Qv, cv = np.zeros((0, n)), np.zeros(0), np.zeros(0)
        out = black_litterman(cov_annual, w_m, rf, s.tau, d, Pv, Qv, cv,
                              omega_method=s.omega_method,
                              use_posterior_cov=s.use_posterior_cov)
        return out["mu_total"], out["Sigma_used"]
    return _builder


def fetch_benchmark_and_fx(uni, s: Settings):
    bench_ticker, bench_label = BENCHMARKS.get(s.market, (None, "Index"))
    bench = (get_series_close(bench_ticker, uni["start_str"], uni["end_str"])
             if bench_ticker else None)
    fx = get_fx_series(uni["native_ccy"], s.currency, uni["start_str"], uni["end_str"])
    return bench, fx, bench_label


def backtest_args(uni, s: Settings, rf=None, delta=None, bench_prices=None,
                  fx_prices=None, bench_label="Index"):
    rf = s.rf_annual if rf is None else float(rf)
    return dict(freq=uni["ret_freq"], use_log=uni["use_log"], cov_method=s.cov_method,
                freq_per_year=uni["freq_per_year"], rf=rf, stance=s.stance,
                max_weight=s.max_weight, gross_limit=s.gross_limit, objective=s.objective,
                alpha=s.cvar_alpha, train_frac=s.train_frac,
                resample_n=min(s.resample_n, 10),
                rebalance_periods=s.reb_periods, rebal_label=s.hcfg["rebal_label"],
                bench_prices=bench_prices, fx_prices=fx_prices, bench_label=bench_label,
                caps_weights=market_weights_at(uni),
                vol_target=(s.vol_target_pct / 100.0
                            if s.use_vol_target and not s.vol_target_relative else None),
                vol_target_frac=(s.vol_target_frac_pct / 100.0
                                 if s.use_vol_target and s.vol_target_relative else None),
                vol_lookback=s.vol_lookback, vol_max_leverage=s.vol_max_lev,
                delta=delta, capital=s.capital, cost_bps=s.cost_bps)


def run_backtests(uni, s: Settings, rf=None, P=None, Q=None, conf=None, engine=None,
                  n_pairs=2, apply_typed_views=False, bench=None, fx=None,
                  bench_label="Index", delta=None, prices=None):
    """(strategy, pure-equilibrium) backtests. The second is None when it would
    be identical to the first (no views in play).

    `prices` overrides uni["prices"] for the walk itself — used by the
    staggered-start study, which re-runs on truncated histories.
    """
    args = backtest_args(uni, s, rf=rf, delta=delta, bench_prices=bench, fx_prices=fx,
                         bench_label=bench_label)
    px = uni["prices"] if prices is None else prices
    eq = make_builder(uni, s, rf=rf)
    if engine is not None:
        bt = run_backtest(px, mu_builder=make_builder(uni, s, rf=rf, engine=engine,
                                                      n_pairs=n_pairs), **args)
        return bt, run_backtest(px, mu_builder=eq, **args)
    if apply_typed_views and P is not None and len(P):
        bt = run_backtest(px, mu_builder=make_builder(uni, s, rf=rf, P=P, Q=Q, conf=conf),
                          **args)
        return bt, run_backtest(px, mu_builder=eq, **args)
    return run_backtest(px, mu_builder=eq, **args), None


# =========================================================
# Robustness: the same test from many start dates
# =========================================================
def rolling_origin(uni, s: Settings, test_years=3.0, step_years=0.5, min_train_years=3.0,
                   progress=None, **run_kwargs):
    """Re-run the backtest over many fixed-length windows with staggered starts.

    One backtest is one path from one start date. This slides a window of
    `test_years` forward in steps of `step_years`; for each origin the history is
    truncated at the window's end and the training fraction set so the
    walk-forward begins exactly at the origin (training still expands from the
    first date, as in the main backtest).

    The comparison line is the pure-equilibrium run when views are in play,
    otherwise the cap-weighted market bought at the window start.

    Windows overlap, so the rows are NOT independent: read the spread across
    them as "how much does the answer depend on when you started", not as a
    confidence interval. The bootstrap in bl_inference does that job.
    """
    px = uni["prices"]
    fpy = uni["freq_per_year"]
    L = int(round(test_years * fpy))
    step = max(1, int(round(step_years * fpy)))
    first = int(round(min_train_years * fpy))
    n_ret = len(px) - 1
    origins = list(range(first, n_ret - L + 1, step))
    rows = []
    for j, o in enumerate(origins):
        sub = px.iloc[:o + L + 1]
        T_sub = len(sub) - 1
        s_j = Settings(**{**s.__dict__, "train_frac": (o + 0.5) / T_sub})
        bt, bt_eq = run_backtests(uni, s_j, prices=sub, **run_kwargs)
        if progress:
            progress((j + 1) / len(origins))
        if bt is None:
            continue
        if bt_eq is not None:
            ref, ref_label = bt_eq["strat"]["ann_ret"], "pure equilibrium"
        elif bt.get("market") is not None:
            ref, ref_label = bt["market"]["ann_ret"], "cap-weighted market"
        else:
            continue
        rows.append({"start": bt["dates"][0], "end": bt["dates"][-1],
                     "strategy": bt["strat"]["ann_ret"], "reference": ref,
                     "gap": bt["strat"]["ann_ret"] - ref, "vs": ref_label,
                     "sharpe": bt["strat"]["sharpe"], "max_dd": bt["strat"]["max_dd"]})
    return pd.DataFrame(rows)


def summarise_rolling(df):
    """Median, spread and sign-consistency of the gap across windows."""
    if df is None or df.empty:
        return None
    g = df["gap"].values
    return {"n_windows": int(len(g)), "median_gap": float(np.median(g)),
            "p10": float(np.quantile(g, 0.10)), "p90": float(np.quantile(g, 0.90)),
            # "Ahead" means by more than 0.01pp a year: a gap of 1e-16 is
            # floating-point noise, not a window the strategy won.
            "share_positive": float(np.mean(g > 1e-4)),
            "share_negative": float(np.mean(g < -1e-4)), "vs": df["vs"].iloc[0]}
