"""Tests for inference, trading costs, the hit-rate sample size and the pipeline settings.

Streamlit and yfinance are stubbed, so nothing here needs a network.

Run:  python test_inference.py
"""
import math
import sys
import types
import numpy as np
import pandas as pd

st = types.ModuleType("streamlit")
st.cache_data = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
st.column_config = types.SimpleNamespace()
sys.modules["streamlit"] = st
yf = types.ModuleType("yfinance")
yf.set_tz_cache_location = lambda *a, **k: None
yf.download = lambda *a, **k: None
yf.Ticker = lambda *a, **k: None
sys.modules["yfinance"] = yf

import black_litterman_portfolio as bl  # noqa: E402
import bl_inference as inf  # noqa: E402
from bl_pipeline import Settings  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


rng = np.random.default_rng(42)

# =========================================================
print("\n--- stationary block bootstrap ---")
T, L = 1500, 12
idx = inf.stationary_bootstrap_indices(T, L, 200, np.random.default_rng(1))
check("indices stay inside the sample", idx.min() >= 0 and idx.max() < T)
runs = (np.diff(idx, axis=1) == 1) | (np.diff(idx, axis=1) == -(T - 1))
mean_block = 1.0 / (1.0 - runs.mean())
check("mean block length matches the requested one", abs(mean_block - L) / L < 0.1,
      f"{mean_block:.1f} vs {L}")
check("default block length grows like T^(1/3)",
      inf.default_block_length(2000) == 13 and inf.default_block_length(8000) == 20)

b = rng.normal(0.0004, 0.01, 2000)
c = 0.0001
res = inf.paired_bootstrap(b + c, b, 252, n_boot=500)
check("a constant per-period edge is recovered exactly",
      abs(res["ann_return"]["diff"] - c * 252) < 1e-12)
check("and, because the noise is shared, its interval has zero width",
      abs(res["ann_return"]["hi"] - res["ann_return"]["lo"]) < 1e-12)
res0 = inf.paired_bootstrap(b, b, 252, n_boot=300)
check("identical strategies: zero gap and p = 1",
      res0["ann_return"]["diff"] == 0 and res0["ann_return"]["p"] == 1.0)

# Positive autocorrelation inflates the variance of a mean by (1+phi)/(1-phi).
# An i.i.d. bootstrap (block length 1) misses that; the block bootstrap should not.
phi, n_ar = 0.3, 3000
e = rng.normal(0, 0.01, n_ar)
x = np.empty(n_ar)
x[0] = e[0]
for t in range(1, n_ar):
    x[t] = phi * x[t - 1] + e[t]
zero = np.zeros(n_ar)
se_block = inf.paired_bootstrap(x, zero, 252, n_boot=800, seed=3)["ann_return"]["se"]
se_iid = inf.paired_bootstrap(x, zero, 252, n_boot=800, mean_block=1, seed=3)["ann_return"]["se"]
ratio = (se_block / se_iid) ** 2
check("a zero-volatility leg gives an undefined Sharpe gap, not a crash",
      np.isnan(inf.paired_bootstrap(x, zero, 252, n_boot=100)["sharpe"]["p"]))
check("block bootstrap captures autocorrelation that an i.i.d. bootstrap misses",
      1.4 < ratio < 2.4, f"variance ratio {ratio:.2f}, theory {(1+phi)/(1-phi):.2f}")

# =========================================================
print("\n--- Ledoit-Wolf Sharpe-difference test ---")
# On i.i.d. normal returns the HAC delta method must reduce to the closed form
#   T * Var(SR1 - SR2) = 2(1 - rho) + 0.5 (SR1^2 + SR2^2 - 2 SR1 SR2 rho^2)
n_big, rho = 200_000, 0.8
sr1, sr2 = 0.06, 0.03                                 # per-period Sharpe ratios
z1 = rng.normal(size=n_big)
z2 = rho * z1 + math.sqrt(1 - rho ** 2) * rng.normal(size=n_big)
ra, rb = sr1 * 0.01 + 0.01 * z1, sr2 * 0.01 + 0.01 * z2
lw = inf.sharpe_diff_test(ra, rb, 1, lag=0)
theory = math.sqrt((2 * (1 - rho) + 0.5 * (sr1 ** 2 + sr2 ** 2 - 2 * sr1 * sr2 * rho ** 2))
                   / n_big)
print(f"      HAC se / closed form = {lw['se'] / theory:.4f}")
check("HAC standard error matches the Jobson-Korkie/Memmel formula on i.i.d. normal data",
      abs(lw["se"] / theory - 1) < 0.03, f"{lw['se']:.5f} vs {theory:.5f}")
check("Sharpe ratios are annualised by sqrt(periods per year)",
      abs(inf.sharpe_diff_test(ra, rb, 252, lag=0)["sharpe_a"] - lw["sharpe_a"] * math.sqrt(252))
      < 1e-9)

rej = 0
n_sims = 1000
for k in range(n_sims):
    r = np.random.default_rng(1000 + k)
    u = r.normal(size=1000)
    v = 0.7 * u + math.sqrt(1 - 0.49) * r.normal(size=1000)
    rej += inf.sharpe_diff_test(0.0003 + 0.01 * u, 0.0003 + 0.01 * v, 252)["p"] < 0.05
print(f"      rejection rate at 5% under the null, {n_sims} simulations = {rej / n_sims:.3f}")
check("under a true null the test rejects about 5% of the time",
      0.025 <= rej / n_sims <= 0.085, f"{rej / n_sims:.3f}")
check("too short a sample returns None rather than a number",
      inf.sharpe_diff_test(ra[:10], rb[:10], 252) is None
      and inf.paired_bootstrap(ra[:10], rb[:10], 252) is None)

# =========================================================
print("\n--- hit rate: independent dates, and its p-value ---")
_n, _T = 10, 2500
_px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0004, 0.012, (_T, _n)), axis=0),
                   index=pd.bdate_range("2014-01-01", periods=_T),
                   columns=[f"H{i}" for i in range(_n)])
_, d1 = bl.engine_hit_rate(_px, "momentum", 252, 21, n_pairs=1)
_, d3 = bl.engine_hit_rate(_px, "momentum", 252, 21, n_pairs=3)
check("the sample size counts dates, so it does not grow with pairs per date",
      d1 == d3 and d1 > 0, f"{d1} vs {d3}")
check("hit-rate p-value: 50% is 0.5, a strong record is small",
      abs(inf.hit_rate_pvalue(0.5, 40) - 0.5) < 1e-12 and inf.hit_rate_pvalue(0.7, 100) < 1e-4)

# =========================================================
print("\n--- trading costs ---")
_nb, _Tb = 8, 1500
_Rb = rng.normal(0.0004, 0.011, (_Tb, _nb))
_pxb = pd.DataFrame(100 * np.cumprod(1 + _Rb, axis=0),
                    index=pd.bdate_range("2016-01-01", periods=_Tb),
                    columns=[f"B{i}" for i in range(_nb)])
_w = np.linspace(1, 2, _nb) / np.linspace(1, 2, _nb).sum()


def _builder(train, cov):
    d, _ = bl.implied_risk_aversion(_w, cov, float(_w @ train.mean().values * 252) - 0.03)
    o = bl.black_litterman(cov, _w, 0.03, 0.05, d, np.zeros((0, _nb)), np.zeros(0), np.zeros(0))
    return o["mu_total"], o["Sigma_used"]


_A = dict(freq="Daily", use_log=True, mu_builder=_builder, cov_method="Ledoit-Wolf shrinkage",
          freq_per_year=252, rf=0.03, stance="Long only", max_weight=0.3, gross_limit=None,
          objective="Max Sharpe", alpha=0.95, train_frac=0.5, resample_n=0,
          rebalance_periods=21, rebal_label="monthly", vol_target=None, caps_weights=_w)
g = bl.run_backtest(_pxb, **_A)
g0 = bl.run_backtest(_pxb, cost_bps=0.0, **_A)
n50 = bl.run_backtest(_pxb, cost_bps=50.0, **_A)
check("zero cost reproduces the gross result exactly",
      np.array_equal(g["pnl"], g0["pnl"]))
check("costs lower the return", n50["strat"]["ann_ret"] < g["strat"]["ann_ret"])
_drag = g["strat"]["ann_ret"] - n50["strat"]["ann_ret"]
check("the drag is about cost x annual turnover",
      abs(_drag / (0.005 * g["ann_turnover"]) - 1) < 0.25,
      f"drag {_drag*100:.3f}pp vs {0.005*g['ann_turnover']*100:.3f}pp")
check("the first purchase from cash is charged (turnover 1.0 on day one)",
      abs((1 + g["pnl"][0]) * (1 - 0.005) - 1 - n50["pnl"][0]) < 1e-15)
check("break-even cost: (return gap) / (turnover gap)",
      abs(inf.breakeven_cost_bps({"strat": {"ann_ret": 0.10}, "ann_turnover": 2.0},
                                 {"strat": {"ann_ret": 0.09}, "ann_turnover": 0.5})
          - 0.01 / 1.5 * 1e4) < 1e-9)
check("no break-even when the busier strategy is not ahead",
      inf.breakeven_cost_bps({"strat": {"ann_ret": 0.08}, "ann_turnover": 2.0},
                             {"strat": {"ann_ret": 0.09}, "ann_turnover": 0.5}) is None)

# =========================================================
print("\n--- pipeline settings ---")
_us = next(m for m in bl.MARKETS if "S&P 500" in m)
s_w = Settings(market=_us, horizon="Monthly — medium-term", ret_freq="Weekly")
check("weekly returns rescale a monthly rebalance to ~4 weeks, not 21",
      s_w.reb_periods == 4, str(s_w.reb_periods))
s_d = Settings(market=_us, horizon="Monthly — medium-term")
check("daily returns keep 21 periods", s_d.reb_periods == 21)
_cross = [m for m, v in bl.MARKETS.items() if v.get("cross_tz")]
check("a universe spanning time zones defaults to weekly returns",
      bool(_cross) and Settings(market=_cross[0]).freq == "Weekly")
_ac = [m for m, v in bl.MARKETS.items() if v.get("kind") == "asset-class"]
check("asset-class universes get the asset-class weight cap",
      Settings(market=_ac[0]).max_weight == bl.DEFAULT_ASSET_CLASS_CAP_PCT / 100)
check("long-short defaults to a 200% gross limit",
      Settings(market=_us, stance="Long-short (both)").gross_limit == 2.0)

# =========================================================
print("\n--- staggered start dates ---")
from bl_pipeline import rolling_origin, summarise_rolling, run_backtests  # noqa: E402

_nr, _Tr = 6, 252 * 9
_pxr = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0004, 0.011, (_Tr + 1, _nr)), axis=0),
                    index=pd.bdate_range("2015-01-01", periods=_Tr + 1),
                    columns=[f"R{i}" for i in range(_nr)])
_uni = {"prices": _pxr, "usable": list(_pxr.columns),
        "caps": dict(zip(_pxr.columns, np.linspace(5e9, 1e9, _nr) * _pxr.iloc[-1].values / 100)),
        "liq": None, "freq_per_year": 252, "ret_freq": "Daily", "use_log": True}
_s = Settings(market=_us, horizon="Yearly — long-term", rf=0.03, use_vol_target=False,
              max_weight_pct=100)
_ro = rolling_origin(_uni, _s, test_years=2, step_years=1, min_train_years=3)
check("windows step forward by the requested amount",
      len(_ro) == 5 and all((_ro["start"].diff().dropna().dt.days > 330)
                            & (_ro["start"].diff().dropna().dt.days < 400)),
      f"{len(_ro)} windows")
_pos = _pxr.index.get_indexer
check("each window spans exactly the requested number of periods",
      all(_pos([e])[0] - _pos([b])[0] + 1 == 2 * 252 for b, e in zip(_ro["start"], _ro["end"])))
# The first window must equal a direct backtest truncated and started by hand.
_o, _L = 252 * 3, 252 * 2
_sub = _pxr.iloc[:_o + _L + 1]
_s0 = Settings(**{**_s.__dict__, "train_frac": (_o + 0.5) / (len(_sub) - 1)})
_bt0, _ = run_backtests(_uni, _s0, prices=_sub)
check("a window reproduces the equivalent direct backtest exactly",
      abs(_ro["strategy"].iloc[0] - _bt0["strat"]["ann_ret"]) < 1e-14
      and _ro["start"].iloc[0] == _pxr.index[_o + 1])
_sm = summarise_rolling(_ro)
check("summary reports median, 10-90 band and sign consistency",
      _sm["p10"] <= _sm["median_gap"] <= _sm["p90"] and 0 <= _sm["share_positive"] <= 1)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
    sys.exit(1)
print("All inference tests passed.")
