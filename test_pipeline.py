"""End-to-end smoke test of the plumbing: synthetic prices -> covariance ->
Black-Litterman posterior -> optimiser -> frontier -> walk-forward backtest.

No network, no Streamlit server. This is the test that catches wiring mistakes
(wrong argument order, a Sigma that never reaches the optimiser, a backtest
that silently falls back to equal weight) as opposed to maths mistakes.
"""
import sys
import types
import numpy as np
import pandas as pd

st = types.ModuleType("streamlit")
st.cache_data = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda fn: fn))
st.column_config = types.SimpleNamespace()
sys.modules["streamlit"] = st
yfm = types.ModuleType("yfinance")
yfm.set_tz_cache_location = lambda *a, **k: None
sys.modules["yfinance"] = yfm

import black_litterman_portfolio as bl  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------
# Synthetic market: 8 assets, one common factor + idiosyncratic noise
# ---------------------------------------------------------
rng = np.random.default_rng(42)
n, T = 8, 1200
tickers = [f"STK{i}" for i in range(n)]
beta = np.linspace(0.6, 1.6, n)
factor = rng.normal(0.0004, 0.010, T)
eps = rng.normal(0, 0.011, (T, n))
rets = factor[:, None] * beta[None, :] + eps
idx = pd.bdate_range("2019-01-01", periods=T)
prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=tickers)

caps = {t: c for t, c in zip(tickers, np.linspace(500e9, 40e9, n))}
rf, tau, delta_manual = 0.04, 0.05, 2.5
freq_per_year = 252

returns = bl.to_returns(prices, "Daily", True)
cov = bl.estimate_cov(returns, "Ledoit-Wolf shrinkage", freq_per_year)
check("covariance is annualised and plausible",
      0.05 < float(np.mean(np.sqrt(np.diag(cov.values)))) < 0.60,
      f"mean vol {np.mean(np.sqrt(np.diag(cov.values))):.3f}")

w_mkt, note = bl.market_weights(tickers, caps)
hist = returns.mean().values * freq_per_year
delta, dnote = bl.implied_risk_aversion(w_mkt, cov.values, float(w_mkt @ hist) - rf)
check("delta implied from synthetic data is in a sane range", 0.5 <= delta <= 10.0, f"{delta:.2f}")

# ---------------------------------------------------------
# No views: the constrained optimum should be close to the market portfolio
# ---------------------------------------------------------
bl0 = bl.black_litterman(cov.values, w_mkt, rf, tau, delta,
                         np.zeros((0, n)), np.zeros(0), np.zeros(0))
w0 = bl.optimize_portfolio("Max Sharpe", bl0["mu_total"], bl0["Sigma_used"], returns.values,
                           rf, "Long only", None, None, 0.95, freq_per_year)
check("no views: long-only weights sum to 1", abs(w0.sum() - 1) < 1e-6, f"{w0.sum():.6f}")
check("no views: optimiser lands on the market portfolio",
      np.abs(w0 - w_mkt).max() < 5e-3, f"max diff {np.abs(w0-w_mkt).max():.4f}")

# ---------------------------------------------------------
# With views, through the real views-table path
# ---------------------------------------------------------
views = pd.DataFrame([
    [True, "Absolute", "STK7", "STK0", 22.0, 70.0],
    [True, "Relative", "STK1", "STK6", 4.0, 40.0],
], columns=["Use", "Type", "Asset", "Versus", "Return % p.a.", "Confidence %"])

P, Q, conf, labels, errs = bl.build_pq(views, tickers, rf)
check("views table parsed with no errors", len(errs) == 0 and P.shape == (2, n), str(errs))

blv = bl.black_litterman(cov.values, w_mkt, rf, tau, delta, P, Q, conf)
wv = bl.optimize_portfolio("Max Sharpe", blv["mu_total"], blv["Sigma_used"], returns.values,
                           rf, "Long only", None, None, 0.95, freq_per_year)
check("with views: weights still sum to 1", abs(wv.sum() - 1) < 1e-6)
check("the bullish view increases that name's weight",
      wv[7] > w0[7], f"{wv[7]:.4f} vs {w0[7]:.4f}")
check("views move the portfolio measurably away from the market",
      np.abs(wv - w_mkt).sum() > np.abs(w0 - w_mkt).sum(),
      f"{np.abs(wv-w_mkt).sum():.4f} vs {np.abs(w0-w_mkt).sum():.4f}")

# ---------------------------------------------------------
# Frontier
# ---------------------------------------------------------
fr = bl.efficient_frontier(blv["mu_total"], blv["Sigma_used"], rf, "Long only", None, None,
                           n_points=12)
check("frontier returns points", len(fr["ret"]) >= 8, f"{len(fr['ret'])} points")
# The frontier is swept across the FULL range of feasible returns, so it
# includes the inefficient lower branch below the minimum-variance point.
# Volatility should therefore be U-shaped in return, not monotone. (This is
# inherited from the Markowitz app, which plots the same full curve.)
v_sorted = fr["vol"][np.argsort(fr["ret"])]
argmin = int(np.argmin(v_sorted))
check("frontier volatility is U-shaped in return (min-variance point in the middle)",
      np.all(np.diff(v_sorted[:argmin + 1]) < 1e-6) and np.all(np.diff(v_sorted[argmin:]) > -1e-6),
      f"min-variance at point {argmin} of {len(v_sorted)}")
pr, pv, ps = bl.portfolio_stats(wv, blv["mu_total"], blv["Sigma_used"], rf)
check("the optimum is not above the frontier", pv >= fr["vol"].min() - 1e-9)

# ---------------------------------------------------------
# Walk-forward backtest through the mu_builder interface
# ---------------------------------------------------------
def make_builder(with_views):
    def _b(train_returns, cov_annual):
        wm, _ = bl.market_weights(tickers, caps)
        h = train_returns.mean().values * freq_per_year
        d, _n = bl.implied_risk_aversion(wm, cov_annual, float(wm @ h) - rf)
        out = bl.black_litterman(cov_annual, wm, rf, tau, d,
                                 P if with_views else np.zeros((0, n)),
                                 Q if with_views else np.zeros(0),
                                 conf if with_views else np.zeros(0))
        return out["mu_total"], out["Sigma_used"]
    return _b


bt = bl.run_backtest(prices, "Daily", True, make_builder(False), "Ledoit-Wolf shrinkage",
                     freq_per_year, rf, "Long only", None, None, "Max Sharpe", 0.95,
                     10.0, 50.0, 0.7, 0, rebalance_periods=21, rebal_label="monthly",
                     caps_weights=w_mkt)
check("backtest returns a result", bt is not None)
if bt:
    check("backtest produced out-of-sample periods", bt["n_test"] > 50, str(bt["n_test"]))
    check("backtest rebalanced more than once", bt["n_rebalances"] > 3, str(bt["n_rebalances"]))
    check("equity curve is finite and positive",
          np.all(np.isfinite(bt["strat"]["equity"])) and bt["strat"]["equity"].min() > 0)
    check("cap-weighted market benchmark was computed", bt["market"] is not None)
    check("turnover was actually charged (weights move)", bt["avg_turnover"] > 0,
          f"{bt['avg_turnover']:.4f}")
    check("dates align with the equity curve", len(bt["dates"]) == len(bt["strat"]["equity"]))

bt_v = bl.run_backtest(prices, "Daily", True, make_builder(True), "Ledoit-Wolf shrinkage",
                       freq_per_year, rf, "Long only", None, None, "Max Sharpe", 0.95,
                       10.0, 50.0, 0.7, 0, rebalance_periods=21, rebal_label="monthly",
                       caps_weights=w_mkt)
check("views-on backtest differs from views-off backtest",
      bt_v is not None and abs(bt_v["strat"]["ann_ret"] - bt["strat"]["ann_ret"]) > 1e-6,
      f"{bt_v['strat']['ann_ret']:.6f} vs {bt['strat']['ann_ret']:.6f}")

# ---------------------------------------------------------
# Long-short and constraints still work off the posterior
# ---------------------------------------------------------
wls = bl.optimize_portfolio("Max Sharpe", blv["mu_total"], blv["Sigma_used"], returns.values,
                            rf, "Long-short (both)", 0.30, 2.0, 0.95, freq_per_year)
check("long-short: net exposure is 1", abs(wls.sum() - 1) < 1e-5, f"{wls.sum():.6f}")
check("long-short: gross exposure respects the 200% limit", np.abs(wls).sum() <= 2.0 + 1e-4,
      f"{np.abs(wls).sum():.4f}")
check("long-short: per-name cap respected", np.abs(wls).max() <= 0.30 + 1e-6,
      f"{np.abs(wls).max():.4f}")

wc = bl.optimize_portfolio("Min variance", blv["mu_total"], blv["Sigma_used"], returns.values,
                           rf, "Long only", 0.20, None, 0.95, freq_per_year)
check("min-variance with a 20% cap respects it", wc.max() <= 0.20 + 1e-6, f"{wc.max():.4f}")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
    sys.exit(1)
print("All pipeline tests passed.")