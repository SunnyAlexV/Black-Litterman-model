"""Reproduce every real-data number quoted in the README.

    python reproduce_results.py                 # all universes, ~10-20 minutes
    python reproduce_results.py --quick         # global multi-asset ETFs only

Needs a network connection (Yahoo Finance). Writes results/results.json and
prints a markdown summary. Every figure in the README's "Results on real data"
section comes from this script, so anyone can check them -- and they will
drift as the data window moves forward, which is the honest behaviour.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import bl_pipeline as P  # noqa: E402
from bl_core import engine_hit_rate, confidence_from_hit_rate  # noqa: E402
from bl_inference import (paired_bootstrap, sharpe_diff_test, breakeven_cost_bps,  # noqa: E402
                          hit_rate_pvalue)

UNIVERSES = ["Global multi-asset (ETFs)", "US sectors (ETFs)",
             "United States — S&P 500", "India — Nifty 500"]
HORIZON = "Yearly — long-term"
ENGINE = "momentum"


def _identity_backcast(caps_today, prices, asof):
    """The OLD behaviour: today's caps at every rebalance."""
    return dict(caps_today)


def study(market):
    t0 = time.time()
    s = P.Settings(market=market, horizon=HORIZON, use_vol_target=False)
    uni = P.load_universe(s)
    bench, fx, bench_label = P.fetch_benchmark_and_fx(uni, s)
    kw = dict(bench=bench, fx=fx, bench_label=bench_label)
    out = {"market": market, "n_assets": uni["n_assets"], "freq": uni["ret_freq"],
           "window": [uni["prices"].index[0].date().isoformat(),
                      uni["prices"].index[-1].date().isoformat()]}

    # 1. How big was the market-cap look-ahead?
    new, _ = P.run_backtests(uni, s, **kw)
    real = P.backcast_caps
    P.backcast_caps = _identity_backcast
    try:
        old, _ = P.run_backtests(uni, s, **kw)
    finally:
        P.backcast_caps = real
    w_old = P.market_weights(uni["usable"], uni["caps"])[0]
    w_new = P.market_weights_at(uni)(new["split_date"])
    curves = {"dates": new["dates"], "point-in-time prior": new["strat"]["equity"],
              "prior on today's caps (old, look-ahead)": old["strat"]["equity"],
              "equal weight": new["bench"]["equity"]}
    out["leak"] = {
        "test_start": str(new["split_date"].date()),
        "prior_old": old["strat"]["ann_ret"], "prior_new": new["strat"]["ann_ret"],
        "market_old": old["market"]["ann_ret"], "market_new": new["market"]["ann_ret"],
        "gap_old": old["strat"]["ann_ret"] - old["market"]["ann_ret"],
        "gap_new": new["strat"]["ann_ret"] - new["market"]["ann_ret"],
        # beat-the-index gap: what remains after the fix is universe selection
        "index_ret": new["index"]["ann_ret"] if new.get("index") else None,
        "vs_index_old": (old["strat"]["ann_ret"] - old["index"]["ann_ret"]
                         if old.get("index") else None),
        "vs_index_new": (new["strat"]["ann_ret"] - new["index"]["ann_ret"]
                         if new.get("index") else None),
        # how far today's weights are from the weights that actually prevailed
        "weight_shift_at_start": float(0.5 * np.abs(w_old - w_new).sum()),
        "biggest_mover": uni["usable"][int(np.argmax(np.abs(w_old - w_new)))],
        "biggest_mover_old_new": [float(w_old[np.argmax(np.abs(w_old - w_new))]),
                                  float(w_new[np.argmax(np.abs(w_old - w_new))])],
    }

    # 2. Does a systematic view add anything? Rule vs pure equilibrium.
    rule, eq = P.run_backtests(uni, s, engine=ENGINE, **kw)
    curves["momentum views"] = rule["strat"]["equity"]
    hr, n_dates = engine_hit_rate(uni["prices"], ENGINE, uni["freq_per_year"],
                                  holding_periods=s.reb_periods)
    conf, _ = confidence_from_hit_rate(hr, n_dates)
    bs = paired_bootstrap(rule["pnl"], eq["pnl"], uni["freq_per_year"], rf=s.rf_annual)
    lw = sharpe_diff_test(rule["pnl"], eq["pnl"], uni["freq_per_year"], rf=s.rf_annual)
    out["views"] = {
        "hit_rate": hr, "independent_dates": n_dates, "hit_p": hit_rate_pvalue(hr, n_dates),
        "confidence": conf,
        "rule_ret": rule["strat"]["ann_ret"], "eq_ret": eq["strat"]["ann_ret"],
        "gap": rule["strat"]["ann_ret"] - eq["strat"]["ann_ret"],
        "boot_diff": bs["ann_return"]["diff"], "boot_lo": bs["ann_return"]["lo"],
        "boot_hi": bs["ann_return"]["hi"], "boot_p": bs["ann_return"]["p"],
        "sharpe_diff": lw["diff"] if lw else None, "sharpe_p": lw["p"] if lw else None,
        "breakeven_bps": breakeven_cost_bps(rule, eq),
        "turnover_rule": rule["ann_turnover"], "turnover_eq": eq["ann_turnover"],
    }

    # 3. How much does it depend on the start date?
    ro_eq = P.rolling_origin(uni, s, **kw)
    ro_rule = P.rolling_origin(uni, s, engine=ENGINE, **kw)
    out["rolling"] = {"prior_vs_market": P.summarise_rolling(ro_eq),
                      "rule_vs_prior": P.summarise_rolling(ro_rule)}
    out["seconds"] = round(time.time() - t0, 1)
    return out, curves


def plot_curves(curves, market, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"point-in-time prior": dict(color="#1f4e79", lw=2.2),
             "momentum views": dict(color="#c0504d", lw=1.4),
             "equal weight": dict(color="#7f7f7f", lw=1.2),
             "prior on today's caps (old, look-ahead)": dict(color="#1f4e79", lw=1.2, ls="--")}
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    d = curves["dates"]
    for k, st_ in style.items():
        y = np.asarray(curves[k])
        n = min(len(d), len(y))
        ax.plot(d[:n], 100 * y[:n], label=k, **st_)
    ax.set_xlim(d[0], d[-1])
    ax.set_ylabel("value of 100 invested")
    ax.set_title(f"{market}: walk-forward, yearly rebalance, gross of costs", fontsize=11)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pp(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return "0.00pp" if abs(x) < 5e-5 else f"{x*100:+.2f}pp"


def report(results):
    lines = ["## Market-cap look-ahead (pure equilibrium, no overlay, yearly)", "",
             "| Universe | Prior − market, today's caps | Prior − market, back-cast | "
             "Weight shift at test start |", "|---|---|---|---|"]
    for r in results:
        L = r["leak"]
        lines.append(f"| {r['market']} | {pp(L['gap_old'])} | {pp(L['gap_new'])} | "
                     f"{L['weight_shift_at_start']*100:.1f}% ({L['biggest_mover']} "
                     f"{L['biggest_mover_old_new'][0]*100:.1f}% → "
                     f"{L['biggest_mover_old_new'][1]*100:.1f}%) |")
    lines += ["", "## Annual return of the pure-equilibrium strategy", "",
              "| Universe | Today's caps (old) | Point-in-time | Overstated by |",
              "|---|---|---|---|"]
    for r in results:
        L = r["leak"]
        lines.append(f"| {r['market']} | {L['prior_old']*100:.2f}% | {L['prior_new']*100:.2f}% | "
                     f"{(L['prior_old'] - L['prior_new'])*100:.1f}pp |")
    lines += ["", "## Strategy minus the index (selection + survivorship, and the old leak)", "",
              "| Universe | Index %/yr | Today's caps | Point-in-time |", "|---|---|---|---|"]
    for r in results:
        L = r["leak"]
        if L.get("index_ret") is not None:
            lines.append(f"| {r['market']} | {L['index_ret']*100:.2f}% | {pp(L['vs_index_old'])} | "
                         f"{pp(L['vs_index_new'])} |")
    lines += ["", "## Momentum views vs pure equilibrium", "",
              "| Universe | Hit rate (dates, p) | Confidence | Gap | 95% interval | p (return) | "
              "p (Sharpe) | Break-even |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        V = r["views"]
        be = f"{V['breakeven_bps']:.0f} bps" if V["breakeven_bps"] else "—"
        hr = (f"{V['hit_rate']*100:.0f}% ({V['independent_dates']}, p={V['hit_p']:.2f})"
              if V["hit_rate"] is not None else "—")
        sp = f"{V['sharpe_p']:.2f}" if V["sharpe_p"] is not None else "—"
        lines.append(f"| {r['market']} | {hr} | {V['confidence']*100:.0f}% | {pp(V['gap'])} | "
                     f"{pp(V['boot_lo'])} to {pp(V['boot_hi'])} | {V['boot_p']:.2f} | "
                     f"{sp} | {be} |")
    lines += ["", "## Across start dates (3-year windows every 6 months)", "",
              "| Universe | Comparison | Windows | Median gap | 10th–90th pct | Ahead / behind |",
              "|---|---|---|---|---|---|"]
    for r in results:
        for key, label in (("prior_vs_market", "prior − market"),
                           ("rule_vs_prior", "momentum − prior")):
            R = r["rolling"][key]
            if R:
                lines.append(f"| {r['market']} | {label} | {R['n_windows']} | "
                             f"{pp(R['median_gap'])} | {pp(R['p10'])} to {pp(R['p90'])} | "
                             f"{R['share_positive']*100:.0f}% / {R['share_negative']*100:.0f}% |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    results = []
    for m in (UNIVERSES[:1] if args.quick else UNIVERSES):
        print(f"... {m}", flush=True)
        r, curves = study(m)
        results.append(r)
        if m == UNIVERSES[0]:
            os.makedirs("results", exist_ok=True)
            plot_curves(curves, m, "results/equity_global_etf.png")
    os.makedirs("results", exist_ok=True)
    with open("results/results.json", "w") as f:
        json.dump({"generated": pd.Timestamp.now().isoformat(timespec="seconds"),
                   "horizon": HORIZON, "results": results}, f, indent=2, default=str)
    md = report(results)
    with open("results/results.md", "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
