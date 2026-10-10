"""Black-Litterman portfolio optimiser — Streamlit UI.

The maths lives in bl_core, the backtest in bl_backtest, market data in bl_data
and order sizing in bl_execution. This file is the interface only, and it
re-exports everything so `import black_litterman_portfolio as bl` keeps working.
"""
import os
import re
import math
import logging
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from concurrent.futures import ThreadPoolExecutor


from bl_config import *  # noqa: F401,F403
from bl_data import *  # noqa: F401,F403
from bl_execution import *  # noqa: F401,F403
from bl_core import *  # noqa: F401,F403
from bl_backtest import *  # noqa: F401,F403
from bl_inference import *  # noqa: F401,F403
from bl_pipeline import Settings, PipelineError, load_universe, run_backtests, \
    fetch_benchmark_and_fx, market_weights_at, rolling_origin, summarise_rolling  # noqa: F401
from bl_core import _scalar  # noqa: F401  (used by the tests)

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

def pretty_df(df, decimals_map=None):
    out = df.copy()
    decimals_map = decimals_map or {}
    for col, dec in decimals_map.items():
        if col in out.columns:
            def _fmt(x):
                if pd.isna(x):
                    return ""
                if isinstance(x, (int, float, np.integer, np.floating)):
                    return f"{float(x):,.{dec}f}"
                return x
            out[col] = out[col].apply(_fmt)
    return out

# =========================================================
# STREAMLIT UI  (the numeric core above imports cleanly without it)
# =========================================================
def _get_selected_points(event):
    if event is None:
        return []
    sel = None
    try:
        sel = event["selection"]
    except Exception:
        sel = getattr(event, "selection", None)
    if not sel:
        return []
    try:
        return sel["points"]
    except Exception:
        return getattr(sel, "points", []) or []



def w_shorten(note, limit=22):
    """Fit a provenance note into a st.metric value slot without wrapping."""
    note = str(note or "")
    head = note.split("(")[0].strip()
    if len(head) <= limit:
        return head or "—"
    return head[:limit - 1] + "…"



def render_frontier(res):
    fr = res["frontier"]
    usable = res["usable"]
    sigma = res["sigma"]; mu_vals = res["mu_vals"]
    port_vol = res["port_vol"]; port_ret = res["port_ret"]
    mkt_vol = res.get("mkt_vol"); mkt_ret = res.get("mkt_ret")

    if len(fr["ret"]) == 0:
        st.info("Could not compute an efficient frontier for these settings.")
        return

    if not HAS_PLOTLY:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(fr["vol"] * 100, fr["ret"] * 100, "-o", color="#1f77b4", ms=4, label="Efficient frontier")
        ax.scatter(sigma * 100, mu_vals * 100, c="black", s=20, label="Individual stocks")
        if mkt_vol is not None:
            ax.scatter([mkt_vol * 100], [mkt_ret * 100], c="#ff7f0e", s=120, marker="D",
                       label="Market portfolio (the prior)", zorder=4)
        ax.scatter([port_vol * 100], [port_ret * 100], c="red", s=160, marker="*",
                   label="Black-Litterman optimal", zorder=5)
        ax.set_xlabel("Annual volatility (%)"); ax.set_ylabel("Annual return (%)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.caption("Install Plotly for the interactive version: `pip install plotly`.")
        return

    left, right = st.columns([3, 1])
    with left:
        fig = go.Figure()
        # curve 0 = efficient frontier (the clickable trace)
        fig.add_trace(go.Scatter(
            x=fr["vol"] * 100, y=fr["ret"] * 100, mode="lines+markers", name="Efficient frontier",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=9, color=fr["sharpe"], colorscale="Viridis",
                        showscale=True, colorbar=dict(title="Sharpe")),
            hovertemplate="Return %{y:.2f}%<br>Volatility %{x:.2f}%<br>Sharpe %{marker.color:.2f}"
                          "<extra>click for allocation</extra>"))
        # background cloud
        if len(res["mc_vols"]):
            fig.add_trace(go.Scatter(
                x=res["mc_vols"] * 100, y=res["mc_rets"] * 100, mode="markers",
                name="Random portfolios", marker=dict(size=3, color="rgba(150,150,150,0.35)"),
                hoverinfo="skip", showlegend=True))
        # individual stocks
        fig.add_trace(go.Scatter(
            x=sigma * 100, y=mu_vals * 100, mode="markers", name="Individual stocks",
            marker=dict(color="black", size=7), text=usable,
            hovertemplate="%{text}<br>Return %{y:.2f}%<br>Vol %{x:.2f}%<extra></extra>"))
        # the prior: the market portfolio itself
        if mkt_vol is not None:
            fig.add_trace(go.Scatter(
                x=[mkt_vol * 100], y=[mkt_ret * 100], mode="markers",
                name="Market portfolio (the prior)",
                marker=dict(color="#ff7f0e", size=14, symbol="diamond"),
                hovertemplate="Market portfolio<br>Return %{y:.2f}%<br>Vol %{x:.2f}%<extra></extra>"))
        # optimal portfolio
        fig.add_trace(go.Scatter(
            x=[port_vol * 100], y=[port_ret * 100], mode="markers",
            name="Your Black-Litterman portfolio",
            marker=dict(color="red", size=18, symbol="star"),
            hovertemplate="Black-Litterman<br>Return %{y:.2f}%<br>Vol %{x:.2f}%<extra></extra>"))
        fig.update_layout(xaxis_title="Annual volatility (%)", yaxis_title="Annual return (%)",
                          height=500, margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
        try:
            event = st.plotly_chart(fig, width='stretch', on_select="rerun",
                                    selection_mode="points", key="frontier_chart")
        except TypeError:
            st.plotly_chart(fig, width='stretch')
            event = None

    sel_idx = None
    for p in _get_selected_points(event):
        if p.get("curve_number") == 0:
            sel_idx = p.get("point_index", p.get("point_number"))
            break

    with right:
        st.markdown("**Portfolio detail**")
        if sel_idx is None:
            st.caption("Hover any point for its return & volatility. **Click a point on the blue "
                       "frontier** to see its exact allocation below. The orange diamond is the "
                       "market portfolio your prior is anchored on — with no views, the red star "
                       "sits on top of it.")
        else:
            sel_idx = int(sel_idx)
            st.metric("Annual return", f"{fr['ret'][sel_idx]*100:,.2f}%")
            st.metric("Annual volatility", f"{fr['vol'][sel_idx]*100:,.2f}%")
            st.metric("Sharpe", f"{fr['sharpe'][sel_idx]:,.2f}")
            if st.button("↩ Back to full chart"):
                st.session_state.pop("frontier_chart", None)
                st.rerun()

    if sel_idx is not None:
        sel_idx = int(sel_idx)
        w = np.asarray(fr["weights"][sel_idx])
        alloc = pd.DataFrame({"Ticker": usable, "WeightPct": w * 100})
        alloc = alloc[np.abs(alloc["WeightPct"]) > 1e-4].sort_values("WeightPct", ascending=False)
        st.markdown(f"**Allocation of the selected frontier portfolio** — return "
                    f"{fr['ret'][sel_idx]*100:.2f}%, volatility {fr['vol'][sel_idx]*100:.2f}%, "
                    f"Sharpe {fr['sharpe'][sel_idx]:.2f}:")
        d1, d2 = st.columns([1, 1])
        with d1:
            st.dataframe(pretty_df(alloc, {"WeightPct": 2}), hide_index=True, width='stretch')
        with d2:
            a2 = alloc.sort_values("WeightPct")
            figb, axb = plt.subplots(figsize=(5, max(2.5, 0.3 * len(a2))))
            axb.barh(a2["Ticker"], a2["WeightPct"],
                     color=["#2ca02c" if x >= 0 else "#d62728" for x in a2["WeightPct"]])
            axb.axvline(0, color="black", lw=0.7); axb.set_xlabel("Weight (%)"); axb.grid(alpha=0.3, axis="x")
            st.pyplot(figb)


def render_view_impact(res):
    """The section that makes the model legible: what the views actually did."""
    usable = res["usable"]
    pi = res["pi_total"]; mu = res["mu_total"]
    w_mkt = res["w_mkt"]; w_bl = res["weights"]; w_mk = res.get("w_markowitz")

    tilt = float(np.sum(np.abs(w_bl - w_mkt))) / 2.0     # active share vs the market
    biggest = int(np.argmax(np.abs(mu - pi))) if len(mu) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active views", f"{res['n_views']}")
    c2.metric("Active share vs market", f"{tilt*100:,.1f}%",
              help="Half the sum of absolute weight differences from the market portfolio. "
                   "0% = you are holding the market; 100% = you share nothing with it.")
    # The ticker goes in the LABEL, not the delta slot: Streamlit draws a green
    # up-arrow beside any delta, which read as "VEDL went up" next to a -4.39%
    # revision. Misleading in exactly the place the user is checking the maths.
    c3.metric(f"Largest revision — {usable[biggest]}" if len(mu) else "Largest revision",
              f"{(mu[biggest]-pi[biggest])*100:+,.2f}%" if len(mu) else "—",
              help="The single biggest gap between the posterior and the equilibrium prior. "
                   "For a relative view the short leg often moves more than the long leg — "
                   "the adjustment lands wherever the covariance matrix says it should.")
    c4.metric("Risk aversion δ", f"{res['delta']:,.2f}",
              help=res.get("delta_note", ""))

    if "default" in str(res.get("delta_note", "")).lower():
        st.error(
            "**Your equilibrium is not anchored to this market.** δ could not be implied from the "
            "data — over this history window the cap-weighted market portfolio did not return more "
            "than the risk-free rate, so there was no risk premium to reverse-engineer, and the "
            "model fell back to the textbook constant 2.5. π is therefore *a* plausible "
            "equilibrium, not *this market's* equilibrium, and the central Black-Litterman claim "
            "(\"these are the returns that justify what the market is holding\") does not hold on "
            "this run. Try a longer horizon, or a market that earned a positive premium over the "
            "window.")

    if not res["has_views"]:
        st.info("**No views are active, so this is the pure equilibrium portfolio.** "
                "μ = π exactly, and the optimal weights are the market weights (up to whatever "
                "constraints and objective you chose). That is the correct Black-Litterman answer "
                "to 'I have no opinion' — and it is a useful baseline to tilt away from.")

    comp = pd.DataFrame({
        "Ticker": usable,
        "Equilibrium π %": pi * 100,
        "Posterior μ_BL %": mu * 100,
        "Revision %": (mu - pi) * 100,
        "Market wt %": w_mkt * 100,
        "BL wt %": w_bl * 100,
        "BL − Market %": (w_bl - w_mkt) * 100,
    })
    if w_mk is not None:
        comp["Markowitz wt %"] = w_mk * 100
    comp = comp.sort_values("BL − Market %", ascending=False)
    st.dataframe(pretty_df(comp, {c: 2 for c in comp.columns if c != "Ticker"}),
                 hide_index=True, width='stretch')

    if res["has_views"]:
        d = (w_bl - w_mkt) * 100
        up = np.argsort(-d)[:3]
        dn = np.argsort(d)[:3]
        fmt = lambda ix: ", ".join(f"{usable[i]} {d[i]:+,.1f}pp" for i in ix if abs(d[i]) > 0.05)
        overs, unders = fmt(up), fmt(dn)
        st.markdown(
            f"**In plain terms:** your views moved the portfolio **{tilt*100:,.1f}%** away from simply "
            f"holding the market. "
            + (f"Biggest overweights: {overs}. " if overs else "")
            + (f"Biggest underweights: {unders}." if unders else ""))
        if tilt < 0.02:
            st.warning(
                "**That tilt is tiny — your views are barely doing anything.** Usual causes, in order: "
                "the view is close to what the equilibrium already implies (check the *Surprise* column "
                "below — a view that restates the equilibrium changes nothing, however confident you "
                "are); confidence is set too low; or the max-weight cap is truncating the tilt before "
                "it reaches the portfolio.")

    if res["has_views"] and len(res["view_labels"]):
        st.markdown("**Your views, and how the model treated each one**")
        vt = pd.DataFrame({
            "View": res["view_labels"],
            "Confidence %": np.asarray(res["view_conf"]) * 100,
            "Surprise vs equilibrium %": np.asarray(res["view_surprise"]) * 100,
            "Implied ω": res["omega_diag"],
            "View uncertainty (σ) %": np.sqrt(np.clip(res["omega_diag"], 0, None)) * 100,
        })
        st.dataframe(pretty_df(vt, {"Confidence %": 0, "Surprise vs equilibrium %": 2,
                                    "Implied ω": 6, "View uncertainty (σ) %": 2}),
                     hide_index=True, width='stretch')
        st.caption("**Surprise** is how far your view sits from what the equilibrium already implies — "
                   "a view that merely restates the equilibrium changes nothing, however confident you "
                   "are. **ω** is the variance the model assigned to your view: smaller = the model "
                   "trusts it more and tilts further. Under Idzorek, ω is solved for so the tilt matches "
                   "your stated confidence.")

        # The trap that catches everyone once: a view can sound bullish and act
        # bearish. "A beats B by 10%" is a DOWNGRADE if the equilibrium already
        # expected 17%. The sign of the surprise, not the sign of your number,
        # decides which way the money moves.
        q_stated = np.asarray(res.get("view_q_stated", []), dtype=float)
        surp = np.asarray(res["view_surprise"], dtype=float)
        for i, lab in enumerate(res["view_labels"]):
            if i >= len(surp):
                break
            s_i = float(surp[i])
            q_i = float(q_stated[i]) if i < len(q_stated) else np.nan
            if np.isfinite(q_i) and q_i > 0 and s_i < -1e-6:
                st.warning(
                    f"**“{lab}” is a BEARISH view, despite the positive number.** The equilibrium "
                    f"already implied a spread of **{(q_i - s_i)*100:,.2f}%**, so asking for only "
                    f"{q_i*100:,.2f}% is a downgrade of **{s_i*100:,.2f}pp**. The model is "
                    f"correctly *reducing* that position. If you meant to be bullish, your number "
                    f"has to exceed what the market already expects — check the **Equilibrium π** "
                    f"column before choosing it.")
            elif np.isfinite(q_i) and q_i < 0 and s_i > 1e-6:
                st.warning(
                    f"**“{lab}” is a BULLISH view, despite the negative number** — the equilibrium "
                    f"implied an even worse spread ({(q_i - s_i)*100:,.2f}%), so your view is an "
                    f"upgrade of **{s_i*100:+,.2f}pp**.")
            elif abs(s_i) < 0.01:
                st.info(
                    f"**“{lab}” is doing almost nothing** — it sits only "
                    f"{s_i*100:+,.2f}pp from the equilibrium, so even at "
                    f"{res['view_conf'][i]*100:,.0f}% confidence it moves μ by about "
                    f"{res['view_conf'][i]*s_i*100:+,.2f}pp. You are telling the model something "
                    f"it already believed.")


def _render_horizon_effect(res):
    """What the chosen holding period did — and, when it did nothing, why."""
    vr = res.get("variance_ratio")
    h = int(res.get("hold_periods", 1) or 1)
    if vr is None or h <= 1:
        return
    vr = np.asarray(vr, dtype=float)
    tick = res["usable"]
    days = h
    st.markdown(f"#### What holding for **{days} trading days** changed")

    lower = int(np.sum(vr < 0.97)); higher = int(np.sum(vr > 1.03))
    st.caption(
        f"Scaling daily volatility by {days} assumes each day is independent of the last. "
        f"Measured over actual {days}-day holds, **{lower}** of these {len(vr)} names turned out "
        f"*less* volatile than that assumption implies and **{higher}** turned out *more*. "
        f"The model uses the measured figure, so a name that settles down over a long hold can "
        f"be held bigger, and one that trends away from you is held smaller.")

    order = np.argsort(vr)
    rows = [{"Ticker": tick[i], "Risk vs daily-scaled": f"{(vr[i]-1)*100:+.0f}%",
             "Reads as": ("settles down over the hold" if vr[i] < 0.97
                          else "drifts further over the hold" if vr[i] > 1.03
                          else "behaves as daily data implies")}
            for i in list(order[:3]) + list(order[-3:])]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    # The honest caveat: with no views and Max Sharpe this cannot move anything.
    active = float(np.sum(np.abs(res["weights"] - res["w_mkt"]))) / 2
    if not res.get("has_views") and res.get("objective") == "Max Sharpe":
        st.info(
            "**Your holding period cannot change this particular portfolio, and that is a "
            "property of Black-Litterman rather than a fault.** With no views, the posterior "
            "*is* the equilibrium prior, and the prior was built by reverse-optimising from the "
            "market's own weights — so optimising it forward hands the market portfolio back, "
            "whatever the covariance says. Rescaling risk for a longer hold rescales the prior "
            "by exactly the same amount and it cancels.\n\n"
            "To make the horizon bite, do one of two things: **add a view** (the tilt is "
            "proportional to Σ, so a long hold tilts further into anything that settles down), "
            "or switch **Optimisation goal** to *Min variance*, *Min CVaR* or *Max Sortino*, "
            "which judge risk directly rather than reconstructing the market. On simulated data "
            "those move the book by 40–65% between a one-day and a one-year hold.")
    elif active < 0.01:
        st.caption("This book is within 1% of the market portfolio, so there is little for the "
                   "holding period to act on.")


def _render_holding_periods(res, bt):
    """What ONE holding period looked like, over and over.

    The headline backtest answers "what if I ran this for eight years?". Most
    people are asking something narrower and more useful: "if I buy this today
    and hold it for a week, what happens?" The walk-forward loop has already
    run that experiment a few hundred times -- every segment between two
    rebalances is exactly one complete hold -- so this reports the spread of
    those outcomes instead of a single compounded path.
    """
    holds = bt.get("holds") or []
    if len(holds) < 3:
        return
    bc = res["base_currency"]; cap = res["total_capital"]
    label = bt.get("rebal_label", "period")
    unit = {"weekly": "week", "monthly": "month", "yearly": "year"}.get(label, "period")

    r = np.array([h["strat"] for h in holds], dtype=float)
    ri = np.array([h["index"] for h in holds], dtype=float)
    have_idx = np.isfinite(ri).sum() >= max(3, len(ri) // 2)

    st.markdown(f"#### If you bought this and held it for **one {unit}**")
    st.caption(
        f"The test window contains **{len(r)}** complete one-{unit} holds. Each one buys the "
        f"portfolio the model recommended using only data available at the time, holds it "
        f"untouched for one {unit}, and records what happened. This is the spread of those "
        f"{len(r)} results — not one path through eight years.")

    med = float(np.median(r)); win = float(np.mean(r > 0) * 100)
    p5, p95 = float(np.percentile(r, 5)), float(np.percentile(r, 95))
    h1, h2, h3, h4 = st.columns(4)
    h1.metric(f"Typical {unit}", f"{med*100:+,.2f}%",
              delta=f"{cap*med:+,.0f} {bc} on {cap:,.0f}", delta_color="off")
    h2.metric(f"{unit.capitalize()}s in profit", f"{win:,.0f}%",
              help=f"Share of the {len(r)} holds that ended above where they started.")
    h3.metric("Best / worst", f"{r.max()*100:+,.1f}% / {r.min()*100:+,.1f}%")
    h4.metric("Usual range (5–95%)", f"{p5*100:+,.1f}% to {p95*100:+,.1f}%",
              help="Nine out of ten holds landed inside this band. The other one did not.")

    st.markdown(
        f"**In plain terms.** Put **{cap:,.0f} {bc}** in, leave it for one {unit}, and the "
        f"middle outcome is **{cap*med:+,.0f} {bc}**. About **{win:,.0f}%** of the time you "
        f"end the {unit} up. The worst single {unit} in this window lost "
        f"**{cap*abs(r.min()):,.0f} {bc}** ({r.min()*100:,.1f}%), the best gained "
        f"**{cap*r.max():,.0f} {bc}**. One {unit} is a coin-flip-ish outcome; the compounded "
        f"eight-year figure above is what repeating it {len(r)} times produced on this "
        f"particular stretch of history.")

    if have_idx:
        beat = float(np.mean(r[np.isfinite(ri)] > ri[np.isfinite(ri)]) * 100)
        st.caption(
            f"Against **{bt.get('index_label', 'the index')}** over the very same days: this "
            f"portfolio finished ahead in **{beat:,.0f}%** of the {int(np.isfinite(ri).sum())} "
            f"holds. A number near 50% means the two are hard to tell apart over a single "
            f"{unit}, whatever the eight-year totals say.")

    if len(r) <= 20:
        st.caption(f"Only {len(r)} holds fit in the test window at this horizon, so read the "
                   f"spread as indicative rather than as a distribution. Shorter horizons give "
                   f"far more samples of the same question.")
        show = pd.DataFrame([{"From": h["start"].date(), "To": h["end"].date(),
                              f"Return %": h["strat"] * 100,
                              f"P&L ({bc})": h["strat"] * cap} for h in holds])
        st.dataframe(pretty_df(show, {f"Return %": 2, f"P&L ({bc})": 0}),
                     hide_index=True, width='stretch')
    else:
        fig, axh = plt.subplots(figsize=(9, 2.6))
        axh.hist(r * 100, bins=min(40, max(10, len(r) // 8)),
                 color="#1f77b4", alpha=0.85, edgecolor="none")
        axh.axvline(0, color="black", lw=1.0)
        axh.axvline(med * 100, color="#d62728", lw=1.4, ls="--",
                    label=f"median {med*100:+.2f}%")
        axh.set_xlabel(f"Return over one {unit} (%)")
        axh.set_ylabel(f"Number of {unit}s")
        axh.legend(fontsize=8)
        axh.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, width='stretch')
        plt.close(fig)


def _render_backtest(res, bt, bt_eq=None):
    bc = res["base_currency"]; total_capital = res["total_capital"]
    ret_freq = res["ret_freq"]; freq_per_year = res["freq_per_year"]
    s, b = bt["strat"], bt["bench"]
    idx = bt.get("index"); idx_label = bt.get("index_label", "Index")
    mkt = bt.get("market")
    test_start, test_end = bt["dates"][0], bt["dates"][-1]
    yrs = bt.get("test_years") or (bt["n_test"] / freq_per_year)
    strat_final = total_capital * float(s["equity"][-1])
    strat_pnl = strat_final - total_capital
    strat_ret_tot = strat_final / total_capital - 1.0 if total_capital else 0.0
    bench_final = total_capital * float(b["equity"][-1])
    bench_pnl = bench_final - total_capital
    idx_final = total_capital * float(idx["equity"][-1]) if idx else None
    mkt_final = total_capital * float(mkt["equity"][-1]) if mkt else None

    reb = bt.get("rebal_label", "periodic")
    n_reb = bt.get("n_rebalances", 1)
    turn = bt.get("avg_turnover", 0.0)
    fx_note = (" Returns are converted into your reporting currency including exchange-rate moves."
               if bt.get("fx_applied") else "")
    st.caption(
        f"Walk-forward test — after an initial {bt['n_train']} {ret_freq.lower()} periods "
        f"(up to {bt['split_date'].date()}), the equilibrium prior was re-derived and the portfolio "
        f"re-optimised **{reb}** using only data available at each point, held until the next "
        f"rebalance: {n_reb} rebalances across {bt['n_test']} out-of-sample periods (~{yrs:.1f} yrs), "
        f"{test_start.date()} to {test_end.date()}. Turnover averaged "
        f"{turn*100:,.0f}% of the book per rebalance.{fx_note}")

    _cbps = float(bt.get("cost_bps", 0.0) or 0.0)
    if _cbps > 0:
        st.caption(
            f"**Net of trading costs at {_cbps:,.0f} bps one-way** on every rebalance, "
            f"including the first purchase: about **{bt.get('cost_drag_ann', 0.0)*100:,.2f}pp "
            f"a year** of drag on {bt.get('ann_turnover', 0.0)*100:,.0f}% annual turnover. "
            f"Taxes and borrowing charges are not modelled.")
    else:
        st.caption(
            "**Gross of all costs.** No trading costs, taxes or borrowing charges are "
            "deducted (set a trading cost in Advanced settings to change that). Costs come off "
            "the top and grow with how often you trade — so read the higher-frequency horizons "
            f"as the most flattered, not the best. This run turned over "
            f"{bt.get('ann_turnover', 0.0)*100:,.0f}% of the book a year.")

    # ---- volatility-targeting overlay: did it earn its keep? ----
    if bt.get("vt_applied") and bt.get("unscaled") is not None:
        u = bt["unscaled"]
        st.markdown("##### Volatility-targeting overlay")
        d_sharpe = s["sharpe"] - u["sharpe"]
        d_dd = s["max_dd"] - u["max_dd"]

        # When the excess return is negative the Sharpe ratio stops being
        # comparable: you are dividing a negative numerator by the volatility,
        # so CUTTING risk makes the ratio look WORSE. Presenting a red
        # "-0.12 vs no overlay" there actively misleads, so we suppress the
        # comparison and say why.
        sharpe_meaningful = (s["ann_ret"] - res["rf"] > 0) and (u["ann_ret"] - res["rf"] > 0)

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Sharpe", f"{s['sharpe']:,.2f}",
                  delta=(f"{d_sharpe:+,.2f} vs no overlay" if sharpe_meaningful else None),
                  help=(None if sharpe_meaningful else
                        "Both portfolios returned less than the risk-free rate over this window, "
                        "so the Sharpe numerator is negative and the ratio is not comparable "
                        "between them — lowering volatility mechanically makes a negative Sharpe "
                        "look worse. Judge the overlay on drawdown and volatility instead."))
        v2.metric("Volatility", f"{s['ann_vol']*100:,.1f}%",
                  delta=f"{(s['ann_vol']-u['ann_vol'])*100:+,.1f}pp vs no overlay",
                  delta_color="inverse")
        v3.metric("Worst drawdown", f"{s['max_dd']*100:,.1f}%",
                  delta=f"{d_dd*100:+,.1f}pp vs no overlay")
        v4.metric("Average exposure", f"{bt['avg_scale']*100:,.0f}%",
                  help=f"Ranged {bt['min_scale']*100:,.0f}%–{bt['max_scale']*100:,.0f}%. "
                       f"The remainder sat in cash earning the risk-free rate.")
        tgt_desc = (f"Target = **{bt['vt_frac']*100:,.0f}% of the portfolio's own volatility**, "
                    f"recomputed at each rebalance from data available then "
                    f"(averaged {bt['vt_target']*100:,.1f}% over the test)"
                    if bt.get("vt_relative") else
                    f"Target {bt['vt_target']*100:,.0f}% annual volatility (fixed)")
        st.caption(
            f"{tgt_desc}, measured over a trailing {bt['vt_lookback']}-period window, capped at "
            f"{bt['vt_max_leverage']:,.1f}x exposure. Realised {s['ann_vol']*100:,.1f}% vs "
            f"{u['ann_vol']*100:,.1f}% without it. Return per year {s['ann_ret']*100:,.2f}% vs "
            f"{u['ann_ret']*100:,.2f}%.")
        if not sharpe_meaningful:
            st.warning(
                f"**Ignore the Sharpe comparison on this run.** Both versions returned less than "
                f"your {res['rf']*100:,.1f}% risk-free rate, which makes the Sharpe numerator "
                f"negative — and a negative Sharpe gets *worse* when you reduce volatility, "
                f"purely as arithmetic. ({u['ann_ret']*100:,.2f}% → {s['ann_ret']*100:,.2f}% "
                f"return; {u['ann_vol']*100:,.1f}% → {s['ann_vol']*100:,.1f}% volatility.) "
                f"The meaningful comparison here is drawdown: "
                f"{u['max_dd']*100:,.1f}% → {s['max_dd']*100:,.1f}%.")

        # The mirror image of "the overlay never engaged": a target far below the
        # portfolio's natural volatility leaves it permanently half-invested,
        # which reads as the model losing to its own prior when really the
        # overlay is parked in cash. Say so rather than letting the user infer it.
        # In relative mode the exposure SHOULD sit near the chosen fraction, so
        # only complain when it lands well below what was asked for.
        _low_bar = (0.70 * float(bt.get("vt_frac") or 1.0)) if bt.get("vt_relative") else 0.70
        if bt["avg_scale"] < _low_bar and bt.get("unscaled") is not None:
            nat = bt["unscaled"]["ann_vol"]
            st.warning(
                f"**The overlay dominated this run — average exposure {bt['avg_scale']*100:,.0f}%.** "
                f"Your {bt['vt_target']*100:,.0f}% target sits far below this portfolio's natural "
                f"{nat*100:,.1f}% volatility, so it sat roughly half in cash throughout and gave up "
                f"**{(bt['unscaled']['ann_ret']-s['ann_ret'])*100:,.1f}pp of return a year**. That is "
                f"the overlay doing exactly what you asked, not a model failure — but a 12% target "
                f"suits a diversified book, not a concentrated high-volatility one. For this "
                f"portfolio a target nearer **{nat*0.75*100:,.0f}%** would de-risk the worst periods "
                f"without permanently halving your exposure.")

        if bt["avg_scale"] > 0.97 and bt["vt_max_leverage"] <= 1.0:
            st.warning(
                f"**The overlay barely engaged** — average exposure {bt['avg_scale']*100:,.0f}%. "
                f"Your target of {bt['vt_target']*100:,.0f}% is at or above the portfolio's own "
                f"volatility ({u['ann_vol']*100:,.1f}%), and max exposure is capped at 1.0x, so "
                f"there was nothing for it to cut. Lower the target below "
                f"{u['ann_vol']*100:,.0f}% to make it bite.")
        elif d_dd > 0.005:
            # State the trade, both sides, and leave the judgement to the reader.
            # The overlay bought lower risk with lower return; whether that was
            # worth it is not something a backtest can settle.
            st.info(
                f"**The overlay traded return for risk on this run.** Drawdown "
                f"{u['max_dd']*100:,.1f}% → {s['max_dd']*100:,.1f}% "
                f"({d_dd*100:+,.1f}pp) and volatility {u['ann_vol']*100:,.1f}% → "
                f"{s['ann_vol']*100:,.1f}%, at a cost of "
                f"{(u['ann_ret']-s['ann_ret'])*100:,.2f}pp of return a year "
                f"({u['ann_ret']*100:,.2f}% → {s['ann_ret']*100:,.2f}%). "
                + ((f"Sharpe {u['sharpe']:,.2f} → {s['sharpe']:,.2f}."
                    if sharpe_meaningful else "")
                   ) +
                " Across simulated markets this overlay reduced drawdown in most runs and moved "
                "Sharpe in either direction about equally often, so treat the risk reduction as "
                "the expected effect and any Sharpe gain as this window's luck. One historical "
                "path either way.")
        else:
            st.info(
                f"**Little effect over this window** — drawdown {u['max_dd']*100:,.1f}% → "
                f"{s['max_dd']*100:,.1f}%, volatility {u['ann_vol']*100:,.1f}% → "
                f"{s['ann_vol']*100:,.1f}%. The overlay earns its keep in turbulent periods; a "
                f"calm test window gives it little to do. Report the result rather than tuning "
                f"the lookback until it looks better — that is how backtest overfitting starts.")

    if res.get("backtest_uses_views"):
        st.error(
            "**Your views are being applied throughout this backtest, which makes it in-sample.** "
            "You typed those views today, knowing how the period turned out. The model is being fed "
            "information it could not have had in 2019. Treat the result as an illustration of how "
            "views propagate, never as evidence the views work. The dashed grey line is the same "
            "strategy with views switched off — that one is honest, and the gap between them is "
            "exactly the size of the look-ahead.")

    beat_market = ""
    if idx is not None:
        diff = strat_final - idx_final
        beat_market = (f" Just buying the {idx_label} index would have become {idx_final:,.0f} {bc}, so "
                       f"the strategy {'beat' if diff >= 0 else 'lagged'} the market by "
                       f"{abs(diff):,.0f} {bc}.")
    # Most people are not asking "what if I ran this for eight years?". They are
    # asking "what if I buy it and hold it for one week?". Answer that FIRST,
    # before the compounded figure, because the compounded figure is the one
    # that gets misread as a single-period outcome.
    _render_horizon_effect(res)
    _render_holding_periods(res, bt)

    st.markdown("#### And if you kept repeating it")
    st.markdown(
        f"**In plain terms:** running this {reb}-rebalanced strategy, your {total_capital:,.0f} {bc} "
        f"would have become **{strat_final:,.0f} {bc}** — a {'profit' if strat_pnl >= 0 else 'loss'} of "
        f"**{strat_pnl:+,.0f} {bc} ({strat_ret_tot*100:+,.1f}%)** over ~{yrs:.1f} years.{beat_market}")
    st.caption(
        f"That is one historical path through one market, with {bt.get('n_rebalances', 0)} "
        f"rebalances. It is a description of what these rules would have done, not an estimate of "
        f"what they will do. The strategy-versus-its-own-prior line below is the more informative "
        f"comparison, because it isolates the model from whatever the market happened to deliver.")

    bcols = st.columns(4)
    bcols[0].metric(f"Test P&L ({bc})", f"{strat_pnl:+,.0f}", delta=f"{strat_ret_tot*100:+,.1f}% total")
    bcols[1].metric("Final value", f"{strat_final:,.0f}",
                    delta=(f"vs {idx_final:,.0f} {idx_label}" if idx else f"vs {bench_final:,.0f} eq-wt"))
    bcols[2].metric("Return per year", f"{s['ann_ret']*100:,.2f}%",
                    delta=(f"{(s['ann_ret']-idx['ann_ret'])*100:+,.2f}% vs {idx_label}" if idx
                           else f"{(s['ann_ret']-b['ann_ret'])*100:+,.2f}% vs eq-wt"))
    bcols[3].metric("Worst drop (drawdown)", f"{s['max_dd']*100:,.1f}%")

    fig4, ax4 = plt.subplots(figsize=(9, 4))
    label_main = ("Black-Litterman with your views" if res.get("backtest_uses_views")
                  else f"Black-Litterman equilibrium ({reb}-rebalanced)")
    if bt.get("vt_applied"):
        label_main += ", vol-targeted"
    ax4.plot(bt["dates"], total_capital * s["equity"], label=label_main, color="#1f77b4", lw=1.9)

    # If the overlay is on, the blue line is volatility-scaled while every
    # benchmark below is plain buy-and-hold. Comparing them directly reads as
    # "the model lost to its own prior" when most of the gap is just the
    # overlay de-risking. Plot the unscaled strategy too so the comparison is
    # like for like.
    if bt.get("vt_applied") and bt.get("unscaled") is not None:
        u_eq = bt["unscaled"]["equity"]
        n_u = min(len(bt["dates"]), len(u_eq))
        ax4.plot(bt["dates"][:n_u], total_capital * u_eq[:n_u],
                 label="Same model, overlay OFF (like-for-like)",
                 color="#1f77b4", lw=1.2, ls="--", alpha=0.8)

    if bt_eq is not None and res.get("backtest_uses_views"):
        n_eq = min(len(bt["dates"]), len(bt_eq["strat"]["equity"]))
        ax4.plot(bt["dates"][:n_eq], total_capital * bt_eq["strat"]["equity"][:n_eq],
                 label="Same model, views OFF (the honest line)", color="#444", lw=1.4, ls="--")
    if mkt is not None:
        ax4.plot(bt["dates"], total_capital * mkt["equity"],
                 label="Cap-weighted market portfolio (the prior)", color="#ff7f0e", lw=1.4)
    if idx is not None:
        ax4.plot(bt["dates"], total_capital * idx["equity"],
                 label=f"{idx_label} (buy & hold the market)", color="#d62728", lw=1.5)
    ax4.plot(bt["dates"], total_capital * b["equity"], label="Equal-weight of same stocks",
             color="#888", lw=1.0, ls=":")
    ax4.axhline(total_capital, color="black", lw=0.6)
    # Matplotlib's default 5% margin pads roughly five months of empty space
    # onto each end, so the axis reads 2018-2027 for data that stops in August
    # 2026 -- and the trailing gap looks like the strategy went flat. Pin the
    # limits to the actual first and last observation.
    if len(bt["dates"]) > 1:
        ax4.set_xlim(bt["dates"][0], bt["dates"][-1])
    ax4.set_ylabel(f"Portfolio value ({bc})")
    ax4.set_title(f"Out-of-sample value of {total_capital:,.0f} {bc} "
                  f"({test_start.date()} → {test_end.date()})")
    ax4.legend(loc="best", fontsize=8); ax4.grid(alpha=0.3)
    st.pyplot(fig4)

    # ---- why is the strategy above or below its own prior? ----
    # This is the question every user asks, and answering it badly (or not at
    # all) is how people conclude the model is broken when it is behaving
    # exactly as the theory says it should.
    if mkt is not None:
        base = bt["unscaled"] if (bt.get("vt_applied") and bt.get("unscaled")) else s
        gap_total = s["ann_ret"] - mkt["ann_ret"]
        gap_ex_overlay = base["ann_ret"] - mkt["ann_ret"]
        overlay_cost = base["ann_ret"] - s["ann_ret"] if bt.get("vt_applied") else 0.0
        with st.expander("Why is the strategy above/below the cap-weighted prior?", expanded=True):
            st.markdown(
                f"- **Strategy:** {s['ann_ret']*100:,.2f}% a year\n"
                + (f"- **Same strategy, overlay off:** {base['ann_ret']*100:,.2f}% a year "
                   f"— the overlay cost **{overlay_cost*100:,.2f}pp** of return in exchange for "
                   f"lower volatility\n" if bt.get("vt_applied") else "")
                + f"- **Cap-weighted prior (buy & hold):** {mkt['ann_ret']*100:,.2f}% a year\n"
                + f"- **Equal-weight of the same names:** {b['ann_ret']*100:,.2f}% a year\n"
                + (f"- **{idx_label}:** {idx['ann_ret']*100:,.2f}% a year\n" if idx else ""))
            st.markdown(
                f"Gap to the prior: **{gap_total*100:+,.2f}pp** a year"
                + (f", of which **{-overlay_cost*100:+,.2f}pp** is the volatility overlay, "
                   f"leaving **{gap_ex_overlay*100:+,.2f}pp** from the model itself."
                   if bt.get("vt_applied") else "."))
            if res.get("sys_engine") and bt_eq is not None:
                # The one comparison in this whole app that answers "do views
                # add value?" with evidence rather than assertion.
                r_rule = s["ann_ret"]; r_eq = bt_eq["strat"]["ann_ret"]
                sh_rule = s["sharpe"]; sh_eq = bt_eq["strat"]["sharpe"]
                d_r = r_rule - r_eq
                st.markdown("##### Did the rule add anything?")
                q1, q2, q3 = st.columns(3)
                q1.metric(f"Rule: {res['sys_engine']}", f"{r_rule*100:,.2f}%/yr",
                          delta=f"{d_r*100:+,.2f}pp vs no views")
                q2.metric("Pure equilibrium", f"{r_eq*100:,.2f}%/yr")
                q3.metric("Sharpe", f"{sh_rule:,.2f}",
                          delta=f"{sh_rule - sh_eq:+,.2f} vs no views")
                _render_inference(bt)
                if abs(d_r) < 0.002:
                    st.info(
                        f"**The rule made essentially no difference** ({d_r*100:+,.2f}pp a year). "
                        f"Usually because it earned little confidence, so the posterior barely "
                        f"moved off the prior — which is the model correctly declining to act on "
                        f"a weak signal.")
                elif d_r > 0:
                    st.info(
                        f"**The rule was ahead of pure equilibrium by {d_r*100:,.2f}pp a year on "
                        f"this path.** The construction is sound — views rebuilt from scratch at "
                        f"every rebalance from data available then, confidence recalibrated from "
                        f"the hit rate at that point, no look-ahead — but that establishes the "
                        f"test was fair, not that the edge is real. Read the interval above, not "
                        f"the point estimate; the confidence the rule earned "
                        f"({res.get('sys_conf', 0)*100:,.0f}%) is a second measure of how weak "
                        f"the signal was. Treat it as a hypothesis worth re-testing on another "
                        f"market, not as a result.")
                else:
                    st.warning(
                        f"**The rule cost {abs(d_r)*100:,.2f}pp a year versus doing nothing.** That "
                        f"is a legitimate finding and worth reporting as-is. Most systematic signals "
                        f"fail this test; it is why the equilibrium anchor does the heavy lifting in "
                        f"Black-Litterman. Do not cycle through rules until one looks good — that "
                        f"search is itself how backtest overfitting happens.")

            if not res.get("backtest_uses_views") and not res.get("sys_engine"):
                st.info(
                    "**Note what this backtest actually tests.** Your views are not applied here "
                    "(switching them on would be look-ahead bias), so the strategy is the *pure "
                    "equilibrium* portfolio. With no views the posterior equals the prior, so in "
                    "theory this should reproduce the market portfolio — and it very nearly does. "
                    "It cannot beat it. Any remaining gap is the cost of the constraints you "
                    "imposed (the weight cap forcing an underweight in the largest names, "
                    "long-only, a Max-Sharpe objective rather than cap weights) plus trading "
                    "costs. **Black-Litterman's value is a defensible starting point and a "
                    "principled way to tilt from it — not mechanical outperformance.** "
                    "Outperformance would have to come from views that carry real information.")
            if idx is not None and b["ann_ret"] > idx["ann_ret"] and mkt["ann_ret"] > idx["ann_ret"]:
                uni_edge = mkt["ann_ret"] - idx["ann_ret"]
                st.warning(
                    f"**Where the outperformance actually came from.** The cap-weighted, "
                    f"equal-weighted *and* optimised versions of these names all beat "
                    f"{idx_label}. When every weighting scheme wins, the edge is in which stocks "
                    f"were selected — here, the liquidity screen — not in how they were weighted. "
                    f"Credit the universe, not the optimiser.")
                if uni_edge > 0.08:
                    st.error(
                        f"**{uni_edge*100:,.1f} percentage points a year of that 'edge' is almost "
                        f"certainly look-ahead bias.** The universe is the most liquid names in "
                        f"the index *today*, and today's most liquid names are disproportionately "
                        f"the ones that won over the test period — you could not have picked this "
                        f"list in {bt['split_date'].year}. A gap this large between the universe "
                        f"({mkt['ann_ret']*100:,.1f}%) and the index it came from "
                        f"({idx['ann_ret']*100:,.1f}%) is a measure of that bias, not of skill. "
                        f"Treat the absolute returns on this run as uninformative; only the "
                        f"comparison between the strategy and its own prior means anything here.")

    metrics = ["Final value", "Profit / loss", "Total return", "Return per year",
               "Volatility per year", "Sharpe", "Worst drawdown"]

    def _col(perf, final):
        return [f"{final:,.0f}", f"{final-total_capital:+,.0f}",
                f"{(final/total_capital-1)*100:+,.1f}%", f"{perf['ann_ret']*100:,.2f}%",
                f"{perf['ann_vol']*100:,.2f}%", f"{perf['sharpe']:,.2f}",
                f"{perf['max_dd']*100:,.1f}%"]

    cols = {"Metric": metrics, f"Black-Litterman ({bc})": _col(s, strat_final)}
    if bt_eq is not None and res.get("backtest_uses_views"):
        eq_final = total_capital * float(bt_eq["strat"]["equity"][-1])
        cols[f"BL, views off ({bc})"] = _col(bt_eq["strat"], eq_final)
    if mkt is not None:
        cols[f"Market portfolio ({bc})"] = _col(mkt, mkt_final)
    if idx is not None:
        cols[f"{idx_label} (index)"] = _col(idx, idx_final)
    cols[f"Equal-weight ({bc})"] = _col(b, bench_final)
    st.dataframe(pd.DataFrame(cols), hide_index=True, width='stretch')

    negatives = [nm for nm, p in [("the strategy", s), ("the market portfolio", mkt),
                                  (idx_label, idx), ("equal-weight", b)]
                 if p is not None and (p["ann_ret"] - res["rf"]) < 0]
    if negatives:
        st.caption(
            "⚠️ **Sharpe rows are not comparable across columns here.** "
            + ", ".join(negatives).capitalize() +
            f" returned less than the {res['rf']*100:,.1f}% risk-free rate, which makes the Sharpe "
            f"numerator negative — and a negative Sharpe improves when volatility *rises*. Compare "
            f"the return and drawdown rows instead, and note that a window where nothing beat cash "
            f"tells you about the market, not about the model.")


def _render_inference(bt):
    """Interval and p-values for the rule's gap to pure equilibrium."""
    inf = bt.get("inference") or {}
    bs, lw, be = inf.get("bootstrap"), inf.get("sharpe_test"), inf.get("breakeven_bps")
    if not bs:
        return
    r = bs["ann_return"]
    i1, i2, i3 = st.columns(3)
    i1.metric("Return gap, 95% interval",
              f"{r['lo']*100:+,.2f} to {r['hi']*100:+,.2f}pp",
              help=f"Stationary block bootstrap of the paired per-period returns "
                   f"({bs['n_boot']:,} resamples, mean block {bs['mean_block']} periods). "
                   f"Resampling both runs on the same dates keeps their correlation, which "
                   f"is what makes a small gap measurable at all. Arithmetic mean, so it can "
                   f"differ slightly from the compounded figure above.")
    i2.metric("p-value, return gap", f"{r['p']:.3f}")
    if lw:
        i3.metric("p-value, Sharpe gap", f"{lw['p']:.3f}",
                  help=f"Ledoit & Wolf (2008) test: delta method with a Newey-West HAC "
                       f"covariance (lag {lw['lag']}), valid under fat tails and serial "
                       f"dependence. Sharpe gap {lw['diff']:+.2f}, standard error "
                       f"{lw['se']:.2f}.")
    inside = r["lo"] <= 0.0 <= r["hi"]
    if inside:
        st.caption(
            f"**Statistically indistinguishable from zero.** The interval runs from "
            f"{r['lo']*100:+,.2f} to {r['hi']*100:+,.2f}pp a year and includes no "
            f"difference at all. This is the precise version of \"the rule added nothing\": "
            f"not proof that it has no edge, but no evidence that it does.")
    else:
        st.caption(
            f"**The gap is significant at 5% on this path** (p = {r['p']:.3f}). Before "
            f"believing it: this app offers four rules, and trying all four makes a "
            f"false positive somewhere roughly four times as likely — a Bonferroni bar would be "
            f"p < 0.0125. Re-test on another market before acting on it.")
    if be is not None:
        st.caption(
            f"**Break-even trading cost ≈ {be:,.0f} bps one-way.** Above that, the extra "
            f"turnover the rule generates costs more than its gross edge over equilibrium "
            f"(first-order estimate). Compare it with what your broker actually charges.")


def _render_rolling(res):
    """The same test from many start dates."""
    df = res.get("rolling")
    sm = summarise_rolling(df)
    if sm is None:
        return
    st.markdown("##### The same test from many start dates")
    r1, r2, r3 = st.columns(3)
    r1.metric(f"Median gap vs {sm['vs']}", f"{sm['median_gap']*100:+,.2f}pp/yr",
              help=f"Across {sm['n_windows']} three-year windows, starting every six months.")
    r2.metric("10th to 90th percentile",
              f"{sm['p10']*100:+,.2f} to {sm['p90']*100:+,.2f}pp")
    r3.metric("Windows ahead / behind",
              f"{sm['share_positive']*100:,.0f}% / {sm['share_negative']*100:,.0f}%",
              help="By more than 0.01pp a year; the rest are ties.")
    show = df.assign(**{
        "Window": [f"{a.date()} → {b.date()}" for a, b in zip(df["start"], df["end"])],
        "Strategy %/yr": df["strategy"] * 100, "Reference %/yr": df["reference"] * 100,
        "Gap pp/yr": df["gap"] * 100})[["Window", "Strategy %/yr", "Reference %/yr", "Gap pp/yr"]]
    st.bar_chart(show.set_index("Window")["Gap pp/yr"], height=220)
    with st.expander("Every window"):
        st.dataframe(pretty_df(show, {"Strategy %/yr": 2, "Reference %/yr": 2, "Gap pp/yr": 2}),
                     hide_index=True, width='stretch')
    st.caption(
        "The windows overlap, so they are not independent samples and the band above is "
        "not a confidence interval. What it shows is how much the headline result depends "
        "on the start date. A gap that changes sign from one window to the next was never "
        "a stable property of the strategy.")


def render_results(res):
    bc = res["base_currency"]
    weights = res["weights"]

    # ---- headline: the answer, before any of the machinery ----
    st.subheader("3) Your portfolio")
    n_held = int(np.sum(np.abs(weights) > 1e-4))
    tilt = float(np.sum(np.abs(weights - res["w_mkt"]))) / 2.0
    top_ix = np.argsort(-np.abs(weights))[:3]
    top_txt = ", ".join(f"**{res['usable'][i]}** {weights[i]*100:,.1f}%" for i in top_ix)
    st.markdown(
        f"**{n_held} positions**, largest: {top_txt}. Model-implied **{res['port_ret']*100:,.1f}%** a year "
        f"with **{res['port_vol']*100:,.1f}%** volatility (Sharpe **{res['port_sharpe']:,.2f}**), "
        f"sitting **{tilt*100:,.1f}%** away from just holding the market."
        + ("  \n⚠️ Expected return is *below* your risk-free rate of "
           f"{res['rf']*100:,.1f}% — this portfolio is not being paid for its risk."
           if res["port_ret"] < res["rf"] else ""))

    _exec_tab = "🔁 What to trade" if res.get("has_holdings") else "🛒 What to buy"
    tabs = st.tabs(["📊 Holdings", "🎯 What your views did", "📈 Efficient frontier",
                    "⏱️ Backtest", _exec_tab, "💾 Compare runs"])

    with tabs[0]:
        _render_holdings(res, bc, weights)
    with tabs[1]:
        st.caption("Black-Litterman starts from the market portfolio and moves away from it only as "
                   "far as your views — and your confidence in them — justify. Here is how far.")
        render_view_impact(res)
    with tabs[2]:
        st.caption("Each point is a possible portfolio. The blue curve is the efficient frontier — "
                   "the highest return for each level of risk, computed from the **posterior** "
                   "returns. Hover for return & volatility; **click a point** for its allocation.")
        render_frontier(res)
    with tabs[3]:
        if not res["do_backtest"]:
            st.info("Backtest is switched off in Advanced settings.")
        elif res["bt"] is None:
            st.info("Not enough history for a reliable backtest — widen the date range, lower the "
                    "training fraction, or use a higher return frequency.")
        else:
            _render_backtest(res, res["bt"], res.get("bt_eq"))
            _render_rolling(res)
    with tabs[4]:
        _render_execution(res, bc)
    with tabs[5]:
        _render_saved_runs(res)

    st.info("**Note on accuracy:** this optimises over each index's *current* constituents, so "
            "delisted or dropped companies never appear — a survivorship bias that tends to overstate "
            "historical returns. The equilibrium prior is only as good as the market-cap weights behind "
            "it, and views are opinions, not forecasts. The out-of-sample backtest is the best "
            "available reality check. Educational tool, not investment advice.")


def _render_holdings(res, bc, weights):
    if res.get("universe_note"):
        st.caption(res["universe_note"])
    st.caption(res["settings_caption"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model-implied annual return", f"{res['port_ret']*100:,.2f}%",
              help="NOT a forecast. This is what the posterior expected returns imply for this "
                   "weighting — and the posterior is built from today's market weights and a "
                   "historical covariance matrix, neither of which predicts anything. Treat it as "
                   "a description of the inputs, not of the future. The out-of-sample backtest is "
                   "the only number here with any predictive claim attached, and it is one path.")
    m2.metric("Annual volatility", f"{res['port_vol']*100:,.2f}%")
    m3.metric("Sharpe ratio", f"{res['port_sharpe']:,.2f}")
    m4.metric("Names held", f"{int(np.sum(np.abs(weights) > 1e-4))}")
    st.caption(res["exposure_caption"])
    st.dataframe(pretty_df(res["weights_df"], {"Equilibrium π %": 2, "Posterior μ_BL %": 2,
                                               "AnnualVolatility": 4, "Market wt %": 2,
                                               "Weight": 4, "WeightPct": 2,
                                               "Weight ± (resampled)": 2}),
                 hide_index=True, width='stretch')
    if res.get("has_stability"):
        st.caption("‘Weight ± (resampled)’ shows how much each weight varied across the resampled "
                   "simulations — smaller = more stable/robust; larger = more sensitive to noise.")


def _render_trade_list(res, bc):
    """Buy / sell / hold against what the user actually owns."""
    df = res["exec_df"]
    probs = res.get("held_problems") or []
    if probs:
        st.warning("**Some lines were not used:**\n\n" + "\n".join(f"- {p}" for p in probs))

    act = df["Action"].fillna("")
    n_hold = int((act == "HOLD").sum())
    n_buy = int((act == "BUY").sum())
    n_sell = int((act == "SELL").sum())
    tv = df[f"TradeValue ({bc})"]
    gross_traded = float(tv.abs().sum(skipna=True))
    cap = res["total_capital"]
    turn_pct = (gross_traded / cap * 100) if cap > 0 else 0.0

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Hold — no action", f"{n_hold}",
              help="Already within one lot of the target, so there is no order to place.")
    t2.metric("Buy", f"{n_buy}")
    t3.metric("Sell", f"{n_sell}")
    t4.metric("Value to trade", f"{gross_traded:,.0f} {bc}",
              delta=f"{turn_pct:,.1f}% of the book", delta_color="off")

    st.caption(
        f"Your holdings are worth **{res['held_value']:,.0f} {bc}**"
        + (f", plus **{res['new_cash']:,.0f} {bc}** of new cash you asked to deploy"
           if res.get("new_cash", 0) > 0 else "")
        + f" — a book of **{cap:,.0f} {bc}**. Rows marked HOLD are already within one "
          f"tradeable lot of their target, so there is nothing to place. The trade list does "
          f"not net costs off: judge for yourself whether a small adjustment is worth your "
          f"brokerage.")


    show = df[df["Action"].isin(["BUY", "SELL"])] if (n_buy + n_sell) else df
    if not (n_buy + n_sell):
        st.success("**Nothing to do.** Every position is already within one lot of its target.")
    st.dataframe(pretty_df(show[res["detail_cols"]], res["exec_fmt"]),
                 hide_index=True, width='stretch')
    if n_hold and (n_buy + n_sell):
        with st.expander(f"Show the {n_hold} position(s) being held unchanged"):
            st.dataframe(pretty_df(df[df["Action"] == "HOLD"][res["detail_cols"]], res["exec_fmt"]),
                         hide_index=True, width='stretch')

    st.download_button("Download trade list (CSV)",
                       data=df.to_csv(index=False).encode("utf-8"),
                       file_name="black_litterman_trades.csv", mime="text/csv")
    st.caption("Share counts round to whole lots. This is a description of what the model's "
               "rules imply given the prices fetched just now — not advice, and not an "
               "estimate of what any of it will earn.")


def _render_execution(res, bc):
    if res.get("has_holdings"):
        _render_trade_list(res, bc)
        return
    # Implementability first — a beautiful set of weights you cannot actually
    # buy is not a portfolio, and the rounding error is invisible in every
    # other tab.
    cap = res["total_capital"]
    lp = res.get("leftover_pct", 0.0)
    e1, e2, e3 = st.columns(3)
    e1.metric("Cash actually invested", f"{(1-lp)*100:,.1f}%",
              help="After rounding every position down to whole shares (or board lots).")
    e2.metric("Left uninvested", f"{res['leftover_total']:,.0f} {bc}",
              delta=f"{lp*100:,.1f}% of capital", delta_color="inverse")
    e3.metric("Positions that round to zero", f"{res.get('n_unaffordable', 0)}",
              delta=f"of {res.get('n_holdings_chosen', 0)} intended", delta_color="off")

    # Severity should match the damage. 0.2% idle with one dropped name is a
    # footnote; 18% idle is a problem that outweighs anything the optimiser
    # decided. Firing a red error at 99.8% invested trains people to ignore it.
    n_zero = int(res.get("n_unaffordable", 0))
    n_int = int(res.get("n_holdings_chosen", 0))
    if res.get("capital_warn"):
        mc = res.get("min_capital", np.nan)
        an = res.get("afford_n", np.nan)
        detail = ""
        if np.isfinite(mc) and n_zero > 0:
            detail += (f" Fitting **every** intended position at these weights would need about "
                       f"**{mc:,.0f} {bc}**; you have {cap:,.0f} {bc}.")
        if np.isfinite(an) and int(an) < n_int:
            detail += (f" About **{int(an)} names** fit comfortably at your capital — reduce "
                       f"*Number of stocks to hold* in Advanced settings.")

        if lp > 0.05:
            st.error(
                f"**{lp*100:,.1f}% of your capital would sit in cash.** That drag is real and "
                f"permanent, and none of the returns shown elsewhere account for it."
                f"{detail}")
        elif n_zero > 0:
            st.warning(
                f"**{n_zero} of {n_int} intended position(s) round down to zero shares** — their "
                f"weight is too small to buy a single share at your capital. The rest of the book "
                f"is fine ({(1-lp)*100:,.1f}% invested), so this is a small tracking error against "
                f"the target weights rather than a cash problem.{detail}")
        else:
            st.info(f"{lp*100:,.1f}% of capital stays in cash after rounding — negligible.")

    st.caption("Share counts use the latest available market price (fetched live where possible, "
               "otherwise the last close in your history window). A negative share count means sell short.")
    st.dataframe(pretty_df(res["exec_df"][res["detail_cols"]], res["exec_fmt"]),
                 hide_index=True, width='stretch')
    lots = res["exec_df"]["LotSize"] if "LotSize" in res["exec_df"] else None
    lot_note = ""
    if lots is not None and (lots > 1).any():
        lot_note = (" This market trades in board lots (e.g. 100 shares), so share counts are rounded "
                    "down to whole lots.")
    st.caption(f"Unallocated cash after rounding to whole lots: {res['leftover_total']:,.2f} {bc}.{lot_note}")

    st.download_button("Download execution table (CSV)",
                       data=res["exec_df"].to_csv(index=False).encode("utf-8"),
                       file_name="black_litterman_execution.csv", mime="text/csv")
    if res.get("fx_warn"):
        st.warning("Some tickers could not be converted into the reporting currency; their shares "
                   "are blank above.")


def _render_saved_runs(res):
    csave, cclear = st.columns([1, 1])
    with csave:
        if st.button("💾 Save this run for comparison"):
            saved = st.session_state.get("saved_runs", [])
            saved.append(res["run_summary"])
            st.session_state["saved_runs"] = saved[-10:]
    with cclear:
        if st.session_state.get("saved_runs") and st.button("Clear saved runs"):
            st.session_state.pop("saved_runs", None)
    saved = st.session_state.get("saved_runs", [])
    if saved:
        st.dataframe(pd.DataFrame(saved), hide_index=True, width='stretch')
    else:
        st.caption("Save a few runs — the same views at 25%, 50% and 75% confidence is the most "
                   "instructive comparison — to see how much the answer depends on your inputs.")


# =========================================================
# EQUILIBRIUM HELPERS FOR THE UI
# =========================================================
def _equilibrium_from_universe(uni, rf, delta_mode, delta_manual):
    """Recompute delta and pi from the cached universe plus the live rf setting."""
    Sigma = uni["cov"].values
    w_mkt = uni["w_mkt"]
    if delta_mode == "Set it myself":
        delta = float(delta_manual)
        note = "set manually"
    else:
        delta, note = implied_risk_aversion(w_mkt, Sigma, uni["mkt_hist_return"] - rf)
    pi = equilibrium_returns(delta, Sigma, w_mkt)
    return delta, note, pi


def _momentum_views(uni, n_pairs=2):
    """Suggest a few relative views from 12-1 month momentum, purely as a
    worked example of how the views table is meant to be filled in."""
    prices = uni["prices"]
    if len(prices) < 30:
        return None
    look = min(len(prices) - 1, int(uni["freq_per_year"]))
    skip = max(1, int(uni["freq_per_year"] / 12))
    px = prices.iloc[-(look + 1):]
    mom = (px.iloc[-1 - skip] / px.iloc[0] - 1.0).sort_values(ascending=False)
    tickers = list(mom.index)
    if len(tickers) < 2 * n_pairs:
        return None
    rows = []
    for i in range(n_pairs):
        win, lose = tickers[i], tickers[-(i + 1)]
        spread = float(mom.iloc[i] - mom.iloc[-(i + 1)])
        rows.append({"Use": True, "Type": "Relative", "Asset": win, "Versus": lose,
                     "Return % p.a.": round(min(spread, 0.40) * 100 * 0.25, 2),
                     "Confidence %": DEFAULT_VIEW_CONFIDENCE})
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="Black-Litterman Optimizer", layout="wide")
    st.title("Black-Litterman Portfolio Optimizer — start from the market, tilt with your views")
    st.caption(
        "Markowitz asks you to forecast every return, then amplifies whatever you got wrong. "
        "Black-Litterman turns the problem around: it starts from the portfolio the market is already "
        "holding, works out what returns would make that portfolio optimal, and then moves away from "
        "it only as far as your own views — and your confidence in them — justify. Pick a market, "
        "load it, type your views, and see exactly how much they move the money."
    )

    st.subheader("1) Set up your portfolio")

    mode = st.radio("Mode", ["Simple (recommended)", "Advanced"], index=0, horizontal=True,
                    help="Simple asks only the essentials and uses well-tested defaults for the rest. "
                         "Advanced exposes τ, δ, the Ω method, estimation, costs and validation.")

    colA, colB = st.columns(2)
    with colA:
        market_name = st.selectbox("Which market to trade", list(MARKETS.keys()), index=0)
    with colB:
        stance = st.selectbox("Strategy", ["Long only", "Short only", "Long-short (both)"], index=0,
                              help="Long only = buy names you expect to rise. Short only = bet against "
                                   "names you expect to fall. Long-short = do both at once.")

    # Currency defaults to the market's own, so no FX conversion happens unless
    # the user deliberately asks for it.
    _ccy_opts = ["USD", "EUR", "GBP", "JPY", "INR", "CAD", "AUD", "HKD"]
    _native = MARKETS[market_name]["currency"]
    col1, col2, col3 = st.columns(3)
    with col1:
        base_currency = st.selectbox(
            "Your currency", _ccy_opts,
            index=_ccy_opts.index(_native) if _native in _ccy_opts else 0,
            help="Defaults to the market's own currency. Change it only if you actually hold "
                 "a different one — the backtest will then include exchange-rate moves.")
    with col2:
        total_capital = st.number_input(f"Amount to invest ({base_currency})", min_value=0.0,
                                        value=100000.0, step=1000.0)
    with col3:
        horizon = st.selectbox("How long will you hold?", list(HORIZON_CFG.keys()), index=1,
                               help="Sets how much history is used and how returns are measured. "
                                    "Longer horizons use more history and steadier (weekly) returns "
                                    "for more reliable estimates.")

    # ---- what you already own -------------------------------------------
    # Without this the app treats every visit as a fresh start from cash, so
    # it hands you a shopping list instead of a trade list -- and it cannot
    # charge for turnover, because as far as it knows every position is a
    # first purchase. The result is that the screen tells you to chase weight
    # changes the backtest itself would have declined to trade on.
    with st.expander("🧾 I already own some of these (optional)", expanded=False):
        st.markdown(
            "Paste what you hold and the **What to buy** tab becomes a **trade list** — "
            "buy, sell, or hold, per position — instead of a from-scratch shopping list. "
            "Positions within one tradeable lot of their target will say *hold*.")
        holdings_text = st.text_area(
            "One position per line: ticker, then number of shares",
            value="", height=140, placeholder="RELIANCE.NS 40\nBHARTIARTL.NS, 55\nHDFCBANK.NS: 48",
            help="The exchange suffix is optional — 'RELIANCE' finds RELIANCE.NS. Lines starting "
                 "with # are ignored. Anything not recognised is reported, never silently dropped.")
        new_cash = st.number_input(
            f"New cash you want to add ({base_currency})", min_value=0.0, value=0.0, step=1000.0,
            help="Left at zero, the trade list rebalances what you already hold and nothing more. "
                 "Set it and the model deploys that cash too, spread across the targets.")

    hcfg = HORIZON_CFG[horizon]
    today = pd.Timestamp.today().normalize()
    auto_start = today - pd.DateOffset(years=hcfg["years"] + 1)
    auto_end = today

    _rf_default = market_default(market_name, "rf")
    rf_annual = st.number_input(
        f"Risk-free rate (annual) — recommended {_rf_default*100:,.1f}% for {_native}",
        value=float(_rf_default), step=0.005, format="%.3f", key=f"rf_{market_name}",
        help="The local short-dated government yield. This is a MEASURING STICK, not an "
             "allocation — none of your money goes into it, and your stock weights still sum "
             "to 100%. It matters because the equilibrium prior π is an EXCESS return, so this "
             "sets the level everything is measured from, and because absolute views are "
             "converted to excess space as (your view − rf). Leaving it at zero silently makes "
             "every view several points more bullish than you intended.")

    # Recommended defaults — used as-is in Simple mode, editable in Advanced.
    objective = "Max Sharpe"; cov_method = "Ledoit-Wolf shrinkage"
    # Return frequency is a question about the ASSETS, not about how often you
    # trade. Daily data gives five times more observations, which is what Sigma
    # wants — but when assets trade in different sessions (SPY at 21:00 IST,
    # EFA at 16:30 CET, EEM spanning both), a day's "simultaneous" returns are
    # not simultaneous at all, and daily correlations come out badly understated.
    # Measured on simulated panels, comparing daily-annualised against
    # weekly-annualised Sigma:
    #
    #   synchronous, no autocorrelation        pi differs by 0.38pp   <- daily fine
    #   synchronous, AR(1)=0.10                pi differs by 1.30pp
    #   NON-synchronous sessions               pi differs by 1.61pp
    #   both (global multi-asset ETFs)         pi differs by 2.36pp   <- daily wrong
    #
    # So: daily for a single-exchange universe (all NSE, all NYSE), weekly when
    # the universe spans time zones. Every horizon within a universe still sees
    # identical data — the frequency varies by what is being held, not by how
    # often it is traded.
    _cross_tz = bool(MARKETS[market_name].get("cross_tz"))
    ret_freq = "Weekly" if _cross_tz else hcfg["freq"]
    # hcfg["rebal"] counts DAILY periods, so it must be rescaled if the returns
    # are weekly — otherwise "rebalance every 5" would mean five weeks.
    _reb_periods = max(1, int(round(hcfg["rebal"] * FREQ_PER_YEAR[ret_freq]
                                    / FREQ_PER_YEAR[hcfg["freq"]])))
    use_log = True; cvar_alpha = 0.95
    resample_n = 0
    do_backtest = True; train_frac = DEFAULT_TRAIN_FRAC
    # A 12% single-name cap is sensible for 25 stocks and wrong for 13 asset
    # classes, where the market portfolio itself is legitimately concentrated
    # (US equity really is a huge share of global assets). Capping at 12% there
    # forces near-equal-weight and makes every market-relative number
    # meaningless — active share of 60%+ against a prior you were never allowed
    # to hold.
    _is_asset_class = MARKETS[market_name].get("kind") == "asset-class"
    max_weight_pct = DEFAULT_ASSET_CLASS_CAP_PCT if _is_asset_class else DEFAULT_MAX_WEIGHT_PCT
    use_vol_target = True
    vol_target_mode = "Relative to this portfolio (recommended)"
    vol_target_pct = DEFAULT_VOL_TARGET_PCT
    vol_target_frac_pct = DEFAULT_VOL_TARGET_FRAC_PCT
    vol_max_lev = 1.0
    vol_lookback = None
    gross_pct = 200 if stance == "Long-short (both)" else None
    n_universe = len(MARKETS[market_name]["tickers"])
    n_holdings = min(AUTO_HOLDINGS, n_universe)
    start_date, end_date = auto_start, auto_end
    # Black-Litterman defaults
    tau = 0.05
    delta_mode = "Imply it from the market"
    delta_manual = 2.5
    omega_method = "Idzorek confidence"
    use_posterior_cov = False
    backtest_uses_views = False
    cost_bps = 0.0
    do_rolling = False

    if mode == "Advanced":
        with st.expander("Black-Litterman settings — τ, δ, Ω", expanded=True):
            b1, b2, b3 = st.columns(3)
            with b1:
                tau = st.slider(
                    "τ (tau) — prior uncertainty", 0.01, 1.00, 0.05, 0.01,
                    help="How uncertain the equilibrium prior is, as a fraction of Σ. "
                         "IMPORTANT: under BOTH Ω methods here, τ has NO effect on the posterior "
                         "expected returns — Ω is itself proportional to τ, so it cancels out of "
                         "the posterior entirely. Verify it yourself: change τ and watch the "
                         "posterior μ_BL column not move. τ only bites when 'Use posterior "
                         "covariance Σ+M' is ticked, where it acts as a diversification dial. "
                         "To change how far your views move the portfolio, use the Confidence "
                         "column in the views table, not this.")
                st.caption("τ does not change μ_BL — see the tooltip. Use **Confidence** in the "
                           "views table to control view strength.")
            with b2:
                delta_mode = st.selectbox("Risk aversion δ",
                                          ["Imply it from the market", "Set it myself"], index=0,
                                          help="δ = (market excess return) / (market variance). "
                                               "Implying it is principled; setting it is what most "
                                               "practitioners do. 2.5–3 is the standard range.")
                delta_manual = st.number_input("δ if setting manually", 0.5, 10.0, 2.5, 0.1)
            with b3:
                omega_method = st.selectbox("Ω (view uncertainty)",
                                            ["Idzorek confidence", "He-Litterman proportional"], index=0,
                                            help="Idzorek: you give a 0-100% confidence per view and Ω "
                                                 "is solved for. He-Litterman: Ω = diag(P τΣ P'), no "
                                                 "confidence input needed — the Confidence column is "
                                                 "then ignored.")
                use_posterior_cov = st.checkbox("Use posterior covariance Σ+M", value=False,
                                                help="Theoretically correct: acknowledges that the "
                                                     "posterior mean is itself uncertain. Produces "
                                                     "slightly more diversified portfolios. Off by "
                                                     "default to match most textbook examples.")

        with st.expander("Advanced settings — universe, dates, estimation, objective, costs, validation",
                         expanded=False):
            u1, u2 = st.columns(2)
            with u1:
                n_holdings = st.slider("Number of stocks to hold", 5, min(60, n_universe),
                                       min(AUTO_HOLDINGS, n_universe), 1,
                                       help="How many of the most liquid names to hold. These become "
                                            "the assets you can express views on.")
            with u2:
                custom_dates = st.checkbox("Use custom history dates", value=False,
                                           help="Off = dates set automatically from your horizon.")
            if custom_dates:
                d1, d2 = st.columns(2)
                with d1:
                    start_date = st.date_input("History start", auto_start)
                with d2:
                    end_date = st.date_input("History end", auto_end)
            g1, g2 = st.columns(2)
            with g1:
                max_weight_pct = st.slider(
                    "Max weight per holding (%)", 5, 100, max_weight_pct, 1,
                    help="Concentration limit; 100% = no cap. The recommended 12% keeps roughly "
                         "8+ real positions. Without a cap, Max Sharpe will happily put 40% into "
                         "one name. Note a tight cap can also truncate a view's tilt.")
            with g2:
                if stance == "Long-short (both)":
                    gross_pct = st.slider("Max gross exposure (%)", 100, 300, 200, 10,
                                          help="Total long + short size per unit of net capital.")
            a1, a2, a3 = st.columns(3)
            with a1:
                objective = st.selectbox("Optimisation goal",
                                         ["Max Sharpe", "Min variance", "Max Sortino", "Min CVaR"], index=0,
                                         help="Applied to the POSTERIOR returns. Sharpe = best "
                                              "risk-adjusted return; Min variance = lowest risk "
                                              "(and ignores your views almost entirely, by design).")
                cvar_alpha = st.slider("CVaR confidence", 0.90, 0.99, 0.95, 0.01)
            with a2:
                cov_method = st.selectbox("Risk (covariance) estimate",
                                          ["Ledoit-Wolf shrinkage", "Sample", "Exponentially weighted"],
                                          index=0,
                                          help="Σ does double duty in Black-Litterman: it sets the risk "
                                               "AND, through π = δΣw, the entire prior. Shrinkage matters "
                                               "more here than in plain Markowitz.")
            with a3:
                ret_freq = st.selectbox("Return frequency", ["Daily", "Weekly", "Monthly"],
                                        index=["Daily", "Weekly", "Monthly"].index(ret_freq),
                                        help="Defaults to daily, or weekly when the universe "
                                             "spans time zones (daily correlations across "
                                             "sessions are understated).")
                use_log = st.checkbox("Use log returns", value=True)
            c1, c3 = st.columns(2)
            with c1:
                cost_bps = st.number_input(
                    "Trading cost (bps, one-way)", 0.0, 200.0, 0.0, 1.0,
                    help="Charged on every rebalance as this many basis points of the value "
                         "traded, including the first purchase. 0 = gross, as before. The "
                         "backtest also reports the BREAK-EVEN cost — the level at which the "
                         "strategy's edge over its own prior is used up — so you can judge "
                         "it against your own broker without the app guessing your fees. "
                         "Taxes and borrowing charges are not modelled.")
            with c3:
                resample_n = st.slider("Resampling draws (Michaud)", 0, 100, 0, 5,
                                       help="0 = off. Higher = sturdier weights but slower.")
            st.markdown("**Volatility targeting overlay**")
            v1, v2, v3 = st.columns(3)
            with v1:
                use_vol_target = st.checkbox(
                    "Scale exposure to a volatility target", value=True,
                    help="Cut exposure when markets get turbulent, restore it when they calm "
                         "down. Volatility is predictable from its own recent past; returns are "
                         "not — so sizing against risk works even though timing returns does "
                         "not. What this reliably buys you is SMALLER DRAWDOWNS and a steadier "
                         "risk level. It does not reliably raise the Sharpe ratio: on simulated "
                         "data it improved Sharpe in only about half of runs, while cutting the "
                         "worst drawdown in every single one.")
            with v2:
                vol_target_mode = st.radio(
                    "Target type", ["Relative to this portfolio (recommended)", "Absolute %"],
                    index=0,
                    help="RELATIVE sets the target as a fraction of the portfolio's own "
                         "volatility, recomputed at every rebalance from data available then. "
                         "It adapts to whatever you are holding — a 12% absolute target is "
                         "sensible for a diversified book and parks a 24%-volatility tech "
                         "portfolio permanently at half exposure. ABSOLUTE fixes the number "
                         "yourself, which is right when you have a specific risk mandate.")
                if vol_target_mode.startswith("Relative"):
                    vol_target_frac_pct = st.slider(
                        "Target as % of the portfolio's own volatility", 40, 100,
                        DEFAULT_VOL_TARGET_FRAC_PCT, 5,
                        help="75% means: run at three-quarters of your normal risk, cutting "
                             "exposure further only when markets get turbulent. 100% means "
                             "de-risk only in genuinely abnormal periods.")
                else:
                    vol_target_pct = st.slider(
                        "Target annual volatility (%)", 4.0, 40.0, DEFAULT_VOL_TARGET_PCT, 0.5,
                        help="The risk level the overlay steers toward. Set it well below the "
                             "portfolio's natural volatility and it will sit in cash most of "
                             "the time.")
            with v3:
                vol_max_lev = st.slider(
                    "Max exposure (x)", 1.0, 2.0, 1.0, 0.1,
                    help="1.0x = the overlay can ONLY move money into cash, never borrow. "
                         "Above 1.0x it will lever up in calm markets, which raises returns "
                         "and also raises the damage when calm ends abruptly.")
                _lb_default = max(10, int(FREQ_PER_YEAR[ret_freq] / 4))   # ~3 months
                vol_lookback = st.slider(
                    "Volatility lookback (periods)", 5, 120, min(_lb_default, 120), 1,
                    help="Trailing window used to measure realised volatility, defaulted to "
                         "about three months. Shorter reacts faster but trades more and, on "
                         "simulated data, does slightly worse than a longer window.")
            if vol_max_lev > 1.0:
                st.caption("⚠️ Above 1.0x the overlay borrows. Leverage magnifies losses as "
                           "readily as gains, and volatility targeting has historically been "
                           "at its worst in sudden crashes that follow calm periods.")

            bb1, bb2, bb3 = st.columns(3)
            with bb1:
                do_backtest = st.checkbox("Run out-of-sample backtest", value=True)
            with bb2:
                train_frac = st.slider(
                    "Backtest training fraction", 0.30, 0.90, DEFAULT_TRAIN_FRAC, 0.05,
                    help="Share of history used to train before out-of-sample testing starts. "
                         "Lower = a longer, more meaningful test window. 0.45 typically gives "
                         "~5 years of out-of-sample instead of ~2.")
            with bb3:
                backtest_uses_views = st.checkbox("Apply my views throughout the backtest", value=False,
                                                  help="OFF (default) = the backtest re-derives the "
                                                       "equilibrium prior at each rebalance and uses no "
                                                       "views. That is genuinely out-of-sample. ON = your "
                                                       "views, typed today, are applied to 2019 — which "
                                                       "is look-ahead bias and cannot be evidence of "
                                                       "anything. Useful only to see how views propagate.")
            do_rolling = st.checkbox(
                "Repeat the backtest from many start dates (slower)", value=False,
                help="Re-runs the walk-forward test over 3-year windows starting every 6 "
                     "months. One backtest is one path from one start date; this shows how "
                     "much the answer depends on when you happened to begin.")


    settings = Settings(
        market=market_name, horizon=horizon, stance=stance, capital=float(total_capital),
        base_currency=base_currency, rf=float(rf_annual), n_holdings=n_holdings,
        start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), ret_freq=ret_freq,
        use_log=use_log, cov_method=cov_method, objective=objective, cvar_alpha=cvar_alpha,
        resample_n=resample_n, max_weight_pct=max_weight_pct, gross_pct=gross_pct, tau=tau,
        delta_mode=delta_mode, delta_manual=delta_manual, omega_method=omega_method,
        use_posterior_cov=use_posterior_cov, use_vol_target=use_vol_target,
        vol_target_relative=vol_target_mode.startswith("Relative"),
        vol_target_pct=vol_target_pct, vol_target_frac_pct=vol_target_frac_pct,
        vol_max_lev=vol_max_lev, vol_lookback=vol_lookback, train_frac=train_frac,
        cost_bps=float(cost_bps), auto_trim=(mode != "Advanced"))
    # Derived once, from the FINAL return frequency. Computing this before the
    # Advanced panel meant a switch to weekly returns turned "every 21 periods"
    # into every 21 weeks.
    _reb_periods = settings.reb_periods

    top_n = n_holdings
    native_ccy = MARKETS[market_name]["currency"]

    st.info(
        f"**Here's the plan.** Take the **{top_n} most liquid** "f"{'asset classes' if _is_asset_class else 'stocks'} in {market_name}, look at how "
        f"much of the market each one is, and work backwards to the returns that would justify those "
        f"sizes — that's the equilibrium, and it's your starting portfolio. Then you tell it where you "
        f"disagree, and it moves only as far as your confidence warrants. "
        f"Built from ~{hcfg['years']} years of {ret_freq.lower()} history, priced in {base_currency}.")

    # All three horizons now see identical data, so none is "the good one".
    # State the actual trade-off neutrally instead of steering.
    _reb_per_year = FREQ_PER_YEAR[ret_freq] / _reb_periods
    st.caption(
        f"**\"How long will you hold?\" sets how long you keep the portfolio before "
        f"adjusting it — not how long you invest for.** This setting holds each portfolio for "
        f"about **{252/_reb_per_year:,.0f} trading days** before re-checking it. All three "
        f"choices read the same {hcfg['years']}-year daily history, so they are directly "
        f"comparable; they differ only in that holding length. The backtest then answers two "
        f"separate questions: what **one** such hold looked like, and what happened when the "
        f"whole thing was repeated across the test window. Neither horizon is more correct."
    )

    if mode != "Advanced":
        st.caption(
            f"**Recommended settings applied automatically for {native_ccy}:** risk-free "
            f"{float(rf_annual)*100:,.1f}% · max {max_weight_pct}% per "
            f"stock · δ implied from the market · Ω Idzorek · τ {tau:.2f} · view confidence "
            f"{DEFAULT_VIEW_CONFIDENCE:,.0f}% · backtest trains on {train_frac:.0%} of history. "
            f"Switch to **Advanced** to change any of them.")

    # ---------------- STAGE A: load the universe and its equilibrium ----------------
    if st.button("① Load universe & equilibrium", type="primary"):
        for k in ("uni", "res", "frontier_chart"):
            st.session_state.pop(k, None)
        try:
            with st.spinner("Downloading market data and market capitalisations "
                            "(the caps are what the prior is built from)..."):
                _uni = load_universe(settings)
        except PipelineError as e:
            st.error(str(e))
            st.stop()
        for _w in _uni["warnings"]:
            st.warning(_w)
        st.session_state["uni"] = _uni

    uni = st.session_state.get("uni")

    # ---------------- STAGE B: equilibrium + views ----------------
    if uni is not None:
        usable = uni["usable"]
        n_assets = uni["n_assets"]
        sig = uni["signature"]
        delta, delta_note, pi = _equilibrium_from_universe(uni, float(rf_annual),
                                                           delta_mode, delta_manual)

        st.subheader("2) The equilibrium, and your views")
        st.caption(uni["universe_note"])

        # Is there enough data to estimate a covariance matrix of this size?
        # Sigma has n(n+1)/2 free parameters; with T observations the sample
        # estimate is rank-deficient once T < n, and unreliable well before
        # that. Ledoit-Wolf keeps it invertible but cannot invent information.
        T_obs = int(uni["returns"].shape[0])
        ratio = T_obs / max(n_assets, 1)
        if ratio < 4:
            st.error(
                f"**Not enough history to estimate risk for {n_assets} assets.** You have "
                f"{T_obs} {uni['ret_freq'].lower()} observations — {ratio:,.1f} per asset, against "
                f"{n_assets*(n_assets+1)//2:,} free parameters in Σ. Ledoit-Wolf shrinkage will "
                f"keep the matrix invertible, but most of what you see will be the shrinkage "
                f"target rather than the data. Since π = δΣw, an unreliable Σ means an unreliable "
                f"prior — not just noisy risk numbers. Use a longer horizon or hold fewer names.")
        elif ratio < 10:
            st.warning(
                f"**Thin history for this many assets** — {T_obs} observations for {n_assets} "
                f"assets ({ratio:,.1f} per asset). Workable with shrinkage, but treat Σ, and "
                f"therefore π, as approximate. 10+ observations per asset is a reasonable bar.")
        e1, e2, e3 = st.columns(3)
        e1.metric("Risk aversion δ", f"{delta:,.2f}", help=delta_note)
        e2.metric("Market portfolio return", f"{(uni['mkt_hist_return'])*100:,.2f}%",
                  help="Realised annualised return of the cap-weighted portfolio of these names "
                       "over the history window. Only used to imply δ.")
        e3.metric("Prior weights from", w_shorten(uni["w_mkt_note"]), help=uni["w_mkt_note"])

        if "traded value" in uni["w_mkt_note"]:
            _wm = uni["w_mkt"]; _top = int(np.argmax(_wm))
            st.error(
                f"**Your prior is built from trading volume, not size — treat it with suspicion.** "
                f"Sizes were unavailable for these symbols, so the app fell back to traded value, "
                f"and volume is a poor proxy for how big an asset class is. SPY turns over roughly "
                f"eighty times more dollars per day than AGG, yet the US bond market is larger than "
                f"the US equity market. Here that has put **{usable[_top]} at "
                f"{_wm[_top]*100:,.1f}%** of the 'market'. π = δΣw is only an equilibrium if w is "
                f"genuinely the market portfolio, so the central claim of the model does not hold "
                f"on this run. Yahoo usually returns fund net assets on a retry — rebuild the "
                f"universe, and if it persists, read the results as a constrained optimisation "
                f"rather than a Black-Litterman equilibrium.")

        with st.expander("What the market is implying right now (the prior π)", expanded=False):
            eq_df = pd.DataFrame({
                "Ticker": usable,
                "Market weight %": uni["w_mkt"] * 100,
                "Equilibrium return π % (total)": (pi + float(rf_annual)) * 100,
                "Annual volatility %": uni["sigma"] * 100,
                "Historical mean %": uni["hist_annual"] * 100,
            }).sort_values("Market weight %", ascending=False)
            st.dataframe(pretty_df(eq_df, {c: 2 for c in eq_df.columns if c != "Ticker"}),
                         hide_index=True, width='stretch')
            st.caption("π is **not** a forecast and it is **not** the historical mean — it is the set of "
                       "returns that would make today's market portfolio the optimal one. Compare the "
                       "last two columns: the historical means are wild, π is smooth and ranks assets "
                       "by their contribution to market risk. That smoothness is the whole point.")

        # ---------------- where do the views come from? ----------------
        st.markdown("**Where should the views come from?**")
        view_source = st.radio(
            "Views source", ["Systematic rule (recommended)", "My own views", "None — pure equilibrium"],
            index=0, horizontal=True, label_visibility="collapsed",
            help="A SYSTEMATIC RULE can be tested: it is a function of past prices, so the "
                 "backtest can rebuild it at every rebalance and measure whether it added "
                 "anything. YOUR OWN VIEWS cannot be tested — you formed them today, so "
                 "applying them to past trades is look-ahead bias. NONE gives the pure "
                 "equilibrium, which is the correct answer to 'I have no opinion'.")

        sys_engine = None
        sys_n_pairs = 2
        sys_conf = 0.0
        sys_conf_note = ""
        sys_labels = []

        if view_source.startswith("Systematic"):
            s1, s2 = st.columns([2, 1])
            with s1:
                engine_name = st.selectbox("Rule", [k for k in VIEW_ENGINES if VIEW_ENGINES[k]],
                                           index=0)
                sys_engine = VIEW_ENGINES[engine_name]
            with s2:
                sys_n_pairs = st.slider("Views to generate", 1, 4, 2, 1,
                                        help="Each view pairs the best-ranked asset against the "
                                             "worst. More views means more tilt and more ways to "
                                             "be wrong.")
            st.caption(ENGINE_NOTES.get(sys_engine, ""))

            with st.spinner("Measuring how often this rule has been right..."):
                hr, nchecks = engine_hit_rate(uni["prices"], sys_engine, uni["freq_per_year"],
                                              holding_periods=_reb_periods, n_pairs=sys_n_pairs)
                sys_conf, sys_conf_note = confidence_from_hit_rate(hr, nchecks)
                Ps, Qs, cs, sys_labels = systematic_views(
                    uni["prices"], usable, sys_engine, uni["freq_per_year"],
                    n_pairs=sys_n_pairs, conf=sys_conf)

            h1, h2, h3 = st.columns(3)
            h1.metric("Historical hit rate", f"{hr*100:,.0f}%" if hr is not None else "—",
                      help="How often this rule got the direction right, walking forward through "
                           "your history window using only data available at each point.")
            _hp = hit_rate_pvalue(hr, nchecks)
            h2.metric("Independent checks", f"{nchecks}",
                      delta=(f"p = {_hp:.2f} vs chance" if _hp is not None else None),
                      delta_color="off",
                      help="Distinct dates the rule was tested on, spaced at least one holding "
                           "period apart so the outcomes do not overlap. Several pairs are "
                           "scored on each date, but they share that date's market move, so "
                           "they count once. The p-value is a one-sided test of 'better than "
                           "a coin flip'.")
            h3.metric("Confidence assigned", f"{sys_conf*100:,.0f}%",
                      help="confidence = 2 x (hit rate - 50%), shrunk for sample size. A rule that "
                           "is right half the time carries no information and gets zero.")

            if sys_conf <= 0.001:
                st.warning(
                    f"**This rule earns zero confidence, so it will change nothing.** {sys_conf_note}. "
                    f"That is the model working correctly, not a failure — a signal that is right "
                    f"half the time should not move your portfolio. Try another rule, or accept the "
                    f"pure equilibrium as the answer.")
            else:
                st.success(f"**{sys_conf_note}.** The views below were generated from prices alone "
                           f"and will be rebuilt at every rebalance in the backtest, so the result "
                           f"is a genuine out-of-sample test of the rule.")

            if sys_labels:
                st.dataframe(pd.DataFrame({"Generated view": sys_labels,
                                           "Confidence %": [sys_conf * 100] * len(sys_labels)}),
                             hide_index=True, width='stretch')

        elif view_source.startswith("None"):
            st.info("**Pure equilibrium.** The posterior equals the prior, so you get the market "
                    "portfolio subject to your constraints. This is the most defensible output the "
                    "model produces — it rests on no opinion of yours at all.")

        manual_active = view_source.startswith("My own")
        if manual_active:
            st.markdown("**Your views** — tick *Use* to activate a row. Leave the table empty and you get "
                        "the pure equilibrium portfolio, which is the correct answer to 'I have no opinion'.")
            st.caption("**Absolute**: *Asset* will return X% per year (a total return — the app converts it "
                       "to excess internally). **Relative**: *Asset* will outperform *Versus* by X% per year; "
                       "*Return %* is the spread, not either leg.")

            with st.expander("What does Confidence mean? (read this once)", expanded=False):
                st.markdown(
                    "**Confidence is not a statistical confidence interval.** It is not a 95%-significance "
                    "thing, and it says nothing about probability. It is a *blending weight* between two "
                    "opinions: the market's and yours.\n\n"
                    "Under Idzorek, the posterior obeys a rule you can do in your head:\n\n"
                    "> **shift in expected return = confidence × (your view − what the equilibrium already implied)**\n\n"
                    "So if the equilibrium says a stock earns 11% and you say 16%, your view is a **5pp "
                    "surprise**. Then:\n\n"
                    "- **0%** — ignore me, keep the market's number (11%).\n"
                    "- **25%** — nudge it to 12.25%.\n"
                    "- **50%** — split the difference, 13.5%.\n"
                    "- **75%** — mostly believe me, 14.75%. *(recommended default)*\n"
                    "- **100%** — the market is simply wrong, use 16%.\n\n"
                    "Two consequences worth internalising. First, **a view that agrees with the equilibrium "
                    "does nothing at any confidence** — if the surprise is zero, so is the shift. Check the "
                    "*Surprise* column after building. Second, **confidence is not free**: at 75% a wrong "
                    "view loses you money three times faster than at 25%. It is a statement about how much "
                    "you are willing to be wrong by, not a dial that makes returns bigger.")
            if mode != "Advanced":
                st.caption(f"Confidence defaults to **{DEFAULT_VIEW_CONFIDENCE:,.0f}%** — the recommended "
                           f"starting point. Edit the column to change it per view.")

            # st.data_editor holds its own state against its key, so replacing the
            # underlying DataFrame is not enough to refresh it — the key itself has
            # to change. Hence the version counter bumped by the buttons below.
            vkey = f"views_df_{sig}"
            vver_key = f"views_ver_{sig}"
            if vkey not in st.session_state:
                st.session_state[vkey] = blank_views(usable, 3)
            st.session_state.setdefault(vver_key, 0)

            sug1, sug2, _sp = st.columns([1, 1, 3])
            with sug1:
                if st.button("📈 Suggest momentum views",
                             help="Fills the table with a couple of relative views built from 12-1 month "
                                  "momentum, at a deliberately modest confidence. This is a worked example "
                                  "of how to fill the table in, not a recommendation."):
                    mv = _momentum_views(uni)
                    if mv is not None:
                        st.session_state[vkey] = mv
                        st.session_state[vver_key] += 1
                        st.rerun()
                    else:
                        st.warning("Not enough price history to build momentum views.")
            with sug2:
                if st.button("🧹 Clear views"):
                    st.session_state[vkey] = blank_views(usable, 3)
                    st.session_state[vver_key] += 1
                    st.rerun()

            edited = st.data_editor(
                st.session_state[vkey],
                key=f"views_editor_{sig}_{st.session_state[vver_key]}", num_rows="dynamic",
                hide_index=True, width='stretch',
                column_config={
                    "Use": st.column_config.CheckboxColumn("Use", help="Activate this view", default=False),
                    "Type": st.column_config.SelectboxColumn("Type", options=VIEW_TYPES, required=True),
                    "Asset": st.column_config.SelectboxColumn("Asset", options=usable, required=True),
                    "Versus": st.column_config.SelectboxColumn(
                        "Versus", options=usable,
                        help="Only used for Relative views — the asset you expect to underperform"),
                    "Return % p.a.": st.column_config.NumberColumn(
                        "Return % p.a.", format="%.2f", step=0.5,
                        help="Absolute: the total annual return. Relative: the annual outperformance."),
                    "Confidence %": st.column_config.NumberColumn(
                        "Confidence %", min_value=0.0, max_value=100.0, step=5.0, format="%.0f"),
                })
            st.session_state[vkey] = edited

            if omega_method == "He-Litterman proportional":
                st.caption("Ω method is He-Litterman, so the **Confidence column is ignored** — each view's "
                           "uncertainty is taken straight from the prior.")

        else:
            # Rule-driven or pure-equilibrium: keep whatever the user typed
            # in session state, but do not render a ticked 75%-confidence
            # table while something else is actually driving the portfolio.
            vkey = f"views_df_{sig}"
            edited = st.session_state.get(vkey, blank_views(usable, 3))
            if bool(np.asarray(edited.get("Use", pd.Series(dtype=bool))).sum()):
                st.caption("You have manual views saved. They are **not** being used — "
                           "switch the source above to *My own views* to activate them.")

        # ---------------- BUILD ----------------
        if st.button("② Build portfolio", type="primary"):
            st.session_state.pop("frontier_chart", None)
            rf = float(rf_annual)
            cov = uni["cov"]; returns = uni["returns"]; prices = uni["prices"]
            freq_per_year = uni["freq_per_year"]
            w_mkt = uni["w_mkt"]

            # ---- the holding period reaches the model ----
            # Scaling daily variance by the number of days assumes returns are
            # independent day to day. They are not. An asset that mean-reverts
            # is genuinely less risky over a year than its daily volatility
            # implies and should be held in larger size for a long hold; a
            # trending one is riskier and should be held in smaller size. This
            # rescales each asset's variance by its measured behaviour over a
            # hold of exactly this length, which is what makes a one-week
            # portfolio differ from a one-year one at all.
            _vr = variance_ratio(returns.values, _reb_periods) if _reb_periods > 1 \
                else np.ones(returns.shape[1])
            Sigma = apply_variance_ratio(cov.values, _vr) if _reb_periods > 1 else cov.values
            # Loss-based objectives judge the hold, not the day.
            _R_obj = horizon_returns(returns.values, _reb_periods) if _reb_periods > 1 \
                else returns.values
            _R_ppy = freq_per_year / float(max(_reb_periods, 1))

            if sys_engine is not None:
                # Rule-driven: P, Q and the confidence all come from price history.
                # Nothing here is a judgement call the user could get wrong.
                P, Q, conf, view_labels = systematic_views(
                    uni["prices"], usable, sys_engine, uni["freq_per_year"],
                    n_pairs=sys_n_pairs, conf=sys_conf)
                view_errors = []
            elif view_source.startswith("None"):
                P, Q, conf = np.zeros((0, len(usable))), np.zeros(0), np.zeros(0)
                view_labels, view_errors = [], []
            else:
                P, Q, conf, view_labels, view_errors = build_pq(edited, usable, rf)
            for e in view_errors:
                st.warning(e)

            bl = black_litterman(Sigma, w_mkt, rf, tau, delta, P, Q, conf,
                                 omega_method=omega_method,
                                 use_posterior_cov=use_posterior_cov)
            mu = pd.Series(bl["mu_total"], index=usable)
            Sigma_used = bl["Sigma_used"]
            sigma_vec = np.sqrt(np.diag(Sigma_used))

            max_weight = None if max_weight_pct >= 100 else max_weight_pct / 100.0
            gross_limit = (gross_pct / 100.0) if gross_pct is not None else None

            settings_caption = (
                f"Objective: {objective} · Ω: {omega_method} · τ = {tau:.2f} · δ = {delta:,.2f} "
                f"({delta_note}) · Σ: {cov_method}"
                f"{' + posterior M' if use_posterior_cov else ''} · {ret_freq} "
                f"{'log' if use_log else 'simple'} returns · rf {rf*100:,.2f}% · "
                f"{len(view_labels)} active view(s)"
                f"{f' from rule ' + str(sys_engine) + f' @ {sys_conf*100:,.0f}% earned confidence' if sys_engine else ''} · "
                f"{'resampled x' + str(resample_n) if resample_n else 'single fit'}.")

            # ---- prices and FX, gathered BEFORE the optimisation ----
            # These used to be fetched further down, only for the execution
            # table. The optimiser needs them now: valuing what you already
            # own is what turns "here is a portfolio" into "here is the trade".
            # Allocating position-by-position is what strands cash -- each name
            # floors independently and loses part of a lot -- so the arrays are
            # gathered once here and allocated across the whole book later.
            execution_date = prices.index[-1]
            last_prices = prices.loc[execution_date]
            with st.spinner("Fetching current prices..."):
                cur_map = get_current_prices(usable)
            meta_all, ccy_all, fx_all, ok_all = [], [], [], []
            lots_all, px_native, px_report = [], [], []
            for i, ticker in enumerate(usable):
                meta = get_ticker_meta(ticker)
                tk_ccy = meta["currency"] or infer_currency_from_suffix(ticker) or ""
                fx_rate, fx_source, converted_ok = get_fx_rate(tk_ccy, base_currency)
                cur_native = cur_map.get(ticker, np.nan)
                if not (pd.notna(cur_native) and cur_native > 0):
                    cur_native = float(last_prices[ticker]) if ticker in last_prices.index else np.nan
                usable_px = (pd.notna(fx_rate) and fx_rate > 0
                             and pd.notna(cur_native) and cur_native > 0)
                meta_all.append(meta); ccy_all.append(tk_ccy)
                fx_all.append(fx_rate); ok_all.append(usable_px)
                lots_all.append(lot_size_for(ticker))
                px_native.append(cur_native)
                px_report.append(cur_native * fx_rate if usable_px else np.nan)

            # ---- what you already hold ----
            held_shares, held_problems = parse_holdings(holdings_text, usable)
            w_prev_live, held_value, n_held_priced = holdings_to_weights(
                held_shares, usable, px_report)
            has_holdings = held_value > 0
            # Rebalancing an existing book is a different question from building
            # one: the money in play is what you already have, plus whatever new
            # cash you said you would add. The 'Amount to invest' box only
            # governs the from-scratch case.
            invest_capital = (held_value + float(new_cash)) if has_holdings else total_capital
            # Charge the live optimiser at the same rate the backtest used, so
            # the portfolio on screen is the one the backtest measured.
            try:
                if resample_n and resample_n > 0:
                    stack = resample_stack(objective, mu.values, Sigma_used, rf, stance, max_weight,
                                           gross_limit, cvar_alpha, freq_per_year, resample_n,
                                           delta=delta)
                    weights = stack.mean(axis=0)
                    wstd = stack.std(axis=0)
                else:
                    weights = optimize_portfolio(objective, mu.values, Sigma_used, _R_obj, rf,
                                                 stance, max_weight, gross_limit, cvar_alpha,
                                                 freq_per_year, delta=delta, R_ppy=_R_ppy)
                    wstd = None
            except Exception as e:
                st.error(f"Optimisation failed: {e}")
                st.stop()

            # plain Markowitz (James-Stein) weights, for the side-by-side comparison
            w_markowitz = None
            try:
                mu_mk = estimate_mu(returns, "James-Stein shrinkage", freq_per_year, rf,
                                    cov_annual=Sigma, market_caps=uni["caps"])
                # Same horizon treatment, so the side-by-side comparison stays
                # like for like.
                w_markowitz = optimize_portfolio(objective, mu_mk.values, Sigma, _R_obj, rf,
                                                 stance, max_weight, gross_limit, cvar_alpha,
                                                 freq_per_year, R_ppy=_R_ppy)
            except Exception:
                w_markowitz = None

            port_ret, port_vol, port_sharpe = portfolio_stats(weights, mu.values, Sigma_used, rf)
            mkt_ret_bl, mkt_vol_bl, _ = portfolio_stats(w_mkt, mu.values, Sigma_used, rf)
            gross = float(np.sum(np.abs(weights)))
            net = float(np.sum(weights))
            short_gross = float(np.sum(np.clip(-weights, 0, None)))
            exposure_caption = (f"Gross exposure {gross*100:,.0f}%, net {net*100:,.0f}%. "
                                f"Negative weights are shorts. No trading, borrowing or tax "
                                f"costs are modelled — every figure here is gross.")

            wdf = {"Ticker": usable,
                   "Equilibrium π %": bl["pi_total"] * 100,
                   "Posterior μ_BL %": bl["mu_total"] * 100,
                   "AnnualVolatility": sigma_vec,
                   "Market wt %": w_mkt * 100,
                   "Weight": weights, "WeightPct": weights * 100}
            if wstd is not None:
                wdf["Weight ± (resampled)"] = wstd * 100
            weights_df = pd.DataFrame(wdf).sort_values("Weight", ascending=False)

            with st.spinner("Computing efficient frontier..."):
                frontier = efficient_frontier(mu.values, Sigma_used, rf, stance, max_weight,
                                              gross_limit, n_points=40)
            mcv, mcr, mcs = monte_carlo_cloud(mu.values, Sigma_used, rf, stance, n_points=1500)

            # ---- walk-forward backtest ----
            bt = bt_eq = None
            if do_backtest:
                _bench, _fx, _bench_lbl = fetch_benchmark_and_fx(uni, settings)
                _spin = ("Back-testing the rule against pure equilibrium (re-deriving views "
                         "at every rebalance)..." if sys_engine is not None else
                         "Back-testing (walk-forward, re-deriving the equilibrium each time)...")
                with st.spinner(_spin):
                    bt, bt_eq = run_backtests(
                        uni, settings, rf=rf, P=P, Q=Q, conf=conf, engine=sys_engine,
                        n_pairs=sys_n_pairs,
                        apply_typed_views=bool(backtest_uses_views and bl["has_views"]),
                        bench=_bench, fx=_fx, bench_label=_bench_lbl, delta=delta)
                if bt is not None and bt_eq is not None:
                    # Is the gap to the prior real, or one path's luck? Paired
                    # tests on the per-period returns of the two runs.
                    bt["inference"] = {
                        "bootstrap": paired_bootstrap(bt["pnl"], bt_eq["pnl"],
                                                      freq_per_year, rf=rf),
                        "sharpe_test": sharpe_diff_test(bt["pnl"], bt_eq["pnl"],
                                                        freq_per_year, rf=rf),
                        "breakeven_bps": breakeven_cost_bps(bt, bt_eq),
                    }
                rolling = None
                if do_rolling and bt is not None:
                    _bar = st.progress(0.0, text="Repeating the test from staggered start dates...")
                    rolling = rolling_origin(
                        uni, settings, rf=rf, P=P, Q=Q, conf=conf, engine=sys_engine,
                        n_pairs=sys_n_pairs,
                        apply_typed_views=bool(backtest_uses_views and bl["has_views"]),
                        fx=_fx, delta=delta,
                        progress=lambda f: _bar.progress(f, text="Repeating the test from "
                                                                 "staggered start dates..."))
                    _bar.empty()

            # ---- execution table ----
            targets_report = [float(weights[i] * invest_capital) for i in range(len(usable))]
            shares_all = allocate_shares(targets_report, px_report, lots_all, invest_capital)

            rows = []
            for i, ticker in enumerate(usable):
                meta = meta_all[i]; tk_ccy = ccy_all[i]
                fx_rate = fx_all[i]; converted_ok = ok_all[i]
                lot = lots_all[i]; cur_native = px_native[i]
                cur_report = px_report[i]
                alloc_base = targets_report[i]
                if ok_all[i]:
                    shares = float(shares_all[i])
                    alloc_native = alloc_base / fx_rate
                    filled_report = shares * cur_report
                    leftover_report = alloc_base - filled_report
                else:
                    alloc_native = shares = filled_report = leftover_report = np.nan
                # What you hold now, and therefore what actually has to change.
                # A target is not an instruction; the difference is.
                held_i = float(held_shares.get(ticker, 0.0) or 0.0)
                if has_holdings and ok_all[i] and pd.notna(shares):
                    trade_i = float(shares) - held_i
                    # Below one lot there is nothing executable to do, so call
                    # it a hold rather than printing a fractional instruction.
                    if abs(trade_i) < max(float(lot), 1.0) - 1e-9:
                        trade_i, action_i = 0.0, "HOLD"
                    else:
                        action_i = "BUY" if trade_i > 0 else "SELL"
                    trade_val_i = trade_i * cur_report
                else:
                    trade_i = trade_val_i = np.nan
                    action_i = "BUY" if weights[i] >= 0 else "SELL SHORT"

                rows.append({
                    "Ticker": ticker, "Name": meta["name"] or "",
                    "Side": "BUY" if weights[i] >= 0 else "SELL SHORT",
                    "Action": action_i, "SharesHeld": held_i,
                    "TradeShares": trade_i, f"TradeValue ({base_currency})": trade_val_i,
                    "HeldWtPct": (w_prev_live[i] * 100) if has_holdings else np.nan,
                    "Exchange": meta["exchange"] or "", "NativeCurrency": tk_ccy or "UNKNOWN",
                    "CurrentPrice": cur_native, f"CurrentPrice ({base_currency})": cur_report,
                    "MarketWtPct": w_mkt[i] * 100, "WeightPct": weights[i] * 100,
                    f"Allocation ({base_currency})": alloc_base,
                    "LotSize": lot, "Shares": shares, f"Cost ({base_currency})": filled_report,
                    f"LeftoverCash ({base_currency})": leftover_report,
                    "Converted": "YES" if converted_ok else "NO", "FXRate": fx_rate,
                })
            exec_df = pd.DataFrame(rows)
            if has_holdings:
                # Lead with the instruction, not the destination.
                detail_cols = ["Ticker", "Name", "Action", "SharesHeld", "TradeShares",
                               f"TradeValue ({base_currency})", "HeldWtPct", "WeightPct",
                               "Shares", "CurrentPrice", f"CurrentPrice ({base_currency})",
                               "MarketWtPct", f"Allocation ({base_currency})", "LotSize",
                               f"Cost ({base_currency})", "Converted", "FXRate"]
            else:
                detail_cols = ["Ticker", "Name", "Side", "Exchange", "NativeCurrency", "CurrentPrice",
                               f"CurrentPrice ({base_currency})", "MarketWtPct", "WeightPct",
                               f"Allocation ({base_currency})", "LotSize", "Shares",
                               f"Cost ({base_currency})", f"LeftoverCash ({base_currency})",
                               "Converted", "FXRate"]
            exec_fmt = {"CurrentPrice": 2, f"CurrentPrice ({base_currency})": 2, "MarketWtPct": 2,
                        "WeightPct": 2, f"Allocation ({base_currency})": 2, "LotSize": 0, "Shares": 0,
                        f"Cost ({base_currency})": 2, f"LeftoverCash ({base_currency})": 2, "FXRate": 6,
                        "SharesHeld": 0, "TradeShares": 0, f"TradeValue ({base_currency})": 2,
                        "HeldWtPct": 2}
            # True idle cash = capital minus what the long book actually costs.
            # (Summing the per-row leftovers would double-count now that the
            # top-up pass lets some rows overshoot their own target slightly.)
            _long_cost = exec_df.loc[exec_df[f"Cost ({base_currency})"] > 0,
                                     f"Cost ({base_currency})"].sum(skipna=True)
            _long_target = exec_df.loc[exec_df[f"Allocation ({base_currency})"] > 0,
                                       f"Allocation ({base_currency})"].sum(skipna=True)
            leftover_total = max(0.0, float(min(total_capital, _long_target) - _long_cost))
            fx_warn = not exec_df[exec_df["Converted"] == "NO"].empty

            intended = exec_df["WeightPct"].abs() > 0.5
            got_zero = intended & (exec_df["Shares"].fillna(0) == 0)
            n_unaffordable = int(got_zero.sum())
            leftover_pct = (leftover_total / total_capital) if total_capital > 0 else 0.0
            capital_warn = (n_unaffordable >= 1) or (leftover_pct > 0.05)

            # How much capital would this portfolio actually need? For each
            # intended position, one board lot must fit inside its own weight:
            #   capital >= (lot * price) / weight
            # The binding name is whichever has the worst price-to-weight ratio,
            # which is usually a small weight on an expensive share.
            need = []
            for _, r in exec_df.iterrows():
                w_abs = abs(float(r["WeightPct"])) / 100.0
                px = r[f"CurrentPrice ({base_currency})"]
                if w_abs > 0.005 and pd.notna(px) and px > 0:
                    need.append((float(r["LotSize"]) * float(px)) / w_abs)
            min_capital = float(max(need)) if need else np.nan
            # A rough "how many names can I actually hold?" figure: assume roughly
            # equal weights, so each name needs lot*price of the capital.
            # Use the SAME heuristic as the universe-load trim, so the two never
            # contradict each other ("you need 256,707" next to "hold 28 names"
            # was a real bug: two different formulas answering one question).
            _lot_prices = [float(r["LotSize"]) * float(r[f"CurrentPrice ({base_currency})"])
                           for _, r in exec_df.iterrows()
                           if pd.notna(r[f"CurrentPrice ({base_currency})"])
                           and float(r[f"CurrentPrice ({base_currency})"]) > 0]
            afford_n = (affordable_holdings(_lot_prices, total_capital, n_assets)
                        if _lot_prices else np.nan)

            run_summary = {
                "Market": market_name.split("—")[0].strip(), "Strategy": stance, "Horizon": horizon,
                "Views": len(view_labels), "tau": tau, "delta": round(delta, 2),
                "Omega": omega_method.split()[0],
                "Exp.ret %": round(port_ret * 100, 2), "Vol %": round(port_vol * 100, 2),
                "Sharpe": round(port_sharpe, 2),
                "Active share %": round(float(np.sum(np.abs(weights - w_mkt))) / 2 * 100, 1),
            }
            if bt is not None:
                run_summary["OOS ann.ret %"] = round(bt["strat"]["ann_ret"] * 100, 2)
                run_summary["OOS Sharpe"] = round(bt["strat"]["sharpe"], 2)

            st.session_state["res"] = {
                "base_currency": base_currency, "total_capital": invest_capital, "usable": usable,
                "max_weight_pct": float(max_weight_pct),
                "variance_ratio": _vr, "hold_periods": int(_reb_periods),
                "objective": objective,
                "has_holdings": has_holdings, "held_value": held_value,
                "held_problems": held_problems, "n_held_priced": n_held_priced,
                "new_cash": float(new_cash), "w_prev_live": w_prev_live,
                "n_assets": n_assets, "freq_per_year": freq_per_year, "ret_freq": ret_freq,
                "weights": weights, "mu_vals": mu.values, "sigma": sigma_vec,
                "pi_total": bl["pi_total"], "mu_total": bl["mu_total"],
                "w_mkt": w_mkt, "w_markowitz": w_markowitz,
                "delta": delta, "delta_note": delta_note, "tau": tau, "rf": rf,
                "has_views": bl["has_views"], "n_views": len(view_labels),
                "view_labels": view_labels, "view_conf": conf,
                "view_surprise": bl["view_surprise"], "omega_diag": bl["omega_diag"],
                "view_q_stated": Q,
                "port_ret": port_ret, "port_vol": port_vol, "port_sharpe": port_sharpe,
                "mkt_ret": mkt_ret_bl, "mkt_vol": mkt_vol_bl,
                "gross": gross, "net": net, "short_gross": short_gross,
                "settings_caption": settings_caption, "exposure_caption": exposure_caption,
                "universe_note": uni["universe_note"],
                "weights_df": weights_df, "has_stability": wstd is not None,
                "frontier": frontier, "mc_vols": mcv, "mc_rets": mcr, "mc_sharpes": mcs,
                "do_backtest": do_backtest, "bt": bt, "bt_eq": bt_eq,
                "rolling": (rolling if do_backtest else None),
                "backtest_uses_views": bool(backtest_uses_views and bl["has_views"]),
                "sys_engine": sys_engine, "sys_conf": sys_conf,
                "sys_conf_note": sys_conf_note,
                "exec_df": exec_df, "detail_cols": detail_cols, "exec_fmt": exec_fmt,
                "leftover_total": leftover_total, "fx_warn": fx_warn,
                "capital_warn": capital_warn, "n_unaffordable": n_unaffordable,
                "leftover_pct": leftover_pct, "min_capital": min_capital,
                "afford_n": afford_n, "n_holdings_chosen": n_assets,
                "run_summary": run_summary,
            }
    else:
        st.info("Choose your settings above, then press **① Load universe & equilibrium**. "
                "The app needs to know which stocks it picked before you can express views on them.")

    if "res" in st.session_state:
        render_results(st.session_state["res"])


if __name__ == "__main__":
    main()
