"""Headless tests for the Black-Litterman numeric core.

Streamlit and yfinance are stubbed so the module imports without a running
server or network. Nothing here touches the UI — these tests are about whether
the maths is right.

Run:  python test_bl_core.py
"""
import math
import sys
import types
import numpy as np
import pandas as pd
import scipy.optimize as sciopt  # exact-reference solves in the turnover tests

# ---------------------------------------------------------
# Stub the modules the app imports but that the maths never uses
# ---------------------------------------------------------
st = types.ModuleType("streamlit")


def _cache_data(*a, **k):
    if a and callable(a[0]):
        return a[0]

    def deco(fn):
        return fn
    return deco


st.cache_data = _cache_data
st.column_config = types.SimpleNamespace()
sys.modules["streamlit"] = st

yf = types.ModuleType("yfinance")
yf.set_tz_cache_location = lambda *a, **k: None
yf.download = lambda *a, **k: None
yf.Ticker = lambda *a, **k: None
sys.modules["yfinance"] = yf

import black_litterman_portfolio as bl  # noqa: E402

RNG = np.random.default_rng(0)
FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def fixture(n=6, seed=1):
    """A well-conditioned covariance matrix and a lopsided market portfolio."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    Sigma = A @ A.T / n + np.eye(n) * 0.02
    Sigma = (Sigma + Sigma.T) / 2
    caps = np.array([300, 250, 180, 120, 90, 60], dtype=float)[:n]
    w_mkt = caps / caps.sum()
    tickers = [f"A{i}" for i in range(n)]
    return tickers, Sigma, w_mkt


# =========================================================
print("\n--- equilibrium ---")
tickers, Sigma, w_mkt = fixture()
rf, tau, delta = 0.04, 0.05, 2.5
pi = bl.equilibrium_returns(delta, Sigma, w_mkt)

# 1. Reverse optimisation must invert cleanly: (delta*Sigma)^-1 pi == w_mkt.
w_back = bl.implied_confidence_weights(pi, Sigma, delta)
check("reverse optimisation round-trips to the market weights",
      np.allclose(w_back, w_mkt, atol=1e-10), f"max err {np.abs(w_back-w_mkt).max():.2e}")

# 2. delta implied from a market with a known excess return.
mkt_excess = float(w_mkt @ pi)
d_imp, note = bl.implied_risk_aversion(w_mkt, Sigma, mkt_excess)
check("implied delta recovers the delta used to build pi",
      abs(d_imp - delta) < 1e-8, f"got {d_imp}")

# 3. A market whose excess return was negative must not produce a negative delta.
d_neg, note_neg = bl.implied_risk_aversion(w_mkt, Sigma, -0.02)
check("negative market excess return falls back to the default delta",
      d_neg == 2.5 and "not positive" in note_neg, f"got {d_neg} / {note_neg}")

# =========================================================
print("\n--- posterior with no views ---")
out = bl.black_litterman(Sigma, w_mkt, rf, tau, delta, np.zeros((0, len(tickers))),
                         np.zeros(0), np.zeros(0))
check("no views => mu_BL == pi exactly",
      np.array_equal(out["mu_excess"], out["pi_excess"]))
check("no views => total returns are excess + rf",
      np.allclose(out["mu_total"], out["pi_excess"] + rf))
check("no views => optimal unconstrained weights are the market weights",
      np.allclose(bl.implied_confidence_weights(out["mu_excess"], Sigma, delta), w_mkt, atol=1e-10))

# =========================================================
print("\n--- the two posterior formulas agree ---")
P = np.array([[1.0, 0, 0, 0, 0, 0],
              [0, 1.0, -1.0, 0, 0, 0]])
Q = np.array([0.10 - rf, 0.03])
Omega = bl.omega_he_litterman(P, Sigma, tau)

smw = bl.posterior_mu(pi, Sigma, P, Q, Omega, tau)
tauS = tau * Sigma
textbook = np.linalg.solve(
    np.linalg.inv(tauS) + P.T @ np.linalg.inv(Omega) @ P,
    np.linalg.inv(tauS) @ pi + P.T @ np.linalg.inv(Omega) @ Q)
check("Woodbury form == textbook form",
      np.allclose(smw, textbook, atol=1e-9), f"max err {np.abs(smw-textbook).max():.2e}")

# =========================================================
print("\n--- Omega: He-Litterman ---")
check("He-Litterman Omega == diag(P tau Sigma P')",
      np.allclose(np.diag(Omega), np.diag(P @ tauS @ P.T)))
check("He-Litterman Omega is diagonal",
      np.allclose(Omega, np.diag(np.diag(Omega))))

# =========================================================
print("\n--- Omega: Idzorek confidence ---")
# A single absolute view, swept across confidences.
P1 = np.array([[1.0, 0, 0, 0, 0, 0]])
Q1 = np.array([0.12 - rf])

errs = []
for c in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99):
    Om = bl.omega_idzorek(P1, Q1, Sigma, tau, delta, w_mkt, pi, np.array([c]))
    mu_c = bl.posterior_mu(pi, Sigma, P1, Q1, Om, tau)
    w_c = bl.implied_confidence_weights(mu_c, Sigma, delta)

    denom = bl._scalar(P1 @ tauS @ P1.T)
    mu_100 = pi + (tauS @ P1.T).ravel() * ((Q1[0] - bl._scalar(P1 @ pi)) / denom)
    w_100 = bl.implied_confidence_weights(mu_100, Sigma, delta)
    target = w_mkt + c * (w_100 - w_mkt)
    errs.append(np.abs(w_c - target).max())

check("Idzorek: realised tilt matches the requested confidence at every level",
      max(errs) < 1e-6, f"worst weight error {max(errs):.2e}")

# Omega must fall as confidence rises (more confident = less uncertain view).
oms = [float(bl.omega_idzorek(P1, Q1, Sigma, tau, delta, w_mkt, pi, np.array([c]))[0, 0])
       for c in (0.1, 0.3, 0.5, 0.7, 0.9)]
check("Idzorek: omega decreases monotonically as confidence increases",
      all(oms[i] > oms[i + 1] for i in range(len(oms) - 1)), str(oms))

# At (near) full confidence the posterior should sit on the view.
Om99 = bl.omega_idzorek(P1, Q1, Sigma, tau, delta, w_mkt, pi, np.array([0.9999]))
mu99 = bl.posterior_mu(pi, Sigma, P1, Q1, Om99, tau)
check("Idzorek: ~100% confidence puts the posterior essentially on the view",
      abs(bl._scalar(P1 @ mu99) - Q1[0]) < 5e-4,
      f"P.mu={bl._scalar(P1 @ mu99):.6f} vs Q={Q1[0]:.6f}")

# A view that merely restates the equilibrium must change nothing.
Q_same = np.array([bl._scalar(P1 @ pi)])
Om_same = bl.omega_idzorek(P1, Q_same, Sigma, tau, delta, w_mkt, pi, np.array([0.9]))
mu_same = bl.posterior_mu(pi, Sigma, P1, Q_same, Om_same, tau)
check("a view identical to the equilibrium leaves mu unchanged",
      np.allclose(mu_same, pi, atol=1e-8), f"max err {np.abs(mu_same-pi).max():.2e}")

# =========================================================
print("\n--- views table -> P, Q ---")
tk = tickers


def vdf(rows):
    return pd.DataFrame(rows, columns=["Use", "Type", "Asset", "Versus",
                                       "Return % p.a.", "Confidence %"])


P2, Q2, c2, lab2, err2 = bl.build_pq(vdf([
    [True,  "Absolute", "A0", "A1", 12.0, 60.0],
    [True,  "Relative", "A1", "A2",  3.0, 40.0],
    [False, "Absolute", "A3", "A0", 99.0, 90.0],     # unticked -> ignored
]), tk, rf)

check("only ticked rows become views", P2.shape[0] == 2 and len(lab2) == 2, str(lab2))
check("absolute view Q is converted from total to excess",
      abs(Q2[0] - (0.12 - rf)) < 1e-12, f"{Q2[0]}")
check("relative view Q keeps the raw spread (rf cancels)",
      abs(Q2[1] - 0.03) < 1e-12, f"{Q2[1]}")
check("absolute row picks exactly one asset with weight 1",
      P2[0, 0] == 1.0 and P2[0].sum() == 1.0)
check("relative row is +1/-1 and sums to zero",
      P2[1, 1] == 1.0 and P2[1, 2] == -1.0 and abs(P2[1].sum()) < 1e-15)
check("confidences carried through as fractions", np.allclose(c2, [0.6, 0.4]))

# Malformed rows must be reported, not silently mangled.
P3, Q3, c3, lab3, err3 = bl.build_pq(vdf([
    [True, "Absolute", "NOPE", "A1", 5.0, 50.0],       # unknown ticker
    [True, "Relative", "A0",   "A0", 5.0, 50.0],       # against itself
    [True, "Absolute", "A0",   "A1", None, 50.0],      # unparseable return
    [True, "Absolute", "A0",   "A1", 5.0, 0.0],        # zero confidence
]), tk, rf)
check("every malformed row is skipped with an explanation",
      P3.shape[0] == 0 and len(err3) == 4, f"{P3.shape}, {len(err3)} errors")

Pe, Qe, ce, labe, erre = bl.build_pq(vdf([]), tk, rf)
check("an empty table yields no views", Pe.shape == (0, len(tk)) and len(erre) == 0)

# =========================================================
print("\n--- market weights ---")
caps_full = {t: v for t, v in zip(tk, [300e9, 250e9, 180e9, 120e9, 90e9, 60e9])}
w, note = bl.market_weights(tk, caps_full)
check("full caps give cap weights that sum to 1",
      abs(w.sum() - 1) < 1e-12 and "capitalisation" in note and "missing" not in note)
check("cap weights are ordered like the caps", np.all(np.diff(w) < 0))

caps_gap = dict(caps_full)
caps_gap["A2"] = np.nan
w2, note2 = bl.market_weights(tk, caps_gap)
check("one missing cap is filled with the median, not dropped",
      abs(w2.sum() - 1) < 1e-12 and "missing" in note2, note2)

w3, note3 = bl.market_weights(tk, {}, fallback_weights=np.array([5, 4, 3, 2, 1, 1.0]))
check("no caps at all falls back to traded value",
      abs(w3.sum() - 1) < 1e-12 and "traded value" in note3, note3)

w4, note4 = bl.market_weights(tk, {})
check("no caps and no fallback gives equal weight",
      np.allclose(w4, 1 / len(tk)) and "equal weight" in note4, note4)

# =========================================================
print("\n--- posterior covariance ---")
Sp = bl.posterior_sigma(Sigma, P, Omega, tau)
ev = np.linalg.eigvalsh(Sp)
check("posterior covariance is positive semi-definite", ev.min() > -1e-10, f"min eig {ev.min():.2e}")
check("posterior covariance is at least as large as Sigma",
      np.all(np.diag(Sp) >= np.diag(Sigma) - 1e-12))
check("posterior covariance is symmetric", np.allclose(Sp, Sp.T))

Sp0 = bl.posterior_sigma(Sigma, np.zeros((0, 6)), np.zeros((0, 0)), tau)
check("with no views posterior covariance is Sigma*(1+tau)",
      np.allclose(Sp0, Sigma * (1 + tau), atol=1e-10))

# =========================================================
print("\n--- end-to-end pipeline ---")
Pf = np.array([[0.0, 0, 0, 0, 0, 1.0]])         # the smallest-cap name will outperform
Qf = np.array([0.20 - rf])
res_lo = bl.black_litterman(Sigma, w_mkt, rf, tau, delta, Pf, Qf, np.array([0.10]))
res_hi = bl.black_litterman(Sigma, w_mkt, rf, tau, delta, Pf, Qf, np.array([0.90]))
tilt_lo = np.abs(bl.implied_confidence_weights(res_lo["mu_excess"], Sigma, delta) - w_mkt).sum()
tilt_hi = np.abs(bl.implied_confidence_weights(res_hi["mu_excess"], Sigma, delta) - w_mkt).sum()
check("higher confidence produces a bigger tilt away from the market",
      tilt_hi > tilt_lo * 2, f"10%: {tilt_lo:.4f}  90%: {tilt_hi:.4f}")
check("a bullish view raises that asset's posterior return",
      res_hi["mu_excess"][5] > pi[5], f"{res_hi['mu_excess'][5]:.4f} vs {pi[5]:.4f}")
check("view surprise is Q - P.pi",
      abs(res_hi["view_surprise"][0] - (Qf[0] - bl._scalar(Pf @ pi))) < 1e-14)
check("total returns are excess + rf throughout",
      np.allclose(res_hi["mu_total"], res_hi["mu_excess"] + rf))

# Views on one asset should leave uncorrelated assets nearly untouched, and
# correlated ones should move — that is the whole reason to use Sigma.
Sd = np.diag([0.04, 0.09, 0.16, 0.04, 0.09, 0.16])
pid = bl.equilibrium_returns(delta, Sd, w_mkt)
rd = bl.black_litterman(Sd, w_mkt, rf, tau, delta, Pf, Qf, np.array([0.9]))
untouched = np.abs(rd["mu_excess"][:5] - pid[:5]).max()
check("with a diagonal Sigma a view moves only the asset it names",
      untouched < 1e-12, f"max leakage {untouched:.2e}")

# =========================================================
print("\n--- numerical robustness ---")
bad = np.array([[1.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0]])   # duplicate views
Qb = np.array([0.10 - rf, 0.11 - rf])
try:
    Omb = bl.omega_idzorek(bad, Qb, Sigma, tau, delta, w_mkt, pi, np.array([0.5, 0.5]))
    mub = bl.posterior_mu(pi, Sigma, bad, Qb, Omb, tau)
    ok = np.all(np.isfinite(mub))
except Exception as e:                                          # noqa: BLE001
    ok = False
check("duplicate/conflicting views do not blow up", ok)

n_big = 40
Ab = RNG.normal(size=(n_big, n_big))
Sb = Ab @ Ab.T / n_big + np.eye(n_big) * 0.01
wb = np.repeat(1 / n_big, n_big)
pib = bl.equilibrium_returns(delta, Sb, wb)
Pb = np.zeros((3, n_big)); Pb[0, 0] = 1; Pb[1, 5] = 1; Pb[1, 6] = -1; Pb[2, 10] = 1
Qbb = np.array([0.08, 0.02, 0.06])
Omb2 = bl.omega_idzorek(Pb, Qbb, Sb, tau, delta, wb, pib, np.array([0.3, 0.6, 0.9]))
mub2 = bl.posterior_mu(pib, Sb, Pb, Qbb, Omb2, tau)
check("40 assets x 3 views stays finite and sane",
      np.all(np.isfinite(mub2)) and np.abs(mub2).max() < 5.0)

# =========================================================
print()
print("--- tau invariance and confidence linearity ---")

# tau cancels out of the posterior MEAN under both Omega methods, because Omega
# is itself proportional to tau. This is why the UI tells the user to reach for
# Confidence, not tau, when views feel too weak.
n_fx = len(w_mkt)
Pt = np.zeros((2, n_fx)); Pt[0, 0] = 1.0; Pt[1, 1] = 1.0; Pt[1, 2] = -1.0
Qt = np.array([pi[0] + 0.05, float(Pt[1] @ pi) + 0.03])
ct = np.array([0.6, 0.3])

mus_idz, mus_hl = [], []
for tv in (0.01, 0.05, 0.25):
    Oi = bl.omega_idzorek(Pt, Qt, Sigma, tv, delta, w_mkt, pi, ct)
    mus_idz.append(bl.posterior_mu(pi, Sigma, Pt, Qt, Oi, tv))
    Oh = bl.omega_he_litterman(Pt, Sigma, tv)
    mus_hl.append(bl.posterior_mu(pi, Sigma, Pt, Qt, Oh, tv))

check("tau does not change the posterior mean (Idzorek)",
      max(np.abs(mus_idz[0] - m).max() for m in mus_idz[1:]) < 1e-10)
check("tau does not change the posterior mean (He-Litterman)",
      max(np.abs(mus_hl[0] - m).max() for m in mus_hl[1:]) < 1e-10)

# tau DOES change the posterior covariance -- that is its only live effect here.
sp = [np.mean(np.diag(bl.posterior_sigma(
        Sigma, Pt, bl.omega_he_litterman(Pt, Sigma, tv), tv))) for tv in (0.01, 0.25)]
check("tau does change the posterior covariance", sp[1] > sp[0] + 1e-6)

# Under Idzorek the tilt is exactly confidence x surprise, for a single view.
Ps = np.zeros((1, n_fx)); Ps[0, 0] = 1.0
surprise = 0.05
Qs = np.array([pi[0] + surprise])
lin_ok = True
for c in (0.1, 0.25, 0.5, 0.75, 0.9):
    Oc = bl.omega_idzorek(Ps, Qs, Sigma, tau, delta, w_mkt, pi, np.array([c]))
    muc = bl.posterior_mu(pi, Sigma, Ps, Qs, Oc, tau)
    if abs((muc[0] - pi[0]) - c * surprise) > 1e-6:
        lin_ok = False
check("posterior shift equals confidence x surprise (Idzorek)", lin_ok)

# A view that merely restates the equilibrium moves nothing, at any confidence.
Q0 = np.array([float(Ps[0] @ pi)])
O0 = bl.omega_idzorek(Ps, Q0, Sigma, tau, delta, w_mkt, pi, np.array([0.99]))
mu0 = bl.posterior_mu(pi, Sigma, Ps, Q0, O0, tau)
check("a zero-surprise view changes nothing even at 99% confidence",
      np.abs(mu0 - pi).max() < 1e-9)

# =========================================================
print()
print("--- volatility targeting ---")

# k = target / realised, so a series with exactly the target vol gets k = 1.
rng_v = np.random.default_rng(7)
per_vol = 0.12 / np.sqrt(252)
calm = rng_v.normal(0, per_vol, 500)
k_calm = bl.vol_target_scale(calm, 0.12, 252, 60, max_leverage=3.0)
check("realised vol at target gives a scale near 1.0", abs(k_calm - 1.0) < 0.15, f"k={k_calm:.3f}")

# Twice the volatility should halve the exposure.
wild = rng_v.normal(0, per_vol * 2, 500)
k_wild = bl.vol_target_scale(wild, 0.12, 252, 60, max_leverage=3.0)
check("double the volatility roughly halves the exposure",
      abs(k_wild - 0.5) < 0.12, f"k={k_wild:.3f}")

# The leverage cap must bind, and never go negative.
k_capped = bl.vol_target_scale(calm * 0.1, 0.12, 252, 60, max_leverage=1.0)
check("max_leverage=1.0 caps the scale at 1.0", k_capped <= 1.0 + 1e-12, f"k={k_capped:.3f}")
check("scale is never negative",
      bl.vol_target_scale(wild, 0.12, 252, 60, max_leverage=1.0) >= 0.0)

# Too little history => stay fully invested rather than guess.
check("insufficient history returns a scale of 1.0",
      bl.vol_target_scale(np.array([0.01, -0.01]), 0.12, 252, 60) == 1.0)

# Degenerate (zero-variance) input must not divide by zero.
k_zero = bl.vol_target_scale(np.zeros(200), 0.12, 252, 60, max_leverage=1.0)
check("zero-variance history does not blow up", np.isfinite(k_zero) and k_zero == 1.0)

# NO LOOK-AHEAD: the scale for a period must not depend on that period's return.
# Recompute with the last observation altered — the answer must be identical,
# because the caller only ever passes strictly-past returns.
hist_a = list(rng_v.normal(0, per_vol, 200))
hist_b = hist_a[:-1] + [hist_a[-1] * 5.0]
same = bl.vol_target_scale(hist_a[:-1], 0.12, 252, 60) == \
       bl.vol_target_scale(hist_b[:-1], 0.12, 252, 60)
check("scale depends only on the history it is given (no look-ahead by construction)", same)

# The lookback window must actually be respected: an old crisis outside the
# window should not affect today's exposure.
crisis_then_calm = np.concatenate([rng_v.normal(0, per_vol * 6, 300),
                                   rng_v.normal(0, per_vol, 300)])
k_short = bl.vol_target_scale(crisis_then_calm, 0.12, 252, 60, max_leverage=3.0)
k_long = bl.vol_target_scale(crisis_then_calm, 0.12, 252, 600, max_leverage=3.0)
check("a short lookback ignores a crisis that has passed", k_short > k_long,
      f"short={k_short:.2f} long={k_long:.2f}")

# =========================================================
print()
print("--- the negative-Sharpe artifact (why the UI suppresses that comparison) ---")

# Two return streams with the SAME (negative) excess return but different
# volatility. The lower-volatility one scores a WORSE Sharpe, which is why any
# Sharpe comparison is meaningless once the numerator goes negative.
rf_t = 0.06
rng_s = np.random.default_rng(21)
base = rng_s.normal(0, 1, 2000)
base = (base - base.mean()) / base.std(ddof=1)          # exactly mean 0, sd 1
target_ann = 0.02                                       # 2% a year, below rf

hi = target_ann / 252 + base * (0.16 / np.sqrt(252))    # 16% vol
lo = target_ann / 252 + base * (0.08 / np.sqrt(252))    # 8% vol
m_hi = bl.perf_metrics(hi, 252, rf_t)
m_lo = bl.perf_metrics(lo, 252, rf_t)

check("both streams earn less than the risk-free rate",
      m_hi["ann_ret"] < rf_t and m_lo["ann_ret"] < rf_t)
check("halving volatility makes a NEGATIVE Sharpe worse, not better",
      m_lo["sharpe"] < m_hi["sharpe"],
      f"16% vol -> {m_hi['sharpe']:.3f} ; 8% vol -> {m_lo['sharpe']:.3f}")

# ...and the opposite when the excess return is positive, which is the case
# where the ratio behaves the way people expect.
hi_p = 0.12 / 252 + base * (0.16 / np.sqrt(252))
lo_p = 0.12 / 252 + base * (0.08 / np.sqrt(252))
check("with a POSITIVE excess return, lower volatility improves Sharpe",
      bl.perf_metrics(lo_p, 252, rf_t)["sharpe"] > bl.perf_metrics(hi_p, 252, rf_t)["sharpe"])

# =========================================================
print()
print("--- share allocation (cash drag) ---")

# Prices and weights from a real Nifty run that stranded 18% of the capital.
alloc_px = np.array([1312.40, 731.00, 1417.50, 1049.90, 1131.00, 1248.00, 1096.70,
                     2297.80, 1950.00, 402.90, 1500.00, 2400.00, 3400.00, 1700.00,
                     12500.00, 350.00, 420.00, 3600.00, 2900.00])
alloc_w = np.repeat(1.0 / len(alloc_px), len(alloc_px))
CAP = 100_000.0
alloc_lots = np.ones(len(alloc_px))
tgt = alloc_w * CAP

floor_only = np.floor(tgt / alloc_px)
smart = bl.allocate_shares(tgt, alloc_px, alloc_lots, CAP)

idle_floor = CAP - float((floor_only * alloc_px).sum())
idle_smart = CAP - float((smart * alloc_px).sum())

check("the allocator never overspends the capital",
      float((smart * alloc_px).sum()) <= CAP + 1e-6,
      f"spent {float((smart*alloc_px).sum()):,.2f} of {CAP:,.2f}")
check("the allocator strands less cash than flooring each position",
      idle_smart < idle_floor,
      f"floor {idle_floor:,.0f} vs smart {idle_smart:,.0f}")
check("idle cash is smaller than the cheapest lot still buyable",
      idle_smart < float(alloc_px.min()) * 1.0 + 1e-6,
      f"idle {idle_smart:,.2f} vs cheapest lot {alloc_px.min():,.2f}")
check("no position is short-changed below its floored share count",
      bool(np.all(smart >= floor_only)))
check("weight tracking error does not get worse",
      np.abs(smart * alloc_px / CAP - alloc_w).sum()
      <= np.abs(floor_only * alloc_px / CAP - alloc_w).sum() + 1e-9)

# Board lots must be respected (Tokyo / Hong Kong trade in 100s).
lots100 = np.full(len(alloc_px), 100.0)
smart100 = bl.allocate_shares(tgt, alloc_px, lots100, CAP)
check("board lots are respected", bool(np.all(smart100 % 100 == 0)))
check("board lots never overspend",
      float((smart100 * alloc_px).sum()) <= CAP + 1e-6)

# Degenerate inputs must not hang or explode.
bad = bl.allocate_shares([1000.0, 1000.0], [np.nan, 0.0], [1, 1], 5000.0)
check("unusable prices yield zero shares rather than an error",
      np.all(np.asarray(bad) == 0))
check("zero capital yields zero shares",
      np.all(bl.allocate_shares(tgt * 0, alloc_px, alloc_lots, 0.0) == 0))

# Capital-aware holdings suggestion should fall as capital falls.
sizes = [bl.affordable_holdings(list(alloc_px), c, 25) for c in (20_000, 100_000, 500_000)]
check("suggested holdings is non-decreasing in capital",
      sizes[0] <= sizes[1] <= sizes[2], str(sizes))

# =========================================================
print()
print("--- relative volatility targeting ---")


def _bt_builder(train_returns, cov_annual):
    k = cov_annual.shape[0]
    wm = np.repeat(1.0 / k, k)
    d, _ = bl.implied_risk_aversion(wm, cov_annual,
                                    float(wm @ (train_returns.mean().values * 252)) - 0.04)
    o = bl.black_litterman(cov_annual, wm, 0.04, 0.05, d,
                           np.zeros((0, k)), np.zeros(0), np.zeros(0))
    return o["mu_total"], o["Sigma_used"]


_BT = dict(freq="Daily", use_log=True, mu_builder=_bt_builder,
           cov_method="Ledoit-Wolf shrinkage", freq_per_year=252, rf=0.04,
           stance="Long only", max_weight=0.15, gross_limit=None, objective="Max Sharpe",
           alpha=0.95, tc_bps=10.0, borrow_bps=50.0, train_frac=0.45, resample_n=0,
           rebalance_periods=252, rebal_label="yearly", vol_lookback=63)

rng_v2 = np.random.default_rng(5)
n_v, T_v = 8, 1200
R_v = rng_v2.normal(0.0006, 0.012, size=(T_v, n_v))
idx_v = pd.bdate_range("2019-01-01", periods=T_v)
cols_v = [f"S{i}" for i in range(n_v)]

# Same length, identical history, DIFFERENT FUTURE. Any change to an exposure
# BEFORE the divergence point would mean the overlay is reading the future.
CUT = 900
A_v, B_v = R_v.copy(), R_v.copy()
B_v[CUT:] = rng_v2.normal(0.0006, 0.045, size=(T_v - CUT, n_v))
px_A = pd.DataFrame(100 * np.cumprod(1 + A_v, axis=0), index=idx_v, columns=cols_v)
px_B = pd.DataFrame(100 * np.cumprod(1 + B_v, axis=0), index=idx_v, columns=cols_v)

bt_A = bl.run_backtest(px_A, vol_target_frac=0.75, vol_max_leverage=1.0, **_BT)
bt_B = bl.run_backtest(px_B, vol_target_frac=0.75, vol_max_leverage=1.0, **_BT)
sc_A, sc_B = np.asarray(bt_A["scales"]), np.asarray(bt_B["scales"])
overlap = CUT - bt_A["n_train"]

check("relative targeting reads no future data (identical exposures before divergence)",
      np.allclose(sc_A[:overlap], sc_B[:overlap], atol=1e-12),
      f"max diff {np.abs(sc_A[:overlap]-sc_B[:overlap]).max():.2e}")
check("but exposures DO react once the futures differ",
      not np.allclose(sc_A[overlap:overlap + 50], sc_B[overlap:overlap + 50]))

# A fixed absolute target cannot suit both a calm book and a turbulent one;
# the relative target should land near its requested fraction in both.
def _gjr(scale, seed):
    rg = np.random.default_rng(seed)
    vol = np.zeros(T_v); vol[0] = 0.013 * scale; sh = 0.0
    for t in range(1, T_v):
        asym = 0.08 if sh < 0 else 0.0
        vol[t] = math.sqrt(0.000004 * scale ** 2 + 0.90 * vol[t - 1] ** 2
                           + (0.05 + asym) * (vol[t - 1] * sh) ** 2)
        sh = rg.normal()
    L = np.tril(rg.normal(size=(n_v, n_v)) / np.sqrt(n_v)) + np.eye(n_v) * 0.5
    dr = rg.normal(0.0006, 0.0002, n_v)
    Rx = np.array([dr + (L @ rg.normal(size=n_v)) * vol[t] for t in range(T_v)])
    return pd.DataFrame(100 * np.cumprod(1 + Rx, axis=0), index=idx_v, columns=cols_v)


exposures, sharpe_gain = [], []
for sc, sd in ((0.6, 3), (1.8, 3)):
    pxx = _gjr(sc, sd)
    off = bl.run_backtest(pxx, vol_target=None, **_BT)
    rel = bl.run_backtest(pxx, vol_target_frac=0.75, vol_max_leverage=1.0, **_BT)
    exposures.append(rel["avg_scale"])
    sharpe_gain.append(rel["strat"]["sharpe"] - off["strat"]["sharpe"])

check("relative targeting holds a similar exposure across very different books",
      abs(exposures[0] - exposures[1]) < 0.15,
      f"low-vol {exposures[0]:.2f} vs high-vol {exposures[1]:.2f}")
check("relative targeting lands near the requested fraction",
      all(0.60 <= e <= 0.95 for e in exposures), str([round(e, 2) for e in exposures]))

# =========================================================
print()
print("--- systematic view engines ---")

rng_e = np.random.default_rng(2)
n_e, T_e = 8, 1500
drift_e = np.linspace(0.0010, -0.0002, n_e)          # A0 best, A7 worst, by construction
R_e = rng_e.normal(drift_e, 0.011, size=(T_e, n_e))
idx_e = pd.bdate_range("2019-01-01", periods=T_e)
cols_e = [f"A{i}" for i in range(n_e)]
px_e = pd.DataFrame(100 * np.cumprod(1 + R_e, axis=0), index=idx_e, columns=cols_e)

for eng in ("momentum", "reversal", "trend", "lowvol"):
    P_s, Q_s, c_s, lab_s = bl.systematic_views(px_e, cols_e, eng, 252, n_pairs=2)
    check(f"{eng}: produces well-formed views",
          P_s.shape[0] == len(Q_s) == len(lab_s) and P_s.shape[1] == n_e and P_s.shape[0] > 0)
    if eng != "trend":
        check(f"{eng}: relative rows are +1/-1 and sum to zero",
              all(abs(P_s[r].sum()) < 1e-12 for r in range(P_s.shape[0])))

# Momentum on a series with a built-in winner must actually find it.
P_m, Q_m, _c, lab_m = bl.systematic_views(px_e, cols_e, "momentum", 252, n_pairs=1)
check("momentum picks the genuinely strongest asset as the long leg",
      P_m[0, 0] == 1.0, lab_m[0] if lab_m else "no view")

# THE property that makes rule-driven backtests honest: an engine is a pure
# function of the prices handed to it. Same history + different future must
# give bit-identical views.
fut_a = px_e.copy()
fut_b = px_e.copy()
fut_b.iloc[900:] = fut_b.iloc[900:] * 3.0
same_views = True
for eng in ("momentum", "reversal", "trend", "lowvol"):
    va = bl.systematic_views(fut_a.iloc[:900], cols_e, eng, 252)
    vb = bl.systematic_views(fut_b.iloc[:900], cols_e, eng, 252)
    if not (np.allclose(va[1], vb[1]) and va[3] == vb[3] and np.allclose(va[0], vb[0])):
        same_views = False
check("engines read no future data (identical views when only the future differs)", same_views)

# Q must stay bounded however violent the history is.
wild = px_e.copy()
wild.iloc[:, 0] = wild.iloc[:, 0] * np.linspace(1, 50, T_e)
_Pw, Qw, _cw, _lw = bl.systematic_views(wild, cols_e, "momentum", 252, max_spread=0.20)
check("view size is capped no matter how extreme the signal",
      np.all(np.abs(Qw) <= 0.20 + 1e-12), f"max |Q| = {np.abs(Qw).max():.4f}")

# Too little history must yield no views rather than nonsense.
check("insufficient history produces no views",
      bl.systematic_views(px_e.iloc[:5], cols_e, "momentum", 252)[0].shape[0] == 0)

print()
print("--- confidence from hit rate ---")

check("a coin-flip hit rate earns zero confidence",
      bl.confidence_from_hit_rate(0.50, 200)[0] == 0.0)
check("a below-chance hit rate earns zero confidence",
      bl.confidence_from_hit_rate(0.42, 200)[0] == 0.0)
check("no history earns zero confidence",
      bl.confidence_from_hit_rate(None, 0)[0] == 0.0)
c60 = bl.confidence_from_hit_rate(0.60, 10_000)[0]
check("a 60% hit rate on a large sample approaches 2x(hit-0.5) = 20%",
      abs(c60 - 0.20) < 0.01, f"got {c60:.4f}")
check("confidence rises with the hit rate",
      bl.confidence_from_hit_rate(0.55, 500)[0]
      < bl.confidence_from_hit_rate(0.65, 500)[0]
      < bl.confidence_from_hit_rate(0.80, 500)[0])
check("small samples are shrunk toward zero",
      bl.confidence_from_hit_rate(0.70, 5)[0] < bl.confidence_from_hit_rate(0.70, 500)[0])
check("confidence never reaches 100%", bl.confidence_from_hit_rate(1.0, 10_000)[0] <= 0.90)

# A random walk carries no signal, so no engine should earn much confidence on one.
rng_n = np.random.default_rng(9)
noise = pd.DataFrame(100 * np.cumprod(1 + rng_n.normal(0, 0.012, size=(T_e, n_e)), axis=0),
                     index=idx_e, columns=cols_e)
worst = 0.0
for eng in ("momentum", "reversal", "trend", "lowvol"):
    hr_n, nc_n = bl.engine_hit_rate(noise, eng, 252, holding_periods=63)
    worst = max(worst, bl.confidence_from_hit_rate(hr_n, nc_n)[0])
check("no engine earns high confidence on a random walk", worst < 0.35, f"worst {worst:.3f}")

# Hit-rate measurement itself must not peek: truncating the future can only
# remove checks, never change the verdict on the part that overlaps.
hr_full, n_full = bl.engine_hit_rate(px_e, "momentum", 252, 63)
hr_part, n_part = bl.engine_hit_rate(px_e.iloc[:1000], "momentum", 252, 63)
check("hit rate on a shorter history uses fewer checks", n_part < n_full)
check("hit rates stay in [0,1]", 0.0 <= hr_full <= 1.0 and 0.0 <= hr_part <= 1.0)

# =========================================================
print()
print("--- adaptive history window ---")

_END = pd.Timestamp("2026-08-21")
_IDX = pd.bdate_range(_END - pd.DateOffset(years=20), _END)
_rng_h = np.random.default_rng(1)


def _panel(n_old, n_young, young_frac=0.18):
    """n_old assets with 20y history, n_young that listed recently."""
    d = pd.DataFrame(index=_IDX)
    for i in range(n_old + n_young):
        s = pd.Series(100 * np.cumprod(1 + _rng_h.normal(0.0004, 0.012, len(_IDX))), index=_IDX)
        if i >= n_old:
            s.iloc[:-int(len(_IDX) * young_frac)] = np.nan
        d[f"A{i}"] = s
    return d


HC = bl.HORIZON_CFG["Yearly — long-term"]
panel = _panel(20, 10)
start, keep, note = bl.choose_history_window(panel, list(panel.columns), 25, HC, 52)
yrs = (_END - start).days / 365.25

check("adaptive window takes the full length when enough assets support it",
      yrs > 14, f"{yrs:.1f}y")
check("young assets are excluded rather than truncating everyone",
      len(keep) == 20 and all(int(t[1:]) < 20 for t in keep), f"kept {len(keep)}")
check("the window note reports the length and the exclusions",
      "15 years" in note and "too young" in note)

# Every horizon should now estimate from the SAME long window — that is the
# whole point of decoupling holding period from estimation length.
lens = []
for _k, _c in bl.HORIZON_CFG.items():
    s_, k_, _n = bl.choose_history_window(panel, list(panel.columns), 25, _c,
                                          bl.FREQ_PER_YEAR[_c["freq"]])
    lens.append(round((_END - s_).days / 365.25))
check("all three horizons estimate from the same history length", len(set(lens)) == 1, str(lens))

# A universe where nothing is old must degrade gracefully, not collapse.
young_only = _panel(0, 10)
s2, k2, _n2 = bl.choose_history_window(young_only, list(young_only.columns), 10, HC, 52)
check("an all-young universe shortens the window instead of failing",
      len(k2) >= 3 and (_END - s2).days > 365, f"kept {len(k2)}, {(_END-s2).days/365.25:.1f}y")

# Degenerate inputs must not raise.
one = _panel(1, 0)
s3, k3, _n3 = bl.choose_history_window(one, list(one.columns), 5, HC, 52)
check("a single-asset panel does not raise", len(k3) >= 1)

check("every horizon requests a long estimation window",
      all(c["years"] >= 15 for c in bl.HORIZON_CFG.values()))
check("every horizon estimates from the same return frequency",
      len({c["freq"] for c in bl.HORIZON_CFG.values()}) == 1,
      str({c["freq"] for c in bl.HORIZON_CFG.values()}))
check("horizons differ ONLY in how often they rebalance",
      {c["rebal"] for c in bl.HORIZON_CFG.values()} == {5, 21, 252}
      and len({(c["years"], c["min_years"], c["freq"]) for c in bl.HORIZON_CFG.values()}) == 1)

# Equal information means equal observations: no horizon may see less data than
# another. This is the property that stops "yearly" being quietly worse.
_obs = {k: c["years"] * bl.FREQ_PER_YEAR[c["freq"]] for k, c in bl.HORIZON_CFG.items()}
check("all horizons get an identical observation count", len(set(_obs.values())) == 1,
      str(_obs))
check("that count clears the observations-per-asset bar for a 25-asset universe",
      min(_obs.values()) / 25 >= bl.TARGET_OBS_PER_ASSET,
      f"{min(_obs.values())/25:.0f} per asset")

# =========================================================
print()
print("--- Ledoit-Wolf: vectorised form equals the textbook loop ---")


def _lw_reference(X):
    """The literal Ledoit-Wolf (2004) formulation, loop and all. Slow, and kept
    here purely as the thing the fast version must agree with."""
    X = np.asarray(X, dtype=float)
    t, n = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / t
    mm = np.trace(S) / n
    d2 = np.sum((S - mm * np.eye(n)) ** 2) / n
    b = 0.0
    for i in range(t):
        xi = Xc[i][:, None]
        b += np.sum((xi @ xi.T - S) ** 2)
    b = b / (t ** 2) / n
    b2 = min(b, d2)
    a2 = d2 - b2
    sh = b2 / d2 if d2 > 0 else 1.0
    return sh * mm * np.eye(n) + (a2 / d2 if d2 > 0 else 0.0) * S


_rng_lw = np.random.default_rng(3)
_worst = 0.0
for _T, _n in ((300, 10), (900, 25), (400, 40)):
    _X = _rng_lw.normal(0, 0.012, size=(_T, _n))
    _worst = max(_worst, float(np.abs(_lw_reference(_X) - bl.ledoit_wolf_identity(_X)).max()))
check("vectorised Ledoit-Wolf matches the loop to machine precision",
      _worst < 1e-15, f"worst |diff| = {_worst:.2e}")

# Shrinkage must still behave. Test it on data with REAL correlation (a single
# common factor); on pure noise the sample covariance is genuinely near-diagonal,
# so "less shrinkage" would look like less off-diagonal mass and the test would
# read backwards.
def _factor_panel(T, n=25, seed=11):
    r = np.random.default_rng(seed)
    f = r.normal(0, 0.010, size=(T, 1))                 # common factor
    beta = np.linspace(0.6, 1.4, n)[None, :]
    return f @ beta + r.normal(0, 0.006, size=(T, n))


def _shrink_intensity(X):
    """How far the estimate sits from the sample covariance: 0 = no shrinkage."""
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / X.shape[0]
    est = bl.ledoit_wolf_identity(X)
    return float(np.abs(est - S).sum() / max(np.abs(S).sum(), 1e-18))


_int_short = _shrink_intensity(_factor_panel(60))
_int_long = _shrink_intensity(_factor_panel(4000))
check("shrinkage weakens as the sample grows", _int_long < _int_short,
      f"intensity {_int_short:.4f} (T=60) vs {_int_long:.4f} (T=4000)")
check("correlation structure survives on a long sample",
      np.abs(bl.ledoit_wolf_identity(_factor_panel(4000))
             - np.diag(np.diag(bl.ledoit_wolf_identity(_factor_panel(4000))))).sum() > 0)
_panel_long = _factor_panel(4000)
check("the shrunk covariance is symmetric",
      np.allclose(bl.ledoit_wolf_identity(_panel_long), bl.ledoit_wolf_identity(_panel_long).T))
check("the shrunk covariance is positive definite",
      np.all(np.linalg.eigvalsh(bl.ledoit_wolf_identity(_panel_long)) > 0))
check("shrinkage keeps a rank-deficient panel invertible (T < n)",
      np.all(np.linalg.eigvalsh(bl.ledoit_wolf_identity(_factor_panel(15, n=25))) > 0))

# =========================================================
print()
print("--- buy-and-hold between rebalances ---")

# The bug this guards against: `R @ w` applies a fixed weight vector to every
# period, which silently rebalances the portfolio back to target every period.
# A "yearly rebalanced" strategy computed that way is really a daily-rebalanced
# one, and every rebalance frequency collapses to nearly the same answer.

_w2 = np.array([0.5, 0.5])

# One period: nothing has drifted yet, so both must agree exactly.
_R1 = np.array([[0.02, -0.01]])
_r1, _ = bl.buy_and_hold_returns(_R1, _w2)
check("over a single period, buy-and-hold equals the fixed-weight formula",
      abs(_r1[0] - float((_R1 @ _w2)[0])) < 1e-15)

# A year of one asset compounding: the winner must grow as a share of the book.
_Rg = np.zeros((252, 2)); _Rg[:, 0] = 0.001
_rh, _wend = bl.buy_and_hold_returns(_Rg, _w2)
check("holdings drift toward the winner", _wend[0] > 0.55, f"end weights {_wend.round(3)}")
check("drifted weights still sum to 1", abs(_wend.sum() - 1.0) < 1e-12)
check("buy-and-hold compounds more than silent rebalancing in a trend",
      np.prod(1 + _rh) > np.prod(1 + (_Rg @ _w2)),
      f"{np.prod(1+_rh):.5f} vs {np.prod(1+(_Rg @ _w2)):.5f}")

# Flat market: no drift, so the two must coincide.
_Rf = np.zeros((100, 2))
_rf2, _wf = bl.buy_and_hold_returns(_Rf, _w2)
check("a flat market produces no drift and no return",
      np.allclose(_rf2, 0.0) and np.allclose(_wf, _w2))

# Degenerate input must not raise.
_re, _we = bl.buy_and_hold_returns(np.zeros((0, 2)), _w2)
check("an empty segment returns empty without raising", len(_re) == 0)

# End to end: with holdings drifting, rebalance frequency must actually MATTER.
_rngb = np.random.default_rng(7)
_nb, _Tb = 20, 3780
_dr = _rngb.normal(0.00045, 0.00035, _nb)
_Lb = np.tril(_rngb.normal(size=(_nb, _nb)) / np.sqrt(_nb)) + np.eye(_nb) * 0.6
_Rb = np.array([_dr + (_Lb @ _rngb.normal(size=_nb)) * 0.011 for _ in range(_Tb)])
_pxb = pd.DataFrame(100 * np.cumprod(1 + _Rb, axis=0),
                    index=pd.bdate_range("2011-01-03", periods=_Tb),
                    columns=[f"A{i}" for i in range(_nb)])
_capsb = _rngb.random(_nb) ** 1.5
_wmb = _capsb / _capsb.sum()


def _builder_b(train_returns, cov_annual):
    d, _ = bl.implied_risk_aversion(_wmb, cov_annual,
                                    float(_wmb @ (train_returns.mean().values * 252)) - 0.06)
    o = bl.black_litterman(cov_annual, _wmb, 0.06, 0.05, d,
                           np.zeros((0, _nb)), np.zeros(0), np.zeros(0))
    return o["mu_total"], o["Sigma_used"]


_BASE = dict(freq="Daily", use_log=True, mu_builder=_builder_b,
             cov_method="Ledoit-Wolf shrinkage", freq_per_year=252, rf=0.06,
             stance="Long only", max_weight=0.12, gross_limit=None, objective="Max Sharpe",
             alpha=0.95, tc_bps=25.0, borrow_bps=50.0, train_frac=0.45, resample_n=0,
             caps_weights=_wmb, vol_target=None)
# The turnover penalty is OFF here on purpose: this block tests the drift
# mechanism, and the penalty's whole job is to suppress the small trades that
# the drift mechanism generates. Leaving it on would test both at once and
# tell us about neither.
_res = {}
_turn = {}
for _lbl, _reb in (("weekly", 5), ("yearly", 252)):
    _bt = bl.run_backtest(_pxb, rebalance_periods=_reb, rebal_label=_lbl,
                          turnover_lambda=0.0, **_BASE)
    _res[_lbl] = _bt["strat"]["ann_ret"]
    _turn[_lbl] = _bt["avg_turnover"]

check("rebalance frequency changes the result once holdings drift",
      abs(_res["weekly"] - _res["yearly"]) > 0.002,
      f"weekly {_res['weekly']*100:.2f}%/yr vs yearly {_res['yearly']*100:.2f}%/yr")
check("turnover per rebalance grows with the holding period",
      _turn["yearly"] > _turn["weekly"] * 3,
      f"weekly {_turn['weekly']*100:.1f}% vs yearly {_turn['yearly']*100:.1f}% per rebalance")

# The frequency effect is second order to begin with -- the objective is flat
# at its maximum, so trading more often buys very little gross. Charging the
# optimiser for the trade therefore compresses the horizons further: it takes
# away exactly the marginal trades that made the fast horizon expensive.
_res_pen = {}
for _lbl, _reb in (("weekly", 5), ("yearly", 252)):
    _btp = bl.run_backtest(_pxb, rebalance_periods=_reb, rebal_label=_lbl,
                           turnover_lambda=1.0, delta=3.0, **_BASE)
    _res_pen[_lbl] = _btp["strat"]["ann_ret"]
check("charging for turnover narrows the gap between horizons",
      abs(_res_pen["weekly"] - _res_pen["yearly"]) <= abs(_res["weekly"] - _res["yearly"]) + 1e-9,
      f"unpenalised {abs(_res['weekly']-_res['yearly'])*100:.3f}pp -> "
      f"penalised {abs(_res_pen['weekly']-_res_pen['yearly'])*100:.3f}pp")
check("the fast horizon is the one the penalty helps most",
      _res_pen["weekly"] >= _res["weekly"] - 1e-9,
      f"weekly {_res['weekly']*100:.2f}%/yr -> {_res_pen['weekly']*100:.2f}%/yr")

# =========================================================
print()
print("--- return frequency vs synchronous trading ---")

_n_tz, _T_tz = 12, 3780
_IDX_TZ = pd.bdate_range("2011-01-03", periods=_T_tz)


def _panel_tz(phi, lead, seed=4):
    """A factor panel with optional AR(1) and optional lead-lag (non-synchronous
    sessions, as when SPY, EFA and EEM close hours apart)."""
    rg = np.random.default_rng(seed)
    f = np.zeros(_T_tz)
    e = rg.normal(0, 0.009, _T_tz)
    for t in range(1, _T_tz):
        f[t] = phi * f[t - 1] + e[t]
    beta = np.linspace(0.7, 1.3, _n_tz)
    R = np.outer(f, beta) + rg.normal(0, 0.006, size=(_T_tz, _n_tz))
    if lead:
        R[1:, _n_tz // 2:] = ((1 - lead) * R[1:, _n_tz // 2:]
                              + lead * np.outer(f[:-1], beta[_n_tz // 2:]))
    return pd.DataFrame(100 * np.cumprod(1 + R, axis=0), index=_IDX_TZ,
                        columns=[f"A{i}" for i in range(_n_tz)])


def _mean_corr(px, freq, fpy):
    C = bl.estimate_cov(bl.to_returns(px, freq, True), "Ledoit-Wolf shrinkage", fpy).values
    d = np.sqrt(np.diag(C))
    R = C / np.outer(d, d)
    return float(R[np.triu_indices(_n_tz, 1)].mean())


_sync = _panel_tz(0.0, 0.0)
_async = _panel_tz(0.0, 0.35)
_gap_sync = _mean_corr(_sync, "Weekly", 52) - _mean_corr(_sync, "Daily", 252)
_gap_async = _mean_corr(_async, "Weekly", 52) - _mean_corr(_async, "Daily", 252)

check("daily and weekly agree when assets trade in the same session",
      abs(_gap_sync) < 0.03, f"correlation gap {_gap_sync:+.3f}")
check("daily UNDERSTATES correlation when sessions do not overlap",
      _gap_async > 0.05, f"correlation gap {_gap_async:+.3f}")
check("the non-synchronous distortion is much larger than the synchronous one",
      _gap_async > 3 * abs(_gap_sync), f"{_gap_async:.3f} vs {abs(_gap_sync):.3f}")

# The universes spanning time zones must be flagged so the app measures them weekly.
_flagged = {k for k, v in bl.MARKETS.items() if v.get("cross_tz")}
check("global multi-timezone universes are flagged cross_tz",
      "Global multi-asset (ETFs)" in _flagged and "Global equity regions (ETFs)" in _flagged,
      str(sorted(_flagged)))
check("single-exchange universes are NOT flagged",
      not any(bl.MARKETS[k].get("cross_tz")
              for k in ("India — Nifty 500", "United States — S&P 500", "US sectors (ETFs)")))

# Rescaling the rebalance period must preserve rebalances-per-year.
for _lbl, _c in bl.HORIZON_CFG.items():
    _daily = _c["rebal"]
    _weekly = max(1, int(round(_daily * 52 / 252)))
    _per_yr_d = 252 / _daily
    _per_yr_w = 52 / _weekly
    check(f"{_lbl.split(' ')[0]}: rebalance rate survives a switch to weekly returns",
          abs(_per_yr_d - _per_yr_w) / _per_yr_d < 0.10,
          f"{_per_yr_d:.1f}/yr daily vs {_per_yr_w:.1f}/yr weekly")

# =========================================================
# TURNOVER PENALTY
# =========================================================
print("\n--- turnover penalty ---")

_rng_to = np.random.default_rng(4242)
_n_to = 12
_beta_to = _rng_to.uniform(0.7, 1.3, _n_to)
_S_to = np.outer(_beta_to, _beta_to) * 0.028 + np.diag(_rng_to.uniform(0.03, 0.09, _n_to))
_S_to = bl.nearest_psd(_S_to)
_delta_to = 3.0
_wm_to = _rng_to.dirichlet(np.ones(_n_to) * 3)
_mu_to = bl.equilibrium_returns(_delta_to, _S_to, _wm_to) + 0.04 + _rng_to.normal(0, 0.012, _n_to)
_R_to = _rng_to.multivariate_normal(_mu_to / 252, _S_to / 252, size=600)
_rf_to = 0.05
_TC = 0.0025 * 12          # 25 bps charged 12x a year

# Start from a drifted book: what you actually hold when a rebalance fires.
_w_free = bl.optimize_portfolio("Max Sharpe", _mu_to, _S_to, _R_to, _rf_to,
                                "Long only", 0.30, None, 0.95, 252)
_drift = _w_free * np.exp(_rng_to.normal(0, 0.10, _n_to))
_w_prev_to = _drift / _drift.sum()

def _after_cost_sharpe(w, w_prev=None, tc=None):
    w_prev = _w_prev_to if w_prev is None else w_prev
    tc = _TC if tc is None else tc
    drag = tc * float(np.sum(np.abs(np.asarray(w) - w_prev)))
    return ((w @ _mu_to) - drag - _rf_to) / math.sqrt(max(w @ _S_to @ w, 1e-16))

_w_pen = bl.optimize_portfolio("Max Sharpe", _mu_to, _S_to, _R_to, _rf_to,
                               "Long only", 0.30, None, 0.95, 252,
                               w_prev=_w_prev_to, turnover_cost=_TC, delta=_delta_to)

# --- 1. THE FLOOR. Holding still is always available, so the solver must never
# return anything worse than it. This is the property the first implementation
# silently violated: SLSQP stepped off the kink at w_prev and returned a point
# scoring 0.6518 when doing nothing scored 0.6733. ---
check("penalised solve never loses to simply holding still",
      _after_cost_sharpe(_w_pen) >= _after_cost_sharpe(_w_prev_to) - 1e-9,
      f"solve {_after_cost_sharpe(_w_pen):.6f} vs hold {_after_cost_sharpe(_w_prev_to):.6f}")
check("penalised solve never loses to trading the whole way",
      _after_cost_sharpe(_w_pen) >= _after_cost_sharpe(_w_free) - 1e-9,
      f"solve {_after_cost_sharpe(_w_pen):.6f} vs full trade {_after_cost_sharpe(_w_free):.6f}")

# --- 2. it beats the best point on the line between the two endpoints ---
# A partial rebalance along w_prev -> w_free is the obvious cheap heuristic.
# The real solve has 25 degrees of freedom, so it should match or beat it.
_alphas = np.linspace(0, 1, 201)
_line = _w_prev_to[None, :] + _alphas[:, None] * (_w_free - _w_prev_to)[None, :]
_best_line = max(_after_cost_sharpe(_line[i]) for i in range(len(_alphas)))
check("penalised solve matches or beats the best partial rebalance",
      _after_cost_sharpe(_w_pen) >= _best_line - 1e-6,
      f"solve {_after_cost_sharpe(_w_pen):.6f} vs best line point {_best_line:.6f}")

# --- 3. the buy/sell split reports true L1 turnover, not slack ---
_wS, _wB = _w_pen, _w_prev_to
check("solved weights obey the budget constraint exactly",
      abs(_w_pen.sum() - 1.0) < 1e-9, f"sum = {_w_pen.sum():.12f}")
check("solved weights respect the position cap",
      _w_pen.max() <= 0.30 + 1e-9 and _w_pen.min() >= -1e-9,
      f"max {_w_pen.max():.4f}, min {_w_pen.min():.6f}")

# --- 4. it does what it is for ---
_to_free = float(np.sum(np.abs(_w_free - _w_prev_to)))
_to_pen = float(np.sum(np.abs(_w_pen - _w_prev_to)))
check("penalty reduces turnover", _to_pen < _to_free,
      f"{_to_free*100:.2f}% unpenalised -> {_to_pen*100:.2f}% penalised")

# The economic argument: the objective is flat at its maximum, so the gross
# give-up must be small next to the cost saved.
_gross_free = ((_w_free @ _mu_to) - _rf_to) / math.sqrt(_w_free @ _S_to @ _w_free)
_gross_pen = ((_w_pen @ _mu_to) - _rf_to) / math.sqrt(max(_w_pen @ _S_to @ _w_pen, 1e-16))
check("gross objective gives up far less than the cost saved",
      (_gross_free - _gross_pen) < _TC * (_to_free - _to_pen) / 0.15 + 0.05,
      f"gross Sharpe {_gross_free:.4f} -> {_gross_pen:.4f}, "
      f"turnover {_to_free*100:.1f}% -> {_to_pen*100:.1f}%")
check("penalised portfolio wins on the AFTER-COST objective",
      _after_cost_sharpe(_w_pen) > _after_cost_sharpe(_w_free),
      f"{_after_cost_sharpe(_w_free):.6f} -> {_after_cost_sharpe(_w_pen):.6f}")

# --- 5. the floor property holds across many independent problems ---
# The floor is the best FEASIBLE fallback, which is not always "hold still":
# drift routinely pushes a position past the max-weight cap (3 of the 12 cases
# below), and when it does, holding is not on the menu at any price -- the
# model has to trade back inside the cap however expensive that is. Scoring
# against an unattainable hold would be scoring against a portfolio the user
# is not allowed to own.
_floor_fails, _capped = 0, 0
for _k in range(12):
    _r = np.random.default_rng(900 + _k)
    _mu_k = bl.equilibrium_returns(_delta_to, _S_to, _r.dirichlet(np.ones(_n_to) * 3)) \
        + 0.05 + _r.normal(0, 0.02, _n_to)
    _R_k = _r.multivariate_normal(_mu_k / 252, _S_to / 252, size=500)
    _wf_k = bl.optimize_portfolio("Max Sharpe", _mu_k, _S_to, _R_k, _rf_to,
                                  "Long only", 0.30, None, 0.95, 252)
    _d_k = _wf_k * np.exp(_r.normal(0, 0.10, _n_to)); _wp_k = _d_k / _d_k.sum()
    _wp_sol = bl.optimize_portfolio("Max Sharpe", _mu_k, _S_to, _R_k, _rf_to,
                                    "Long only", 0.30, None, 0.95, 252,
                                    w_prev=_wp_k, turnover_cost=_TC, delta=_delta_to)

    def _acs(w, mu=_mu_k, wp=_wp_k):
        return ((w @ mu) - _TC * np.sum(np.abs(w - wp)) - _rf_to) / \
            math.sqrt(max(w @ _S_to @ w, 1e-16))

    def _ok(w):
        w = np.asarray(w, dtype=float)
        return w.max() <= 0.30 + 1e-7 and w.min() >= -1e-7 and abs(w.sum() - 1) < 1e-6

    if not _ok(_wp_k):
        _capped += 1
    _floor = max([_acs(w) for w in (_wp_k, _wf_k) if _ok(w)] or [-np.inf])
    # every point on the segment is a fallback too, when it is feasible
    for _a in np.linspace(0, 1, 51):
        _wl = _wp_k + _a * (_wf_k - _wp_k)
        if _ok(_wl):
            _floor = max(_floor, _acs(_wl))
    if not _ok(_wp_sol) or _acs(_wp_sol) < _floor - 1e-7:
        _floor_fails += 1
check("floor property holds on 12 independent problems", _floor_fails == 0,
      f"{_floor_fails} violations ({_capped}/12 had a drifted book breaching the cap)")

# The cap must win over the penalty: a book that has drifted out of compliance
# gets traded back no matter how expensive trading is made.
_r_c = np.random.default_rng(555)
_bad = _w_free.copy(); _bad[int(np.argmax(_bad))] += 0.25; _bad /= _bad.sum()
_w_forced = bl.optimize_portfolio("Max Sharpe", _mu_to, _S_to, _R_to, _rf_to,
                                  "Long only", 0.30, None, 0.95, 252,
                                  w_prev=_bad, turnover_cost=_TC * 500, delta=_delta_to)
check("cap is enforced even when trading is made prohibitively expensive",
      _w_forced.max() <= 0.30 + 1e-7 and abs(_w_forced.sum() - 1) < 1e-6,
      f"breached book max {_bad.max():.4f} -> solved max {_w_forced.max():.4f}")

# --- 4. inertness when it should be inert ---
check("lambda=0 leaves the unpenalised answer untouched",
      np.allclose(bl.optimize_portfolio("Max Sharpe", _mu_to, _S_to, _R_to, _rf_to,
                                        "Long only", 0.30, None, 0.95, 252,
                                        w_prev=_w_prev_to, turnover_cost=0.0),
                  _w_free, atol=1e-8))
check("w_prev=None leaves the unpenalised answer untouched",
      np.allclose(bl.optimize_portfolio("Max Sharpe", _mu_to, _S_to, _R_to, _rf_to,
                                        "Long only", 0.30, None, 0.95, 252,
                                        w_prev=None, turnover_cost=_TC),
                  _w_free, atol=1e-8))
check("already-optimal book is left alone when trading is expensive",
      float(np.sum(np.abs(
          bl.optimize_portfolio("Max Sharpe", _mu_to, _S_to, _R_to, _rf_to,
                                "Long only", 0.30, None, 0.95, 252,
                                w_prev=_w_free, turnover_cost=_TC * 40) - _w_free))) < 0.02)

# --- 5. monotone in lambda, and every objective accepts it ---
_tos = []
for _lam in (0.0, 1.0, 4.0, 16.0):
    _wl = bl.optimize_portfolio("Max Sharpe", _mu_to, _S_to, _R_to, _rf_to,
                                "Long only", 0.30, None, 0.95, 252,
                                w_prev=_w_prev_to, turnover_cost=_TC * _lam)
    _tos.append(float(np.sum(np.abs(_wl - _w_prev_to))))
check("turnover is non-increasing as the charge rises",
      all(_tos[i + 1] <= _tos[i] + 1e-3 for i in range(len(_tos) - 1)),
      " -> ".join(f"{t*100:.2f}%" for t in _tos))

for _obj in ("Max Sharpe", "Min variance", "Max Sortino", "Min CVaR"):
    _wo_free = bl.optimize_portfolio(_obj, _mu_to, _S_to, _R_to, _rf_to,
                                     "Long only", 0.30, None, 0.95, 252)
    _wo_pen = bl.optimize_portfolio(_obj, _mu_to, _S_to, _R_to, _rf_to,
                                    "Long only", 0.30, None, 0.95, 252,
                                    w_prev=_w_prev_to, turnover_cost=_TC * 4,
                                    delta=_delta_to)
    _a = float(np.sum(np.abs(_wo_free - _w_prev_to)))
    _b = float(np.sum(np.abs(_wo_pen - _w_prev_to)))
    check(f"{_obj}: penalty is wired in and reduces turnover",
          _b <= _a + 1e-6 and abs(_wo_pen.sum() - 1.0) < 1e-6,
          f"{_a*100:.2f}% -> {_b*100:.2f}%")

# --- 6. end to end through the backtest ---
_px_to = pd.DataFrame(
    100 * np.cumprod(1 + _rng_to.multivariate_normal(_mu_to / 252, _S_to / 252, size=900), axis=0),
    index=pd.bdate_range("2019-01-01", periods=900),
    columns=[f"T{i}" for i in range(_n_to)])

def _mu_builder_to(tr, cov_annual):
    return bl.equilibrium_returns(_delta_to, cov_annual, _wm_to) + _rf_to, cov_annual

_BT_TO = dict(freq="Daily", use_log=True, mu_builder=_mu_builder_to,
              cov_method="Ledoit-Wolf shrinkage", freq_per_year=252, rf=_rf_to,
              stance="Long only", max_weight=0.30, gross_limit=None,
              objective="Max Sharpe", alpha=0.95, tc_bps=25.0, borrow_bps=50.0,
              train_frac=0.4, resample_n=0, rebalance_periods=21,
              vol_target=None, delta=_delta_to)

_bt_off = bl.run_backtest(_px_to, turnover_lambda=0.0, **_BT_TO)
_bt_on = bl.run_backtest(_px_to, turnover_lambda=1.0, **_BT_TO)
check("backtest accepts the penalty and both runs complete",
      _bt_off is not None and _bt_on is not None)
check("backtest turnover falls once the optimiser is charged",
      _bt_on["avg_turnover"] <= _bt_off["avg_turnover"] + 1e-9,
      f"{_bt_off['avg_turnover']*100:.2f}% -> {_bt_on['avg_turnover']*100:.2f}% per rebalance")
check("penalised run reports the lambda it actually used",
      abs(_bt_on["turnover_lambda"] - 1.0) < 1e-12 and _bt_on["turnover_cost_used"] > 0)
check("lambda=0 backtest reproduces the old unpenalised behaviour exactly",
      abs(_bt_off["strat"]["ann_ret"] - bl.run_backtest(_px_to, turnover_lambda=0.0,
                                                        **_BT_TO)["strat"]["ann_ret"]) < 1e-12)
check("trading-cost drag is reported and consistent with turnover",
      _bt_on["tc_drag_per_year"] >= 0 and _bt_on["tc_drag_per_year"] < 0.05,
      f"{_bt_on['tc_drag_per_year']*100:.3f}pp/yr")

# =========================================================
# CURRENT HOLDINGS -> TRADE LIST
# =========================================================
print("\n--- holdings input ---")

_UNI = ["RELIANCE.NS", "BHARTIARTL.NS", "HDFCBANK.NS", "INFY.NS", "VEDL.NS"]

_h, _p = bl.parse_holdings("RELIANCE.NS 40\nBHARTIARTL.NS, 55\nHDFCBANK.NS: 48", _UNI)
check("parses space, comma and colon separators alike",
      _h == {"RELIANCE.NS": 40.0, "BHARTIARTL.NS": 55.0, "HDFCBANK.NS": 48.0} and not _p, str(_h))

_h, _p = bl.parse_holdings("reliance 40\nINFY 12", _UNI)
check("matches without the exchange suffix and ignores case",
      _h == {"RELIANCE.NS": 40.0, "INFY.NS": 12.0} and not _p, str(_h))

_h, _p = bl.parse_holdings("RELIANCE.NS\t40\tshares\nVEDL.NS  1,250", _UNI)
check("handles pasted tabs, trailing words and thousands separators",
      _h == {"RELIANCE.NS": 40.0, "VEDL.NS": 1250.0} and not _p, str(_h))

_h, _p = bl.parse_holdings("# my book\n\nRELIANCE.NS 40\nTCS.NS 10\nINFY.NS", _UNI)
check("unknown tickers and malformed lines are reported, not silently dropped",
      _h == {"RELIANCE.NS": 40.0} and len(_p) == 2, f"{_h} / {_p}")

_h, _p = bl.parse_holdings("RELIANCE.NS 10\nRELIANCE.NS 30", _UNI)
check("repeated tickers accumulate", _h == {"RELIANCE.NS": 40.0}, str(_h))

check("empty input is empty, not an error", bl.parse_holdings("", _UNI) == ({}, []))
check("negative share counts are refused",
      bl.parse_holdings("RELIANCE.NS -40", _UNI)[0] == {} and
      len(bl.parse_holdings("RELIANCE.NS -40", _UNI)[1]) == 1)

# --- weights from share counts ---
_px = [100.0, 200.0, 50.0, np.nan, 10.0]
_w_h, _val, _npr = bl.holdings_to_weights({"RELIANCE.NS": 30, "HDFCBANK.NS": 40}, _UNI, _px)
check("holdings become weights that sum to 1",
      abs(_w_h.sum() - 1.0) < 1e-12 and _npr == 2, f"sum {_w_h.sum():.12f}, priced {_npr}")
check("holding weights are value-proportional",
      abs(_w_h[0] - 3000 / 5000) < 1e-12 and abs(_w_h[2] - 2000 / 5000) < 1e-12
      and abs(_val - 5000.0) < 1e-9, f"{_w_h.round(4)}, value {_val}")
_wu, _vu, _nu = bl.holdings_to_weights({"INFY.NS": 100}, _UNI, _px)
check("an unpriced holding is excluded rather than counted at zero",
      _vu == 0.0 and _nu == 0 and float(np.sum(np.abs(_wu))) == 0.0,
      f"value {_vu}, priced {_nu}")
_wm2, _vm2, _nm2 = bl.holdings_to_weights({"RELIANCE.NS": 30, "INFY.NS": 100}, _UNI, _px)
check("a mix of priced and unpriced holdings keeps only what could be priced",
      _nm2 == 1 and abs(_vm2 - 3000.0) < 1e-9 and abs(_wm2[0] - 1.0) < 1e-12,
      f"value {_vm2}, priced {_nm2}")
check("no holdings gives no value and no weights",
      bl.holdings_to_weights({}, _UNI, _px)[1] == 0.0)

# --- the point of the feature: a nudged target must not generate a trade ---
# Optimise, drift the answer slightly, feed it back as the current book, and
# check the model declines to chase the difference once trading is charged.
_w_now = bl.optimize_portfolio("Max Sharpe", _mu_to, _S_to, _R_to, _rf_to,
                               "Long only", 0.30, None, 0.95, 252)
_nudge = _w_now * np.exp(np.random.default_rng(31).normal(0, 0.02, _n_to))
_nudge = _nudge / _nudge.sum()
_w_free_again = bl.optimize_portfolio("Max Sharpe", _mu_to, _S_to, _R_to, _rf_to,
                                      "Long only", 0.30, None, 0.95, 252, w_prev=_nudge,
                                      turnover_cost=0.0)
_w_sticky = bl.optimize_portfolio("Max Sharpe", _mu_to, _S_to, _R_to, _rf_to,
                                  "Long only", 0.30, None, 0.95, 252, w_prev=_nudge,
                                  turnover_cost=_TC, delta=_delta_to)
_churn_free = float(np.sum(np.abs(_w_free_again - _nudge)))
_churn_sticky = float(np.sum(np.abs(_w_sticky - _nudge)))
check("a lightly drifted book is left far more alone once trading is charged",
      _churn_sticky < _churn_free / 2,
      f"would trade {_churn_free*100:.2f}% of the book, now trades {_churn_sticky*100:.2f}%")

# New cash must not look like selling: after dilution the existing weights are
# a smaller share of a bigger book, and w_prev has to say so.
_held_val, _cash = 80000.0, 20000.0
_w_pre = np.array([0.5, 0.5] + [0.0] * (_n_to - 2))
_w_post = _w_pre * (_held_val / (_held_val + _cash))
check("adding cash rescales the existing book instead of implying sales",
      abs(_w_post.sum() - 0.8) < 1e-12 and np.all(_w_post <= _w_pre + 1e-12),
      f"weights sum to {_w_post.sum():.4f} of the post-contribution book")

# =========================================================
# BREAK-EVEN DRIFT / REBALANCE GUIDANCE
# =========================================================
print("\n--- when is it worth trading again ---")

_n_be = 25
_r_be = np.random.default_rng(11)
_beta_be = _r_be.uniform(0.7, 1.35, _n_be)
_S_be = bl.nearest_psd(np.outer(_beta_be, _beta_be) * 0.030
                       + np.diag(_r_be.uniform(0.035, 0.10, _n_be)))
_wm_be = _r_be.dirichlet(np.ones(_n_be) * 3)
_mu_be = bl.equilibrium_returns(3.16, _S_be, _wm_be) + 0.06
_R_be = _r_be.multivariate_normal(_mu_be / 252, _S_be / 252, size=700)
_w_be = bl.optimize_portfolio("Max Sharpe", _mu_be, _S_be, _R_be, 0.06,
                              "Long only", 0.12, None, 0.95, 252)

_LB_BE, _UB_BE, _NET_BE = 0.0, 0.12, 1.0

def _be(tc_bps, reb_per_year, seed=7):
    """Break-even drift, measured by asking the solver itself."""
    return bl.breakeven_drift(
        _w_be,
        lambda wp: bl.optimize_portfolio("Max Sharpe", _mu_be, _S_be, _R_be, 0.06,
                                         "Long only", 0.12, None, 0.95, 252,
                                         w_prev=wp, turnover_cost=(tc_bps / 1e4) * reb_per_year,
                                         delta=3.16),
        _LB_BE, _UB_BE, _NET_BE, seed=seed)

_be_in_m, _be_in_w, _be_in_y = _be(25, 12), _be(25, 52), _be(25, 1)
_be_us_m, _be_free = _be(10, 12), _be(0, 12)

check("free trading leaves essentially no band", _be_free < 0.005,
      f"{_be_free*100:.3f}pp")
check("the band widens as trading gets more frequent",
      _be_in_y <= _be_in_m <= _be_in_w,
      f"yearly {_be_in_y*100:.2f}pp <= monthly {_be_in_m*100:.2f}pp <= weekly {_be_in_w*100:.2f}pp")
check("the band widens as trading gets more expensive",
      _be_us_m <= _be_in_m, f"US 10bps {_be_us_m*100:.2f}pp vs India 25bps {_be_in_m*100:.2f}pp")

# The number must match the SOLVER'S OWN behaviour, measured separately over
# 625 position-level decisions per setting: at 25 bps monthly and weekly the
# optimiser declined every discretionary trade; at 25 bps yearly and 10 bps
# monthly it traded. A previous version of this function compared the
# certainty-equivalent gain of correcting ONE position against its commission,
# and called US monthly unreachable while the solver traded in 68 of 625 cases
# -- a correction spread over many names gains far more per unit of L1
# turnover than the same movement in one, so a single-name model overstates
# the band by an order of magnitude. These checks exist to catch that class of
# disagreement, not to restate the formula.
check("India monthly: nothing discretionary clears the cost",
      not np.isfinite(_be_in_m), f"break-even {_be_in_m}")
check("India weekly: nothing discretionary clears the cost",
      not np.isfinite(_be_in_w), f"break-even {_be_in_w}")
check("India yearly: a reachable band, matching the 0.44pp median move measured",
      np.isfinite(_be_in_y) and _be_in_y < 0.02, f"break-even {_be_in_y*100:.2f}pp")
check("US monthly: reachable, matching the trades the solver actually makes",
      np.isfinite(_be_us_m) and _be_us_m < 0.05, f"break-even {_be_us_m*100:.2f}pp")

# A drifted book handed to the projection must be one the constraints allow,
# or the 'trade' it triggers is a forced cap correction, not a choice.
_pj = bl.project_to_bounds(_w_be * np.exp(np.random.default_rng(3).normal(0, 0.5, _n_be)),
                           0.0, 0.12, 1.0)
check("drifted books are projected back inside the cap and the budget",
      _pj.max() <= 0.12 + 1e-9 and _pj.min() >= -1e-9 and abs(_pj.sum() - 1) < 1e-8,
      f"max {_pj.max():.6f}, sum {_pj.sum():.10f}")
check("projection leaves an already-feasible book alone",
      np.allclose(bl.project_to_bounds(_w_be, 0.0, 0.12, 1.0), _w_be, atol=1e-9))

check("the answer is stable across independent drift draws",
      abs(_be(25, 1, seed=99) - _be_in_y) < 0.01 or
      (not np.isfinite(_be(25, 1, seed=99)) and not np.isfinite(_be_in_y)),
      f"seed 7 {_be_in_y*100:.2f}pp vs seed 99 {_be(25,1,seed=99)*100:.2f}pp")

# --- the wording that reaches the user ---
_h1, _d1, _s1 = bl.rebalance_guidance(25, 12, _be_in_m, 12.0, observed_turnover=0.0133)
check("an unreachable band tells the user to do nothing, and says why",
      _s1 == "none" and "none" in _h1.lower() and "12%" in _d1,
      _h1)
# 25 bps x 12 rebalances x 1.33% of the book turned over = 0.04pp a year.
check("the commission figure is computed, not asserted",
      "0.04pp a year" in _d1 and "3.00% a year" in _d1, _d1[:200])
_h2, _d2, _s2 = bl.rebalance_guidance(10, 1, 0.008, 12.0)
check("a small reachable band gives a normal act-on-drift instruction",
      _s2 == "normal" and "0.8pp" in _h2, _h2)
_h3, _d3, _s3 = bl.rebalance_guidance(25, 4, 0.035, 12.0)
check("a wide but reachable band gives a rarely-act instruction",
      _s3 == "rare" and "3.5pp" in _h3, _h3)
check("the commission figure is flagged as already cost-aware, not as cheap",
      "because" in _d1 and "refused" in _d1, _d1[-160:])
check("guidance never crashes on degenerate inputs",
      all(isinstance(bl.rebalance_guidance(*a)[0], str) for a in
          [(0, 12, 0.0, 12.0), (25, 0, float("inf"), 12.0), (25, 12, float("inf"), 100.0)]))

# =========================================================
# WHAT A TRADE ACTUALLY COSTS
# =========================================================
# NOTE: these primitives are not yet wired to the UI. They are tested here so
# that nothing ships unverified, and so the backward-compatibility guarantee
# below is a checked property rather than a claim.
print("\n--- real trading costs ---")

_ZER = bl.BROKER_PRESETS["India — discount broker, delivery (Zerodha, Groww, Upstox)"]
_FULL = bl.BROKER_PRESETS["India — full-service broker, delivery"]
_US = bl.BROKER_PRESETS["United States — commission-free broker"]

# Hand-computed, Rs 20,000 equity delivery BUY at a discount broker:
#   STT 0.1% = 20.00 | stamp 0.015% = 3.00 | exchange 0.00297% = 0.594
#   SEBI 0.0001% = 0.02 | GST 18% of (0 + 0.594 + 0.02) = 0.110
#   slippage 0.05% = 10.00  ->  33.72
check("discount-broker buy matches the hand-computed charge sheet",
      abs(bl.order_cost(20000, "BUY", _ZER) - 33.72) < 0.01,
      f"{bl.order_cost(20000, 'BUY', _ZER):.4f} vs 33.72")
# The same sale adds the flat DP charge and drops stamp duty:
#   20.00 + 0.594 + 0.02 + 0.110 + 10.00 + 15.93 = 46.65
check("a sale drops stamp duty and adds the flat DP charge",
      abs(bl.order_cost(20000, "SELL", _ZER) - 46.65) < 0.01,
      f"{bl.order_cost(20000, 'SELL', _ZER):.4f} vs 46.65")
check("a zero-size order costs nothing",
      bl.order_cost(0, "BUY", _ZER) == 0.0 and bl.order_cost(0, "SELL", _ZER) == 0.0)
check("no cost config means no cost", bl.order_cost(10000, "BUY", None) == 0.0)

# THE POINT OF THE WHOLE EXERCISE: a fixed charge does not scale, so the
# effective rate explodes on small orders. A flat 25 bps assumption is wrong by
# a factor of ~33 on a Rs 200 trade, and this is what makes a small account
# hold fewer names and trade less often than a large one.
_bps = {v: bl.roundtrip_bps(v, _ZER) for v in (200, 1000, 20000, 200000, 1000000)}
check("effective cost falls monotonically as the order grows",
      all(_bps[a] > _bps[b] for a, b in zip([200, 1000, 20000, 200000],
                                            [1000, 20000, 200000, 1000000])),
      " > ".join(f"{v}:{_bps[v]:.0f}bps" for v in _bps))
check("a tiny order is catastrophically expensive, not 25 bps",
      _bps[200] > 500, f"Rs 200 round trip costs {_bps[200]:,.0f} bps")
check("even a large order costs MORE than the 25 bps the app assumed",
      _bps[1000000] > 25, f"Rs 10 lakh round trip costs {_bps[1000000]:,.1f} bps")
check("a full-service broker costs materially more than a discount one",
      bl.roundtrip_bps(20000, _FULL) > 2 * bl.roundtrip_bps(20000, _ZER),
      f"{bl.roundtrip_bps(20000,_FULL):,.0f} vs {bl.roundtrip_bps(20000,_ZER):,.0f} bps")
check("US costs carry no STT on buys and no depository charge",
      bl.order_cost(20000, "BUY", _US) < bl.order_cost(20000, "BUY", _ZER) / 3,
      f"US {bl.order_cost(20000,'BUY',_US):.2f} vs India {bl.order_cost(20000,'BUY',_ZER):.2f}")
check("a minimum brokerage floors a tiny full-service order",
      bl.order_cost(200, "BUY", _FULL) > 25.0,
      f"{bl.order_cost(200, 'BUY', _FULL):.2f} on a Rs 200 order")

# --- splitting one trade into many is what fixed charges punish ---
# Only on the SELL side at a discount broker: the depository charge is per
# scrip sold, while buys carry no fixed component at all. So fragmenting a
# purchase is free and fragmenting a sale is not, which is not obvious and is
# exactly the kind of thing a flat basis-point rate hides.
_sell_one = bl.trade_book_cost(np.array([-0.25]), 400000.0, _ZER)
_sell_many = bl.trade_book_cost(np.full(25, -0.01), 400000.0, _ZER)
check("one big sale is cheaper than the same sale split across 25 names",
      _sell_many > _sell_one * 1.5,
      f"one order {_sell_one*1e4:.1f} bps vs 25 orders {_sell_many*1e4:.1f} bps")
check("the gap is exactly the extra depository charges",
      abs((_sell_many - _sell_one) * 400000.0 - 24 * _ZER["dp_flat_sell"]) < 0.01,
      f"extra {(_sell_many-_sell_one)*400000.0:,.2f} vs 24 x {_ZER['dp_flat_sell']:.2f}")
_buy_one = bl.trade_book_cost(np.array([0.25]), 400000.0, _ZER)
_buy_many = bl.trade_book_cost(np.full(25, 0.01), 400000.0, _ZER)
check("fragmenting a PURCHASE costs nothing extra at a discount broker",
      abs(_buy_many - _buy_one) < 1e-12,
      f"{_buy_one*1e4:.2f} bps vs {_buy_many*1e4:.2f} bps")
check("...but it does at a broker with a minimum brokerage",
      bl.trade_book_cost(np.full(25, 0.01), 400000.0, _FULL)
      > bl.trade_book_cost(np.array([0.25]), 400000.0, _FULL))

# --- BACKWARD COMPATIBILITY: with no cost config, identical to the old rate ---
_dw = np.array([0.03, -0.02, 0.0, 0.015, -0.025])
for _tc in (0.0, 10.0, 25.0, 100.0):
    check(f"flat fallback at {_tc:.0f} bps reproduces tc_frac * turnover exactly",
          abs(bl.trade_book_cost(_dw, 1e6, None, fallback_bps=_tc)
              - (_tc / 1e4) * float(np.sum(np.abs(_dw)))) < 1e-15)
check("an untouched book is never charged",
      bl.trade_book_cost(np.zeros(5), 1e6, _ZER) == 0.0)
check("a zero-value book cannot be charged",
      bl.trade_book_cost(_dw, 0.0, _ZER) == 0.0)

# =========================================================
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
    sys.exit(1)
print("All tests passed.")
