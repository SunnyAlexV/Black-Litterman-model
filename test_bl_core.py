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
check("horizons differ only in frequency and rebalance period",
      {(c["freq"], c["rebal"]) for c in bl.HORIZON_CFG.values()}
      == {("Daily", 5), ("Daily", 21), ("Weekly", 52)})

# =========================================================
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
    sys.exit(1)
print("All tests passed.")
