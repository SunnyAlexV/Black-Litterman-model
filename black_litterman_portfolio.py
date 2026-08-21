import os
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

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# ---------------------------------------------------------
# Private, app-controlled yfinance cache (kept out of OneDrive / synced
# folders, which can corrupt the shared cache -> "database disk image is
# malformed"). If the shared cache breaks, close the app and delete
# %LOCALAPPDATA%\py-yfinance, then relaunch.
# ---------------------------------------------------------
_YF_CACHE = os.path.join(tempfile.gettempdir(), "yf_cache_black_litterman")
try:
    os.makedirs(_YF_CACHE, exist_ok=True)
    yf.set_tz_cache_location(_YF_CACHE)
except Exception:
    pass

# Quieten yfinance's per-ticker "possibly delisted / no timezone found" chatter —
# failed symbols are retried and then simply skipped, so these are noise.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

FREQ_PER_YEAR = {"Daily": 252, "Weekly": 52, "Monthly": 12}
FREQ_RULE = {"Daily": None, "Weekly": "W-FRI", "Monthly": "ME"}

# Investment horizon -> years of history, return frequency, and how often the
# walk-forward backtest rebalances (in return-periods) + a human label.
# History windows. Longer is better for the COVARIANCE matrix, which is what
# Black-Litterman actually leans on (pi = delta*Sigma*w), and the estimation
# error there falls roughly as 1/sqrt(T). It is much less obviously better for
# anything mean-like: a sample mean's standard error is driven by volatility,
# not sample length, so decades are needed to pin one down — and the further
# back you reach, the more you are averaging across regimes that no longer
# exist. These windows are a compromise: long enough that Sigma is well
# estimated, short enough that the data still describes the current world.
# How long you intend to HOLD and how much history you should ESTIMATE FROM are
# two different questions, and tying them together was the flaw in the earlier
# design: choosing "weekly" quietly cut the estimation window to two years and
# left the covariance matrix — and therefore pi = delta*Sigma*w — running on
# fumes. Wanting to trade weekly is a preference; how much data Sigma needs is
# a statistical fact, and the answer to the second is always "as much as is
# reliably available".
#
# So every horizon now estimates from the same long window. The horizon sets
# only the return frequency and how often the portfolio is rebalanced.
HORIZON_CFG = {
    "Weekly — short-term": {"years": 15, "min_years": 5, "freq": "Daily",
                            "rebal": 5, "rebal_label": "weekly"},
    "Monthly — medium-term": {"years": 15, "min_years": 5, "freq": "Daily",
                              "rebal": 21, "rebal_label": "monthly"},
    # Yearly estimates from DAILY returns too. Tying the return frequency to the
    # holding period was the same mistake as tying the window length to it: a
    # yearly rebalancer gained nothing from throwing away four fifths of its
    # observations (780 weekly instead of 3,780 daily). All three horizons now
    # see identical data and differ only in how often they trade.
    "Yearly — long-term": {"years": 15, "min_years": 5, "freq": "Daily",
                           "rebal": 252, "rebal_label": "yearly"},
}

# A single young ticker must not truncate everyone else. Keep a name only if it
# covers at least this share of the chosen window.
MIN_HISTORY_COVERAGE = 0.95
# Sigma has n(n+1)/2 free parameters; this is the observations-per-asset bar the
# adaptive window aims to clear before it stops lengthening.
TARGET_OBS_PER_ASSET = 25


def choose_history_window(close, candidates, want_n, hcfg, freq_per_year):
    """Pick the longest history window that keeps a full-sized universe.

    Forcing a fixed 15 years would silently discard every asset younger than
    that — half a Nifty universe, in practice. Forcing a short window would
    leave Sigma undernourished. So walk the window down from the maximum and
    stop at the first length that still supports a proper universe and clears
    the observations-per-asset bar.

    Returns (start_timestamp, kept_tickers, note).
    """
    end = close.index.max()
    max_y, min_y = float(hcfg["years"]), float(hcfg["min_years"])
    per_year = freq_per_year
    first_valid = {}
    for t in candidates:
        s = close[t].dropna()
        if not s.empty:
            first_valid[t] = s.index.min()
    if not first_valid:
        return close.index.min(), list(candidates), "no usable history"

    need_n = max(5, int(math.ceil(0.8 * want_n)))
    best = None
    years = max_y
    while years >= min_y - 1e-9:
        start = end - pd.DateOffset(years=years)
        cutoff = start + pd.Timedelta(days=int(365.25 * years * (1 - MIN_HISTORY_COVERAGE)))
        keep = [t for t, d in first_valid.items() if d <= cutoff]
        obs = years * per_year
        if len(keep) >= need_n and (len(keep) == 0 or obs / max(len(keep), 1) >= TARGET_OBS_PER_ASSET):
            best = (start, keep, years)
            break
        if best is None and len(keep) >= need_n:
            best = (start, keep, years)
        years -= 1.0

    if best is None:
        # Nothing clears the bar — take the longest window that keeps the most
        # names, which is what the user would do by hand anyway.
        start = end - pd.DateOffset(years=min_y)
        keep = [t for t, d in first_valid.items() if d <= start + pd.Timedelta(days=180)]
        if len(keep) < 3:
            keep = sorted(first_valid, key=lambda t: first_valid[t])[:max(3, want_n)]
            start = max(first_valid[t] for t in keep)
        best = (start, keep, min_y)

    start, keep, years = best
    keep = [t for t in candidates if t in keep]          # preserve liquidity order
    dropped = [t for t in candidates if t not in keep]
    note = (f"Estimating from **{years:,.0f} years** of history "
            f"({int(years * per_year):,} {hcfg['freq'].lower()} observations, "
            f"~{int(years * per_year / max(len(keep), 1)):,} per asset). ")
    if dropped:
        note += (f"{len(dropped)} name(s) too young for that window were replaced by "
                 f"longer-lived ones: {', '.join(dropped[:6])}"
                 f"{'…' if len(dropped) > 6 else ''}. ")
    return start, keep, note
AUTO_HOLDINGS = 25  # stocks auto-selected in Simple mode

# Market -> (broad index ticker, label) for the "beat the market" benchmark.
BENCHMARKS = {
    "United States — S&P 500": ("^GSPC", "S&P 500"),
    "United Kingdom — FTSE 350": ("^FTSE", "FTSE 100"),
    "Germany — HDAX": ("^GDAXI", "DAX"),
    "France — SBF 120": ("^FCHI", "CAC 40"),
    "Japan — Nikkei 225": ("^N225", "Nikkei 225"),
    "India — Nifty 500": ("^NSEI", "Nifty 50"),
    "Canada — S&P/TSX Composite": ("^GSPTSE", "S&P/TSX"),
    "Australia — S&P/ASX 200": ("^AXJO", "ASX 200"),
    "Hong Kong — Hang Seng Composite": ("^HSI", "Hang Seng"),
}

# Per-market recommended defaults, applied automatically when the market is
# chosen. rf is the local short-dated government yield (the opportunity cost the
# equilibrium is measured against) and tc is a realistic all-in round-trip
# trading cost for liquid large caps in that market. Both are set once here so
# nobody has to remember that India charges STT and the US does not.
#   rf sources: 1-year government bill yields, August 2026.
MARKET_DEFAULTS = {
    "United States — S&P 500":         {"rf": 0.041, "tc_bps": 10.0},
    "United Kingdom — FTSE 350":       {"rf": 0.040, "tc_bps": 20.0},   # incl. 0.5% stamp on buys
    "Germany — HDAX":                  {"rf": 0.023, "tc_bps": 12.0},
    "France — SBF 120":                {"rf": 0.024, "tc_bps": 15.0},   # incl. FTT on large caps
    "Japan — Nikkei 225":              {"rf": 0.008, "tc_bps": 12.0},
    "India — Nifty 500":               {"rf": 0.060, "tc_bps": 25.0},   # incl. STT, stamp, exchange
    "Canada — S&P/TSX Composite":      {"rf": 0.031, "tc_bps": 12.0},
    "Australia — S&P/ASX 200":         {"rf": 0.037, "tc_bps": 15.0},
    "Hong Kong — Hang Seng Composite": {"rf": 0.035, "tc_bps": 22.0},   # incl. stamp duty
}
DEFAULT_RF = 0.04
DEFAULT_TC_BPS = 15.0

# Recommended defaults for the settings that actually change the answer.
DEFAULT_MAX_WEIGHT_PCT = 12     # single-STOCK cap; keeps ~8+ real positions
DEFAULT_ASSET_CLASS_CAP_PCT = 70  # asset classes: the market really is concentrated,
                                  # so a tight cap would forbid holding the prior
DEFAULT_TRAIN_FRAC = 0.45       # leaves a long enough out-of-sample window to mean something
DEFAULT_VIEW_CONFIDENCE = 75.0  # Idzorek confidence: tilt 75% of the way from market to view
DEFAULT_VOL_TARGET_PCT = 12.0        # absolute-mode target, when the user picks a fixed number
DEFAULT_VOL_TARGET_FRAC_PCT = 75     # relative-mode default: run at 75% of the book's own risk


def market_default(market_name, key):
    return MARKET_DEFAULTS.get(market_name, {}).get(
        key, DEFAULT_RF if key == "rf" else DEFAULT_TC_BPS)


def affordable_holdings(last_prices, capital, target_n, min_n=5, lots_per_name=3.0):
    """How many names can this capital actually express?

    A portfolio of 19 names funded with 100,000 INR sounds fine until you try to
    buy it: each position is allocated ~5,000, share prices run 400-2,300, and
    every position rounds down and strands most of a share. In one real run that
    left 18% of the money in cash — a drag larger than anything the optimiser
    was arguing about.

    Rule of thumb used here: each position should be able to buy at least
    `lots_per_name` board lots, so its weight is expressible rather than
    quantised to zero or one share. Returns the largest n <= target_n that
    clears that bar.
    """
    px = np.asarray([p for p in last_prices if np.isfinite(p) and p > 0], dtype=float)
    if px.size == 0 or capital <= 0:
        return target_n
    px = np.sort(px)[::-1]                      # dearest first: the binding ones
    for n in range(int(target_n), int(min_n) - 1, -1):
        if n > px.size:
            continue
        typical = float(np.median(px[:n]))      # median lot cost among the n held
        if capital / n >= lots_per_name * typical:
            return n
    return int(min_n)


def allocate_shares(targets, prices, lots, capital):
    """Turn target cash allocations into whole-share (or whole-lot) counts.

    Flooring every position independently is what strands the cash: each name
    loses up to one lot, and nineteen names lose nineteen part-lots. So we floor
    first, then spend what's left greedily on whichever position sits furthest
    BELOW its target and can still fit another lot.

    This never overspends, and never pushes a position more than one lot past
    its target (once a name reaches target its shortfall goes non-positive and
    it stops being a candidate). Leftover cash ends up smaller than the cheapest
    lot still buyable, instead of the sum of nineteen roundings.

    targets/prices/lots are aligned sequences; only LONG positions are topped up
    (a short's cash mechanics are not the same trade-off). Returns share counts.
    """
    t = np.asarray(targets, dtype=float)
    p = np.asarray(prices, dtype=float)
    L = np.asarray(lots, dtype=float)
    n = len(t)
    shares = np.zeros(n)
    ok = np.isfinite(p) & (p > 0) & np.isfinite(t) & (L > 0)

    # 1) floor every position to whole lots
    for i in range(n):
        if not ok[i]:
            continue
        units = int(math.floor(abs(t[i]) / p[i]))
        shares[i] = (units // int(L[i])) * int(L[i]) * (1.0 if t[i] >= 0 else -1.0)

    # 2) greedily spend the remainder on the most under-filled long positions
    spent = float(np.sum(np.where(shares > 0, shares * p, 0.0)))
    long_target = float(np.sum(np.where(t > 0, t, 0.0)))
    cash = min(capital, long_target) - spent
    for _ in range(10000):                       # bounded: each pass buys one lot
        best_i, best_gap = -1, 0.0
        for i in range(n):
            if not ok[i] or t[i] <= 0:
                continue
            lot_cost = L[i] * p[i]
            if lot_cost > cash + 1e-9:
                continue
            gap = t[i] - shares[i] * p[i]        # how far below target we still are
            if gap > best_gap:
                best_i, best_gap = i, gap
        if best_i < 0:
            break
        shares[best_i] += L[best_i]
        cash -= L[best_i] * p[best_i]
    return shares


def lot_size_for(ticker):
    """Board-lot size for markets that don't trade single shares (approximate)."""
    t = (ticker or "").upper()
    if t.endswith(".T"):
        return 100      # Tokyo: 100-share units
    if t.endswith(".HK"):
        return 100      # Hong Kong: varies by stock; 100 is a common default
    return 1

# =========================================================
# MARKET / INDEX UNIVERSES
# =========================================================
MARKETS = {
    # ---- Asset-class universes -------------------------------------------
    # These fix the deepest flaw in the stock universes below. Black-Litterman
    # needs w to be THE market portfolio; a hand-picked slice of 25 large caps
    # is not one, and picking today's most liquid names and back-testing them
    # imports the answer (survivorship + selection bias). An ETF spanning an
    # asset class has none of that: it existed throughout, and constituent
    # churn is handled inside the fund. This is also the problem Black and
    # Litterman (1992) actually wrote about — global allocation across markets,
    # not stock selection.
    "Global multi-asset (ETFs)": {"currency": "USD", "kind": "asset-class", "tickers": [
        "SPY",    # US large cap equity
        "IWM",    # US small cap equity
        "EFA",    # developed ex-US equity
        "EEM",    # emerging market equity
        "AGG",    # US aggregate bonds
        "TLT",    # US long treasuries
        "IEF",    # US intermediate treasuries
        "LQD",    # investment grade credit
        "HYG",    # high yield credit
        "TIP",    # inflation-linked
        "GLD",    # gold
        "DBC",    # broad commodities
        "VNQ",    # US real estate
    ]},
    "US sectors (ETFs)": {"currency": "USD", "kind": "asset-class", "tickers": [
        "XLK", "XLF", "XLV", "XLY", "XLP", "XLE",
        "XLI", "XLB", "XLU", "XLRE", "XLC",
    ]},
    "Global equity regions (ETFs)": {"currency": "USD", "kind": "asset-class", "tickers": [
        "SPY", "IWM", "EFA", "EEM", "EWJ", "EWU", "EWG", "EWY", "EWZ", "INDA", "FXI", "EWC",
    ]},
    "United States — S&P 500": {"currency": "USD", "tickers": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "BRK-B", "JPM",
        "V", "MA", "UNH", "HD", "PG", "JNJ", "XOM", "CVX", "KO", "PEP",
        "ABBV", "COST", "WMT", "BAC", "AVGO", "LLY", "MRK", "PFE", "ORCL", "CRM",
        "ADBE", "NFLX", "AMD", "INTC", "CSCO", "TMO", "ABT", "ACN", "MCD", "NKE",
        "DIS", "WFC", "VZ", "QCOM", "TXN", "IBM", "GE", "CAT", "BA", "GS",
        "AMGN", "HON", "UNP", "LOW", "SBUX", "BLK", "C", "AXP", "PM", "RTX",
        "INTU", "AMAT", "BKNG", "ISRG", "ADP", "MU", "LRCX", "GILD", "REGN", "VRTX",
        "MDLZ", "ADI", "PANW", "KLAC", "SNPS", "CDNS", "MRVL", "ORLY", "CSX", "DE",
        "MMM", "CB", "MO", "DUK", "SO", "PLD", "CI", "ELV", "BMY", "TJX",
        "SPGI", "SCHW", "T", "MMC", "GM", "F", "COP", "EOG", "SLB", "NEE"]},
    "United Kingdom — FTSE 350": {"currency": "GBP", "tickers": [
        "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L", "DGE.L",
        "GLEN.L", "BATS.L", "LSEG.L", "REL.L", "NG.L", "VOD.L", "BARC.L", "LLOY.L",
        "PRU.L", "TSCO.L", "AAL.L", "NWG.L", "STAN.L", "BT-A.L", "IMB.L", "CPG.L",
        "RKT.L", "AV.L", "SSE.L", "BA.L", "LGEN.L", "ANTO.L", "NXT.L", "SGRO.L",
        "EXPN.L", "RR.L", "HLN.L", "BNZL.L", "SN.L", "WTB.L", "III.L", "ADM.L",
        "AHT.L", "IHG.L", "FRES.L", "SMIN.L", "PSN.L", "SVT.L", "UU.L", "WEIR.L",
        "PSON.L", "CNA.L", "SBRY.L", "KGF.L", "HWDN.L", "BDEV.L", "LAND.L", "BLND.L",
        "RMV.L", "ITRK.L", "MNDI.L", "DPLM.L", "HL.L", "PHNX.L", "HIK.L", "CCH.L",
        "SDR.L", "BKG.L", "SMT.L", "TW.L", "WPP.L", "AUTO.L", "ABF.L", "CTEC.L"]},
    "Germany — HDAX": {"currency": "EUR", "tickers": [
        "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "AIR.DE", "MRK.DE", "MBG.DE", "BMW.DE",
        "VOW3.DE", "BAS.DE", "BAYN.DE", "ADS.DE", "DBK.DE", "IFX.DE", "DB1.DE", "MUV2.DE",
        "RWE.DE", "EOAN.DE", "DHL.DE", "HEN3.DE", "VNA.DE", "FRE.DE", "CON.DE", "DTG.DE",
        "PAH3.DE", "SHL.DE", "ZAL.DE", "HNR1.DE", "BEI.DE", "SY1.DE", "MTX.DE", "QIA.DE",
        "BNR.DE", "RHM.DE", "CBK.DE", "P911.DE", "HEI.DE", "FME.DE", "SRT3.DE", "PUM.DE",
        "LHA.DE", "TKA.DE", "HFG.DE", "EVK.DE", "FRA.DE", "G24.DE", "1COV.DE", "AIXA.DE",
        "SDF.DE", "UN01.DE", "WCH.DE", "LEG.DE", "CTS.DE", "KGX.DE", "NDA.DE", "TEG.DE"]},
    "France — SBF 120": {"currency": "EUR", "tickers": [
        "MC.PA", "OR.PA", "RMS.PA", "TTE.PA", "SAN.PA", "AIR.PA", "SU.PA", "AI.PA",
        "EL.PA", "BNP.PA", "CS.PA", "DG.PA", "SAF.PA", "BN.PA", "KER.PA", "STLAP.PA",
        "CAP.PA", "ENGI.PA", "GLE.PA", "ACA.PA", "DSY.PA", "LR.PA", "PUB.PA", "VIE.PA",
        "ML.PA", "RI.PA", "ORA.PA", "STMPA.PA", "HO.PA", "SGO.PA", "EN.PA", "VIV.PA",
        "CA.PA", "BVI.PA", "RNO.PA", "EDEN.PA", "ERF.PA", "SW.PA", "TEP.PA", "ALO.PA",
        "FR.PA", "AKE.PA", "FDJ.PA", "GET.PA", "COFA.PA", "RXL.PA", "SOI.PA", "AMUN.PA"]},
    "Japan — Nikkei 225": {"currency": "JPY", "tickers": [
        "7203.T", "6758.T", "6861.T", "8306.T", "9984.T", "6098.T", "9432.T", "7974.T",
        "8035.T", "6501.T", "4063.T", "6902.T", "8058.T", "8316.T", "4568.T", "9433.T",
        "6367.T", "7267.T", "6594.T", "8001.T", "4519.T", "6981.T", "8031.T", "7741.T",
        "6273.T", "9434.T", "6752.T", "6503.T", "7751.T", "8766.T", "8411.T", "8802.T",
        "4661.T", "9022.T", "2914.T", "4502.T", "6301.T", "7011.T", "6702.T", "4901.T",
        "6326.T", "5108.T", "8053.T", "6954.T", "7269.T", "7201.T", "6971.T", "4543.T",
        "4523.T", "6857.T", "7735.T", "4578.T", "9020.T", "9101.T", "5401.T", "3382.T",
        "9983.T", "8267.T", "4452.T", "2802.T", "4507.T", "7013.T", "5020.T", "8604.T",
        "8750.T", "9501.T", "9531.T", "6146.T", "4689.T"]},
    "India — Nifty 500": {"currency": "INR", "tickers": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LT.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS", "SUNPHARMA.NS",
        "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "TATAMOTORS.NS",
        "TATASTEEL.NS", "POWERGRID.NS", "NTPC.NS", "BAJAJFINSV.NS", "ADANIENT.NS",
        "ADANIPORTS.NS", "COALINDIA.NS", "JSWSTEEL.NS", "HINDALCO.NS", "GRASIM.NS",
        "TECHM.NS", "ONGC.NS", "BPCL.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
        "BRITANNIA.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "INDUSINDBK.NS",
        "APOLLOHOSP.NS", "TATACONSUM.NS", "HDFCLIFE.NS", "SBILIFE.NS", "LTIM.NS",
        "M&M.NS", "SHRIRAMFIN.NS", "ADANIGREEN.NS", "DMART.NS", "PIDILITIND.NS",
        "GODREJCP.NS", "DABUR.NS", "HAVELLS.NS", "SIEMENS.NS", "BOSCHLTD.NS", "ABB.NS",
        "VEDL.NS", "AMBUJACEM.NS", "SHREECEM.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS",
        "GAIL.NS", "IOC.NS", "DLF.NS", "TRENT.NS", "NAUKRI.NS", "INDIGO.NS",
        "MARICO.NS", "COLPAL.NS", "BERGEPAINT.NS", "ICICIPRULI.NS", "ICICIGI.NS",
        "MUTHOOTFIN.NS", "CHOLAFIN.NS", "TVSMOTOR.NS", "SRF.NS", "UPL.NS",
        "AUROPHARMA.NS", "LUPIN.NS", "TORNTPHARM.NS", "MPHASIS.NS", "PERSISTENT.NS"]},
    "Canada — S&P/TSX Composite": {"currency": "CAD", "tickers": [
        "RY.TO", "TD.TO", "ENB.TO", "CNQ.TO", "BNS.TO", "BMO.TO", "CP.TO", "CNR.TO",
        "TRP.TO", "SU.TO", "BCE.TO", "MFC.TO", "ATD.TO", "CM.TO", "SHOP.TO", "SLF.TO",
        "TRI.TO", "NTR.TO", "GIB-A.TO", "WCN.TO", "NA.TO", "IMO.TO", "PPL.TO", "FTS.TO",
        "WPM.TO", "AEM.TO", "ABX.TO", "POW.TO", "MG.TO", "CVE.TO", "T.TO", "BAM.TO",
        "CCO.TO", "DOL.TO", "L.TO", "QSR.TO", "MRU.TO", "SAP.TO", "TECK-B.TO", "FNV.TO",
        "GIL.TO", "CTC-A.TO", "WN.TO", "EMA.TO", "H.TO", "FM.TO", "IFC.TO", "OTEX.TO",
        "CSU.TO", "AGI.TO", "NPI.TO", "STN.TO", "TOU.TO", "ARX.TO", "KEY.TO", "CAE.TO"]},
    "Australia — S&P/ASX 200": {"currency": "AUD", "tickers": [
        "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WBC.AX", "ANZ.AX", "WES.AX", "MQG.AX",
        "FMG.AX", "WOW.AX", "TLS.AX", "RIO.AX", "GMG.AX", "TCL.AX", "WDS.AX", "ALL.AX",
        "COL.AX", "STO.AX", "QBE.AX", "REA.AX", "ORG.AX", "SUN.AX", "AMC.AX", "RMD.AX",
        "JHX.AX", "COH.AX", "BXB.AX", "SCG.AX", "S32.AX", "IAG.AX", "MPL.AX", "ASX.AX",
        "XRO.AX", "WTC.AX", "PLS.AX", "NST.AX", "TWE.AX", "CPU.AX", "SGP.AX", "GPT.AX",
        "MGR.AX", "SHL.AX", "RHC.AX", "ALD.AX", "ORI.AX", "AGL.AX", "APA.AX", "SEK.AX",
        "CAR.AX", "REH.AX", "MIN.AX", "PME.AX", "EDV.AX", "QAN.AX", "BSL.AX", "JBH.AX",
        "SOL.AX", "IGO.AX", "LYC.AX", "WHC.AX", "VCX.AX", "BEN.AX", "BOQ.AX", "FPH.AX"]},
    "Hong Kong — Hang Seng Composite": {"currency": "HKD", "tickers": [
        "0700.HK", "9988.HK", "0939.HK", "1299.HK", "0941.HK", "3690.HK", "1810.HK",
        "0388.HK", "2318.HK", "0005.HK", "1398.HK", "0883.HK", "2628.HK", "0016.HK",
        "1211.HK", "0011.HK", "0002.HK", "0003.HK", "0001.HK", "0027.HK", "0175.HK",
        "1024.HK", "9618.HK", "9999.HK", "2020.HK", "2331.HK", "0669.HK", "1113.HK",
        "0288.HK", "0386.HK", "3968.HK", "0688.HK", "2688.HK", "1109.HK", "0762.HK",
        "0857.HK", "1088.HK", "2382.HK", "0968.HK", "2015.HK", "9868.HK", "3888.HK",
        "0981.HK", "1177.HK", "2269.HK", "0291.HK", "2319.HK", "1093.HK", "6690.HK",
        "0316.HK", "1928.HK", "0012.HK", "0006.HK", "0066.HK", "0017.HK", "1044.HK"]},
}

# =========================================================
# DATA DOWNLOAD
# =========================================================
def _download_batch(tickers, start, end, interval="1d"):
    raw = yf.download(tickers=list(tickers), start=start, end=end, interval=interval,
                      auto_adjust=True, progress=False, group_by="column")
    if raw is None or raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    def _extract(field):
        if isinstance(raw.columns, pd.MultiIndex):
            if field in raw.columns.get_level_values(0):
                out = raw[field].copy()
            else:
                return pd.DataFrame()
        else:
            if field in raw.columns:
                out = raw[[field]].copy()
                if len(tickers) == 1:
                    out.columns = [tickers[0]]
            else:
                return pd.DataFrame()
        if isinstance(out, pd.Series):
            out = out.to_frame(name=tickers[0])
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(-1)
        return out

    return _extract("Close"), _extract("Volume")


@st.cache_data(show_spinner=False)
def download_prices_volume(tickers, start, end, retries=2, interval="1d"):
    """Download close+volume, retrying tickers that come back empty (recovers
    transient Yahoo rate-limit / hiccup failures rather than treating them as
    delisted). Genuinely unavailable symbols are simply left out.

    `interval` matters for speed: the yearly horizon works in weekly returns, so
    pulling 15 years of DAILY bars for ~100 symbols downloads and parses five
    times more data than it can ever use."""
    close, volume = _download_batch(tickers, start, end, interval)

    def _missing(cl):
        done = set(cl.columns) if not cl.empty else set()
        good = {c for c in done if not cl[c].dropna().empty}
        return [t for t in tickers if t not in good]

    missing = _missing(close)
    for _ in range(retries):
        if not missing:
            break
        c2, v2 = _download_batch(missing, start, end, interval)
        if not c2.empty:
            for col in c2.columns:
                close[col] = c2[col]
            if not v2.empty:
                for col in v2.columns:
                    volume[col] = v2[col]
        missing = _missing(close)

    if not close.empty:
        close = close.dropna(how="all")
    return close, volume


@st.cache_data(show_spinner=False)
def get_series_close(ticker, start, end):
    """Download a single symbol's adjusted close as a Series (for index / FX)."""
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True,
                          progress=False, group_by="column")
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" not in raw.columns.get_level_values(0):
                return None
            s = raw["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
        else:
            if "Close" not in raw.columns:
                return None
            s = raw["Close"]
        return s.dropna()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_fx_series(native, report, start, end):
    """Time series of (report per native) FX rate; None if same currency/unavailable."""
    native = (native or "").upper().strip()
    report = (report or "").upper().strip()
    if not native or not report or native == report:
        return None
    s = get_series_close(f"{native}{report}=X", start, end)
    if s is not None and not s.empty:
        return s
    inv = get_series_close(f"{report}{native}=X", start, end)
    if inv is not None and not inv.empty:
        return 1.0 / inv
    return None


def rank_by_liquidity(close, volume, top_n):
    if close.empty:
        return []
    if volume is None or volume.empty:
        return close.count().sort_values(ascending=False).head(top_n).index.tolist()
    common = [c for c in close.columns if c in volume.columns]
    dv = (close[common] * volume[common]).median(axis=0, skipna=True)
    dv = dv.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    return dv.head(top_n).index.tolist()


@st.cache_data(show_spinner=False)
def get_market_caps(tickers):
    """Size of each holding, for the market portfolio the prior is built from.

    For a company that is market capitalisation. For a FUND there is no market
    cap, so we fall back to net assets (AUM) — and that distinction matters more
    than it sounds. The obvious alternative, traded value, is badly wrong for
    ETFs: SPY turns over roughly eighty times more dollars per day than AGG,
    yet the US bond market is larger than the US equity market. Sizing an
    asset-class prior by volume produces a "market portfolio" that is 64% SPY
    and 15% bonds, which is not the market by any definition.
    """
    def _one(t):
        mc = None
        try:
            fi = yf.Ticker(t).fast_info
            if fi:
                mc = fi.get("market_cap") or fi.get("marketCap")
        except Exception:
            mc = None
        if not mc:
            # Funds have no market cap, so fall back to net assets. This hits
            # .info, which is an order of magnitude slower than fast_info — for
            # an ETF universe that is EVERY symbol, so it must not run serially.
            try:
                info = yf.Ticker(t).info or {}
                mc = (info.get("totalAssets") or info.get("netAssets")
                      or info.get("totalAsset") or None)
            except Exception:
                mc = None
        try:
            return t, (float(mc) if mc else np.nan)
        except (TypeError, ValueError):
            return t, np.nan

    tickers = list(tickers)
    caps = {}
    # These are network-bound, so threads help enormously and cost nothing.
    # Serial, an ETF universe took the better part of a minute here.
    try:
        with ThreadPoolExecutor(max_workers=min(12, max(1, len(tickers)))) as ex:
            for t, v in ex.map(_one, tickers):
                caps[t] = v
    except Exception:
        for t in tickers:
            k, v = _one(t)
            caps[k] = v
    return caps


@st.cache_data(show_spinner=False)
def get_current_prices(tickers):
    """Latest available market price per ticker (live, one batched download)."""
    tickers = list(tickers)
    try:
        raw = yf.download(tickers, period="5d", auto_adjust=True, progress=False, group_by="column")
    except Exception:
        return {}
    if raw is None or raw.empty:
        return {}
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return {}
        close = raw["Close"]
    else:
        if "Close" not in raw.columns:
            return {}
        close = raw[["Close"]]
        if len(tickers) == 1:
            close.columns = [tickers[0]]
    out = {}
    for t in tickers:
        try:
            s = close[t].dropna() if t in close.columns else pd.Series(dtype=float)
            out[t] = float(s.iloc[-1]) if len(s) else np.nan
        except Exception:
            out[t] = np.nan
    return out

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
    b_bar2 = 0.0
    for i in range(t):
        xi = Xc[i][:, None]
        diff = xi @ xi.T - S
        b_bar2 += np.sum(diff ** 2)
    b_bar2 = b_bar2 / (t ** 2) / n
    b2 = min(b_bar2, d2)
    a2 = d2 - b2
    shrink = b2 / d2 if d2 > 0 else 1.0
    sigma = shrink * m * np.eye(n) + (a2 / d2 if d2 > 0 else 0.0) * S
    return sigma


def _ewma_weights(t, halflife):
    lam = 0.5 ** (1.0 / max(halflife, 1e-9))
    w = lam ** np.arange(t - 1, -1, -1)
    return w / w.sum()


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


# =========================================================
# BLACK-LITTERMAN CORE
# =========================================================
# Everything in this block works in EXCESS-return space (returns over the
# risk-free rate), because that is the space the equilibrium is defined in:
#   pi = delta * Sigma * w_mkt   is an excess return by construction.
# Total returns are recovered at the very end as  mu_total = rf + mu_excess.
# Getting this wrong is the single most common Black-Litterman bug: it silently
# shifts every view by the risk-free rate.

VIEW_TYPES = ["Absolute", "Relative"]

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


# ---------------------------------------------------------
# Omega — how uncertain each view is
# ---------------------------------------------------------
# =========================================================
# SYSTEMATIC VIEW ENGINES
# =========================================================
# Typed views cannot be tested. You form them today, so applying them to 2022
# is look-ahead bias — which is why the backtest can only ever run views-off.
# A RULE can be evaluated at every rebalance using only the data available at
# that moment, so the backtest can finally answer the question that matters:
# do views add anything at all?
#
# Every engine here is a pure function of a price history. Give it prices up to
# time t and it returns views as of time t. Nothing else. That property is what
# makes the whole thing honest, and it is enforced by a test.

VIEW_ENGINES = {
    "None — pure equilibrium": None,
    "Momentum (12-1)": "momentum",
    "Short-term reversal (1m)": "reversal",
    "Trend (price vs 200d)": "trend",
    "Low volatility": "lowvol",
}

ENGINE_NOTES = {
    "momentum": ("Ranks assets on their return over the past 12 months, skipping the most "
                 "recent month (the skip avoids short-term reversal contaminating the signal). "
                 "Expects past winners to keep beating past losers. The most robust anomaly in "
                 "the literature, and also among the most crowded."),
    "reversal": ("Ranks on the last month's return and bets the other way — recent losers "
                 "bounce. Works at short horizons where momentum does not, and is the reason "
                 "momentum skips the most recent month."),
    "trend": ("Compares each asset's price to its own 200-day average. Above = hold, below = "
              "expect weakness. A time-series signal rather than a cross-sectional one: it can "
              "be bearish on everything at once."),
    "lowvol": ("Ranks on realised volatility and expects the calmer assets to deliver better "
               "risk-adjusted returns — the low-volatility anomaly. Note this partly duplicates "
               "what the optimiser already does, so its views often add little."),
}


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

    Returns (hit_rate, n_checks). A rule with no information lands at 0.5.
    """
    px = prices.dropna(how="all")
    T = len(px)
    hp = max(1, int(holding_periods))
    per_month = max(1, int(round(freq_per_year / 12)))
    warm = 13 * per_month                      # momentum needs 12m + 1m skip
    if T < warm + hp + 5:
        return None, 0

    step = max(hp, (T - warm - hp) // max_checks + 1)
    hits = total = 0
    for t in range(warm, T - hp, step):
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
    if total == 0:
        return None, 0
    return hits / total, total


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


def _solve(obj_fun, n, lb, ub, net, gross, seed_w=None):
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - net}]
    if gross is not None:
        cons.append({"type": "ineq", "fun": lambda w, g=gross: g - np.sum(np.abs(w))})
    bounds = tuple((lb, ub) for _ in range(n))
    starts = [np.clip(np.repeat(net / n, n), lb, ub)]
    if seed_w is not None:
        starts.append(np.clip(seed_w, lb, ub))
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


def _optimize_once(objective, mu, cov, R, rf, stance, cap, gross, alpha, freq_per_year):
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)
    lb, ub, net = _stance_bounds(stance, cap, n)
    target_period = rf / freq_per_year

    if objective == "Min variance":
        obj = lambda w: float(w @ cov @ w)
    elif objective == "Max Sortino":
        obj = lambda w: -((w @ mu) - rf) / max(_downside_dev(w, R, target_period, freq_per_year), 1e-9)
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
                   alpha, freq_per_year, resample_n, seed=7):
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
                                      cap, gross_limit, alpha, freq_per_year))
        except Exception:
            continue
    if not acc:
        raise ValueError("Resampled optimisation failed on all draws.")
    return np.array(acc)


def optimize_portfolio(objective, mu, cov, R, rf, stance, max_weight, gross_limit,
                       alpha, freq_per_year, resample_n=0, seed=7):
    cap = max_weight if max_weight is not None else 1.0
    if resample_n and resample_n > 0:
        return resample_stack(objective, mu, cov, rf, stance, max_weight, gross_limit,
                              alpha, freq_per_year, resample_n, seed).mean(axis=0)
    return _optimize_once(objective, mu, cov, R, rf, stance, cap, gross_limit, alpha, freq_per_year)


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

# =========================================================
# BACKTEST
# =========================================================
def perf_metrics(period_returns, freq_per_year, rf):
    pr = np.asarray(period_returns, dtype=float)
    if len(pr) == 0:
        return None
    growth = np.prod(1.0 + pr)
    ann_ret = growth ** (freq_per_year / len(pr)) - 1.0 if growth > 0 else -1.0
    ann_vol = pr.std(ddof=1) * math.sqrt(freq_per_year) if len(pr) > 1 else 0.0
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


def run_backtest(prices, freq, use_log, mu_builder, cov_method, freq_per_year, rf,
                 stance, max_weight, gross_limit, objective, alpha,
                 tc_bps, borrow_bps, train_frac, resample_n,
                 rebalance_periods=21, rebal_label="periodic",
                 bench_prices=None, fx_prices=None, bench_label="Index",
                 caps_weights=None,
                 vol_target=None, vol_lookback=None, vol_max_leverage=1.0,
                 vol_target_frac=None):
    """Walk-forward backtest: starting after an initial training window, re-build
    the expected returns every `rebalance_periods` periods using all data up to
    that point (expanding window) and hold those weights until the next
    rebalance. All returns are converted into the reporting currency using the
    FX path (fx_prices), and an index buy-and-hold (bench_prices) is included as
    a 'beat the market' benchmark.

    The only structural difference from the Markowitz version is `mu_builder`:
    a callable (train_returns, cov_annual) -> (mu_annual_total, Sigma_to_use).
    That lets the same engine drive either a plain historical/shrinkage mu or a
    Black-Litterman posterior, and lets the BL prior be re-derived at every
    rebalance from data available at that time — which is what keeps the test
    genuinely out-of-sample.
    """
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
    borrow_frac = borrow_bps / 1e4
    tc_frac = tc_bps / 1e4

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
            mu_tr, sigma_tr = mu_builder(tr, cov_tr.values)
            w = optimize_portfolio(objective, np.asarray(mu_tr, dtype=float), sigma_tr,
                                   tr.values, rf, stance, max_weight, gross_limit,
                                   alpha, freq_per_year, resample_n=resample_n)
        except Exception:
            w = prev_w if n_rebalances > 0 else np.repeat(1.0 / n, n)

        turnover = float(np.sum(np.abs(w - prev_w)))
        turnover_sum += turnover
        n_rebalances += 1
        prev_w = w

        end_j = min(i + rebalance_periods, T)
        seg = (R_simple.iloc[i:end_j].values @ w).astype(float)      # native
        seg -= borrow_frac * float(np.sum(np.clip(-w, 0, None))) / freq_per_year
        if len(seg):
            seg[0] -= tc_frac * turnover
        seg_report = to_report(seg, i, end_j)                       # -> reporting ccy

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
                # cost of changing exposure, charged on the change in gross
                lev_cost = tc_frac * abs(k - prev_k) * float(np.sum(np.abs(w)))
                # the un-invested fraction earns the risk-free rate
                scaled[t] = k * r_t + (1.0 - k) * rf_period - lev_cost
                scales.append(k)
                prev_k = k
                hist_unscaled.append(float(r_t))       # only past data feeds the next k
            pnl.extend(scaled.tolist())
        else:
            pnl.extend(seg_report.tolist())

        pnl_unscaled.extend(seg_report.tolist())        # same strategy, always fully invested
        dates.extend(list(R_simple.index[i:end_j]))
        i = end_j

    if not pnl:
        return None

    m = len(pnl)
    wb = np.repeat(1.0 / n, n)
    pb_native = R_simple.iloc[start_i:start_i + m].values @ wb
    pb = to_report(pb_native, start_i, start_i + m)                  # equal-weight, reporting ccy

    strat = perf_metrics(np.array(pnl), freq_per_year, rf)
    bench = perf_metrics(pb, freq_per_year, rf)

    # Buy-and-hold the CAP-WEIGHTED market portfolio of the same names. This is
    # the benchmark that actually matters for Black-Litterman: with no views the
    # model is anchored on exactly this portfolio, so it is the honest
    # 'did the machinery add anything?' line.
    market_perf = None
    if caps_weights is not None:
        wm = np.asarray(caps_weights, dtype=float)
        if wm.shape[0] == n and np.isfinite(wm).all():
            pm_native = R_simple.iloc[start_i:start_i + m].values @ wm
            market_perf = perf_metrics(to_report(pm_native, start_i, start_i + m),
                                       freq_per_year, rf)

    index_perf = None
    if bench_prices is not None and np.any(idx_ret[start_i:start_i + m] != 0):
        idx_report = to_report(idx_ret[start_i:start_i + m], start_i, start_i + m)
        index_perf = perf_metrics(idx_report, freq_per_year, rf)

    # The same strategy without the volatility overlay — the honest like-for-like
    # comparison that shows whether the overlay earned its keep.
    unscaled_perf = perf_metrics(np.array(pnl_unscaled), freq_per_year, rf) if do_vt else None

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
            "n_test": m, "n_train": start_i,
            "n_rebalances": n_rebalances, "rebal_label": rebal_label,
            "avg_turnover": turnover_sum / max(n_rebalances, 1)}

# =========================================================
# METADATA + CURRENCY / FX
# =========================================================
def infer_currency_from_suffix(ticker: str):
    ticker = (ticker or "").upper().strip()
    suffix_map = {".L": "GBP", ".T": "JPY", ".NS": "INR", ".BO": "INR", ".HK": "HKD",
                  ".TO": "CAD", ".V": "CAD", ".AX": "AUD", ".NZ": "NZD", ".SI": "SGD",
                  ".KS": "KRW", ".KQ": "KRW", ".SS": "CNY", ".SZ": "CNY", ".AS": "EUR",
                  ".PA": "EUR", ".MI": "EUR", ".DE": "EUR", ".MC": "EUR", ".SW": "CHF",
                  ".OL": "NOK", ".CO": "DKK", ".ST": "SEK", ".HE": "EUR", ".JO": "ZAR",
                  ".MX": "MXN"}
    for suffix, ccy in suffix_map.items():
        if ticker.endswith(suffix):
            return ccy
    if "." not in ticker:
        return "USD"
    return None


@st.cache_data(show_spinner=False)
def get_ticker_meta(ticker):
    tk = yf.Ticker(ticker)
    meta = {"currency": None, "exchange": None, "name": None}
    try:
        fi = getattr(tk, "fast_info", None)
        if fi:
            meta["currency"] = fi.get("currency") or meta["currency"]
            meta["exchange"] = fi.get("exchange") or meta["exchange"]
    except Exception:
        pass
    try:
        info = tk.info
        meta["currency"] = meta["currency"] or info.get("currency")
        meta["exchange"] = meta["exchange"] or info.get("exchange")
        meta["name"] = info.get("shortName") or info.get("longName") or meta["name"]
    except Exception:
        pass
    if not meta["currency"]:
        meta["currency"] = infer_currency_from_suffix(ticker)
    meta["currency"] = (meta["currency"] or "").upper().strip() or None
    meta["exchange"] = meta["exchange"] or ""
    meta["name"] = meta["name"] or ""
    return meta


@st.cache_data(show_spinner=False)
def _last_close(symbol):
    try:
        df = yf.download(symbol, period="10d", interval="1d",
                         auto_adjust=True, progress=False, group_by="column")
        if df is None or df.empty:
            return np.nan
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" not in df.columns.get_level_values(0):
                return np.nan
            s = df["Close"].dropna()
        else:
            if "Close" not in df.columns:
                return np.nan
            s = df["Close"].dropna()
        if len(s) == 0:
            return np.nan
        return float(s.iloc[-1].item() if hasattr(s.iloc[-1], "item") else s.iloc[-1])
    except Exception:
        return np.nan


def get_fx_rate(native_ccy, report_ccy):
    native_ccy = (native_ccy or "").upper().strip()
    report_ccy = (report_ccy or "").upper().strip()
    if not native_ccy:
        return np.nan, "Missing native currency", False
    if not report_ccy:
        return np.nan, "Missing reporting currency", False
    if native_ccy == report_ccy:
        return 1.0, "No conversion needed", True
    direct = f"{native_ccy}{report_ccy}=X"
    r = _last_close(direct)
    if pd.notna(r) and r > 0:
        return r, direct, True
    inverse = f"{report_ccy}{native_ccy}=X"
    r_inv = _last_close(inverse)
    if pd.notna(r_inv) and r_inv > 0:
        return 1.0 / r_inv, f"Inverse of {inverse}", True
    if native_ccy != "USD" and report_ccy != "USD":
        n_usd = _last_close(f"{native_ccy}USD=X")
        if not (pd.notna(n_usd) and n_usd > 0):
            usd_n = _last_close(f"USD{native_ccy}=X")
            if pd.notna(usd_n) and usd_n > 0:
                n_usd = 1.0 / usd_n
        r_usd = _last_close(f"{report_ccy}USD=X")
        if not (pd.notna(r_usd) and r_usd > 0):
            usd_r = _last_close(f"USD{report_ccy}=X")
            if pd.notna(usd_r) and usd_r > 0:
                r_usd = 1.0 / usd_r
        if pd.notna(n_usd) and n_usd > 0 and pd.notna(r_usd) and r_usd > 0:
            return n_usd / r_usd, f"USD bridge: {native_ccy}->USD and {report_ccy}->USD", True
    return np.nan, f"FX unavailable for {native_ccy}->{report_ccy}", False


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


def _render_backtest(res, bt, bt_eq=None):
    bc = res["base_currency"]; total_capital = res["total_capital"]
    ret_freq = res["ret_freq"]; freq_per_year = res["freq_per_year"]
    s, b = bt["strat"], bt["bench"]
    idx = bt.get("index"); idx_label = bt.get("index_label", "Index")
    mkt = bt.get("market")
    test_start, test_end = bt["dates"][0], bt["dates"][-1]
    yrs = bt["n_test"] / freq_per_year
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
        f"{test_start.date()} to {test_end.date()}. Trading costs charged on turnover "
        f"(avg {turn*100:,.0f}% per rebalance); borrow costs included.{fx_note}")

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
                        f"test was fair, not that the edge is real. A single historical path "
                        f"cannot distinguish a {d_r*100:,.2f}pp edge from noise, and the confidence "
                        f"the rule earned ({res.get('sys_conf', 0)*100:,.0f}%) is itself a measure "
                        f"of how weak the signal was. Treat it as a hypothesis worth re-testing on "
                        f"another market, not as a result.")
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

    tabs = st.tabs(["📊 Holdings", "🎯 What your views did", "📈 Efficient frontier",
                    "⏱️ Backtest", "🛒 What to buy", "💾 Compare runs"])

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
                   "the only number here with any predictive claim attached, and it is one path.",
              delta=f"{-res['borrow_cost_annual']*100:,.2f}% borrow" if res["short_gross"] > 0 else None)
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


def _render_execution(res, bc):
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
    ret_freq = hcfg["freq"]; use_log = True; cvar_alpha = 0.95
    tc_bps = market_default(market_name, "tc_bps"); borrow_bps = 50.0; resample_n = 0
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
                                        index=["Daily", "Weekly", "Monthly"].index(hcfg["freq"]))
                use_log = st.checkbox("Use log returns", value=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                tc_bps = st.number_input(
                    "Transaction cost (bps)", 0.0, 500.0,
                    float(market_default(market_name, "tc_bps")), 1.0,
                    key=f"tc_{market_name}",
                    help="All-in round-trip cost, pre-set per market. India is higher than the US "
                         "because of STT and stamp duty; the UK and Hong Kong carry stamp too.")
            with c2:
                borrow_bps = st.number_input("Short borrow cost (bps/yr)", 0.0, 2000.0, 50.0, 5.0)
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
    _reb_per_year = FREQ_PER_YEAR[hcfg["freq"]] / hcfg["rebal"]
    st.caption(
        f"**All three horizons estimate from the same {hcfg['years']}-year daily history.** They "
        f"differ only in how often you trade: this one rebalances about "
        f"**{_reb_per_year:,.0f} times a year**. More rebalances means more decisions to judge the "
        f"strategy on, and more trading costs; fewer means a cleaner read on the long-run "
        f"allocation, from a smaller sample of decisions. Neither is more correct."
    )

    if mode != "Advanced":
        st.caption(
            f"**Recommended settings applied automatically for {native_ccy}:** risk-free "
            f"{float(rf_annual)*100:,.1f}% · trading cost {tc_bps:,.0f} bps · max {max_weight_pct}% per "
            f"stock · δ implied from the market · Ω Idzorek · τ {tau:.2f} · view confidence "
            f"{DEFAULT_VIEW_CONFIDENCE:,.0f}% · backtest trains on {train_frac:.0%} of history. "
            f"Switch to **Advanced** to change any of them.")

    # ---------------- STAGE A: load the universe and its equilibrium ----------------
    if st.button("① Load universe & equilibrium", type="primary"):
        for k in ("uni", "res", "frontier_chart"):
            st.session_state.pop(k, None)
        if start_date >= end_date:
            st.error("History start must be earlier than history end.")
            st.stop()

        seed_tickers = MARKETS[market_name]["tickers"]
        with st.spinner("Downloading market data..."):
            close, volume = download_prices_volume(
                seed_tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
                interval=("1wk" if ret_freq == "Weekly" else "1d"))
        if close.empty:
            st.error(
                "No price data returned. Common causes:\n\n"
                "1. **Corrupted yfinance cache** (`DatabaseError: database disk image is malformed`). "
                "Close the app and run in PowerShell: "
                "`Remove-Item -Recurse -Force \"$env:LOCALAPPDATA\\py-yfinance\"`, then relaunch.\n\n"
                "2. **Rate-limited by Yahoo** (`Too Many Requests`). Wait a few minutes.\n\n"
                "3. **Outdated library**: `python -m pip install --upgrade yfinance curl_cffi`."
            )
            st.stop()

        top_tickers = rank_by_liquidity(close, volume, top_n)

        # Capital-aware trim. Rank by liquidity first, then check whether the
        # money can actually express that many positions in whole shares. In
        # Simple mode we cut automatically and say so; in Advanced we respect
        # the user's choice and warn instead.
        _cand_px = [float(close[t].dropna().iloc[-1]) for t in top_tickers
                    if t in close.columns and not close[t].dropna().empty]
        _lots = [lot_size_for(t) for t in top_tickers if t in close.columns]
        _lot_px = [p * l for p, l in zip(_cand_px, _lots)]
        fit_n = affordable_holdings(_lot_px, total_capital, len(top_tickers))
        auto_trimmed = None
        if fit_n < len(top_tickers):
            if mode != "Advanced":
                auto_trimmed = (len(top_tickers), fit_n)
                top_tickers = top_tickers[:fit_n]
            else:
                st.warning(
                    f"**{total_capital:,.0f} {base_currency} may not stretch to "
                    f"{len(top_tickers)} positions.** At these share prices roughly "
                    f"**{fit_n} names** can be held with enough granularity to express "
                    f"their weights; beyond that, positions round down hard and money "
                    f"sits in cash. Lower *Number of stocks to hold*, or accept the "
                    f"rounding — the 'What to buy' tab reports exactly how much stays idle.")

        # Choose the estimation window from the data rather than imposing one.
        # A naive dropna() over a fixed 15-year request would silently truncate
        # EVERY asset to the youngest one's history; a fixed short window would
        # starve Sigma. This picks the longest window that still supports a
        # full-sized universe, and backfills the dropped names from the next
        # most liquid long-lived candidates.
        _all_ranked = rank_by_liquidity(close, volume, min(len(close.columns), 4 * top_n))
        _win_start, _keep, _win_note = choose_history_window(
            close, _all_ranked, top_n, hcfg, FREQ_PER_YEAR[ret_freq])
        _cand = _keep[:top_n] if len(_keep) >= top_n else _keep
        prices = close.loc[close.index >= _win_start, _cand].dropna()
        history_note = _win_note
        if prices.shape[1] < 2 or prices.empty:
            st.error("Could not assemble at least 2 liquid tickers with overlapping history. "
                     "Widen the date range or increase the number of stocks.")
            st.stop()

        usable = list(prices.columns)
        n_assets = len(usable)
        freq_per_year = FREQ_PER_YEAR[ret_freq]
        returns = to_returns(prices, ret_freq, use_log)
        if returns.shape[0] < 10:
            st.error(f"Not enough {ret_freq.lower()} observations ({returns.shape[0]}) to estimate "
                     f"risk/return. Widen the date range or use a higher frequency.")
            st.stop()

        max_weight = None if max_weight_pct >= 100 else max_weight_pct / 100.0
        if max_weight is not None and max_weight * n_assets < 1.0 - 1e-9:
            st.error(f"Max weight per stock ({max_weight_pct}%) is too low for {n_assets} stocks. "
                     f"Raise it to at least {math.ceil(100 / n_assets)}% or add more stocks.")
            st.stop()

        with st.spinner("Fetching market capitalisations (this is what the prior is built from)..."):
            caps = get_market_caps(usable)

        cov = estimate_cov(returns, cov_method, freq_per_year)
        # liquidity fallback for the market portfolio if caps are unusable
        liq = None
        if volume is not None and not volume.empty:
            common = [t for t in usable if t in volume.columns]
            if len(common) == n_assets:
                dv = (prices[common] * volume[common]).median(axis=0, skipna=True)
                if np.isfinite(dv.values).all() and dv.values.sum() > 0:
                    liq = dv.values
        w_mkt, w_mkt_note = market_weights(usable, caps, fallback_weights=liq)

        hist_annual = returns.mean().values * freq_per_year
        n_available = int(sum(1 for c in close.columns if not close[c].dropna().empty))

        st.session_state["uni"] = {
            "signature": f"{market_name}|{top_n}|{ret_freq}|{start_date}|{end_date}|{cov_method}",
            "market_name": market_name, "usable": usable, "n_assets": n_assets,
            "prices": prices, "returns": returns, "cov": cov,
            "sigma": np.sqrt(np.diag(cov.values)),
            "caps": caps, "w_mkt": w_mkt, "w_mkt_note": w_mkt_note,
            "mkt_hist_return": float(w_mkt @ hist_annual),
            "hist_annual": hist_annual,
            "freq_per_year": freq_per_year, "ret_freq": ret_freq, "use_log": use_log,
            "native_ccy": native_ccy,
            "start_str": start_date.strftime("%Y-%m-%d"), "end_str": end_date.strftime("%Y-%m-%d"),
            "universe_note": (
                history_note +
                (f"**Trimmed from {auto_trimmed[0]} to {auto_trimmed[1]} names to fit "
                 f"{total_capital:,.0f} {base_currency}.** At these share prices, "
                 f"{auto_trimmed[0]} positions would round down hard and leave a large "
                 f"slice of your money sitting in cash — a drag bigger than anything the "
                 f"optimiser is deciding. Override in Advanced settings. "
                 if auto_trimmed else "") +
                f"Selected the {n_assets} most liquid of {n_available} stocks that downloaded "
                f"successfully from {market_name} ({n_universe} in the index). "
                + ("" if n_available >= n_universe else
                   f"{n_universe - n_available} symbol(s) were unavailable on Yahoo and skipped.")),
        }

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
                                              holding_periods=hcfg["rebal"], n_pairs=sys_n_pairs)
                sys_conf, sys_conf_note = confidence_from_hit_rate(hr, nchecks)
                Ps, Qs, cs, sys_labels = systematic_views(
                    uni["prices"], usable, sys_engine, uni["freq_per_year"],
                    n_pairs=sys_n_pairs, conf=sys_conf)

            h1, h2, h3 = st.columns(3)
            h1.metric("Historical hit rate", f"{hr*100:,.0f}%" if hr is not None else "—",
                      help="How often this rule got the direction right, walking forward through "
                           "your history window using only data available at each point.")
            h2.metric("Checks", f"{nchecks}",
                      help="Independent out-of-sample observations behind that hit rate.")
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
            Sigma = cov.values
            w_mkt = uni["w_mkt"]

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

            try:
                if resample_n and resample_n > 0:
                    stack = resample_stack(objective, mu.values, Sigma_used, rf, stance, max_weight,
                                           gross_limit, cvar_alpha, freq_per_year, resample_n)
                    weights = stack.mean(axis=0)
                    wstd = stack.std(axis=0)
                else:
                    weights = optimize_portfolio(objective, mu.values, Sigma_used, returns.values, rf,
                                                 stance, max_weight, gross_limit, cvar_alpha,
                                                 freq_per_year)
                    wstd = None
            except Exception as e:
                st.error(f"Optimisation failed: {e}")
                st.stop()

            # plain Markowitz (James-Stein) weights, for the side-by-side comparison
            w_markowitz = None
            try:
                mu_mk = estimate_mu(returns, "James-Stein shrinkage", freq_per_year, rf,
                                    cov_annual=Sigma, market_caps=uni["caps"])
                w_markowitz = optimize_portfolio(objective, mu_mk.values, Sigma, returns.values, rf,
                                                 stance, max_weight, gross_limit, cvar_alpha,
                                                 freq_per_year)
            except Exception:
                w_markowitz = None

            port_ret, port_vol, port_sharpe = portfolio_stats(weights, mu.values, Sigma_used, rf)
            mkt_ret_bl, mkt_vol_bl, _ = portfolio_stats(w_mkt, mu.values, Sigma_used, rf)
            gross = float(np.sum(np.abs(weights)))
            net = float(np.sum(weights))
            short_gross = float(np.sum(np.clip(-weights, 0, None)))
            borrow_cost_annual = (borrow_bps / 1e4) * short_gross
            tc_cost = (tc_bps / 1e4) * gross
            net_expected = port_ret - borrow_cost_annual

            exposure_caption = (f"Gross exposure {gross*100:,.0f}%, net {net*100:,.0f}%. "
                                f"Est. build cost {tc_cost*total_capital:,.0f} {base_currency} "
                                f"({tc_bps:.0f} bps of turnover); annual short-borrow cost "
                                f"{borrow_cost_annual*total_capital:,.0f} {base_currency}. "
                                f"Return after borrow ≈ {net_expected*100:,.2f}%. "
                                f"Negative weights are shorts.")

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
                bench_ticker, bench_lbl = BENCHMARKS.get(market_name, (None, "Index"))
                bench_prices = (get_series_close(bench_ticker, uni["start_str"], uni["end_str"])
                                if bench_ticker else None)
                fx_prices = get_fx_series(uni["native_ccy"], base_currency,
                                          uni["start_str"], uni["end_str"])
                caps_map = uni["caps"]; liq_fallback = None

                # Re-walking the entire history to re-measure a hit rate at every
                # single rebalance costs ~90ms a time — 40 seconds across a
                # weekly backtest, for a number that barely moves between
                # adjacent weeks. Recompute it only when the training window has
                # grown by 5% or more; that is still strictly backward-looking.
                _hr_cache = {}

                def _cached_hit_rate(px_now, engine, fpy, hp):
                    bucket = int(math.log(max(len(px_now), 2)) / math.log(1.05))
                    key = (engine, hp, bucket)
                    if key not in _hr_cache:
                        _hr_cache[key] = engine_hit_rate(px_now, engine, fpy, holding_periods=hp)
                    return _hr_cache[key]

                def _make_builder(with_views, engine=None):
                    def _builder(train_returns, cov_annual):
                        # re-derive the prior from data available at this point only
                        w_m, _ = market_weights(usable, caps_map, fallback_weights=liq_fallback)
                        hist = train_returns.mean().values * freq_per_year
                        if delta_mode == "Set it myself":
                            d = float(delta_manual)
                        else:
                            d, _n = implied_risk_aversion(w_m, cov_annual, float(w_m @ hist) - rf)

                        if engine is not None:
                            # THE HONEST PATH. Rebuild the views from the rule using
                            # only prices up to this rebalance, and recalibrate the
                            # confidence from the rule's hit rate over that same
                            # window. Nothing from the future enters, so unlike typed
                            # views this genuinely tests whether views add value.
                            px_now = prices.loc[:train_returns.index[-1]]
                            hr_t, nc_t = _cached_hit_rate(px_now, engine, freq_per_year,
                                                          hcfg["rebal"])
                            c_t, _ = confidence_from_hit_rate(hr_t, nc_t)
                            Pe, Qe, ce, _lab = systematic_views(
                                px_now, usable, engine, freq_per_year,
                                n_pairs=sys_n_pairs, conf=c_t)
                            out = black_litterman(cov_annual, w_m, rf, tau, d, Pe, Qe, ce,
                                                  omega_method=omega_method,
                                                  use_posterior_cov=use_posterior_cov)
                        elif with_views:
                            out = black_litterman(cov_annual, w_m, rf, tau, d, P, Q, conf,
                                                  omega_method=omega_method,
                                                  use_posterior_cov=use_posterior_cov)
                        else:
                            out = black_litterman(cov_annual, w_m, rf, tau, d,
                                                  np.zeros((0, len(usable))), np.zeros(0), np.zeros(0),
                                                  omega_method=omega_method,
                                                  use_posterior_cov=use_posterior_cov)
                        return out["mu_total"], out["Sigma_used"]
                    return _builder

                bt_args = dict(freq=ret_freq, use_log=use_log, cov_method=cov_method,
                               freq_per_year=freq_per_year, rf=rf, stance=stance,
                               max_weight=max_weight, gross_limit=gross_limit, objective=objective,
                               alpha=cvar_alpha, tc_bps=tc_bps, borrow_bps=borrow_bps,
                               train_frac=train_frac, resample_n=min(resample_n, 10),
                               rebalance_periods=hcfg["rebal"], rebal_label=hcfg["rebal_label"],
                               bench_prices=bench_prices, fx_prices=fx_prices, bench_label=bench_lbl,
                               caps_weights=w_mkt,
                               vol_target=((vol_target_pct / 100.0) if use_vol_target
                                           and not vol_target_mode.startswith("Relative")
                                           else None),
                               vol_target_frac=((vol_target_frac_pct / 100.0) if use_vol_target
                                                and vol_target_mode.startswith("Relative")
                                                else None),
                               vol_lookback=vol_lookback, vol_max_leverage=vol_max_lev)
                run_views = bool(backtest_uses_views and bl["has_views"])
                if sys_engine is not None:
                    # Systematic views can be tested honestly, so always run the
                    # pair: rule-driven vs pure equilibrium. The gap between them
                    # IS the answer to "do views add value?".
                    with st.spinner("Back-testing the rule against pure equilibrium "
                                    "(re-deriving views at every rebalance)..."):
                        bt = run_backtest(prices, mu_builder=_make_builder(False, engine=sys_engine),
                                          **bt_args)
                        bt_eq = run_backtest(prices, mu_builder=_make_builder(False), **bt_args)
                else:
                    with st.spinner("Back-testing (walk-forward, re-deriving the equilibrium each time)..."):
                        bt = run_backtest(prices, mu_builder=_make_builder(run_views), **bt_args)
                        if run_views:
                            bt_eq = run_backtest(prices, mu_builder=_make_builder(False), **bt_args)

            # ---- execution table ----
            execution_date = prices.index[-1]
            last_prices = prices.loc[execution_date]
            with st.spinner("Fetching current prices..."):
                cur_map = get_current_prices(usable)
            # ---- gather prices/FX first, then allocate ACROSS the whole book ----
            # Allocating position-by-position is what strands cash: each name
            # floors independently and loses part of a lot. allocate_shares()
            # floors once and then spends the remainder where it does most good.
            meta_all, ccy_all, fx_all, ok_all = [], [], [], []
            lots_all, px_native, px_report, targets_report = [], [], [], []
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
                targets_report.append(float(weights[i] * total_capital))

            shares_all = allocate_shares(targets_report, px_report, lots_all, total_capital)

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
                rows.append({
                    "Ticker": ticker, "Name": meta["name"] or "",
                    "Side": "BUY" if weights[i] >= 0 else "SELL SHORT",
                    "Exchange": meta["exchange"] or "", "NativeCurrency": tk_ccy or "UNKNOWN",
                    "CurrentPrice": cur_native, f"CurrentPrice ({base_currency})": cur_report,
                    "MarketWtPct": w_mkt[i] * 100, "WeightPct": weights[i] * 100,
                    f"Allocation ({base_currency})": alloc_base,
                    "LotSize": lot, "Shares": shares, f"Cost ({base_currency})": filled_report,
                    f"LeftoverCash ({base_currency})": leftover_report,
                    "Converted": "YES" if converted_ok else "NO", "FXRate": fx_rate,
                })
            exec_df = pd.DataFrame(rows)
            detail_cols = ["Ticker", "Name", "Side", "Exchange", "NativeCurrency", "CurrentPrice",
                           f"CurrentPrice ({base_currency})", "MarketWtPct", "WeightPct",
                           f"Allocation ({base_currency})", "LotSize", "Shares",
                           f"Cost ({base_currency})", f"LeftoverCash ({base_currency})",
                           "Converted", "FXRate"]
            exec_fmt = {"CurrentPrice": 2, f"CurrentPrice ({base_currency})": 2, "MarketWtPct": 2,
                        "WeightPct": 2, f"Allocation ({base_currency})": 2, "LotSize": 0, "Shares": 0,
                        f"Cost ({base_currency})": 2, f"LeftoverCash ({base_currency})": 2, "FXRate": 6}
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
                "base_currency": base_currency, "total_capital": total_capital, "usable": usable,
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
                "borrow_cost_annual": borrow_cost_annual, "tc_cost": tc_cost,
                "settings_caption": settings_caption, "exposure_caption": exposure_caption,
                "universe_note": uni["universe_note"],
                "weights_df": weights_df, "has_stability": wstd is not None,
                "frontier": frontier, "mc_vols": mcv, "mc_rets": mcr, "mc_sharpes": mcs,
                "do_backtest": do_backtest, "bt": bt, "bt_eq": bt_eq,
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
