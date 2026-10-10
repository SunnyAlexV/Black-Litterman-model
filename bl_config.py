"""Static configuration: horizons, markets, defaults, view-engine names.

No imports from the rest of the project, so every other module can depend on it.
"""
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
# chosen. rf is the local short-dated government yield -- the opportunity cost
# the equilibrium prior is measured against.
#   rf sources: 1-year government bill yields, August 2026.
MARKET_DEFAULTS = {
    "United States — S&P 500":         {"rf": 0.041},
    "United Kingdom — FTSE 350":       {"rf": 0.040},
    "Germany — HDAX":                  {"rf": 0.023},
    "France — SBF 120":                {"rf": 0.024},
    "Japan — Nikkei 225":              {"rf": 0.008},
    "India — Nifty 500":               {"rf": 0.060},
    "Canada — S&P/TSX Composite":      {"rf": 0.031},
    "Australia — S&P/ASX 200":         {"rf": 0.037},
    "Hong Kong — Hang Seng Composite": {"rf": 0.035},
}
DEFAULT_RF = 0.04


# Recommended defaults for the settings that actually change the answer.
DEFAULT_MAX_WEIGHT_PCT = 12     # single-STOCK cap; keeps ~8+ real positions
DEFAULT_ASSET_CLASS_CAP_PCT = 70  # asset classes: the market really is concentrated,
                                  # so a tight cap would forbid holding the prior
DEFAULT_TRAIN_FRAC = 0.45       # leaves a long enough out-of-sample window to mean something
DEFAULT_VIEW_CONFIDENCE = 75.0  # Idzorek confidence: tilt 75% of the way from market to view
DEFAULT_VOL_TARGET_PCT = 12.0        # absolute-mode target, when the user picks a fixed number
DEFAULT_VOL_TARGET_FRAC_PCT = 75     # relative-mode default: run at 75% of the book's own risk


def market_default(market_name, key):
    return MARKET_DEFAULTS.get(market_name, {}).get(key, DEFAULT_RF)

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
    "Global multi-asset (ETFs)": {"currency": "USD", "kind": "asset-class",
                                 "cross_tz": True, "tickers": [
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
    "Global equity regions (ETFs)": {"currency": "USD", "kind": "asset-class",
                                    "cross_tz": True, "tickers": [
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
# BLACK-LITTERMAN CORE
# =========================================================
# Everything in this block works in EXCESS-return space (returns over the
# risk-free rate), because that is the space the equilibrium is defined in:
#   pi = delta * Sigma * w_mkt   is an excess return by construction.
# Total returns are recovered at the very end as  mu_total = rf + mu_excess.
# Getting this wrong is the single most common Black-Litterman bug: it silently
# shifts every view by the risk-free rate.

VIEW_TYPES = ["Absolute", "Relative"]


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
