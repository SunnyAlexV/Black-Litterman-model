"""Market data: Yahoo Finance downloads, market caps, prices, FX.

The only module that touches the network. Everything here is cached with
st.cache_data, which degrades to a plain function call outside a Streamlit run.
"""
import math
import os
import logging
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

from bl_config import *  # noqa: F401,F403

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
