"""Turning target weights into whole-share orders: lots, capital, holdings."""
import math
import re
import numpy as np
import pandas as pd

from bl_config import *  # noqa: F401,F403

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


def parse_holdings(text, tickers):
    """Read a free-text list of what the user already owns.

    Accepts one position per line in whatever punctuation people actually
    type -- 'RELIANCE.NS 40', 'RELIANCE.NS, 40', 'RELIANCE.NS: 40 shares',
    tabs from a pasted spreadsheet -- and matches tickers case-insensitively,
    with or without the exchange suffix, so a paste from a broker statement
    that says plain 'RELIANCE' still lands on RELIANCE.NS.

    Returns (shares_by_ticker, problems). Anything unrecognised is reported
    rather than silently dropped: a holding the app quietly ignores would
    understate what you own and overstate what you need to buy.
    """
    shares, problems = {}, []
    if not text or not str(text).strip():
        return shares, problems

    exact = {t.upper(): t for t in tickers}
    stem = {}
    for t in tickers:
        stem.setdefault(t.upper().split(".")[0], t)

    for raw in str(text).replace(";", "\n").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # A comma is both a field separator and a thousands separator, and
        # Indian brokers export both on the same line. Strip the ones sitting
        # between digits first, or '1,250' splits into 1 and 250 and the
        # position silently shrinks by three orders of magnitude.
        line = re.sub(r"(?<=\d),(?=\d{2,3}(?:\D|$))", "", line)
        parts = [p for p in re.split(r"[\s,:|\t]+", line) if p]
        if len(parts) < 2:
            problems.append(f"`{line}` — need a ticker and a number of shares")
            continue
        sym = parts[0].upper().strip("'\"")
        qty_txt = None
        for p in parts[1:]:                       # first token that parses as a number
            cleaned = p.replace(",", "").replace("_", "")
            try:
                float(cleaned)
                qty_txt = cleaned
                break
            except ValueError:
                continue
        if qty_txt is None:
            problems.append(f"`{line}` — no share count found")
            continue
        tick = exact.get(sym) or stem.get(sym.split(".")[0])
        if tick is None:
            problems.append(f"`{sym}` — not in this universe, ignored")
            continue
        try:
            q = float(qty_txt)
        except ValueError:
            problems.append(f"`{line}` — could not read the share count")
            continue
        if q < 0:
            problems.append(f"`{sym}` — negative share counts are not supported here")
            continue
        shares[tick] = shares.get(tick, 0.0) + q
    return shares, problems


def holdings_to_weights(shares_by_ticker, tickers, prices_report):
    """Current portfolio weights from share counts and per-share prices.

    `prices_report` must already be in the reporting currency. Returns
    (weights, total_value, n_priced). Weights sum to 1 over what could be
    priced; a name with no usable price is treated as not held rather than
    as held at zero, and is counted out via n_priced so the caller can say so.
    """
    n = len(tickers)
    vals = np.zeros(n, dtype=float)
    n_priced = 0
    for i, t in enumerate(tickers):
        q = float(shares_by_ticker.get(t, 0.0) or 0.0)
        px = prices_report[i] if i < len(prices_report) else np.nan
        if q > 0 and pd.notna(px) and float(px) > 0:
            vals[i] = q * float(px)
            n_priced += 1
    total = float(np.sum(vals))
    if total <= 0:
        return np.zeros(n, dtype=float), 0.0, 0
    return vals / total, total, n_priced


def lot_size_for(ticker):
    """Board-lot size for markets that don't trade single shares (approximate)."""
    t = (ticker or "").upper()
    if t.endswith(".T"):
        return 100      # Tokyo: 100-share units
    if t.endswith(".HK"):
        return 100      # Hong Kong: varies by stock; 100 is a common default
    return 1
