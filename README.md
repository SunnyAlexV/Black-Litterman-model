# Black-Litterman Portfolio Model

📊 **Black-Litterman Portfolio Optimizer** — start from the market, tilt with your views.

An interactive Streamlit app that builds a portfolio the way institutional allocators actually do it: instead of asking you to forecast every return (which Markowitz does, and then amplifies whatever you got wrong), it starts from the portfolio the market is already holding, reverse-engineers the returns that would make that portfolio optimal, and moves away from it only as far as your own views — and your confidence in them — justify.

## 🔗 Live demo

https://black-litterman-model.streamlit.app/

On Streamlit's free tier the app sleeps after inactivity and wakes on your first visit (give it ~30 seconds to spin up).

Companion to the [Markowitz Optimum Portfolio Model](https://github.com/SunnyAlexV/Markowitz-Optimum-Portfolio-model). Same markets, same execution output, same backtest engine — a different, and considerably more robust, way of getting to expected returns.

## The idea in one paragraph

Mean-variance optimisation has a well-known failure mode: it is exquisitely sensitive to the expected returns you feed it. Small estimation errors produce wildly concentrated portfolios. Black-Litterman fixes this by changing the question. Rather than "what will each stock return?", it asks "what returns would justify what the market is currently holding?" — that is the equilibrium prior **π = δΣw**, obtained by reverse optimisation. Your views are then blended into that prior in proportion to how confident you are, using Bayes' rule. With no views you get the market portfolio back. With total confidence you get your view exactly. In between, you get a smooth, diversified, defensible tilt.

You can see this working in the app: the **Markowitz wt %** column sits next to the BL weights, and on live data it typically shows the classic corner solution — everything pinned at the weight cap or at zero — while Black-Litterman produces a graded portfolio from the same covariance matrix and the same constraints.

## Features

### The model

- **Equilibrium prior** π = δΣw from cap-weighted market weights, with δ implied from the market's own realised risk premium (or set manually).
- **Absolute and relative views** — "RELIANCE will return 15%" or "HINDALCO beats VEDL by 10%" — entered in an editable table, activated per row.
- **Two Ω constructions**: **Idzorek** (you give a 0–100% confidence per view and ω is solved for numerically) and **He-Litterman** (Ω = diag(P τΣ P′), no confidence input needed).
- **Optional posterior covariance** Σ + M, for when you want to acknowledge that the posterior mean is itself uncertain.
- Correct **excess-return handling** throughout: absolute views are converted as `q − rf`, relative views leave rf to cancel. Getting this wrong is the most common Black-Litterman bug and it silently shifts every view by the risk-free rate.

### Asset-class universes, and why they matter

Alongside the stock universes there are three ETF universes — **Global multi-asset**, **US sectors**, **Global equity regions**. These are not a convenience; they fix the deepest flaw in the stock version.

Black-Litterman's prior only means something if **w is the market portfolio**. A hand-picked slice of 25 large caps is not one, and selecting today's most liquid names then back-testing them imports the answer: in the US stock run, the universe beat its own index by 16.6pp a year purely through survivorship and selection bias. An ETF spanning an asset class has none of that — it existed throughout the test window, and constituent churn is handled inside the fund.

It is also the problem Black and Litterman actually wrote about in 1992: global allocation across markets, not stock selection. And the estimation burden collapses — 13 assets means 91 covariance parameters instead of 325, on inputs whose correlations are far more stable.

### Systematic view engines

Typed views cannot be tested. You form them today, so applying them to past trades is look-ahead bias — which is why a backtest of manual views is meaningless. A **rule** can be rebuilt at every rebalance from data available at that moment, so the backtest finally answers the question that matters: *do views add anything?*

Four engines, each a pure function of price history: **momentum (12-1)**, **short-term reversal**, **trend (price vs 200d)**, **low volatility**.

**Confidence is earned, not typed.** Each engine's realised hit rate is measured by walking forward through the history — form the view using only past prices, check the sign of what actually happened — and converted by

```
confidence = 2 x (hit rate - 50%)
```

shrunk for sample size and capped below 100%. A rule that is right half the time carries no information and gets **zero** confidence, so the posterior ignores it entirely. On a random walk, no engine earns meaningful confidence — that is a test, not a claim.

The backtest then runs the rule against pure equilibrium and reports the difference. A universe you did not pick, views you did not guess, confidence you did not choose.

### Volatility-targeting overlay

Scales exposure to a volatility target, computed from a trailing window of realised portfolio returns. Volatility clusters — a turbulent week follows a turbulent week — while returns don't, so yesterday's volatility forecasts tomorrow's even though yesterday's return doesn't. Capped at 1.0x by default, so it can only ever move into cash, never borrow; the un-invested fraction earns the risk-free rate.

**The target is set as a fraction of the portfolio's own volatility (default 75%), not as a fixed number.** This matters more than it sounds. A fixed 12% target is sensible for a diversified book and actively harmful for a concentrated one — on a 24%-volatility US portfolio it parks you at 56% exposure permanently and costs 12.6pp of return a year. Tested across two very different simulated books:

```
                    exposure   ret/yr     vol    maxDD  Sharpe
LOW-VOL book (natural 3.6%)
  absolute 12%          100%   16.04%    3.6%    -2.1%    3.30   <- inert
  relative 75%           77%   13.38%    2.7%    -1.6%    3.46
HIGH-VOL book (natural 10.9%)
  absolute 12%           96%   16.22%   10.3%   -12.1%    1.17   <- barely engages
  relative 75%           77%   14.61%    8.0%    -9.8%    1.31
```

The relative target is recomputed at each rebalance **from the training window only** — computing it from full-sample volatility would be look-ahead bias. A regression test proves it: two price series with identical history and different futures produce bit-identical exposures before the divergence point.

The backtest runs the strategy both ways and plots both, so you can see what the overlay cost or earned rather than taking it on faith.

### Results on real data

**Headline: Global multi-asset ETFs, yearly, 2022-04 to 2026-08.** This is the run whose absolute numbers mean something — every one of the thirteen ETFs existed throughout the test window, so there is no survivorship bias and no selection bias. 100,000 USD becomes **150,198 USD** (9.63% a year) with a −13.8% worst drawdown, against **132,968** for an equal-weight basket of the same assets.

**The model reproduces its prior, exactly as the theory requires.** With no meaningful views the posterior equals the prior, so the unconstrained optimum *is* the market portfolio. On the ETF universe, where no constraint binds, this is directly visible in the holdings table:

```
        BL wt %   Market wt %
SPY      53.80       53.80
IWM       5.42        5.42
GLD       8.82        8.82
EEM       1.97        1.97
```

Identical to two decimals. And out of sample, across four different universes and horizons:

| Run | BL (no overlay) vs its own prior |
|---|---|
| India Nifty, yearly | −0.23pp |
| US S&P, yearly | −0.60pp |
| US S&P, monthly | +0.05pp |
| **Global multi-asset ETFs, yearly** | **−0.22pp** |

Four markets, δ from 2.04 to 4.69 — and the gap never exceeds two-thirds of a percentage point. That is the central validation of this implementation.

**Views added nothing, and the app measured it rather than assuming it.** Momentum's realised hit rate was 51% over 74 checks on the ETF universe and 53% over 68 on Indian equities — barely distinguishable from chance — so the calibration assigned 2% and 4% confidence respectively, and the portfolio tilted 0.7% and 3.0% away from the market. That reproduces the null result found across three view engines and two universes in the underlying CQF study, and it is worth contrasting with what happens when confidence is *typed*: at a hand-picked 75%, the same portfolio tilted 37.4% — roughly nineteen times harder than the evidence supported.

**What the optimiser is worth, in one column.** On the same thirteen assets, plain mean-variance puts **66.8% in GLD and 33.2% in SPY** and nothing anywhere else — the classic corner solution. Black-Litterman produces a graded thirteen-asset allocation from the identical covariance matrix and constraints.

**The volatility overlay buys drawdown, not return.** On the US stock universe it took Sharpe from 1.02 to 1.05 with a drawdown nine points shallower than the prior's; on the ETF universe it cut drawdown from −18.9% to −13.8% and cost 1.6pp of return. Reliable on risk, neutral-to-negative on Sharpe — which is what the simulations said to expect.

**A caveat on the stock universes.** Their absolute returns are not meaningful: the universe is the most liquid names in each index *today*, which are disproportionately the ones that won over the test period. The US stock universe beat the S&P 500 by 16.6pp a year on that basis alone, and the app flags any gap above 8pp as look-ahead bias. Only the strategy-versus-its-own-prior comparison survives there. The ETF universes exist precisely so there is a run without that problem.

### Honest diagnostics

The app is built to tell you when its own output is unreliable:

- **δ fallback warning** — if the market portfolio didn't beat cash over the estimation window, δ can't be implied and the model falls back to a constant. It says so, loudly, because π is then no longer *this market's* equilibrium.
- **Covariance reliability check** — flags when you have too few observations per asset for Σ (and therefore π) to mean much.
- **Negative-Sharpe guard** — when excess returns are negative the Sharpe ratio stops being comparable, because reducing volatility makes a negative ratio look *worse*. The app suppresses those comparisons and points you at drawdown instead.
- **Capital adequacy** — tells you how much capital this portfolio actually needs to be implementable, how many names you can hold at your current capital, and how much of your money would sit idle after rounding to whole shares.
- **Whole-book share allocation** — most tools floor each position independently, so nineteen names lose nineteen part-lots and a large slice of the capital never gets invested. This one floors once, then spends the remainder greedily on whichever position sits furthest below its target. On a real ₹100,000 / 19-name run that cut idle cash from **14.5% to 0.24%** and *improved* weight tracking error at the same time. Board lots (Tokyo, Hong Kong) are respected and the capital is never overspent.
- **Look-ahead warning** — applying today's views throughout a historical backtest is look-ahead bias; if you switch it on, the app says so and plots the honest no-views line alongside.
- **Horizon guidance** — Black-Litterman is a strategic model whose prior is market capitalisation, which barely moves week to week. The app warns when you pick a horizon it wasn't designed for.

### Everything from the Markowitz app

Nine markets (S&P 500, FTSE 350, HDAX, SBF 120, Nikkei 225, Nifty 500, S&P/TSX, ASX 200, Hang Seng), automatic liquidity screening, Simple and Advanced modes, an interactive Plotly efficient frontier with click-through allocations, a walk-forward out-of-sample backtest that re-derives the equilibrium at every rebalance, FX conversion, board-lot rounding, an executable trade list with CSV export, and run-to-run comparison.

### History windows — all three horizons see identical data

How long you intend to *hold* and how much history you should *estimate from* are different questions, and conflating them was a real flaw in an earlier version: choosing "weekly" quietly cut the estimation window to two years and left the covariance matrix — and therefore π = δΣw — running on fumes. A later version fixed the window but still gave the yearly horizon weekly returns, so it saw 780 observations where the others saw 3,780.

Both are now decoupled. Every horizon estimates from the same 15-year daily history and differs **only** in how often it rebalances:

| Horizon | Estimation window | Returns | Rebalances | Observations |
|---|---|---|---|---|
| Weekly — short-term | up to 15 years | daily | ~50/year | ~3,780 |
| Monthly — medium-term | up to 15 years | daily | ~12/year | ~3,780 |
| Yearly — long-term | up to 15 years | daily | ~1/year | ~3,780 |

Neither setting is the "good" one. More rebalances give more decisions to judge the strategy on and cost more in turnover; fewer give a cleaner read on the long-run allocation from a smaller sample of decisions. The app says exactly that and does not steer.

**The window is chosen from the data, not imposed.** Demanding a fixed 15 years would silently discard every asset younger than that — half a Nifty universe in practice — while a naive `dropna()` would truncate *every* asset to the youngest one's history. Instead the app walks the window down from 15 years and stops at the longest length that still supports a full-sized universe, backfilling excluded names from the next most liquid long-lived candidates. A universe where nothing is old shortens the window gracefully rather than failing.

Why longer helps where it does: Σ's estimation error falls roughly as 1/√T, and since the prior is built from Σ, a better Σ means a better prior rather than merely better risk numbers. It helps far less with anything mean-like — a sample mean's standard error is driven by volatility, not sample length — which is precisely why Black-Litterman does not estimate expected returns from history in the first place.

### On reading the output

The app reports a **model-implied** annual return, not a forecast. It is what the posterior implies for the chosen weights, and the posterior is built from today's market capitalisations and a historical covariance matrix — neither of which predicts anything. The backtest is the only figure carrying a predictive claim, and it is a single historical path.

Where the app compares things, it states both sides of the trade rather than declaring a winner. The volatility overlay, for instance, reduced drawdown in most simulated runs and moved the Sharpe ratio in either direction about equally often — so the app reports the risk reduction as the expected effect and any Sharpe gain as that window's luck.

### Per-market recommended settings

Choosing a market applies sensible defaults automatically — local risk-free rate and realistic all-in trading costs (India carries STT and stamp duty; the UK and Hong Kong carry stamp; the US does not). Alongside: 12% single-stock cap, δ implied, Ω Idzorek, 75% view confidence, a volatility target at 75% of the portfolio's own risk, and a 0.45 backtest training fraction.

The risk-free rates are August 2026 government bill yields, hardcoded in `MARKET_DEFAULTS` at the top of the file. They will drift — update them there.

## Run it locally

```bash
git clone https://github.com/SunnyAlexV/Black-Litterman-model.git
cd Black-Litterman-model
pip install -r requirements.txt
streamlit run black_litterman_portfolio.py
```

On Windows, if `streamlit` isn't on your PATH, use `python -m streamlit run black_litterman_portfolio.py`.

## Tests

```bash
python test_bl_core.py
```

99 headless checks covering the numeric core, with Streamlit and yfinance stubbed so nothing needs a server or a network connection. They check reverse optimisation round-trips, that the two posterior formulas agree, Ω construction under both methods, view parsing and excess-return conversion, τ invariance, confidence linearity, the volatility overlay's no-look-ahead property, the negative-Sharpe artifact, the share allocator (never overspends, respects board lots, strands less than one lot), the relative volatility target, the systematic view engines — both proven free of look-ahead by divergent-futures tests — and the adaptive history window.

Two properties worth knowing about, both locked down by tests:

**τ has no effect on posterior returns.** Under both Ω methods, Ω is itself proportional to τ, so it cancels out of the posterior entirely. τ only bites when posterior covariance is enabled. To change how far your views move the portfolio, use confidence.

**Under Idzorek, the tilt is exactly linear in confidence:**

```
shift in expected return = confidence × (your view − what the equilibrium already implied)
```

A view 5pp above equilibrium at 75% confidence moves μ by exactly 3.75pp. It follows that a view agreeing with the equilibrium changes nothing at any confidence — check the *Surprise* column before reaching for the confidence dial.

## How to use it

1. Pick a market, currency, amount and horizon. **Yearly — long-term** is what the model is designed for.
2. Click **Load universe & equilibrium** and look at what the market is implying before you type anything.
3. Add views. Start with one — the confidence identity is exact for a single view, so you can verify every number by hand. (With multiple correlated views the tilts interact and the identity becomes approximate.)
4. Build, then read the tabs: Holdings, What your views did, Efficient frontier, Backtest, What to buy, Compare runs.
5. Save runs at 25%, 50% and 75% confidence and compare. That single sweep teaches more about the model than any amount of reading.

**The null test:** leave the views table empty and build. With no views the posterior must equal the prior exactly, so μ_BL should match π in every row. That's the check with a known answer.

## Limitations

Read these before drawing conclusions, and certainly before allocating real money.

- **The universe is a hardcoded list, not a real index.** "Nifty 500" is 86 tickers; "S&P 500" is 100. The theoretical claim of Black-Litterman rests on the prior being *the market portfolio* — a cap-weighted slice of 25 liquid large caps is not that.
- **Market caps come from Yahoo and are unverified.** The whole prior is built on them. Missing caps are filled with the median and reported; a cap that is present but wrong fails silently.
- **Survivorship bias.** Optimises over each index's *current* constituents, so delisted and demoted companies never appear. This systematically overstates historical returns.
- **The backtest is a single path.** One market, one sequence of history, a handful of rebalances. It cannot separate skill from luck.
- **Views are opinions.** The model faithfully amplifies them in proportion to your stated confidence. Confident and wrong is the expensive quadrant.
- Costs are modelled as flat basis points on turnover. No market impact, no slippage, no taxes.

Closing this gap is mostly a data problem, not a coding one: point-in-time index membership, vendor market caps, and rolling-window backtests across many start dates.

## Tech stack

Python · Streamlit · yfinance · pandas · NumPy · SciPy · Matplotlib · Plotly

## Disclaimer

Educational and analytical tool. **Not investment advice.** Expected returns are estimates, not forecasts; back-tested performance does not guarantee future results; the equilibrium prior is only as good as the market-cap data behind it. Market data comes from Yahoo Finance and may be delayed or imperfect. Do your own research and consider consulting a licensed financial professional before investing.

## References

- Black, F. and Litterman, R. (1992), "Global Portfolio Optimization", *Financial Analysts Journal*
- He, G. and Litterman, R. (1999), "The Intuition Behind Black-Litterman Model Portfolios", Goldman Sachs
- Idzorek, T. (2005), "A Step-by-Step Guide to the Black-Litterman Model"
- Ledoit, O. and Wolf, M. (2004), "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices"

## License

MIT.
