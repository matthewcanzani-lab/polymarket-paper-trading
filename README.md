# polymarket-model-v2

A **paper-trading-only** prediction model for Polymarket. It reads public market
data, estimates the true probability of YES, and logs *fake* bets so we can
build an honest, forward-tested track record.

> **Safety rule:** this project never places real orders, never connects a
> wallet, and never handles private keys. All network access goes through
> `pmodel/safe_http.py`, which can only make GET requests to a short allowlist
> of public read-only hosts. `tests/test_safety.py` fails if anyone adds order
> or wallet code.

## Status

| Phase | What | Status |
|---|---|---|
| 1 | Data: markets, price history, outcomes in SQLite; volume report | **done** |
| 2 | Model: XGBoost, no look-ahead, time-based split, calibration + Brier | **done — no edge found yet** |
| 3 | Paper trading log: edges > 5 pts, quarter Kelly, slippage, auto-settle, append-only | **done — live ledger started 2026-10-22** |
| 4 | Flask dashboard (+ static export) | **done** |
| 5 | Automation: scheduled scoring, retrain every ~50 resolutions | **done (GitHub Actions)** |
| + | Weather strategy: forecasts + market blend, forward paper test | **live** |

## Set up on a new computer (from zero)

These steps are for a Mac. On Linux they're the same minus Homebrew.

1. **Install the tools** (one time per computer). Open Terminal and run:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   brew install python@3.12 git
   ```
   Python **3.11 or newer** is required (the Mac's built-in Python is too old).

2. **Get the code:**
   ```bash
   cd ~
   git clone https://github.com/matthewcanzani/polymarket-model-v2.git
   cd polymarket-model-v2
   ```
   GitHub will ask you to sign in because the repo is private. The easiest way
   is `brew install gh && gh auth login` before cloning.

3. **Create the virtual environment** (a private folder of Python packages just
   for this project, so it can't clash with anything else on the computer):
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate        # do this every time you open a new Terminal
   pip install -r requirements.txt
   ```
   Your prompt shows `(venv)` when it's active.

4. **Settings:**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` (e.g. `open -e .env`). Nothing in it is secret.

5. **Check everything works:**
   ```bash
   python -m pytest -q
   python -m pmodel status
   ```

6. **Restore your prediction log** — nothing to do: `data/predictions.db` comes
   down with `git clone`. If git's copy is somehow lost, copy the newest
   `predictions_*.db` from your `BACKUP_DIR` (iCloud) to `data/predictions.db`.

7. **Re-download market data** (it is *not* in git; it's big and can always be
   re-pulled): see "Phase 1 commands" below.

## Which markets we use (the "universe")

Defined once in `pmodel/universe.py`, settings in `pmodel/config.py`:

- **Lifetime volume >= $10,000** (`MIN_VOLUME_USD`). Chosen from the Phase 1
  volume report: below ~$5k the bid-ask spread doubles; above $10k it barely
  improves, and ~1,000 open markets above $10k resolve in any 30-day window.
- **No crypto "Up or Down" coin flips** (5-min/15-min/hourly/daily BTC, ETH, SOL,
  XRP markets). They were ~170k of the ~190k crypto markets above $10k.

Result (Oct 2026): ~270k resolved markets since 2024 and ~5,300 open ones.

## Phase 1 commands

```bash
source venv/bin/activate
python -m pmodel ingest-markets active                      # open markets (+ a snapshot of each)
python -m pmodel ingest-markets resolved --min-volume 1000  # resolved since HISTORY_START_DATE (~1.5h)
python -m pmodel volume-report                              # table + reports/volume_distribution.png
python -m pmodel ingest-history --sample --limit 50000      # price history for a fixed random sample
python -m pmodel status                                     # row counts
```

`--min-volume 1000` skips ~450k tiny markets the universe never uses. Later
top-ups only need recent markets: `ingest-markets resolved --since 2026-10-01 --min-volume 1000`.

**Price history quirks** (Polymarket's public API, checked Oct 2026): closed
markets only return hourly data in windows of <= 14 days, so markets that lived
<= 60 days get hourly history and longer ones get one point per 12 hours.

Running `ingest-markets active` again later adds a new *snapshot* of every active
market, so over time we build our own record of what prices and volumes looked
like at each moment.

## Phase 2: the model

```bash
python -m pmodel ingest-trades      # hourly traded-$ per market (needed for volume-at-the-time)
python -m pmodel train              # builds snapshots, trains, backtests, saves models/<version>/
```

Each version folder has `model.json`, `meta.json` (settings, split dates,
metrics), `backtest_report.txt` and `calibration.png`. Everything there is a
**BACKTEST** (historical, out-of-sample by time), never forward paper results.

**How look-ahead bias is prevented** (details in `pmodel/model/features.py`,
`dataset.py`, `train.py`):
1. Features only see an allowlist of facts known when a market opens
   (category, creation date, scheduled end, Yes/No type). Final volume, close
   time, final prices and the outcome are never passed in.
2. Prices and trades are used only once their time bucket has fully finished
   before the prediction moment.
3. A market is only sampled at moments when it was still open AND had already
   reached $10k of volume *by then* (selecting on final volume leaked outcomes —
   see `models/v20261005_031649/SUPERSEDED.md`).
4. Hourly-vs-12h price data is chosen from the *scheduled* length, not the actual one.
5. Only markets scheduled to end 7+ days before the data pull are used, so
   early-resolving markets aren't over-represented at the recent end.
6. Time split: train only on markets resolved before the validation period;
   test on the most recent ~20% of predictions.
7. `tests/test_features.py` scrambles all future prices/trades and checks
   features don't change.

**Result (v20261005_054807, 4,101 test markets, Jun–Sep 2026):** model Brier
0.1283 = market 0.1283; XGBoost found no correction that helped on validation.
Polymarket prices in this universe are already well calibrated, and price/
volume history alone doesn't beat them.

## Phase 3: paper trading

```bash
python -m pmodel score     # score open markets, log PAPER bets to data/predictions.db
python -m pmodel settle    # settle paper bets whose markets resolved
python -m pmodel ledger    # show bankroll/bets and verify nothing was edited
```

Rules (all in `pmodel/config.py`): open markets in the universe with >= $10k
volume so far, ending within 30 days; bet when the model and the market midpoint
differ by > 5 points; fill = walk the real order book + 1 cent per share
slippage; stake = quarter Kelly on a fake $1,000, max 5% per bet, never more
than available cash; one bet per market; only if still +EV after costs.

The ledger is append-only (SQLite triggers block UPDATE/DELETE), every row is
hash-chained to the previous one (`ledger` detects any edit), and the file is
committed to git so GitHub holds dated copies. Trading logic lives in
`pmodel/trading/` and only talks to the `MarketSource` interface.

## Phase 4: dashboard

```bash
python -m pmodel dashboard     # live Flask app at http://127.0.0.1:5000
python -m pmodel build-site    # same pages as static HTML in site/ — open site/index.html anywhere
```

Pages: **Overview** (bankroll, success-criteria progress, latest run), **Live
Markets**, **Edge Finder**, **Paper Portfolio** (P&L curve, open and settled
bets), **Model Stats** (forward calibration, Brier, win rate, return on stake,
by-category P&L). FORWARD results (green, solid border) and BACKTEST results
(grey, dashed box) are always shown separately and labelled.

The static export needs no server, so it works on a locked-down computer and
can be published (e.g. GitHub Pages) for others to see.

## Weather strategy

Daily high/low temperature markets (49 cities). Probability = a blend of the
market midpoint and a forecast-based probability (as-issued Open-Meteo forecast,
corrected by each station's learned bias and spread). See
`models/weather/FINDINGS.md` for the backtest: forecasts beat recorded prices on
Brier score, but the backtest can't prove the edge was tradeable, so it is
being tested FORWARD with every fake fill priced from the live order book.

Rules: bet only 1-2 days ahead (never on the day), > 5-point disagreement with
the live midpoint, walk the real order book + 1c, quarter Kelly, max 5% of the
bankroll per bet and per city-day, max 50% of the bankroll at risk overall.

```bash
python -m pmodel weather-score     # score open temperature markets, log PAPER bets
python -m pmodel weather-fit       # full refit (needs the market DB; run in a cloud session)
python -m pmodel weather-refit     # light refit of station bias/spread from fresh API data
```

Polymarket changed the resolution rule on 23 Oct 2026 (hourly routine METAR
reports instead of NOAA "all times"); measured effect on 1,794 recent
station-days: ~0.0C (`models/weather/rule_change_shift_2026-10.json`).

## Phase 5: automation (GitHub Actions)

No computer of yours needs to stay on. Two workflows in `.github/workflows/`:

* **Paper trading** — 05:17 and 17:17 UTC daily: verify the ledger, settle
  resolved bets, run the weather and general scorers, rebuild the dashboard,
  and commit `data/predictions.db` back to the repo (each commit is a dated,
  outside record of every bet). The dashboard is attached to each run
  (Actions tab -> the run -> Artifacts -> dashboard).
* **Weather model refit** — Sundays: if 50+ weather markets resolved since
  the current version, re-learn station bias/spread and save a NEW version
  (`models/weather/weather-v*.json`); old versions are kept.

The general model needs the multi-GB market database to retrain, so that is
done by hand in a cloud session (`ingest-*`, then `train`); every version is
kept in `models/`.

Run either workflow by hand: GitHub -> Actions -> pick it -> "Run workflow".

## Where the data lives — and why there are two databases

| File | Contents | In git? | Backed up? |
|---|---|---|---|
| `data/market_data.db` | markets, snapshots, price history, outcomes | **No** (large, re-downloadable) | not needed |
| `data/predictions.db` | the paper-bet / prediction log (Phase 3) | **Yes** | git **and** `BACKUP_DIR` |

They're separate so that rebuilding or deleting the market cache can never touch
the prediction log. The prediction log is the one thing that can't be recreated:
it proves each prediction was made *before* the market resolved.

**Backups:** set `BACKUP_DIR` in `.env` to an iCloud Drive folder, e.g.
`~/Library/Mobile Documents/com~apple~CloudDocs/polymarket-model-v2-backups`,
then run `python -m pmodel backup`. Phase 5 schedules this nightly. Keeps the
newest 60 copies.

## Project layout

```
pmodel/
  config.py            all settings (read from .env)
  safe_http.py         the only network code: GET-only, allowlisted hosts
  sources/
    base.py            platform-neutral Market / Resolution / PricePoint + MarketSource interface
    polymarket.py      Polymarket Gamma + CLOB adapter (the only Polymarket-specific code)
  db.py                market database schema + writes
  ingest.py            pulls from any MarketSource into the database
  analysis/volume.py   volume distribution report + chart
  backup.py            prediction-log backup
  __main__.py          command line (`python -m pmodel ...`)
tests/                 run with `python -m pytest -q`
```

To add another exchange later (e.g. a US-regulated one), write
`pmodel/sources/<exchange>.py` implementing `MarketSource`. Nothing else should
need to know which platform the data came from.

## Saving your work

At the end of every session:
```bash
git add -A
git commit -m "describe what changed"
git push
```
`data/predictions.db` is included on purpose. `.env`, `venv/` and the market
database are excluded by `.gitignore`.
