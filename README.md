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
| 1 | Data: markets, price history, outcomes in SQLite; volume report | **built** |
| 2 | Model: XGBoost, no look-ahead, time-based split, calibration + Brier | not started |
| 3 | Paper trading log: edges > 5 pts, quarter Kelly, slippage, auto-settle, append-only | not started |
| 4 | Flask dashboard | not started |
| 5 | Automation: scheduled scoring, retrain every ~50 resolutions | not started |

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

## Phase 1 commands

```bash
source venv/bin/activate
python -m pmodel ingest-markets all          # active + resolved market listings and outcomes
python -m pmodel volume-report               # prints the table, saves reports/volume_distribution.png
# ...choose MIN_VOLUME_USD and put it in .env, then:
python -m pmodel ingest-history              # price history for resolved markets above the cutoff
python -m pmodel status                      # row counts
```

Quick test without pulling everything: `python -m pmodel ingest-markets resolved --max 2000`.

Running `ingest-markets active` again later adds a new *snapshot* of every active
market, so over time we build our own record of what prices and volumes looked
like at each moment.

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
