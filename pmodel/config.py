"""All tunable settings live here, so nothing important is buried in the code.

Values come from environment variables (or a .env file in the project root),
falling back to the defaults below.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(name, default):
    value = os.environ.get(name)
    return default if value in (None, "") else value


DATA_DIR = Path(_env("DATA_DIR", PROJECT_ROOT / "data"))
REPORTS_DIR = PROJECT_ROOT / "reports"

# Two separate databases, on purpose:
#  - market_data.db: big, and re-downloadable from the API at any time. Not in git.
#  - predictions.db: the paper-trading log. Small and irreplaceable. Committed to
#    git AND copied to BACKUP_DIR. Kept separate so a corrupted or deleted market
#    cache can never take the prediction log down with it.
MARKET_DB_PATH = Path(_env("MARKET_DB_PATH", DATA_DIR / "market_data.db"))
PREDICTIONS_DB_PATH = Path(_env("PREDICTIONS_DB_PATH", DATA_DIR / "predictions.db"))

# Minimum lifetime volume (USD) for a market to count. Chosen from the Phase 1
# volume report (see pmodel/universe.py for the reasoning).
MIN_VOLUME_USD = float(_env("MIN_VOLUME_USD", 10_000))

# Markets excluded from the universe regardless of volume: rapid-fire recurring
# crypto coin flips. Matched on series recurrence and on question wording.
# Sub-daily only: sports leagues also run "daily"/"weekly" series, so those stay in.
# (The daily crypto "Up or Down" markets are caught by the question pattern.)
EXCLUDE_RECURRENCES = ("1m", "5m", "15m", "30m", "1h", "hourly", "4h")
EXCLUDE_QUESTION_PATTERNS = ("Up or Down",)

# Ignore resolved markets that ended before this date (older ones are thin and
# the platform behaved differently).
HISTORY_START_DATE = _env("HISTORY_START_DATE", "2024-01-01")

# Markets that traded for at most this many days get hourly price history;
# longer ones get one point every 12 hours (hourly would take many requests each).
HOURLY_MAX_DAYS = int(_env("HOURLY_MAX_DAYS", 60))

# Be polite to the public API: pause between requests.
REQUEST_PAUSE_SEC = float(_env("REQUEST_PAUSE_SEC", 0.25))

BACKUP_DIR = _env("BACKUP_DIR", None)
if BACKUP_DIR:
    BACKUP_DIR = Path(os.path.expanduser(BACKUP_DIR))

# ---------------------------------------------------------------------------
# Paper trading (Phase 3). FAKE money only — nothing here can place an order.
# ---------------------------------------------------------------------------
BANKROLL_START_USD = float(_env("BANKROLL_START_USD", 1000))
# Bet only when model and market midpoint disagree by more than this (5 points).
EDGE_THRESHOLD = float(_env("EDGE_THRESHOLD", 0.05))
# Fraction of the full Kelly stake (0.25 = quarter Kelly).
KELLY_MULTIPLIER = float(_env("KELLY_MULTIPLIER", 0.25))
# Extra cost per share on top of walking the real order book (1 cent).
# Phase 1: median bid-ask spread at >= $10k volume was ~1.0c; paying the ask
# already covers the spread, this covers latency and price moving against us.
SLIPPAGE_PER_SHARE = float(_env("SLIPPAGE_PER_SHARE", 0.01))
# Never stake more than this share of the bankroll on one bet.
MAX_STAKE_FRACTION = float(_env("MAX_STAKE_FRACTION", 0.05))
MIN_STAKE_USD = float(_env("MIN_STAKE_USD", 1))
# Only score markets scheduled to end within this many days (faster settlement).
SCORE_MAX_DAYS_TO_END = int(_env("SCORE_MAX_DAYS_TO_END", 30))

# Latest scorer output (every market scored, with model vs market). Overwritten
# each run; feeds the dashboard. Not part of the tamper-evident ledger.
LATEST_SCORES_PATH = Path(_env("LATEST_SCORES_PATH", DATA_DIR / "latest_scores.json"))
