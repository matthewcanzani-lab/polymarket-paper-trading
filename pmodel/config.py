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

# Minimum lifetime volume (USD) for a market to count. 0 means "no cutoff yet".
MIN_VOLUME_USD = float(_env("MIN_VOLUME_USD", 0))

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
