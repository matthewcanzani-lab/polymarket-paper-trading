"""Platform-neutral data shapes and the interface every data source implements.

To plug in a different (e.g. US-regulated) exchange later, write a new module
in this folder with a class that implements MarketSource. The database, model,
scorer and dashboard only ever see these dataclasses.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class Market:
    platform: str               # e.g. "polymarket"
    market_id: str              # the platform's own ID
    question: str
    slug: Optional[str]
    category: Optional[str]
    event_title: Optional[str]
    outcome_yes: str            # label of the outcome we call "YES" (usually "Yes")
    outcome_no: str
    is_yes_no: bool             # True when the outcomes are literally Yes/No
    yes_token_id: Optional[str] # ID used to fetch YES price history
    created_at: Optional[str]   # ISO-8601 UTC strings throughout
    start_date: Optional[str]
    end_date: Optional[str]     # scheduled end
    closed_time: Optional[str]  # when trading actually stopped
    active: bool
    closed: bool
    volume_usd: float           # lifetime volume AS OF NOW (see note in db.py)
    liquidity_usd: float
    yes_price: Optional[float]  # current/last YES price, 0..1
    best_bid: Optional[float]
    best_ask: Optional[float]
    tags: list = field(default_factory=list)  # raw platform tags, for later analysis
    series: Optional[str] = None              # recurring series this market belongs to, if any
    recurrence: Optional[str] = None          # how often that series repeats, e.g. "5m", "daily"


@dataclass
class Resolution:
    platform: str
    market_id: str
    resolved_yes: Optional[int]  # 1 = YES won, 0 = NO won, None = unclear/void
    status: str                  # "resolved" or "ambiguous"
    resolved_at: Optional[str]


@dataclass
class PricePoint:
    ts: int       # unix seconds, UTC
    price: float  # YES price, 0..1


class MarketSource(ABC):
    platform: str

    @abstractmethod
    def iter_markets(self, closed: bool, end_date_min: Optional[str] = None,
                     volume_min: Optional[float] = None) -> Iterator[dict]:
        """Yield raw market records from the platform API.

        end_date_min / volume_min let the platform filter server-side, so we don't
        download hundreds of thousands of tiny markets we'd never use.
        """

    @abstractmethod
    def parse_market(self, raw: dict) -> Optional[Market]:
        """Convert a raw record to a Market, or None if it's not a usable binary market."""

    @abstractmethod
    def parse_resolution(self, raw: dict) -> Optional[Resolution]:
        """Return the outcome for a closed market, or None if it's still open."""

    @abstractmethod
    def price_history(self, market: Market) -> list[PricePoint]:
        """Return the YES price history for a market, oldest first."""

    @abstractmethod
    def history_step_seconds(self, market: Market) -> int:
        """Spacing between price points for this market. Features treat a point
        stamped `ts` as only knowable at `ts + step` (it may summarise that bucket)."""
