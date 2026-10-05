"""Polymarket adapter: public Gamma API (markets) + public CLOB price history.

Everything here is read-only public data. No API keys, no wallet, no orders.
"""
import json
import re
from datetime import datetime, timezone
from typing import Iterator, Optional

from pmodel import config, safe_http
from pmodel.sources.base import Market, MarketSource, PricePoint, Resolution

GAMMA_URL = "https://gamma-api.polymarket.com/markets/keyset"  # cursor paging (plain /markets caps the offset)
PRICES_URL = "https://clob.polymarket.com/prices-history"
PAGE_SIZE = 100  # the API's maximum page size

# Price history quirks (checked against the live API, Oct 2026):
#  - for closed markets, interval=max only returns data at >= 12h steps;
#  - hourly data must be requested in windows of at most 14 days.
HOURLY_WINDOW_SEC = 14 * 24 * 3600
COARSE_FIDELITY_MIN = 720

# Polymarket has no single "category" field; markets carry many free-form tags.
# We map them onto a few broad categories, checked in this order.
CATEGORY_TAGS = [
    ("Esports", {"esports", "counter-strike", "cs2", "league of legends", "dota 2", "valorant"}),
    ("Sports", {"sports", "nba", "nfl", "mlb", "nhl", "soccer", "tennis", "ufc", "golf", "f1",
                "cricket", "college football", "ncaa", "epl", "boxing"}),
    ("Crypto", {"crypto", "bitcoin", "ethereum", "solana", "crypto prices"}),
    ("Politics", {"politics", "elections", "us election", "trump", "global elections", "congress"}),
    ("Geopolitics", {"geopolitics", "world", "middle east", "ukraine", "israel", "china", "iran"}),
    ("Economy", {"economy", "finance", "business", "fed", "stocks", "earnings", "inflation", "ipos"}),
    ("Tech & Science", {"tech", "ai", "science", "space"}),
    ("Culture", {"culture", "pop culture", "movies", "music", "awards", "celebrities", "mentions"}),
    ("Weather", {"weather", "climate", "temperature"}),
]

# A closed market counts as cleanly resolved when its final YES price is
# at least this close to 1 or 0.
RESOLVED_THRESHOLD = 0.99


def _maybe_json_list(value):
    """Gamma returns some list fields as JSON *strings* (e.g. '["Yes","No"]')."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None


def _float(value) -> Optional[float]:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def to_iso_utc(value) -> Optional[str]:
    """Normalise the API's assorted date formats to 'YYYY-MM-DDTHH:MM:SSZ' (UTC)."""
    if not value:
        return None
    text = str(value).strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "T" in text:  # "...12:00:00+00" -> "...12:00:00+00:00" (date-only strings untouched)
        text = re.sub(r"([+-]\d{2})$", r"\1:00", text)
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_to_unix(iso: Optional[str]) -> Optional[int]:
    if not iso:
        return None
    return int(datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())


def tag_labels(raw: dict) -> list[str]:
    tags = list(raw.get("tags") or [])
    for event in raw.get("events") or []:
        tags += event.get("tags") or []
    labels = [t.get("label") for t in tags if isinstance(t, dict) and t.get("label")]
    return sorted(set(labels))


def _series(raw: dict):
    """(series slug, recurrence) for markets in a recurring series like 'btc-up-or-down-5m'."""
    for event in raw.get("events") or []:
        for series in event.get("series") or []:
            if isinstance(series, dict) and series.get("slug"):
                return series["slug"], series.get("recurrence")
        if event.get("seriesSlug"):
            return event["seriesSlug"], None
    return None, None


def category_from_tags(labels: list[str], fallback: Optional[str] = None) -> Optional[str]:
    lowered = {label.lower() for label in labels}
    for category, keys in CATEGORY_TAGS:
        if lowered & keys:
            return category
    return fallback


class PolymarketSource(MarketSource):
    platform = "polymarket"

    def iter_markets(self, closed: bool, end_date_min: Optional[str] = None,
                     volume_min: Optional[float] = None) -> Iterator[dict]:
        cursor = None
        while True:
            params = {"limit": PAGE_SIZE, "closed": str(closed).lower(), "include_tag": "true"}
            if not closed:
                params["active"] = "true"
            if end_date_min:
                params["end_date_min"] = end_date_min
            if volume_min:
                params["volume_num_min"] = volume_min
            if cursor:
                params["after_cursor"] = cursor
            page = safe_http.get_json(GAMMA_URL, params=params)
            markets = page.get("markets") or []
            yield from markets
            cursor = page.get("next_cursor")
            if not markets or not cursor:
                return

    def parse_market(self, raw: dict) -> Optional[Market]:
        outcomes = _maybe_json_list(raw.get("outcomes"))
        tokens = _maybe_json_list(raw.get("clobTokenIds"))
        prices = _maybe_json_list(raw.get("outcomePrices"))
        if not outcomes or len(outcomes) != 2:
            return None  # only two-outcome markets
        events = raw.get("events") or []
        yes_price = _float(prices[0]) if prices and len(prices) == 2 else _float(raw.get("lastTradePrice"))
        volume = _float(raw.get("volumeNum"))
        if volume is None:
            volume = _float(raw.get("volume")) or 0.0
        liquidity = _float(raw.get("liquidityNum"))
        if liquidity is None:
            liquidity = _float(raw.get("liquidity")) or 0.0
        return Market(
            platform=self.platform,
            market_id=str(raw["id"]),
            question=raw.get("question") or "",
            slug=raw.get("slug"),
            category=category_from_tags(tag_labels(raw), raw.get("category")),
            tags=tag_labels(raw),
            series=_series(raw)[0],
            recurrence=_series(raw)[1],
            event_title=events[0].get("title") if events else None,
            outcome_yes=str(outcomes[0]),
            outcome_no=str(outcomes[1]),
            is_yes_no=[str(o).lower() for o in outcomes] == ["yes", "no"],
            yes_token_id=str(tokens[0]) if tokens and len(tokens) == 2 else None,
            created_at=to_iso_utc(raw.get("createdAt")),
            start_date=to_iso_utc(raw.get("startDate")),
            end_date=to_iso_utc(raw.get("endDate")),
            closed_time=to_iso_utc(raw.get("closedTime")),
            active=bool(raw.get("active")),
            closed=bool(raw.get("closed")),
            volume_usd=volume,
            liquidity_usd=liquidity,
            yes_price=yes_price,
            best_bid=_float(raw.get("bestBid")),
            best_ask=_float(raw.get("bestAsk")),
        )

    def parse_resolution(self, raw: dict) -> Optional[Resolution]:
        if not raw.get("closed"):
            return None
        prices = _maybe_json_list(raw.get("outcomePrices")) or []
        yes = _float(prices[0]) if len(prices) == 2 else None
        uma = (raw.get("umaResolutionStatus") or "").lower()
        resolved_at = to_iso_utc(raw.get("closedTime")) or to_iso_utc(raw.get("endDate"))
        # If the oracle reports a status and it isn't "resolved", don't trust the prices yet.
        oracle_ok = uma in ("", "resolved")
        if oracle_ok and yes is not None and yes >= RESOLVED_THRESHOLD:
            return Resolution(self.platform, str(raw["id"]), 1, "resolved", resolved_at)
        if oracle_ok and yes is not None and yes <= 1 - RESOLVED_THRESHOLD:
            return Resolution(self.platform, str(raw["id"]), 0, "resolved", resolved_at)
        # Closed but not a clean 1/0: 50-50 split, voided, or still disputed.
        return Resolution(self.platform, str(raw["id"]), None, "ambiguous", resolved_at)

    def price_history(self, market: Market) -> list[PricePoint]:
        """Hourly history for markets SCHEDULED to last <= HOURLY_MAX_DAYS, else 12-hourly.

        Hourly costs one request per 14 days of market life, so long-running
        markets get the cheaper coarse series (one request).
        """
        if not market.yes_token_id:
            return []
        start = _iso_to_unix(market.created_at or market.start_date)
        scheduled_end = _iso_to_unix(market.end_date)
        # LOOK-AHEAD GUARD: choose the granularity from the SCHEDULED length, which is
        # known when the market opens. Using the actual close time would make "hourly
        # data" a hint that the market closed early (often meaning it resolved YES early).
        hourly = bool(start and scheduled_end
                      and 0 < scheduled_end - start <= config.HOURLY_MAX_DAYS * 86400)
        # How far to fetch doesn't leak anything (features only use points before the
        # prediction time), so cover the full span the market actually traded.
        end = max(filter(None, [scheduled_end, _iso_to_unix(market.closed_time)]), default=None)
        if hourly and end:
            seen = {}
            t = start
            while t < end:
                chunk = safe_http.get_json(PRICES_URL, params={
                    "market": market.yes_token_id, "startTs": t,
                    "endTs": min(t + HOURLY_WINDOW_SEC, end), "fidelity": 60})
                for p in _parse_history(chunk):
                    seen[p.ts] = p
                t += HOURLY_WINDOW_SEC
            return [seen[k] for k in sorted(seen)]
        data = safe_http.get_json(PRICES_URL, params={
            "market": market.yes_token_id, "interval": "max", "fidelity": COARSE_FIDELITY_MIN})
        return _parse_history(data)


def _parse_history(data) -> list[PricePoint]:
    rows = (data or {}).get("history") or []
    points = []
    for row in rows:
        ts, price = row.get("t"), _float(row.get("p"))
        if ts is not None and price is not None:
            points.append(PricePoint(int(ts), price))
    return sorted(points, key=lambda p: p.ts)
