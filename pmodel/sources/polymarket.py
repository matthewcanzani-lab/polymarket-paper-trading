"""Polymarket adapter: public Gamma API (markets) + public CLOB price history.

Everything here is read-only public data. No API keys, no wallet, no orders.
"""
import json
import re
from datetime import datetime, timezone
from typing import Iterator, Optional

from pmodel import config, safe_http
from pmodel.sources.base import Market, MarketSource, PricePoint, Resolution

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
PRICES_URL = "https://clob.polymarket.com/prices-history"
PAGE_SIZE = 500

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


def _category(raw: dict) -> Optional[str]:
    if raw.get("category"):
        return raw["category"]
    for event in raw.get("events") or []:
        if event.get("category"):
            return event["category"]
    for tag in raw.get("tags") or []:
        if isinstance(tag, dict) and tag.get("label"):
            return tag["label"]
    return None


class PolymarketSource(MarketSource):
    platform = "polymarket"

    def iter_markets(self, closed: bool, end_date_min: Optional[str] = None) -> Iterator[dict]:
        offset = 0
        while True:
            params = {
                "limit": PAGE_SIZE,
                "offset": offset,
                "closed": str(closed).lower(),
                "order": "id",
                "ascending": "true",
            }
            if not closed:
                params["active"] = "true"
            if end_date_min:
                params["end_date_min"] = end_date_min
            page = safe_http.get_json(GAMMA_URL, params=params)
            if isinstance(page, dict):  # tolerate a {"data": [...]} wrapper
                page = page.get("data", [])
            if not page:
                return
            yield from page
            if len(page) < PAGE_SIZE:
                return
            offset += PAGE_SIZE

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
            category=_category(raw),
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
        if not market.yes_token_id:
            return []
        fidelity = config.PRICE_HISTORY_FIDELITY_MIN
        data = safe_http.get_json(
            PRICES_URL,
            params={"market": market.yes_token_id, "interval": "max", "fidelity": fidelity},
        )
        points = _parse_history(data)
        if points:
            return points

        # Fallback: some closed markets return nothing for interval=max, but do
        # answer explicit time windows. Walk the market's lifetime in 30-day chunks.
        start = _iso_to_unix(market.created_at or market.start_date)
        end = _iso_to_unix(market.closed_time or market.end_date)
        if not start or not end or end <= start:
            return []
        window = 30 * 24 * 3600
        seen = {}
        t = start
        while t < end:
            chunk = safe_http.get_json(
                PRICES_URL,
                params={"market": market.yes_token_id, "startTs": t,
                        "endTs": min(t + window, end), "fidelity": fidelity},
            )
            for p in _parse_history(chunk):
                seen[p.ts] = p
            t += window
        return [seen[k] for k in sorted(seen)]


def _parse_history(data) -> list[PricePoint]:
    rows = (data or {}).get("history") or []
    points = []
    for row in rows:
        ts, price = row.get("t"), _float(row.get("p"))
        if ts is not None and price is not None:
            points.append(PricePoint(int(ts), price))
    return sorted(points, key=lambda p: p.ts)
