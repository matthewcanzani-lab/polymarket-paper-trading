"""Polymarket daily temperature markets: find them and read what each one asks.

A typical event is "Highest temperature in NYC on October 3?" with one binary
market per range ("between 76-77°F", "15°C", "21°C or higher", ...). Each
event resolves on ONE weather station's reading in whole degrees, named in the
market rules (e.g. NOAA LaGuardia, KLGA).
"""
import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Iterator, Optional

from pmodel import safe_http
from pmodel.sources.polymarket import _float, _maybe_json_list, to_iso_utc

EVENTS_URL = "https://gamma-api.polymarket.com/events/keyset"
TAG = "daily-temperature"

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}

_Q = re.compile(
    r"Will the (?P<kind>highest|lowest) temperature in (?P<city>.+?) "
    r"(?:be )?(?:between )?(?P<a>-?\d+)\s*(?:[-–]\s*(?P<b>-?\d+))?\s*°\s*(?P<unit>[CF])"
    r"(?P<tail> or higher| or below)? on (?P<month>[A-Za-z]+) (?P<day>\d+)",
    re.IGNORECASE)
_STATION = re.compile(r"(?:site=|/(?:history/daily/[a-z]{2}/[^/]+/))([A-Za-z0-9]{4})\b")


@dataclass
class TempMarket:
    market_id: str
    event_id: str
    city: str
    kind: str            # "high" or "low"
    unit: str            # "C" or "F"
    lo: float            # range in whole degrees, inclusive; -inf / inf for open ends
    hi: float
    day: str             # local calendar date the temperature is measured, YYYY-MM-DD
    station: Optional[str]
    yes_token_id: Optional[str]
    created_at: Optional[str]
    closed_time: Optional[str]
    volume_usd: float
    yes_payout: Optional[float]  # 1/0 once resolved cleanly, else None


def parse_question(question: str, year: int):
    """-> dict(kind, city, unit, lo, hi, day) or None if it isn't a temperature range."""
    m = _Q.search(question)
    if not m or m["month"].lower() not in MONTHS:
        return None
    a = int(m["a"])
    b = int(m["b"]) if m["b"] else a
    lo, hi = float(min(a, b)), float(max(a, b))
    tail = (m["tail"] or "").strip().lower()
    if tail == "or higher":
        hi = float("inf")
    elif tail == "or below":
        lo = float("-inf")
    try:
        day = date(year, MONTHS[m["month"].lower()], int(m["day"])).isoformat()
    except ValueError:
        return None
    return {"kind": "high" if m["kind"].lower() == "highest" else "low",
            "city": m["city"].strip(), "unit": m["unit"].upper(), "lo": lo, "hi": hi, "day": day}


def station_from_rules(description: str) -> Optional[str]:
    m = _STATION.search(description or "")
    return m.group(1).upper() if m else None


def _year_hint(raw_event: dict) -> int:
    end = to_iso_utc(raw_event.get("endDate")) or to_iso_utc(raw_event.get("startDate"))
    return int(end[:4]) if end else date.today().year


def parse_event(raw_event: dict) -> list[TempMarket]:
    out = []
    year = _year_hint(raw_event)
    for raw in raw_event.get("markets") or []:
        parsed = parse_question(raw.get("question") or "", year)
        if not parsed:
            continue
        tokens = _maybe_json_list(raw.get("clobTokenIds")) or []
        prices = _maybe_json_list(raw.get("outcomePrices")) or []
        yes = _float(prices[0]) if len(prices) == 2 else None
        resolved = (raw.get("closed") and (raw.get("umaResolutionStatus") or "resolved") == "resolved"
                    and yes is not None and (yes >= 0.99 or yes <= 0.01))
        out.append(TempMarket(
            market_id=str(raw["id"]), event_id=str(raw_event["id"]), station=station_from_rules(
                raw.get("description") or raw_event.get("description")),
            yes_token_id=str(tokens[0]) if tokens else None,
            created_at=to_iso_utc(raw.get("createdAt")), closed_time=to_iso_utc(raw.get("closedTime")),
            volume_usd=_float(raw.get("volumeNum")) or 0.0,
            yes_payout=(1.0 if yes >= 0.99 else 0.0) if resolved else None, **parsed))
    return out


def iter_events(closed: bool, end_date_min: str, end_date_max: Optional[str] = None) -> Iterator[dict]:
    cursor = None
    while True:
        params = {"tag_slug": TAG, "closed": str(closed).lower(), "limit": 100,
                  "end_date_min": end_date_min}
        if end_date_max:
            params["end_date_max"] = end_date_max
        if cursor:
            params["after_cursor"] = cursor
        page = safe_http.get_json(EVENTS_URL, params=params)
        events = page.get("events") or []
        yield from events
        cursor = page.get("next_cursor")
        if not events or not cursor:
            return
