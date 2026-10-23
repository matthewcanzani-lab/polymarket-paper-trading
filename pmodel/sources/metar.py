"""Airport weather reports (METAR) from aviationweather.gov, read the way
Polymarket's temperature markets resolve from 23 Oct 2026 onward:

  * routine reports only (a line starting "SPECI" doesn't count);
  * use the hourly remark T-group (tenths °C) when present, else the body's
    whole-degree temperature;
  * a report counts if its time, in the station's local time, is on the date;
  * the day's high (or low) is then rounded to a whole degree, halves up
    (after converting to °F for Fahrenheit markets).
"""
import math
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from pmodel import safe_http

METAR_URL = "https://aviationweather.gov/api/data/metar"
_TGROUP = re.compile(r"\bT([01])(\d{3})([01])\d{3}\b")
_BODY = re.compile(r"\s(M?\d{2})/(M?\d{2})?\s")


def report_temp_c(raw: str):
    """Temperature in °C from one report, or None if it doesn't count / has none."""
    raw = raw.strip()
    if raw.startswith("SPECI"):
        return None
    m = _TGROUP.search(raw)
    if m:
        v = int(m.group(2)) / 10
        return -v if m.group(1) == "1" else v
    m = _BODY.search(" " + raw.split(" RMK")[0] + " ")
    if m:
        t = m.group(1)
        return -int(t[1:]) if t.startswith("M") else int(t)
    return None


def round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def qualifying_value(temps_c: list, kind: str, unit: str):
    """The whole-degree value a market would resolve on, from the day's report temps."""
    if not temps_c:
        return None
    v = max(temps_c) if kind == "high" else min(temps_c)
    return round_half_up(v * 9 / 5 + 32 if unit == "F" else v)


def day_temps(station: str, day: str, tz: str) -> list:
    """All counting report temperatures (°C) for a station's local calendar day."""
    zone = ZoneInfo(tz)
    start = datetime.fromisoformat(day).replace(tzinfo=zone)
    end_utc = (start + timedelta(days=1)).astimezone(timezone.utc)
    data = safe_http.get_json(METAR_URL, params={
        "ids": station, "format": "json", "hours": 30,
        "date": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")}) or []
    temps = []
    for rep in data:
        obs = datetime.fromtimestamp(int(rep["obsTime"]), timezone.utc).astimezone(zone)
        if obs.date().isoformat() != day:
            continue
        t = report_temp_c(rep.get("rawOb") or "")
        if t is not None:
            temps.append(t)
    return temps
