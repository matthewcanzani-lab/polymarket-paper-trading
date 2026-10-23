"""Weather forecasts AS THEY WERE ISSUED, from Open-Meteo's Previous Runs API,
plus weather-station coordinates from aviationweather.gov.

Look-ahead rule. `temperature_2m_previous_dayN` is "the value predicted N x 24
hours before valid time" (Open-Meteo docs). A forecast of a whole local day's
max/min uses hours 00:00-23:00, so its newest ingredient was predicted at
(23:00 local on that day) - N days. We add PUBLISH_DELAY for model run and
upload time, and record that as `issued_by_utc`: the forecast may only be used
by a prediction made at or after that moment.
"""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from pmodel import safe_http

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
STATION_URL = "https://aviationweather.gov/api/data/stationinfo"
PUBLISH_DELAY = timedelta(hours=6)
LEADS = (1, 2, 3)


def station_coords(icao_ids: list[str]) -> dict:
    out = {}
    for i in range(0, len(icao_ids), 50):
        data = safe_http.get_json(STATION_URL, params={"ids": ",".join(icao_ids[i:i + 50]),
                                                       "format": "json"})
        for s in data or []:
            out[s["icaoId"].upper()] = (float(s["lat"]), float(s["lon"]), s.get("site"))
    return out


def daily_forecasts(lat: float, lon: float, start: str, end: str, leads=LEADS) -> list[dict]:
    """One row per (local day, lead): forecast daily max/min in °C and when it was knowable."""
    hourly = [f"temperature_2m_previous_day{n}" for n in leads]
    data = safe_http.get_json(PREVIOUS_RUNS_URL, params={
        "latitude": lat, "longitude": lon, "hourly": ",".join(hourly), "timezone": "auto",
        "start_date": start, "end_date": end}, timeout=90)
    tz = ZoneInfo(data["timezone"])
    times = data["hourly"]["time"]
    rows = []
    for n, var in zip(leads, hourly):
        by_day = defaultdict(list)
        for ts, v in zip(times, data["hourly"][var]):
            if v is not None:
                by_day[ts[:10]].append(v)
        for day, vals in by_day.items():
            if len(vals) < 24:  # incomplete day: skip rather than guess
                continue
            last_hour = datetime.fromisoformat(f"{day}T23:00").replace(tzinfo=tz)
            issued_by = (last_hour - timedelta(days=n) + PUBLISH_DELAY).astimezone(timezone.utc)
            rows.append({"day": day, "lead": n, "tmax_c": max(vals), "tmin_c": min(vals),
                         "tz": data["timezone"],
                         "issued_by_utc": issued_by.strftime("%Y-%m-%dT%H:%M:%SZ")})
    return rows
