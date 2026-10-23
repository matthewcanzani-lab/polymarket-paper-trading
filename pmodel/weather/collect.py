"""Collect daily temperature markets, station coordinates, as-issued forecasts,
and price history. All weather markets are collected REGARDLESS of volume, so
the sample isn't selected on anything that happened after a prediction."""
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from pmodel import db
from pmodel.sources import openmeteo, polymarket_weather as pw
from pmodel.sources.base import Market
from pmodel.sources.polymarket import PolymarketSource

INF = 1e9


def collect_markets(conn, start: str, end: str, log=print) -> int:
    n = 0
    for i, ev in enumerate(pw.iter_events(True, f"{start}T00:00:00Z", f"{end}T23:59:59Z"), 1):
        for m in pw.parse_event(ev):
            conn.execute("""INSERT INTO weather_markets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?, NULL)
                ON CONFLICT(market_id) DO UPDATE SET yes_payout=excluded.yes_payout,
                    closed_time=excluded.closed_time, volume_usd=excluded.volume_usd""",
                (m.market_id, m.event_id, m.city, m.kind, m.unit, max(m.lo, -INF), min(m.hi, INF),
                 m.day, m.station, m.yes_token_id, m.created_at, m.closed_time, m.volume_usd,
                 m.yes_payout))
            n += 1
        if i % 1000 == 0:
            conn.commit()
            log(f"  ...{i:,} events")
    conn.commit()
    return n


def collect_forecasts(conn, start: str, end: str, log=print) -> int:
    stations = [r[0] for r in conn.execute(
        "SELECT DISTINCT station FROM weather_markets WHERE station IS NOT NULL")]
    known = {r[0] for r in conn.execute("SELECT station FROM weather_stations")}
    for icao, (lat, lon, name) in openmeteo.station_coords([s for s in stations if s not in known]).items():
        conn.execute("INSERT OR REPLACE INTO weather_stations VALUES (?,?,?,?)", (icao, lat, lon, name))
    conn.commit()
    coords = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT station, lat, lon FROM weather_stations")}
    log(f"{len(coords)} of {len(stations)} stations located: missing {sorted(set(stations) - set(coords))}")
    n = 0
    # One request per station per ~3 months keeps each response modest.
    d0, d1 = date.fromisoformat(start) - timedelta(days=1), date.fromisoformat(end)
    for st, (lat, lon) in sorted(coords.items()):
        a = d0
        while a <= d1:
            b = min(a + timedelta(days=90), d1)
            for r in openmeteo.daily_forecasts(lat, lon, a.isoformat(), b.isoformat()):
                conn.execute("INSERT OR REPLACE INTO weather_forecasts VALUES (?,?,?,?,?,?,?)",
                             (st, r["day"], r["lead"], r["tmax_c"], r["tmin_c"], r["tz"],
                              r["issued_by_utc"]))
                n += 1
            a = b + timedelta(days=1)
        conn.commit()
    return n


def collect_prices(conn, start: str, end: str, workers=8, log=print) -> int:
    """YES price history for weather markets measured between start and end."""
    src = PolymarketSource()
    rows = conn.execute("""SELECT * FROM weather_markets WHERE day BETWEEN ? AND ?
                           AND yes_token_id IS NOT NULL AND history_fetched_at IS NULL""",
                        (start, end)).fetchall()
    log(f"Fetching price history for {len(rows):,} weather markets")

    def as_market(r):
        # Minimal Market for the price-history call (hourly windows over its short life).
        return Market("polymarket", r["market_id"], "", None, "Weather", None, "Yes", "No", True,
                      r["yes_token_id"], r["created_at"], None, f"{r['day']}T12:00:00Z",
                      r["closed_time"], False, True, 0.0, 0.0, None, None, None)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        markets = [as_market(r) for r in rows]
        for i, (m, pts) in enumerate(zip(markets, pool.map(src.price_history, markets)), 1):
            db.save_price_history(conn, "polymarket", m.market_id, pts, db.utc_now())
            conn.execute("UPDATE weather_markets SET history_fetched_at=? WHERE market_id=?",
                         (db.utc_now(), m.market_id))
            if i % 1000 == 0:
                conn.commit()
                log(f"  ...{i:,}/{len(rows):,} ({i / (time.monotonic() - started):.1f}/s)")
    conn.commit()
    return len(rows)
