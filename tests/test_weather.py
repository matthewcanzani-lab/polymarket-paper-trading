"""Weather model: parsing, probabilities, and the forecast-timing (look-ahead) rule."""
import numpy as np
import pandas as pd
import pytest

from pmodel.sources import polymarket_weather as pw
from pmodel.weather import backtest as bt
from pmodel.weather.model import ErrorModel, range_prob


@pytest.mark.parametrize("q,expect", [
    ("Will the highest temperature in New York City be between 76-77°F on October 3?",
     ("high", "New York City", "F", 76, 77, "2026-10-03")),
    ("Will the highest temperature in London be 21°C or higher on May 2?",
     ("high", "London", "C", 21, float("inf"), "2026-05-02")),
    ("Will the lowest temperature in Sao Paulo be 14°C or below on October 5?",
     ("low", "Sao Paulo", "C", float("-inf"), 14, "2026-10-05")),
    ("Will the highest temperature in Seoul be -3°C on January 9?",
     ("high", "Seoul", "C", -3, -3, "2026-01-09")),
    ("Will Bitcoin be above $100k on May 2?", None),
])
def test_parse_question(q, expect):
    p = pw.parse_question(q, 2026)
    if expect is None:
        assert p is None
    else:
        assert (p["kind"], p["city"], p["unit"], p["lo"], p["hi"], p["day"]) == expect


def test_station_from_rules():
    assert pw.station_from_rules("...https://www.weather.gov/wrh/timeseries?site=klga") == "KLGA"
    assert pw.station_from_rules(
        "https://www.wunderground.com/history/daily/gb/london/EGLC.") == "EGLC"


def test_ranges_partition_to_one():
    total = sum(range_prob(20.3, 1.6, k, k) for k in range(0, 41))
    total += range_prob(20.3, 1.6, -1e9, -1) + range_prob(20.3, 1.6, 41, 1e9)
    assert total == pytest.approx(1.0)
    # whole-degree rounding: "21" covers [20.5, 21.5)
    assert range_prob(21.0, 0.01, 21, 21, df=1000) == pytest.approx(1.0)


def test_error_model_shrinks_small_stations_toward_global():
    rng = np.random.default_rng(0)
    big = pd.DataFrame({"station": "A", "kind": "high", "lead": 1, "err_c": rng.normal(0, 1, 500)})
    small = pd.DataFrame({"station": "B", "kind": "high", "lead": 1, "err_c": [5.0, 5.0]})
    m = ErrorModel.fit(pd.concat([big, small]))
    bias_b, _ = m.params("B", "high", 1)
    assert 0 < bias_b < 1.0  # two +5 errors barely move it


def _toy(issued_by):
    mk = pd.DataFrame([{"market_id": "m1", "event_id": "e1", "city": "NYC", "kind": "high",
                        "unit": "F", "lo": 76.0, "hi": 77.0, "day": "2026-09-10",
                        "station": "KLGA", "yes_token_id": "t", "created_at": "2026-09-07T00:00:00Z",
                        "closed_time": "2026-09-11T05:00:00Z", "volume_usd": 1.0,
                        "yes_payout": 1.0, "history_fetched_at": None}])
    fc = pd.DataFrame([{"station": "KLGA", "day": "2026-09-10", "lead": n, "tmax_c": 24.0 + n,
                        "tmin_c": 15.0, "tz": "America/New_York", "issued_by_utc": issued_by[n]}
                       for n in issued_by])
    return mk, fc


def test_snapshot_never_uses_a_forecast_issued_after_t():
    # lead 1 issued 05:00 local on the day; lead 2 the day before; lead 3 two days before.
    mk, fc = _toy({1: "2026-09-10T09:00:00Z", 2: "2026-09-09T09:00:00Z", 3: "2026-09-08T09:00:00Z"})
    s = bt.snapshots(mk, fc, "2026-09-01", "2026-10-01")
    for r in s.itertuples():
        issued = bt._unix(fc[fc.lead == r.lead].issued_by_utc.iloc[0])
        assert issued <= r.t
    leads = dict(zip(s.snapshot, s.lead))
    assert leads == {"2 days before, 18:00": 3, "day before, 12:00": 2, "same day, 08:00": 1}


def test_snapshot_skipped_when_no_forecast_was_available_yet():
    mk, fc = _toy({1: "2026-09-10T09:00:00Z"})  # only the same-day forecast exists
    s = bt.snapshots(mk, fc, "2026-09-01", "2026-10-01")
    assert list(s.snapshot) == ["same day, 08:00"]


def test_price_lookup_respects_hourly_bucket(tmp_path):
    from pmodel import db
    c = db.connect(tmp_path / "m.db")
    c.executemany("INSERT INTO price_history VALUES ('polymarket','m1',?,?)",
                  [(1000, 0.2), (1000 + 3600, 0.9)])
    price_at = bt.price_lookup(c)
    assert price_at("m1", 1000 + 3600 + 10) == 0.2   # second bucket not finished yet
    assert price_at("m1", 1000 + 7200) == 0.9
    assert price_at("m1", 500) is None


def test_bets_charge_costs_and_bet_once_per_market():
    test = pd.DataFrame({"market_id": ["a", "a", "b"], "event": ["e", "e", "e"], "t": [1, 2, 1],
                         "price": [0.30, 0.30, 0.50], "label": [1.0, 1.0, 0.0]})
    out = bt.bets(test, np.array([0.40, 0.45, 0.52]), threshold=0.05, cost=0.02)
    assert list(out.market_id) == ["a"]                      # b: gap too small
    assert out.ret.iloc[0] == pytest.approx(1 / 0.32 - 1)    # paid 30c + 2c


# ---------------------------------------------------------------- live scorer
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from pmodel.sources.base import OrderBook
from pmodel.trading import ledger
from pmodel.weather import scorer as ws


def test_window_and_lead():
    tz = ZoneInfo("America/New_York")
    day = "2026-10-10"
    at = lambda s: datetime.fromisoformat(s).replace(tzinfo=tz)
    assert not ws.in_window(at("2026-10-08T11:59"), day, tz)   # too early
    assert ws.in_window(at("2026-10-08T12:00"), day, tz)
    assert ws.in_window(at("2026-10-09T23:59"), day, tz)
    assert not ws.in_window(at("2026-10-10T00:00"), day, tz)   # never on the day itself
    end = at("2026-10-11T00:00")
    assert ws.lead_for(at("2026-10-08T18:00"), end) == 3        # matches backtest snapshots
    assert ws.lead_for(at("2026-10-09T12:00"), end) == 2


def _event(bid=0.20, ask=0.22):
    def mk(i, lo, hi, q):
        return {"id": str(i), "question": q, "clobTokenIds": f'["tok{i}", "no{i}"]',
                "outcomePrices": '["0.2", "0.8"]', "bestBid": bid, "bestAsk": ask,
                "acceptingOrders": True, "closed": False, "volumeNum": 5000,
                "description": "station https://www.weather.gov/wrh/timeseries?site=klga"}
    return {"id": "e1", "endDate": "2026-10-10T12:00:00Z", "markets": [
        mk(1, 76, 77, "Will the highest temperature in New York City be between 76-77°F on October 10?"),
        mk(2, 78, 79, "Will the highest temperature in New York City be between 78-79°F on October 10?")]}


class _Src:
    def __init__(self, book):
        self.book = book
    def order_book(self, market):
        return self.book


def _params():
    return {"version": "weather-vTEST", "blend_weights": [1.0, 1.0, 0.0], "scale": 1.0, "df": 6.0,
            "global": {"high|1": [0.0, 1.0], "high|2": [0.0, 1.0], "high|3": [0.0, 1.0]},
            "by_key": {}, "stations": {"KLGA": {"lat": 40.78, "lon": -73.88, "name": "LGA", "tz": "America/New_York"}}}


def _patch(monkeypatch, forecast_c):
    monkeypatch.setattr(ws.artifact, "load_latest",
                        lambda: (_params(), ws.artifact.to_error_model(_params())))
    monkeypatch.setattr(ws.pw, "iter_events", lambda *a, **k: iter([_event()]))
    monkeypatch.setattr(ws, "current_forecasts",
                        lambda lat, lon, start, end: {"2026-10-10": (forecast_c, 10.0)})


def test_weather_scorer_bets_within_caps_and_once(tmp_path, monkeypatch):
    _patch(monkeypatch, forecast_c=(76.5 - 32) * 5 / 9)   # forecast right on the 76-77°F range
    conn = ledger.connect(tmp_path / "p.db")
    now = datetime(2026, 10, 9, 16, 0, tzinfo=timezone.utc)  # 12:00 NY the day before
    src = _Src(OrderBook(bids=[(0.20, 5000)], asks=[(0.22, 5000)]))
    snap = tmp_path / "w.json"
    out = ws.run(conn, log=lambda *a: None, now=now, source=src, snapshot_path=snap)
    assert out["bets_placed"] >= 1
    import json
    rows = json.loads(snap.read_text())["markets"]
    assert any(r["status"].startswith("PAPER BET") for r in rows)
    rows = conn.execute("SELECT * FROM bets").fetchall()
    assert sum(r["stake"] for r in rows) <= 0.05 * 1000 + 1e-6   # per city-day cap
    assert all(r["fill_price"] >= 0.23 - 1e-9 for r in rows if r["side"] == "YES")  # ask + 1c
    ws.run(conn, log=lambda *a: None, now=now, source=src)        # second run
    ids = [r["market_id"] for r in conn.execute("SELECT market_id FROM bets")]
    assert len(ids) == len(set(ids))                               # never two bets on one market
    assert ledger.verify(conn) == []


def test_weather_scorer_never_bets_on_the_day(tmp_path, monkeypatch):
    _patch(monkeypatch, forecast_c=(76.5 - 32) * 5 / 9)
    conn = ledger.connect(tmp_path / "p.db")
    now = datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc)  # 10:00 NY on the day
    out = ws.run(conn, log=lambda *a: None, now=now,
                 source=_Src(OrderBook(bids=[(0.20, 5000)], asks=[(0.22, 5000)])))
    assert (out["markets_scored"], out["signals"], out["bets_placed"]) == (0, 0, 0)
