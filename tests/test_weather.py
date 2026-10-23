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
