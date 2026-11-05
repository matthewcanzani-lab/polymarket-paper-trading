"""The prediction log: every scored market, once per model version, append-only."""
import json
from datetime import datetime, timezone

from pmodel.dashboard import data
from pmodel.dashboard.app import create_app
from pmodel.sources.base import OrderBook
from pmodel.trading import ledger, predlog
from pmodel.weather import scorer as ws
from tests.test_weather import _Src, _patch as _patch_weather


def _row(mid, p=0.6, price=0.5, end="2026-10-10T12:00:00Z"):
    return {"market_id": mid, "question": f"Q{mid}?", "category": "Weather", "end_date": end,
            "model_prob_yes": p, "last_price": price}


def test_logs_each_market_once_per_version(tmp_path):
    assert predlog.log_predictions([_row("1"), _row("2")], "weather", "v1", "r1", tmp_path) == 2
    assert predlog.log_predictions([_row("1", p=0.9), _row("3")], "weather", "v1", "r2", tmp_path) == 1
    assert predlog.log_predictions([_row("1")], "weather", "v2", "r3", tmp_path) == 1   # new version
    rows = predlog.read("predictions", tmp_path)
    assert [(r["model_version"], r["market_id"]) for r in rows] == [
        ("v1", "1"), ("v1", "2"), ("v1", "3"), ("v2", "1")]
    assert rows[0]["model_prob_yes"] == "0.600000"            # first prediction kept, never replaced
    assert predlog.verify(tmp_path) == []


def test_verify_catches_edits_and_deletions(tmp_path):
    predlog.log_predictions([_row("1"), _row("2"), _row("3")], "weather", "v1", "r1", tmp_path)
    path = next(tmp_path.glob("predictions-*.csv"))
    original = path.read_text()
    path.write_text(original.replace("0.600000", "0.700000", 1))
    assert any("contents changed" in p for p in predlog.verify(tmp_path))
    lines = original.splitlines(keepends=True)
    path.write_text("".join(lines[:2] + lines[3:]))          # drop the second prediction
    assert any("chain broken" in p for p in predlog.verify(tmp_path))


class _Resolver:
    def markets_by_id(self, ids):
        return [{"id": i} for i in ids]

    def yes_payout(self, raw):
        return {"1": 1.0, "2": 0.0}.get(raw["id"])           # market 3 not resolved yet


def test_resolve_records_outcomes_once_and_record_scores_them(tmp_path):
    predlog.log_predictions([_row("1", p=0.8, price=0.6), _row("2", p=0.3, price=0.4), _row("3")],
                            "weather", "v1", "r1", tmp_path)
    predlog.log_predictions([_row("4", end="2099-01-01T00:00:00Z")], "weather", "v1", "r1", tmp_path)
    now = "2026-10-11T00:00:00Z"
    log = lambda *a: None
    assert predlog.resolve(_Resolver(), tmp_path, now, log) == {"due": 3, "resolved": 2}
    assert predlog.resolve(_Resolver(), tmp_path, now, log) == {"due": 1, "resolved": 0}
    (g,) = predlog.record(tmp_path)
    assert (g["logged"], g["resolved"]) == (4, 2)
    assert abs(g["brier_model"] - (0.2 ** 2 + 0.3 ** 2) / 2) < 1e-9
    assert abs(g["brier_market"] - (0.4 ** 2 + 0.4 ** 2) / 2) < 1e-9
    assert predlog.verify(tmp_path) == []


def test_weather_scorer_logs_every_scored_market(tmp_path, monkeypatch):
    _patch_weather(monkeypatch, forecast_c=(76.5 - 32) * 5 / 9)
    conn = ledger.connect(tmp_path / "p.db")
    now = datetime(2026, 10, 9, 16, 0, tzinfo=timezone.utc)
    src = _Src(OrderBook(bids=[(0.20, 5000)], asks=[(0.22, 5000)]))
    out = ws.run(conn, log=lambda *a: None, now=now, source=src, predlog_dir=tmp_path / "pl")
    assert out["predictions_logged"] == out["markets_scored"] == 2
    out = ws.run(conn, log=lambda *a: None, now=now, source=src, predlog_dir=tmp_path / "pl")
    assert out["predictions_logged"] == 0                     # already logged for this version
    rows = predlog.read("predictions", tmp_path / "pl")
    assert {r["strategy"] for r in rows} == {"weather"} and all(r["raw_model_prob_yes"] for r in rows)


def test_scorer_without_predlog_dir_writes_nothing(tmp_path, monkeypatch):
    _patch_weather(monkeypatch, forecast_c=20.0)
    conn = ledger.connect(tmp_path / "p.db")
    out = ws.run(conn, log=lambda *a: None, now=datetime(2026, 10, 9, 16, 0, tzinfo=timezone.utc),
                 source=_Src(OrderBook(bids=[(0.20, 5000)], asks=[(0.22, 5000)])))
    assert "predictions_logged" not in out


def test_dashboard_shows_prediction_record(tmp_path):
    predlog.log_predictions([_row("1", p=0.8, price=0.6)], "weather", "weather-vX", "r1", tmp_path / "pl")
    predlog.resolve(_Resolver(), tmp_path / "pl", "2026-10-11T00:00:00Z", lambda *a: None)
    pr = data.prediction_record(tmp_path / "pl")
    assert (pr["logged"], pr["resolved"]) == (1, 1)
    client = create_app(ledger_path=tmp_path / "p.db", predlog_dir=tmp_path / "pl").test_client()
    for page in ("/index.html", "/stats.html", "/results/"):
        html = client.get(page).get_data(as_text=True)
        assert "weather-vX" in html or "1 predictions so far" in html
