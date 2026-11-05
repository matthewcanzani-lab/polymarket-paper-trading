"""The 2026-11-05 weather rule changes: calibration, city-day cap across runs,
spread/depth filters, stop-loss, old-rules comparison, retraining guard, and
per-version results on the dashboard."""
import json
from datetime import datetime, timezone

import pytest

from pmodel.dashboard import data
from pmodel.dashboard.app import create_app
from pmodel.sources.base import OrderBook
from pmodel.trading import ledger
from pmodel.weather import artifact, rules as wr, scorer as ws
from tests.test_weather import _Src, _event, _params

NOW = datetime(2026, 10, 9, 16, 0, tzinfo=timezone.utc)   # 12:00 NY the day before
ON_RANGE_C = (76.5 - 32) * 5 / 9                           # forecast right on 76-77°F


def _new_params(**rule_overrides):
    return {**_params(), "version": "weather-vNEW",
            "calibration": {"method": "shrink_toward_market", "k": 1.0},
            "rules": {**artifact.CALIBRATED_RULES, **rule_overrides},
            "starting_bankroll": 1000.0}


def _patch(monkeypatch, params, events=None):
    monkeypatch.setattr(ws.artifact, "load_latest",
                        lambda: (params, ws.artifact.to_error_model(params)))
    monkeypatch.setattr(ws.pw, "iter_events", lambda *a, **k: iter(events or [_event()]))
    monkeypatch.setattr(ws, "current_forecasts",
                        lambda lat, lon, start, end: {"2026-10-10": (ON_RANGE_C, 10.0)})


def _run(tmp_path, book, conn=None):
    conn = conn or ledger.connect(tmp_path / "p.db")
    snap = tmp_path / "w.json"
    out = ws.run(conn, log=lambda *a: None, now=NOW, source=_Src(book), snapshot_path=snap)
    return conn, out, json.loads(snap.read_text())


DEEP = OrderBook(bids=[(0.20, 5000)], asks=[(0.22, 5000)])


# ---------------------------------------------------------------- calibration
def test_calibration_k0_is_market_and_k1_is_model():
    assert wr.calibrate({"calibration": {"k": 0.0}}, 0.9, 0.6) == pytest.approx(0.6)
    assert wr.calibrate({"calibration": {"k": 1.0}}, 0.9, 0.6) == pytest.approx(0.9)
    assert wr.calibrate({}, 0.9, 0.6) == 0.9                     # old versions: unchanged
    half = wr.calibrate({"calibration": {"k": 0.5}}, 0.9, 0.6)
    assert 0.6 < half < 0.9


def test_fit_calibration_uses_only_bets_settled_before_the_date(tmp_path):
    conn = ledger.connect(tmp_path / "p.db")
    # Early bets: the model was overconfident and wrong. A later bet must be ignored.
    for i, settled in enumerate(["2026-11-01T00:00:00Z"] * 20 + ["2026-11-06T00:00:00Z"]):
        ledger.append_bet(conn, _bet(f"b{i}", f"m{i}", p=0.9, mid=0.5))
        ledger.append_settlement(conn, {"bet_id": f"b{i}", "settled_at": settled,
                                        "yes_payout": float(i % 2), "payout": 0.0, "pnl": 0.0})
    fit = wr.fit_calibration(conn, "2026-11-05")
    assert fit["fit_on_bets"] == 20
    assert 0.0 <= fit["k"] <= 1.0 and fit["k"] < 0.2          # coin-flip outcomes: trust the market
    assert fit["brier_after"] <= fit["brier_before"]


def test_k0_version_places_no_bets_but_counts_old_rules(tmp_path, monkeypatch):
    _patch(monkeypatch, {**_new_params(), "calibration": {"k": 0.0}})
    conn, out, snap = _run(tmp_path, DEEP)
    assert out["bets_placed"] == 0
    assert out["old_rules_would_have_bet"] >= 1                # the old rules would have bet
    assert conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0] == 0
    row = snap["markets"][0]
    assert row["model_prob_yes"] == pytest.approx(row["last_price"])
    assert row["raw_model_prob_yes"] != pytest.approx(row["last_price"])


# ---------------------------------------------------------------- filters
def test_edge_threshold_comes_from_the_version(tmp_path, monkeypatch):
    _patch(monkeypatch, _new_params(edge_threshold=0.99))
    _, out, _ = _run(tmp_path, DEEP)
    assert out["bets_placed"] == 0


def test_wide_spread_is_skipped(tmp_path, monkeypatch):
    _patch(monkeypatch, _new_params())
    _, out, snap = _run(tmp_path, OrderBook(bids=[(0.15, 5000)], asks=[(0.23, 5000)]))
    assert out["bets_placed"] == 0
    assert any("spread" in r["status"] for r in snap["markets"])


def test_thin_book_is_skipped(tmp_path, monkeypatch):
    _patch(monkeypatch, _new_params())
    # The model leans NO here; buying NO sells into the YES bids. Only 5 shares rest
    # at the best bid, so the stake can't fill without moving the price.
    thin = OrderBook(bids=[(0.20, 5), (0.19, 5000)], asks=[(0.22, 5000)])
    _, out, snap = _run(tmp_path, thin)
    assert out["bets_placed"] == 0
    assert any("best price" in r["status"] for r in snap["markets"])
    _, out, _ = _run(tmp_path / "deep", DEEP)                 # same market, deep book: bets
    assert out["bets_placed"] >= 1


# ---------------------------------------------------------------- city-day cap
def test_city_day_cap_counts_earlier_runs_and_high_and_low(tmp_path, monkeypatch):
    _patch(monkeypatch, _new_params())
    conn = ledger.connect(tmp_path / "p.db")
    # An earlier run already put $45 on a LOW-temperature market for the same city and date.
    ledger.append_bet(conn, {**_bet("old", "low1", p=0.5, mid=0.5), "stake": 45.0,
                             "question": "Will the lowest temperature in New York City be "
                                         "between 50-51°F on October 10?",
                             "end_date": "2026-10-10T12:00:00Z"})
    conn.commit()
    _run(tmp_path, DEEP, conn)
    total = conn.execute("SELECT SUM(stake) FROM bets").fetchone()[0]
    assert total <= 0.05 * 1000 + 1e-6


def test_city_day_key_parses_ledger_questions():
    assert wr.city_day("Will the highest temperature in Paris be 18°C on October 25?",
                       "2026-10-25T12:00:00Z") == ("Paris", "2026-10-25")


# ---------------------------------------------------------------- stop-loss
def _lose(conn, version, amount, placed="2026-10-08T00:00:00Z"):
    ledger.append_bet(conn, {**_bet("loss", "lossm", p=0.9, mid=0.5), "model_version": version,
                             "placed_at": placed, "stake": amount})
    ledger.append_settlement(conn, {"bet_id": "loss", "settled_at": "2026-10-09T00:00:00Z",
                                    "yes_payout": 0.0, "payout": 0.0, "pnl": -amount})
    conn.commit()


def test_stop_loss_halts_new_bets(tmp_path, monkeypatch):
    _patch(monkeypatch, _new_params())
    conn = ledger.connect(tmp_path / "p.db")
    _lose(conn, "weather-vNEW", 150.0)                          # 15% of 1,000
    _, out, snap = _run(tmp_path, DEEP, conn)
    assert out["stop_loss_halted"] and out["bets_placed"] == 0
    assert snap["stop_loss"]["halted"]
    assert any("stop-loss" in r["status"] for r in snap["markets"])


def test_stop_loss_ignores_other_versions_and_clears_after_review(tmp_path):
    conn = ledger.connect(tmp_path / "p.db")
    _lose(conn, "weather-vOLD", 500.0)
    params = _new_params()
    assert not wr.stop_loss_status(conn, params, tmp_path / "none.json")["halted"]
    conn2 = ledger.connect(tmp_path / "q.db")
    _lose(conn2, "weather-vNEW", 150.0)
    assert wr.stop_loss_status(conn2, params, tmp_path / "none.json")["halted"]
    reviews = tmp_path / "reviews.json"
    reviews.write_text(json.dumps([{"version": "weather-vNEW", "reviewed_at": "2026-10-10T00:00:00Z"}]))
    assert not wr.stop_loss_status(conn2, params, reviews)["halted"]


def test_old_versions_have_no_stop_loss(tmp_path):
    conn = ledger.connect(tmp_path / "p.db")
    assert wr.stop_loss_status(conn, _params())["enabled"] is False


# ---------------------------------------------------------------- retraining guard
def test_retraining_waits_for_150_settled_bets(tmp_path):
    conn = ledger.connect(tmp_path / "p.db")
    ok, why = artifact.retrain_allowed(conn, {"version": "weather-vNEW"})
    assert not ok and "150" in why


def test_refit_workflow_is_manual_only():
    from pmodel import config
    text = (config.PROJECT_ROOT / ".github" / "workflows" / "weather-refit.yml").read_text()
    assert "schedule:" not in text and "workflow_dispatch" in text


# ---------------------------------------------------------------- dashboard
def _dash_models(tmp_path, params):
    (tmp_path / "models" / "weather").mkdir(parents=True)
    (tmp_path / "models" / "weather" / f"{params['version']}.json").write_text(json.dumps(params))
    return tmp_path / "models"


def test_dashboard_shows_results_per_version(tmp_path):
    conn = ledger.connect(tmp_path / "p.db")
    _lose(conn, "weather-vOLD", 20.0)
    models = _dash_models(tmp_path, {**_new_params(), "label": "Calibrated test version"})
    rows = data.by_version(conn, models)
    assert [r["version"] for r in rows] == ["weather-vOLD", "weather-vNEW"]
    assert rows[1]["current"] and rows[1]["bets"] == 0          # listed before its first bet
    assert rows[0]["pnl"] == -20.0
    client = create_app(ledger_path=tmp_path / "p.db", models_dir=models).test_client()
    for page in ("/results/", "/stats.html", "/index.html"):
        html = client.get(page).get_data(as_text=True)
        assert "weather-vNEW" in html and "Calibrated test version" in html


def test_dashboard_flags_stop_loss(tmp_path):
    conn = ledger.connect(tmp_path / "p.db")
    _lose(conn, "weather-vNEW", 200.0)
    models = _dash_models(tmp_path, _new_params())
    html = create_app(ledger_path=tmp_path / "p.db", models_dir=models).test_client().get(
        "/index.html").get_data(as_text=True)
    assert "Stop-loss triggered" in html and "HALTED" in html


def _bet(bet_id, market_id, p, mid):
    return {"bet_id": bet_id, "run_id": "r0", "placed_at": "2026-10-01T00:00:00Z",
            "platform": "polymarket", "market_id": market_id,
            "question": "Will the highest temperature in Paris be 18°C on October 25?",
            "category": "Weather", "end_date": "2026-10-25T12:00:00Z", "side": "YES",
            "model_prob_yes": p, "market_mid": mid, "best_bid": mid - 0.01, "best_ask": mid + 0.01,
            "fill_price": mid + 0.02, "slippage_per_share": 0.01, "shares": 10.0, "stake": 5.0,
            "kelly_full": 0.1, "bankroll_at_bet": 1000.0, "model_version": "weather-vOLD"}
