import json

from pmodel.dashboard import data
from pmodel.dashboard.app import PAGES, build_site, create_app
from pmodel.trading import ledger
from tests.test_trading import _bet


def _seed(path):
    conn = ledger.connect(path)
    ledger.append_bet(conn, _bet(bet_id="b1", market_id="1", category="Politics"))
    ledger.append_bet(conn, _bet(bet_id="b2", market_id="2", side="NO", category="Sports",
                                 model_prob_yes=0.3))
    ledger.append_bet(conn, _bet(bet_id="b3", market_id="3", category="Sports"))
    ledger.append_settlement(conn, {"bet_id": "b1", "settled_at": "2026-10-25T00:00:00Z",
                                    "yes_payout": 1.0, "payout": 100.0, "pnl": 48.0})
    ledger.append_settlement(conn, {"bet_id": "b2", "settled_at": "2026-10-26T00:00:00Z",
                                    "yes_payout": 1.0, "payout": 0.0, "pnl": -52.0})
    ledger.append_run(conn, {"run_id": "r1", "started_at": "2026-10-22T00:00:00Z",
                             "finished_at": "2026-10-22T00:10:00Z", "model_version": "v1",
                             "markets_scored": 10, "signals": 3, "bets_placed": 3,
                             "settings_json": "{}"})
    conn.commit()
    return conn


def test_forward_stats(tmp_path):
    conn = _seed(tmp_path / "p.db")
    fs = data.forward_stats(conn)
    assert fs["settled"] == 2 and fs["open"] == 1
    assert fs["total_pnl"] == -4.0
    assert fs["win_rate"] == 0.5
    assert fs["return_on_stake"] == -4.0 / 104.0
    # Brier on P(YES): bet1 0.6 vs 1, bet2 0.3 vs 1
    assert abs(fs["brier"] - ((0.4 ** 2 + 0.7 ** 2) / 2)) < 1e-9
    crit = data.success_criteria(fs)
    assert [c["met"] for c in crit] == [False, False, False, False]
    assert data.pnl_curve(conn) == [("2026-10-25T00:00:00Z", 48.0), ("2026-10-26T00:00:00Z", -4.0)]


def test_all_pages_render_and_export(tmp_path):
    _seed(tmp_path / "p.db")
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"run_id": "r1", "scored_at": "2026-10-22T00:00:00Z",
        "model_version": "v1", "edge_threshold": 0.05, "markets": [
            {"market_id": "9", "question": "Will it rain?", "category": "Weather",
             "end_date": "2026-10-30T00:00:00Z", "volume_usd": 20000, "last_price": 0.4,
             "best_bid": 0.39, "best_ask": 0.41, "model_prob_yes": 0.48, "status": "PAPER BET YES"}]}))
    kw = dict(ledger_path=tmp_path / "p.db", scores_path=scores)
    client = create_app(**kw).test_client()
    for page, _ in PAGES:
        html = client.get(f"/{page}").get_data(as_text=True)
        assert "paper trading only" in html
    stats = client.get("/stats.html").get_data(as_text=True)
    assert "FORWARD" in stats and "BACKTEST" in stats
    assert "Will it rain?" in client.get("/edges.html").get_data(as_text=True)
    out = build_site(tmp_path / "site", **kw)
    assert sorted(p.name for p in out.iterdir()) == sorted(p for p, _ in PAGES)


def test_dashboard_works_with_empty_ledger(tmp_path):
    client = create_app(ledger_path=tmp_path / "empty.db", scores_path=tmp_path / "none.json").test_client()
    for page, _ in PAGES:
        assert client.get(f"/{page}").status_code == 200


def _ledger_with_run(path, finished_at):
    conn = ledger.connect(path)
    ledger.append_run(conn, {"run_id": "r", "started_at": finished_at, "finished_at": finished_at,
                             "model_version": "v1", "markets_scored": 1, "signals": 0,
                             "bets_placed": 0, "settings_json": "{}"})
    conn.commit()


def test_stale_banner_and_eastern_time(tmp_path):
    from datetime import datetime, timezone
    old, new = tmp_path / "old.db", tmp_path / "new.db"
    _ledger_with_run(old, "2020-01-15T17:00:00Z")
    _ledger_with_run(new, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    stale = create_app(ledger_path=old).test_client().get("/index.html").get_data(as_text=True)
    fresh = create_app(ledger_path=new).test_client().get("/index.html").get_data(as_text=True)
    assert 'id="stale-banner" role="alert">' in stale            # shown
    assert 'id="stale-banner" role="alert" hidden>' in fresh     # hidden
    assert "Jan 15, 2020, 12:00 PM EST" in stale                 # Eastern time
    assert 'data-last-run="2020-01-15T17:00:00Z"' in stale       # browser re-checks age


def test_site_uses_only_relative_paths(tmp_path):
    """GitHub Pages serves the site under /polymarket-paper-trading/, so nothing
    may point at the domain root."""
    import re
    _seed(tmp_path / "p.db")
    out = build_site(tmp_path / "site", ledger_path=tmp_path / "p.db")
    for page in out.iterdir():
        html = page.read_text()
        assert not re.search(r'''(?:href|src)=["']/(?!/)''', html), page.name
        assert "url(/" not in html, page.name
