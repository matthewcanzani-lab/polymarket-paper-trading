from datetime import datetime, timezone

from pmodel import backup, config, db
from pmodel.analysis.volume import build_report, save_chart
from pmodel.ingest import ingest_markets, ingest_price_history
from pmodel.sources import polymarket
from pmodel.sources.polymarket import PolymarketSource
from tests.conftest import gamma_market

RESOLVED = [
    gamma_market(1, closed=True, prices=("1", "0"), volume=50),
    gamma_market(2, closed=True, prices=("0", "1"), volume=2_000),
    gamma_market(3, closed=True, prices=("1", "0"), volume=40_000, category="Sports"),
    gamma_market(4, closed=True, prices=("0.5", "0.5"), volume=9_000),
    gamma_market(5, closed=True, outcomes=("A", "B", "C"), prices=("1", "0", "0")),
]
ACTIVE = [gamma_market(10, volume=500), gamma_market(11, volume=80_000, bid=0.40, ask=0.43)]


def fake_api(url, params=None):
    if "prices-history" in url:
        return {"history": [{"t": 1_700_000_000, "p": 0.3}, {"t": 1_700_003_600, "p": 0.35}]}
    return {"markets": RESOLVED if params["closed"] == "true" else ACTIVE}


def test_ingest_report_and_history(conn, monkeypatch, tmp_path):
    monkeypatch.setattr(polymarket.safe_http, "get_json", fake_api)
    src = PolymarketSource()
    s1 = ingest_markets(src, conn, closed=True, log=lambda *_: None)
    assert s1 == {"seen": 5, "stored": 4, "skipped_non_binary": 1, "resolved": 3, "ambiguous": 1}
    s2 = ingest_markets(src, conn, closed=False, log=lambda *_: None)
    assert s2["stored"] == 2
    assert conn.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()[0] == 2

    # Re-running is safe (upserts, no duplicate rows).
    ingest_markets(src, conn, closed=True, log=lambda *_: None)
    assert conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 6

    # History only for clean-resolved markets above the cutoff.
    monkeypatch.setattr(config, "MIN_VOLUME_USD", 1_000)
    h = ingest_price_history(src, conn, log=lambda *_: None)
    assert h["markets"] == 2 and h["points"] == 4
    again = ingest_price_history(src, conn, log=lambda *_: None)
    assert again["markets"] == 0  # already fetched

    report = build_report(conn, now=datetime(2026, 10, 4, tzinfo=timezone.utc))
    assert "Resolved markets (clean YES/NO outcome): 3" in report
    assert "Closed but ambiguous/void (excluded):    1" in report
    assert "3.0c" in report  # spread on the $80k active market
    assert save_chart(conn, tmp_path / "v.png").exists()


def test_backup_copies_prediction_log(tmp_path):
    src = tmp_path / "predictions.db"
    c = db.connect(src)
    c.execute("CREATE TABLE t (x)")
    c.execute("INSERT INTO t VALUES (42)")
    c.commit()
    c.close()
    dest = backup.backup_predictions(src, tmp_path / "icloud")
    import sqlite3
    assert sqlite3.connect(dest).execute("SELECT x FROM t").fetchone()[0] == 42


def test_universe_excludes_crypto_coin_flips(conn, monkeypatch):
    from pmodel import universe
    from tests.conftest import gamma_market
    monkeypatch.setattr(config, "MIN_VOLUME_USD", 10_000)
    src = PolymarketSource()
    flip = gamma_market(20, volume=90_000)
    flip["question"] = "Bitcoin Up or Down - October 4, 6:55PM-7:00PM ET"
    series = gamma_market(21, volume=90_000)
    series["events"][0]["series"] = [{"slug": "eth-up-or-down-15m", "recurrence": "15m"}]
    daily_sports = gamma_market(22, volume=90_000)
    daily_sports["events"][0]["series"] = [{"slug": "nba", "recurrence": "daily"}]
    small = gamma_market(23, volume=9_999)
    good = gamma_market(24, volume=10_000)
    markets = [src.parse_market(r) for r in (flip, series, daily_sports, small, good)]
    assert markets[1].series == "eth-up-or-down-15m" and markets[1].recurrence == "15m"
    assert [universe.is_in_universe(m) for m in markets] == [False, False, True, False, True]

    now = db.utc_now()
    for m in markets:
        db.upsert_market(conn, m, now)
    where, params = universe.sql_filter("m")
    ids = {r[0] for r in conn.execute(f"SELECT market_id FROM markets m WHERE {where}", params)}
    assert ids == {"22", "24"}  # SQL and Python agree
