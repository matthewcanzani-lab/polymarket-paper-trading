"""Paper-trading rules: sizing, fills, append-only ledger, settlement, scorer."""
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from pmodel import config
from pmodel.sources.base import OrderBook, TradeBucket, TradeHistory, PricePoint
from pmodel.sources.polymarket import PolymarketSource
from pmodel.trading import ledger, scorer, settle
from pmodel.trading.sizing import kelly_fraction, no_side_levels, size_bet, walk_book
from tests.conftest import gamma_market


def test_kelly():
    assert kelly_fraction(0.6, 0.5) == pytest.approx(0.2)
    assert kelly_fraction(0.4, 0.5) == 0.0      # negative edge -> no bet
    assert kelly_fraction(0.9, 1.0) == 0.0


def test_walk_book_pays_deeper_levels_plus_slippage():
    price, shares = walk_book([(0.50, 10), (0.60, 1000)], 20, 0.01)
    # 10 shares at 0.51 = 5.10, then 14.90 / 0.61 = 24.43 shares
    assert shares == pytest.approx(10 + 14.9 / 0.61)
    assert price == pytest.approx(20 / shares)
    assert walk_book([(0.5, 1)], 100, 0.01) == (None, 0.0)  # book too thin


def test_no_side_prices_from_yes_bids():
    assert no_side_levels([(0.70, 5)]) == [(pytest.approx(0.30), 5)]


def test_quarter_kelly_and_caps():
    bet = size_bet(0.62, [(0.50, 10_000)], 1000, 1000, config)
    assert bet["stake"] == 50.0                     # quarter Kelly $56 capped at 5%
    bet = size_bet(0.55, [(0.50, 10_000)], 1000, 1000, config)
    full = kelly_fraction(0.55, 0.51)
    assert bet["stake"] == pytest.approx(round(0.25 * full * 1000, 2))
    assert size_bet(0.62, [(0.50, 10_000)], 1000, 10, config)["stake"] == 10.0  # cash cap
    assert size_bet(0.505, [(0.50, 10_000)], 1000, 1000, config) is None  # edge eaten by slippage


def _bet(**over):
    f = dict(bet_id="b1", run_id="r1", placed_at="2026-10-22T00:00:00Z", platform="polymarket",
             market_id="1", question="Q?", category="Politics", end_date="2026-11-01T00:00:00Z",
             side="YES", model_prob_yes=0.6, market_mid=0.5, best_bid=0.49, best_ask=0.51,
             fill_price=0.52, slippage_per_share=0.01, shares=100.0, stake=52.0, kelly_full=0.1,
             bankroll_at_bet=1000.0, model_version="v1")
    f.update(over)
    return f


def test_ledger_is_append_only(tmp_path):
    conn = ledger.connect(tmp_path / "p.db")
    ledger.append_bet(conn, _bet())
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE bets SET stake = 1")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM bets")
    assert ledger.verify(conn) == []


def test_tampering_is_detected_even_if_triggers_are_dropped(tmp_path):
    conn = ledger.connect(tmp_path / "p.db")
    ledger.append_bet(conn, _bet())
    ledger.append_bet(conn, _bet(bet_id="b2", market_id="2"))
    conn.commit()
    conn.execute("DROP TRIGGER bets_no_update")
    conn.execute("UPDATE bets SET stake = 1 WHERE bet_id = 'b1'")
    problems = ledger.verify(conn)
    assert any("contents changed" in p for p in problems)
    assert any("bets_no_update is missing" in p for p in problems)


class FakeSource(PolymarketSource):
    """Real Polymarket parsing, fake network."""
    def __init__(self, raws, book, final=None):
        self.raws, self.book, self.final = raws, book, final or {}

    def iter_markets(self, **kw):
        return iter(self.raws)

    def price_history(self, market):
        t0 = 1_792_000_000
        return [PricePoint(t0 + i * 3600, 0.50) for i in range(48)]

    def trade_history(self, market):
        return TradeHistory([TradeBucket(1_792_000_000, 20_000, 50)], True, 1_792_000_000)

    def order_book(self, market):
        return self.book

    def markets_by_id(self, ids):
        return [self.final[i] for i in ids if i in self.final]


class FixedModel:
    version = "vTEST"
    def __init__(self, p):
        self.p = p
    def predict(self, rows):
        return np.full(len(rows), self.p)


def _open_market(id):
    raw = gamma_market(id, volume=50_000, end="2026-10-30T00:00:00Z", bid=0.49, ask=0.51)
    raw["acceptingOrders"] = True
    raw["createdAt"] = "2026-10-01T00:00:00Z"
    raw["conditionId"] = f"0xcond{id}"
    return raw


def test_scorer_bets_settles_and_never_bets_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(scorer.registry, "latest", lambda: FixedModel(0.62))
    monkeypatch.setattr(scorer.time, "time", lambda: 1_792_200_000)
    conn = ledger.connect(tmp_path / "p.db")
    book = OrderBook(bids=[(0.49, 5000)], asks=[(0.51, 5000)])
    src = FakeSource([_open_market(1)], book)
    out = scorer.run(src, conn, workers=1, log=lambda *a: None)
    assert out == {"markets_scored": 1, "signals": 1, "bets_placed": 1}
    bet = conn.execute("SELECT * FROM bets").fetchone()
    assert bet["side"] == "YES" and bet["fill_price"] == pytest.approx(0.52)
    assert bet["stake"] == pytest.approx(min(0.25 * kelly_fraction(0.62, 0.52) * 1000, 50), abs=0.01)

    # Second run: same market is skipped.
    out = scorer.run(src, conn, workers=1, log=lambda *a: None)
    assert out["bets_placed"] == 0

    # Market resolves YES -> settled with profit.
    final = _open_market(1)
    final.update(closed=True, outcomePrices='["1", "0"]', umaResolutionStatus="resolved")
    src.final = {"1": final}
    assert settle.run(src, conn, log=lambda *a: None)["settled"] == 1
    s = conn.execute("SELECT * FROM settlements").fetchone()
    assert s["pnl"] == pytest.approx(bet["shares"] - bet["stake"])
    assert settle.run(src, conn, log=lambda *a: None)["open"] == 0   # nothing left to settle
    assert ledger.verify(conn) == []
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2


def test_no_bet_when_model_agrees_with_market(tmp_path, monkeypatch):
    monkeypatch.setattr(scorer.registry, "latest", lambda: FixedModel(0.53))
    monkeypatch.setattr(scorer.time, "time", lambda: 1_792_200_000)
    conn = ledger.connect(tmp_path / "p.db")
    src = FakeSource([_open_market(1)], OrderBook(bids=[(0.49, 5000)], asks=[(0.51, 5000)]))
    assert scorer.run(src, conn, workers=1, log=lambda *a: None)["bets_placed"] == 0


@pytest.mark.parametrize("prices,uma,expected", [
    ('["1", "0"]', "resolved", 1.0), ('["0", "1"]', "resolved", 0.0),
    ('["0.5", "0.5"]', "resolved", 0.5), ('["0.5", "0.5"]', "disputed", None)])
def test_yes_payout(prices, uma, expected):
    raw = gamma_market(1, closed=True)
    raw.update(outcomePrices=prices, umaResolutionStatus=uma)
    assert PolymarketSource().yes_payout(raw) == expected


def test_scorer_uses_live_midpoint_not_stale_history(monkeypatch):
    """History says 0.50 but the market now quotes 0.97/0.99 -> model input is 0.98."""
    src = FakeSource([], None)
    raw = _open_market(1)
    raw.update(bestBid=0.97, bestAsk=0.99)
    m = src.parse_market(raw)
    f = scorer._features_now(src, m, 1_792_200_000)
    assert f["price"] == pytest.approx(0.98)
    assert f["staleness_h"] == 0


def test_total_exposure_cap(tmp_path):
    conn = ledger.connect(tmp_path / "p.db")
    for i in range(9):   # $52 x 9 = $468 open
        ledger.append_bet(conn, _bet(bet_id=f"b{i}", market_id=str(i)))
    money = ledger.bankroll(conn)
    assert money["cash"] == pytest.approx(1000 - 468)
    assert money["available"] == pytest.approx(500 - 468)     # 50% cap binds before cash
    ledger.append_bet(conn, _bet(bet_id="b9", market_id="9"))
    assert ledger.bankroll(conn)["available"] == 0.0


def test_model_registry_ignores_other_folders(tmp_path):
    import json
    from pmodel.model import registry
    (tmp_path / "weather").mkdir()                      # sorts after "v..." folders
    (tmp_path / "weather" / "weather-v1.json").write_text("{}")
    for v, sup in (("v20260101_000000", None), ("v20260201_000000", "leaky")):
        (tmp_path / v).mkdir()
        (tmp_path / v / "meta.json").write_text(json.dumps({"version": v, "superseded": sup}))
    assert [p.name for p in registry.versions(tmp_path)] == ["v20260101_000000", "v20260201_000000"]
