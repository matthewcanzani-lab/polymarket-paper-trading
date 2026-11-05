"""Sportsbook-odds predictions: de-vig, matching, key safety, prediction-only runs."""
from datetime import datetime, timezone

import pytest

from pmodel import config
from pmodel.sources import odds_api
from pmodel.sports import scorer as sp
from pmodel.trading import ledger, predlog

NOW = datetime(2026, 11, 5, 18, 0, tzinfo=timezone.utc)


def _book(key, prices):
    return {"key": key, "markets": [{"key": "h2h", "outcomes": [
        {"name": n, "price": p} for n, p in prices.items()]}]}


def _event(home, away, start="2026-11-06T00:05:00Z", books=None):
    return {"home_team": home, "away_team": away, "commence_time": start,
            "bookmakers": books or [_book("pinnacle", {home: 1.80, away: 2.10})]}


def _raw(id, question, outcomes, title, slug="nhl-dal-cbj-2026-11-05",
         start="2026-11-06 00:00:00+00", bid=0.54, ask=0.56):
    return {"id": id, "question": question, "outcomes": f'["{outcomes[0]}", "{outcomes[1]}"]',
            "sportsMarketType": "moneyline", "slug": slug, "gameStartTime": start,
            "events": [{"title": title}], "bestBid": bid, "bestAsk": ask,
            "endDate": "2026-11-06T00:00:00Z", "active": True, "closed": False}


def test_fair_probs_remove_the_bookmaker_margin():
    p = odds_api.fair_probs(_book("x", {"A": 1.80, "B": 2.10}))
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["A"] == pytest.approx((1 / 1.8) / (1 / 1.8 + 1 / 2.1))


def test_consensus_prefers_pinnacle_else_median():
    ev = _event("A", "B", books=[_book("x", {"A": 1.5, "B": 2.5}), _book("pinnacle", {"A": 2.0, "B": 2.0})])
    assert odds_api.consensus(ev) == ({"A": 0.5, "B": 0.5}, "pinnacle")
    ev = _event("A", "B", books=[_book("x", {"A": 2.0, "B": 2.0}), _book("y", {"A": 1.5, "B": 3.0}),
                                 _book("z", {"A": 1.9, "B": 2.1})])
    probs, src = odds_api.consensus(ev)
    assert src == "median of 3 books" and sum(probs.values()) == pytest.approx(1.0)


def test_team_names_match_nicknames_and_full_names():
    assert sp.same_team("Stars", "Dallas Stars")
    assert sp.same_team("Blue Jackets", "Columbus Blue Jackets")
    assert sp.same_team("Everton FC", "Everton")
    assert sp.same_team("Al Fateh Saudi Club", "Al-Fateh")
    assert not sp.same_team("Stars", "Columbus Blue Jackets")


def test_describe_game_markets():
    d = sp.describe(_raw(1, "Stars vs. Blue Jackets", ("Stars", "Blue Jackets"), "Stars vs. Blue Jackets"))
    assert d["target"] == "Stars" and d["teams"] == ("Stars", "Blue Jackets")
    assert d["start"] == datetime(2026, 11, 6, tzinfo=timezone.utc)
    d = sp.describe(_raw(2, "Will Everton FC win on 2026-11-06?", ("Yes", "No"),
                         "Everton FC vs. Fulham FC", slug="epl-eve-ful-2026-11-06-eve"))
    assert d["target"] == "Everton FC"
    d = sp.describe(_raw(3, "Will Everton FC vs. Fulham FC end in a draw?", ("Yes", "No"),
                         "Everton FC vs. Fulham FC", slug="epl-eve-ful-2026-11-06-draw"))
    assert d["target"] == "Draw"
    assert sp.describe(_raw(4, "Stars vs. Blue Jackets: O/U 5.5", ("Over", "Under"), "x")
                       | {"sportsMarketType": "totals"}) is None
    assert sp.describe(_raw(5, "DNK vs LPH", ("DNK", "LPH"), "DNK vs LPH", slug="cs2-x")) is None


def test_match_needs_both_teams_and_close_start_time():
    d = sp.describe(_raw(1, "Stars vs. Blue Jackets", ("Stars", "Blue Jackets"), "Stars vs. Blue Jackets"))
    ev = _event("Columbus Blue Jackets", "Dallas Stars")
    assert sp.match(d, [ev]) == (ev, "Dallas Stars")
    assert sp.match(d, [_event("Columbus Blue Jackets", "Dallas Stars", start="2026-11-07T00:00:00Z")]) == (None, None)
    assert sp.match(d, [_event("Columbus Blue Jackets", "Boston Bruins")]) == (None, None)
    assert sp.match(d, [ev, ev]) == (None, None)                 # ambiguous: skip


class _Src:
    def __init__(self, raws):
        self.raws = raws

    def iter_markets(self, **kw):
        return iter(self.raws)

    def is_accepting_orders(self, raw):
        return True


def test_run_logs_predictions_and_places_no_bets(tmp_path):
    raws = [_raw(1, "Stars vs. Blue Jackets", ("Stars", "Blue Jackets"), "Stars vs. Blue Jackets"),
            _raw(2, "Will Everton FC win on 2026-11-06?", ("Yes", "No"), "Everton FC vs. Fulham FC",
                 slug="epl-eve-ful-2026-11-06-eve", start="2026-11-06 15:00:00+00"),
            _raw(3, "Rangers vs. Flyers", ("Rangers", "Flyers"), "Rangers vs. Flyers",
                 start="2026-11-05 17:00:00+00")]                 # already started: not logged
    odds = {"icehockey_nhl": [_event("Columbus Blue Jackets", "Dallas Stars")],
            "soccer_epl": [_event("Everton", "Fulham", start="2026-11-06T15:00:00Z", books=[
                _book("pinnacle", {"Everton": 2.5, "Fulham": 3.0, "Draw": 3.2})])]}
    conn = ledger.connect(tmp_path / "p.db")
    out = sp.run(conn, log=lambda *a: None, now=NOW, source=_Src(raws),
                 fetch=lambda key: (odds[key], 400), predlog_dir=tmp_path / "pl")
    assert out["matched"] == out["predictions_logged"] == 2
    rows = {r["market_id"]: r for r in predlog.read("predictions", tmp_path / "pl")}
    assert set(rows) == {"1", "2"} and all(r["strategy"] == "sports" for r in rows.values())
    stars = odds_api.fair_probs(_book("p", {"Columbus Blue Jackets": 1.80, "Dallas Stars": 2.10}))["Dallas Stars"]
    assert float(rows["1"]["model_prob_yes"]) == pytest.approx(stars, abs=1e-6)
    assert float(rows["1"]["market_mid"]) == pytest.approx(0.55)
    assert conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0] == 0


def test_missing_key_is_skipped_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ODDS_API_KEY", "")
    conn = ledger.connect(tmp_path / "p.db")
    raws = [_raw(1, "Stars vs. Blue Jackets", ("Stars", "Blue Jackets"), "Stars vs. Blue Jackets")]
    out = sp.run(conn, log=lambda *a: None, now=NOW, source=_Src(raws), predlog_dir=tmp_path / "pl")
    assert out["matched"] == 0 and out["skipped"]["no odds"] == 1


def test_api_key_never_appears_in_errors(monkeypatch):
    monkeypatch.setattr(config, "ODDS_API_KEY", "SECRET123")

    def boom(url, params=None, **kw):
        raise RuntimeError(f"410 for url {url}?apiKey={params['apiKey']}")
    monkeypatch.setattr(odds_api.safe_http, "get_json", boom)
    with pytest.raises(odds_api.OddsUnavailable) as err:
        odds_api.fetch("icehockey_nhl")
    assert "SECRET123" not in str(err.value) and "***" in str(err.value)
