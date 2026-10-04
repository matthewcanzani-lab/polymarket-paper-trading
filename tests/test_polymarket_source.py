from pmodel.sources import polymarket
from pmodel.sources.polymarket import PolymarketSource, to_iso_utc
from tests.conftest import gamma_market


def test_date_normalisation():
    assert to_iso_utc("2026-03-01 12:00:00+00") == "2026-03-01T12:00:00Z"
    assert to_iso_utc("2026-01-01T00:00:00.000Z") == "2026-01-01T00:00:00Z"
    assert to_iso_utc("2026-01-05") == "2026-01-05T00:00:00Z"
    assert to_iso_utc("2026-01-05T10:00:00-05:00") == "2026-01-05T15:00:00Z"
    assert to_iso_utc("garbage") is None
    assert to_iso_utc(None) is None


def test_parse_market_decodes_json_string_fields():
    m = PolymarketSource().parse_market(gamma_market(7, volume=1234.5))
    assert m.market_id == "7"
    assert m.is_yes_no and m.outcome_yes == "Yes"
    assert m.yes_token_id == "tok7yes"
    assert m.yes_price == 0.6
    assert m.volume_usd == 1234.5
    assert m.category == "Politics"
    assert m.closed_time is None


def test_non_binary_market_skipped():
    raw = gamma_market(8, outcomes=("A", "B", "C"), prices=("0.3", "0.3", "0.4"))
    assert PolymarketSource().parse_market(raw) is None


def test_two_outcome_non_yes_no_kept_but_flagged():
    m = PolymarketSource().parse_market(gamma_market(9, outcomes=("Lakers", "Celtics")))
    assert m is not None and not m.is_yes_no and m.outcome_yes == "Lakers"


def test_resolution_rules():
    src = PolymarketSource()
    assert src.parse_resolution(gamma_market(1)) is None  # still open
    yes = src.parse_resolution(gamma_market(1, closed=True, prices=("1", "0")))
    assert (yes.resolved_yes, yes.status, yes.resolved_at) == (1, "resolved", "2026-03-01T12:00:00Z")
    no = src.parse_resolution(gamma_market(1, closed=True, prices=("0.0005", "0.9995")))
    assert (no.resolved_yes, no.status) == (0, "resolved")
    split = src.parse_resolution(gamma_market(1, closed=True, prices=("0.5", "0.5")))
    assert (split.resolved_yes, split.status) == (None, "ambiguous")
    disputed = src.parse_resolution(gamma_market(1, closed=True, prices=("1", "0"), uma="disputed"))
    assert disputed.status == "ambiguous"


def test_pagination_stops_on_short_page(monkeypatch):
    monkeypatch.setattr(polymarket, "PAGE_SIZE", 2)
    pages = {0: [gamma_market(1), gamma_market(2)], 2: [gamma_market(3)]}
    calls = []

    def fake_get(url, params=None):
        calls.append(params)
        return pages.get(params["offset"], [])

    monkeypatch.setattr(polymarket.safe_http, "get_json", fake_get)
    ids = [r["id"] for r in PolymarketSource().iter_markets(closed=True, end_date_min="2024-01-01")]
    assert ids == ["1", "2", "3"]
    assert len(calls) == 2
    assert calls[0]["closed"] == "true" and calls[0]["end_date_min"] == "2024-01-01"


def test_price_history_falls_back_to_windows(monkeypatch):
    src = PolymarketSource()
    m = src.parse_market(gamma_market(5, closed=True))

    def fake_get(url, params=None):
        if params.get("interval") == "max":
            return {"history": []}
        return {"history": [{"t": params["startTs"], "p": 0.4}, {"t": params["startTs"] + 60, "p": "0.45"}]}

    monkeypatch.setattr(polymarket.safe_http, "get_json", fake_get)
    pts = src.price_history(m)
    assert len(pts) >= 4
    assert [p.ts for p in pts] == sorted(p.ts for p in pts)
    assert all(0 <= p.price <= 1 for p in pts)
