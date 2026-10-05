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
    assert m.tags == ["All", "Politics", "Some Person"]
    assert m.closed_time is None


def test_category_mapping_priority():
    from pmodel.sources.polymarket import category_from_tags
    assert category_from_tags(["Trump", "NFL"]) == "Sports"  # sports checked before politics
    assert category_from_tags(["Counter-Strike", "Sports"]) == "Esports"
    assert category_from_tags(["Random"]) is None


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


def test_keyset_pagination_follows_cursor(monkeypatch):
    pages = {None: {"markets": [gamma_market(1), gamma_market(2)], "next_cursor": "c1"},
             "c1": {"markets": [gamma_market(3)], "next_cursor": "c2"},
             "c2": {"markets": []}}
    calls = []

    def fake_get(url, params=None):
        calls.append(params)
        return pages[params.get("after_cursor")]

    monkeypatch.setattr(polymarket.safe_http, "get_json", fake_get)
    ids = [r["id"] for r in PolymarketSource().iter_markets(
        closed=True, end_date_min="2024-01-01", volume_min=1000)]
    assert ids == ["1", "2", "3"]
    assert calls[0]["volume_num_min"] == 1000
    assert len(calls) == 3
    assert calls[0]["closed"] == "true" and calls[0]["end_date_min"] == "2024-01-01"
    assert calls[0]["include_tag"] == "true"


def test_short_market_gets_hourly_history_in_14_day_windows(monkeypatch):
    src = PolymarketSource()
    m = src.parse_market(gamma_market(5, closed=True, end="2026-03-01T00:00:00Z"))  # scheduled 59 days
    calls = []

    def fake_get(url, params=None):
        calls.append(params)
        assert params["endTs"] - params["startTs"] <= polymarket.HOURLY_WINDOW_SEC
        return {"history": [{"t": params["startTs"], "p": 0.4}, {"t": params["startTs"] + 3600, "p": "0.45"}]}

    monkeypatch.setattr(polymarket.safe_http, "get_json", fake_get)
    pts = src.price_history(m)
    assert len(calls) == 5 and all(c["fidelity"] == 60 for c in calls)
    assert len(pts) == 10
    assert [p.ts for p in pts] == sorted(p.ts for p in pts)


def test_long_market_gets_coarse_history_in_one_request(monkeypatch):
    src = PolymarketSource()
    raw = gamma_market(6, closed=True)
    raw["createdAt"] = "2025-01-01T00:00:00Z"
    m = src.parse_market(raw)
    calls = []

    def fake_get(url, params=None):
        calls.append(params)
        return {"history": [{"t": 1, "p": 0.5}]}

    monkeypatch.setattr(polymarket.safe_http, "get_json", fake_get)
    assert len(src.price_history(m)) == 1
    assert calls == [{"market": "tok6yes", "interval": "max", "fidelity": 720}]


def test_granularity_uses_scheduled_length_not_actual_close(monkeypatch):
    """Look-ahead guard: a long market that happened to close early must NOT get
    hourly data, or the data's resolution would hint at the outcome."""
    src = PolymarketSource()
    raw = gamma_market(7, closed=True)
    raw["createdAt"] = "2026-01-01T00:00:00Z"
    raw["endDate"] = "2026-12-31T00:00:00Z"         # scheduled for a year...
    raw["closedTime"] = "2026-01-05 00:00:00+00"    # ...but closed after 4 days
    m = src.parse_market(raw)
    calls = []
    monkeypatch.setattr(polymarket.safe_http, "get_json",
                        lambda url, params=None: calls.append(params) or {"history": []})
    src.price_history(m)
    assert calls == [{"market": "tok7yes", "interval": "max", "fidelity": 720}]
