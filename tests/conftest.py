import pytest

from pmodel import config, db


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    monkeypatch.setattr(config, "REQUEST_PAUSE_SEC", 0)


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "market_data.db")
    yield c
    c.close()


def gamma_market(id, *, closed=False, prices=("0.6", "0.4"), outcomes=("Yes", "No"),
                 volume=5000.0, uma=None, category="Politics", bid=0.59, ask=0.61,
                 end="2026-10-20T00:00:00Z"):
    """A market record shaped like the Gamma API returns it (list fields are JSON strings)."""
    import json
    raw = {
        "id": str(id),
        "question": f"Will thing {id} happen?",
        "slug": f"thing-{id}",
        "outcomes": json.dumps(list(outcomes)),
        "outcomePrices": json.dumps(list(prices)),
        "clobTokenIds": json.dumps([f"tok{id}yes", f"tok{id}no"]),
        "volume": str(volume),
        "volumeNum": volume,
        "liquidityNum": 1000.0,
        "active": True,
        "closed": closed,
        "createdAt": "2026-01-01T00:00:00.000Z",
        "startDate": "2026-01-02T00:00:00Z",
        "endDate": end,
        "closedTime": "2026-03-01 12:00:00+00" if closed else None,
        "bestBid": bid,
        "bestAsk": ask,
        "tags": [{"label": category}, {"label": "Some Person"}],
        "events": [{"title": f"Event {id}", "tags": [{"label": "All"}]}],
    }
    if uma:
        raw["umaResolutionStatus"] = uma
    return raw
