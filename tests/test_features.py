"""Look-ahead guards for the model's features. If one of these fails, the
backtest numbers can't be trusted."""
import numpy as np
import pandas as pd
import pytest

from pmodel.model import features
from pmodel.model.dataset import snapshot_times
from pmodel.model.train import time_split

H = 3600
INFO = {"category": "Politics", "created_at": "2026-01-01T00:00:00Z",
        "end_date": "2026-02-01T00:00:00Z", "is_yes_no": 1}
T0 = features.iso_to_unix("2026-01-01T00:00:00Z")


def history(step=H, days=31, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.arange(T0, T0 + days * 86400, step, dtype=np.int64)
    return ts, rng.uniform(0.05, 0.95, size=ts.size)


@pytest.mark.parametrize("step", [H, 12 * H])
def test_features_ignore_everything_after_cutoff(step):
    ts, ps = history(step)
    t = T0 + 10 * 86400 + 1234
    before = features.features_at(INFO, ts, ps, t, step)
    scrambled = ps.copy()
    future = ts + step > t                     # points not yet knowable at t
    scrambled[future] = 1 - scrambled[future]  # change the future completely
    after = features.features_at(INFO, ts, scrambled, t, step)
    assert before == pytest.approx(after, nan_ok=True)


def test_point_is_only_known_after_its_bucket_closes():
    ts = np.array([T0, T0 + H], dtype=np.int64)
    ps = np.array([0.2, 0.9])
    f = features.features_at(INFO, ts, ps, T0 + H + 1, H)  # second bucket not finished
    assert f["price"] == 0.2
    assert features.features_at(INFO, ts, ps, T0 + H - 1, H) is None  # nothing known yet


def test_forbidden_fields_never_reach_features():
    market = dict(INFO, volume_usd=1e9, liquidity_usd=1e9, closed_time="2026-01-02T00:00:00Z",
                  yes_price=1.0, best_bid=0.99, best_ask=1.0, resolved_yes=1)
    info = features.static_info(market)
    assert set(info) == set(features.ALLOWED_STATIC)
    for banned in ("volume_usd", "liquidity_usd", "closed_time", "yes_price", "resolved_yes"):
        assert banned not in info


def test_feature_names_match():
    ts, ps = history()
    f = features.features_at(INFO, ts, ps, T0 + 5 * 86400, H)
    assert list(f) == features.FEATURE_NAMES


def test_snapshots_only_while_open():
    created, end = T0, T0 + 30 * 86400
    closed = T0 + 2 * 86400  # closed early
    times = snapshot_times(created, end, closed)
    assert times and all(created + H <= t <= closed - H for t in times)


def test_time_split_never_trains_on_unresolved_or_future():
    rng = np.random.default_rng(1)
    n = 5000
    t = rng.integers(0, 1_000_000, n)
    df = pd.DataFrame({"t": t, "closed_ts": t + rng.integers(1, 200_000, n),
                       "market_id": [str(i) for i in range(n)]})
    train, valid, test, valid_start, test_start = time_split(df)
    assert train["t"].max() < valid_start and train["closed_ts"].max() < valid_start
    assert valid["t"].max() < test_start and valid["closed_ts"].max() < test_start
    assert test["t"].min() >= test_start
    assert not set(train["market_id"]) & set(test["market_id"])
