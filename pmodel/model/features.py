"""Features for predicting P(YES) at a moment in time — with no look-ahead.

HOW LOOK-AHEAD IS PREVENTED (read this before changing anything here)

1. Allowlisted static facts only. `static_info()` copies just the market fields
   that are known the moment a market opens (category, scheduled end date,
   creation date, outcome labels). Lifetime volume, liquidity, the actual close
   time, final prices and the outcome are NOT copied, so `features_at()` cannot
   see them even by accident.

2. Prices strictly from the past, with a safety margin. A price point stamped
   `ts` may summarise the whole bucket [ts, ts + step), so it is treated as
   knowable only from `ts + step`. At prediction time `t` we use only points
   with `ts + step <= t`.

3. Trading activity the same way: an hourly bucket of trades [h, h+1h) is only
   known from h+1h. We never use the market's final volume.

4. Tested. tests/test_features.py scrambles every price and trade after the
   cutoff and checks that the features don't change, and checks that forbidden
   market fields never reach this module.

Snapshot selection (which moments we sample) lives in dataset.py and is
allowed to use the close time — only to make sure the market was still open
at `t`, which the live scorer also knows. It never becomes a feature.
"""
import math
from datetime import datetime, timezone

import numpy as np

# Fields known when a market opens. Anything else is off limits for features.
ALLOWED_STATIC = ("category", "created_at", "end_date", "is_yes_no")

CATEGORIES = ("Sports", "Esports", "Crypto", "Weather", "Politics", "Economy",
              "Culture", "Tech & Science", "Geopolitics")

FEATURE_NAMES = [
    "price", "logit_price", "dist_from_half", "staleness_h",
    "chg_6h", "chg_1d", "chg_3d", "chg_7d",
    "max_7d", "min_7d", "vol_7d",
    "hours_to_end", "log_hours_to_end", "age_h", "frac_elapsed", "scheduled_days",
    "is_yes_no",
    "log_traded_usd", "log_usd_24h", "trades_24h",
] + [f"cat_{c}" for c in CATEGORIES] + ["cat_other"]

H = 3600
EPS = 1e-4


def iso_to_unix(iso):
    if not iso:
        return None
    return int(datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def static_info(market) -> dict:
    """The ONLY market metadata features may use. Accepts a Market or a DB row."""
    get = (lambda k: market[k]) if not hasattr(market, "__dataclass_fields__") else (
        lambda k: getattr(market, k))
    return {k: get(k) for k in ALLOWED_STATIC}


def _price_before(ts, ps, cutoff):
    """Last known price at or before `cutoff` (arrays sorted by ts), or None."""
    i = np.searchsorted(ts, cutoff, side="right") - 1
    return (float(ps[i]), int(ts[i])) if i >= 0 else (None, None)


def traded_before(trades, t):
    """(dollars traded before t, dollars in the 24h before t, trades in the 24h before t).

    trades = (hour_ts, usd, count) arrays, sorted. Only fully finished hours count.
    """
    hour_ts, usd, cnt = trades
    done = hour_ts + H <= t
    recent = done & (hour_ts + H > t - 24 * H)
    return float(usd[done].sum()), float(usd[recent].sum()), float(cnt[recent].sum())


def features_at(info: dict, ts: np.ndarray, ps: np.ndarray, t: int, step: int, trades=None):
    """Feature dict for predicting at unix time `t`, or None if no usable price yet.

    info: output of static_info(); ts/ps: price history (sorted); step: bucket size;
    trades: optional (hour_ts, usd, count) arrays of hourly trading activity.
    """
    # Shift every point to the moment it became knowable, then never look past t.
    known_ts = ts + step
    n = np.searchsorted(known_ts, t, side="right")
    if n == 0:
        return None
    kts, kps = known_ts[:n], ps[:n]
    price, last_ts = float(kps[-1]), int(kts[-1])

    def change(hours):
        old, _ = _price_before(kts, kps, t - hours * H)
        return price - old if old is not None else np.nan

    week = kts >= t - 7 * 24 * H
    wk = kps[week]
    created = iso_to_unix(info["created_at"])
    end = iso_to_unix(info["end_date"])
    hours_to_end = (end - t) / H if end else np.nan
    age_h = (t - created) / H if created else np.nan
    sched = (end - created) / H if end and created else np.nan

    f = {
        "price": price,
        "logit_price": logit(price),
        "dist_from_half": abs(price - 0.5),
        "staleness_h": (t - last_ts) / H,
        "chg_6h": change(6), "chg_1d": change(24), "chg_3d": change(72), "chg_7d": change(168),
        "max_7d": float(wk.max()) if wk.size else np.nan,
        "min_7d": float(wk.min()) if wk.size else np.nan,
        "vol_7d": float(np.diff(wk).std()) if wk.size > 2 else np.nan,
        "hours_to_end": hours_to_end,
        "log_hours_to_end": math.log1p(max(hours_to_end, 0)) if end else np.nan,
        "age_h": age_h,
        "frac_elapsed": age_h / sched if sched and sched > 0 else np.nan,
        "scheduled_days": sched / 24 if sched == sched else np.nan,
        "is_yes_no": float(bool(info["is_yes_no"])),
    }
    if trades is not None:
        total, usd24, n24 = traded_before(trades, t)
        f.update(log_traded_usd=math.log1p(total), log_usd_24h=math.log1p(usd24), trades_24h=n24)
    else:
        f.update(log_traded_usd=np.nan, log_usd_24h=np.nan, trades_24h=np.nan)
    cat = info.get("category")
    for c in CATEGORIES:
        f[f"cat_{c}"] = float(cat == c)
    f["cat_other"] = float(cat not in CATEGORIES)
    return f
