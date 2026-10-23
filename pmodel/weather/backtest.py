"""Backtest the weather model against Polymarket prices — without look-ahead.

Timeline (all splits by calendar date):
  fit    : days before VALID_START   -> learn each station's forecast bias/spread
  valid  : VALID_START..TEST_START   -> choose the spread scale and tail shape
  test   : TEST_START..             -> refit on everything before TEST_START,
                                       then compare to market prices. Touched once.

At each prediction moment t we use only:
  * a forecast whose issued_by_utc <= t (smallest lead that satisfies that),
  * the market's last price that was fully known by t (hourly bucket + 1h),
  * errors from days that ended before the period being predicted.
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pmodel.weather.model import ErrorModel

H = 3600
# (name, days relative to the measured day, local hour, kinds allowed)
SNAPSHOTS = [("2 days before, 18:00", -2, 18, ("high", "low")),
             ("day before, 12:00", -1, 12, ("high", "low")),
             # By 08:00 the overnight low has usually happened: lows excluded.
             ("same day, 08:00", 0, 8, ("high",))]
VALID_START, TEST_START = "2026-08-01", "2026-09-01"


def _unix(iso):
    return int(datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())


def load(conn):
    mk = pd.read_sql("SELECT * FROM weather_markets WHERE station IS NOT NULL", conn)
    fc = pd.read_sql("SELECT * FROM weather_forecasts", conn)
    return mk, fc


def forecast_errors(mk, fc, before_day):
    """observed - forecast (°C) from events whose winning range was a closed range."""
    win = mk[(mk.yes_payout == 1) & (mk.lo > -1e8) & (mk.hi < 1e8) & (mk.day < before_day)]
    d = win.merge(fc, on=["station", "day"])
    f_c = np.where(d.kind == "high", d.tmax_c, d.tmin_c)
    obs = (d.lo + d.hi) / 2
    obs_c = np.where(d.unit == "F", (obs - 32) * 5 / 9, obs)
    return pd.DataFrame({"station": d.station, "kind": d.kind, "lead": d.lead,
                         "err_c": obs_c - f_c, "day": d.day})


def snapshots(mk, fc, day_from, day_to, prices=None):
    """One row per (market, snapshot) with the forecast usable at t and, if given, the price."""
    fc_idx = {(r.station, r.day, r.lead): r for r in fc.itertuples()}
    tz_of = fc.groupby("station").tz.first().to_dict()
    rows = []
    ev = mk[(mk.day >= day_from) & (mk.day < day_to) & mk.yes_payout.notna()]
    for (station, day, kind), g in ev.groupby(["station", "day", "kind"]):
        if station not in tz_of:
            continue
        tz = ZoneInfo(tz_of[station])
        for name, offset, hour, kinds in SNAPSHOTS:
            if kind not in kinds:
                continue
            local = datetime.fromisoformat(day).replace(hour=hour, tzinfo=tz) + timedelta(days=offset)
            t = int(local.timestamp())
            lead = next((n for n in (1, 2, 3) if (station, day, n) in fc_idx
                         and _unix(fc_idx[(station, day, n)].issued_by_utc) <= t), None)
            if lead is None:
                continue
            f = fc_idx[(station, day, lead)]
            f_c = f.tmax_c if kind == "high" else f.tmin_c
            for m in g.itertuples():
                if m.created_at and _unix(m.created_at) > t:
                    continue
                if m.closed_time and _unix(m.closed_time) <= t:
                    continue
                price = None
                if prices is not None:
                    price = prices(m.market_id, t)
                    if price is None:
                        continue
                rows.append({"event": m.event_id, "market_id": m.market_id, "city": m.city,
                             "station": station, "kind": kind, "unit": m.unit, "lo": m.lo,
                             "hi": m.hi, "day": day, "snapshot": name, "t": t, "lead": lead,
                             "forecast_c": f_c, "price": price, "label": m.yes_payout,
                             "volume": m.volume_usd})
    return pd.DataFrame(rows)


def price_lookup(conn):
    """price_at(market_id, t): last hourly price fully known by t (bucket end <= t)."""
    cache = {}

    def price_at(market_id, t):
        if market_id not in cache:
            cache[market_id] = np.array(conn.execute(
                "SELECT ts, price FROM price_history WHERE platform='polymarket' AND market_id=? "
                "ORDER BY ts", (market_id,)).fetchall() or np.empty((0, 2)))
        a = cache[market_id]
        if not len(a):
            return None
        i = np.searchsorted(a[:, 0] + H, t, side="right") - 1
        return float(a[i, 1]) if i >= 0 else None
    return price_at


def predict(model: ErrorModel, snaps):
    return model.prob_batch(snaps.forecast_c, snaps.station, snaps.kind, snaps.lead,
                            snaps.unit, snaps.lo, snaps.hi)


def log_loss(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tune(mk, fc, log=print):
    """Pick spread scale and tail shape on the validation month (outcomes only, no prices)."""
    errs = forecast_errors(mk, fc, VALID_START)
    snaps = snapshots(mk, fc, VALID_START, TEST_START)
    best = None
    for df in (3.0, 6.0, 30.0):
        for scale in (0.8, 0.9, 1.0, 1.1, 1.25, 1.5):
            p = predict(ErrorModel.fit(errs, scale, df), snaps)
            ll = log_loss(snaps.label.to_numpy(), p)
            if best is None or ll < best[0]:
                best = (ll, scale, df)
    log(f"validation ({len(snaps):,} market-snapshots): best log loss {best[0]:.4f} "
        f"at spread scale {best[1]}, tail df {best[2]}")
    return best[1], best[2]


def bets(test, p_model, threshold=0.05, cost=0.02):
    """One bet per market at the first snapshot where the model and the price differ by
    more than `threshold` AND the bet still has positive value after `cost` per share.
    Returns a DataFrame of bets with profit per $1 staked."""
    d = test.assign(p=p_model).sort_values("t")
    yes = (d.p - d.price > threshold) & (d.p > d.price + cost)
    no = (d.price - d.p > threshold) & ((1 - d.p) > (1 - d.price) + cost)
    d = d[yes | no].assign(side=np.where(yes[yes | no], "YES", "NO"))
    d = d.groupby("market_id").head(1)
    pay = np.where(d.side == "YES", d.label, 1 - d.label)
    fill = np.where(d.side == "YES", d.price, 1 - d.price) + cost
    fill = np.minimum(fill, 0.99)
    return d.assign(fill=fill, ret=pay / fill - 1)


def bootstrap_mean(values, groups, n=2000, seed=0):
    """95% range of the mean, resampling whole events (ranges in one event share an outcome)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    sums = np.bincount(codes, values, len(uniq))
    cnts = np.bincount(codes, minlength=len(uniq))
    draws = []
    for _ in range(n):
        k = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        draws.append((k * sums).sum() / max((k * cnts).sum(), 1))
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def _logit(x):
    x = np.clip(np.asarray(x, dtype=float), 1e-3, 1 - 1e-3)
    return np.log(x / (1 - x))


def fit_blend(snaps):
    """Logistic blend: logit P = a*logit(market) + b*logit(forecast) + c.
    b > 0 means the forecast adds information the market price doesn't have."""
    from scipy.optimize import minimize
    X = np.c_[_logit(snaps.price), _logit(snaps.p), np.ones(len(snaps))]
    y = snaps.label.to_numpy()
    w = minimize(lambda w: np.mean(np.logaddexp(0, X @ w) - y * (X @ w)), [1.0, 0.0, 0.0]).x
    return w


def apply_blend(w, snaps):
    X = np.c_[_logit(snaps.price), _logit(snaps.p), np.ones(len(snaps))]
    return 1 / (1 + np.exp(-(X @ w)))


def run(conn, log=print):
    """The one-shot test. Returns a dict of results (also printed)."""
    mk, fc = load(conn)
    priced = mk[mk.history_fetched_at.notna()]
    price_at = price_lookup(conn)
    scale, df = tune(mk, fc, log)

    # Validation month: forecast model fitted on data before it, then the blend fitted on it.
    val = snapshots(priced, fc, VALID_START, TEST_START, prices=price_at)
    val["p"] = predict(ErrorModel.fit(forecast_errors(mk, fc, VALID_START), scale, df), val)
    w = fit_blend(val)
    log(f"blend weights (market, forecast, constant): {np.round(w, 3).tolist()}")

    # Test: refit the error model on everything before TEST_START; blend weights frozen.
    model = ErrorModel.fit(forecast_errors(mk, fc, TEST_START), scale, df)
    test = snapshots(priced, fc, TEST_START, "2099-01-01", prices=price_at)
    test["p"] = predict(model, test)
    test["blend"] = apply_blend(w, test)
    y = test.label.to_numpy()
    res = {"scale": scale, "df": df, "blend_weights": w.tolist(),
           "test_snapshots": len(test), "test_city_days": int(test.event.nunique()),
           "test_days": [test.day.min(), test.day.max()]}
    for col in ("price", "p", "blend"):
        res[f"brier_{col}"] = float(np.mean((test[col] - y) ** 2))
    diff = (test.blend - y) ** 2 - (test.price - y) ** 2
    res["brier_blend_minus_market_ci95"] = bootstrap_mean(diff.to_numpy(), test.event)
    res["by_snapshot"] = {
        n: {"n": len(g), "market": float(np.mean((g.price - g.label) ** 2)),
            "blend": float(np.mean((g.blend - g.label) ** 2))} for n, g in test.groupby("snapshot")}
    res["bets"] = {}
    for cost in (0.01, 0.02, 0.03, 0.04):
        b = bets(test, test.blend.to_numpy(), threshold=0.05, cost=cost)
        res["bets"][cost] = {"n": len(b), "avg_return": float(b.ret.mean()) if len(b) else None,
                             "ci95": bootstrap_mean(b.ret.to_numpy(), b.event) if len(b) else None}
    b = bets(test, test.blend.to_numpy(), threshold=0.05, cost=0.02)
    res["bets_by_snapshot_2c"] = {n: {"n": len(g), "avg_return": float(g.ret.mean())}
                                  for n, g in b.groupby("snapshot")}
    tiers = pd.cut(b.volume, [-1, 1000, 5000, 20000, 1e12], labels=["<$1k", "$1-5k", "$5-20k", ">$20k"])
    res["bets_by_final_volume_2c"] = {str(k): {"n": len(g), "avg_return": float(g.ret.mean())}
                                      for k, g in b.groupby(tiers, observed=True)}
    res["calibration"] = {}
    for col in ("price", "blend"):
        bins = pd.cut(test[col], np.linspace(0, 1, 11), include_lowest=True)
        res["calibration"][col] = [
            {"band": str(k), "forecast": float(g[col].mean()), "actual": float(g.label.mean()), "n": len(g)}
            for k, g in test.groupby(bins, observed=True)]
    return res, test
