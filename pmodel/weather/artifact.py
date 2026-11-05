"""Save / load the weather model as a small JSON file (models/weather/v<date>.json).

The file holds everything the live scorer needs: per-station forecast bias and
spread, the spread scale and tail shape, the market/forecast blend weights, and
station coordinates. So the scorer runs anywhere (e.g. GitHub Actions) without
the multi-GB market database. Old versions are never overwritten.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pmodel import config
from pmodel.weather import backtest as bt
from pmodel.weather.model import ErrorModel

WEATHER_DIR = config.PROJECT_ROOT / "models" / "weather"
MIN_PRICE_MOVES = 3


def fit(conn, log=print) -> dict:
    """Fit on all resolved history. Blend weights come from out-of-sample predictions
    (the error model used for them was fitted only on days before each period)."""
    mk, fc = bt.load(conn)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    scale, df = bt.tune(mk, fc, log)
    errors = bt.forecast_errors(mk, fc, today)
    model = ErrorModel.fit(errors, scale, df)

    priced = mk[mk.history_fetched_at.notna()]
    snaps = bt.snapshots(priced, fc, bt.VALID_START, today, prices=bt.price_lookup(conn))
    # Out-of-sample forecast probabilities: refit the error model month by month.
    parts = []
    for month, g in snaps.groupby(snaps.day.str[:7]):
        g = g.copy()
        g["p"] = bt.predict(ErrorModel.fit(bt.forecast_errors(mk, fc, f"{month}-01"), scale, df), g)
        parts.append(g)
    snaps = pd.concat(parts)
    # Same-day snapshots are excluded live (no edge there), so fit the blend without them.
    snaps = snaps[snaps.snapshot != "same day, 08:00"]
    # Learn the blend only where the recorded price was actively moving: stale
    # placeholder prices make the market look worse than live quotes really are,
    # which would make the model over-trust the forecast.
    snaps = snaps[bt.price_moves(conn, snaps) >= MIN_PRICE_MOVES]
    w = bt.fit_blend(snaps)
    log(f"blend weights from {len(snaps):,} out-of-sample, actively-priced snapshots: "
        f"{np.round(w, 3).tolist()}")

    tz = dict(conn.execute("SELECT station, MAX(tz) FROM weather_forecasts GROUP BY station"))
    stations = {r[0]: {"lat": r[1], "lon": r[2], "name": r[3], "tz": tz.get(r[0])}
                for r in conn.execute("SELECT station, lat, lon, name FROM weather_stations")}
    return {
        "version": "weather-v" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fitted_on_days_before": today,
        "resolved_markets_used": int(mk.yes_payout.notna().sum()),
        "scale": scale, "df": df, "blend_weights": [float(x) for x in w],
        "blend_fit_on": f"snapshots with >= {MIN_PRICE_MOVES} price moves in the prior 12h",
        "global": {f"{k}|{l}": list(v) for (k, l), v in model.global_.items()},
        "by_key": {f"{s}|{k}|{l}": list(v) for (s, k, l), v in model.by_key.items()},
        "stations": stations,
    }


def save(params: dict):
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    path = WEATHER_DIR / f"{params['version']}.json"
    path.write_text(json.dumps(params, indent=1))
    return path


def load_latest(directory=None):
    directory = directory or WEATHER_DIR
    files = sorted(directory.glob("weather-v*.json"))
    if not files:
        raise SystemExit("No weather model yet — run `python -m pmodel weather-fit`.")
    params = json.loads(files[-1].read_text())
    return params, to_error_model(params)


def to_error_model(params) -> ErrorModel:
    m = ErrorModel(scale=params["scale"], df=params["df"])
    for key, v in params["global"].items():
        kind, lead = key.split("|")
        m.global_[(kind, int(lead))] = tuple(v)
    for key, v in params["by_key"].items():
        st, kind, lead = key.split("|")
        m.by_key[(st, kind, int(lead))] = tuple(v)
    return m


def refit_errors(conn, base: dict, log=print) -> dict:
    """Light refit for automation: re-learn per-station bias/spread from resolved
    outcomes + archived forecasts (no price history needed). Keeps the spread
    scale, tail shape and blend weights of `base` (those need the full fit)."""
    mk, fc = bt.load(conn)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    errors = bt.forecast_errors(mk, fc, today)
    model = ErrorModel.fit(errors, base["scale"], base["df"])
    tz = dict(conn.execute("SELECT station, MAX(tz) FROM weather_forecasts GROUP BY station"))
    stations = dict(base["stations"])
    for st, lat, lon, name in conn.execute("SELECT station, lat, lon, name FROM weather_stations"):
        stations[st] = {"lat": lat, "lon": lon, "name": name, "tz": tz.get(st) or stations.get(st, {}).get("tz")}
    log(f"refit on {len(errors):,} forecast errors from {errors.day.min()} to {errors.day.max()}")
    return {**base,
            "version": "weather-v" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fitted_on_days_before": today, "refit_of": base["version"], "refit": "errors only",
            "resolved_markets_used": int(mk.yes_payout.notna().sum()),
            "global": {f"{k}|{l}": list(v) for (k, l), v in model.global_.items()},
            "by_key": {f"{s}|{k}|{l}": list(v) for (s, k, l), v in model.by_key.items()},
            "stations": stations}


# ---------------------------------------------------------------- retraining guard
MIN_SETTLED_BEFORE_RETRAIN = 150


def settled_bets(ledger_conn, version: str) -> int:
    return ledger_conn.execute(
        """SELECT COUNT(*) FROM bets b JOIN settlements s USING (bet_id)
           WHERE b.model_version = ?""", (version,)).fetchone()[0]


def retrain_allowed(ledger_conn, params: dict) -> tuple[bool, str]:
    """Refits are manual only (since 2026-11-05) and need the current version to
    have a big enough forward record to judge the next one against."""
    n = settled_bets(ledger_conn, params["version"])
    if n < MIN_SETTLED_BEFORE_RETRAIN:
        return False, (f"{params['version']} has {n} settled bets; retraining waits until it has "
                       f"{MIN_SETTLED_BEFORE_RETRAIN}+.")
    return True, f"{params['version']} has {n} settled bets."


# ---------------------------------------------------------------- calibrated version
CALIBRATED_RULES = {
    "edge_threshold": 0.10,            # chosen AFTER seeing results: needs forward validation
    "max_spread": 0.06,
    "fill_at_best_price_only": True,
    "city_day_cap_fraction": 0.05,     # all bets on one city + date, across runs and high/low
    "stop_loss_fraction": 0.15,
}


def calibrated_version(ledger_conn, base: dict, before: str) -> dict:
    """A new version: `base`'s forecast model + a calibration toward the market
    fitted only on paper bets settled before `before`, + the stricter rules."""
    from pmodel.trading import ledger
    from pmodel.weather import rules
    now = datetime.now(timezone.utc)
    return {**base,
            "version": "weather-v" + now.strftime("%Y%m%d_%H%M%S") + "-calibrated",
            "label": "Calibrated to market, 10-pt edge, spread/depth filters, city-day cap, stop-loss",
            "created_at": now.isoformat(timespec="seconds"),
            "based_on": base["version"],
            "calibration": rules.fit_calibration(ledger_conn, before),
            "rules": dict(CALIBRATED_RULES),
            "starting_bankroll": round(ledger.bankroll(ledger_conn)["bankroll"], 2)}
