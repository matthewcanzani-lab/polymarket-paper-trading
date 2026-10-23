"""Paper-trade daily temperature markets with the forecast + market blend.

Rules (each one is in config.py or this file, and tested):
  * Only the window the backtest found useful: from 12:00 local two days before
    the measured day until the measured day begins. Never on the day itself.
  * Forecast: the freshest forecast for that day. The error width used is the
    one learned for the matching lead time (2 days -> lead 3, 1 day -> lead 2),
    which is slightly cautious because live forecasts are a little fresher than
    the archived ones the model learned from.
  * Probability = blend(market midpoint, forecast probability).
  * Bet when it differs from the midpoint by > EDGE_THRESHOLD, priced by walking
    the LIVE order book + slippage, quarter Kelly, only if still +EV after the
    fill. One bet per market; at most WEATHER_MAX_EVENT_FRACTION of the bankroll
    on one city-day (its ranges move together).
Paper only — the order book is read, never traded on.
"""
import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from pmodel import config, safe_http
from pmodel.db import utc_now
from pmodel.model.train import _git_commit
from pmodel.sources import openmeteo, polymarket_weather as pw
from pmodel.sources.base import Market
from pmodel.sources.polymarket import PolymarketSource, _float
from pmodel.trading import ledger
from pmodel.trading.sizing import no_side_levels, size_bet
from pmodel.weather import artifact
from pmodel.weather.backtest import _logit

WINDOW_START_DAYS_BEFORE = 2   # from 12:00 local, 2 days before the measured day
WINDOW_START_HOUR = 12


def lead_for(now: datetime, day_end_local: datetime) -> int:
    """Error-width bucket matching the backtest at the same horizon."""
    hours = (day_end_local - now).total_seconds() / 3600
    return int(min(max(math.ceil(hours / 24), 1), 3))


def in_window(now: datetime, day: str, tz: ZoneInfo) -> bool:
    start = (datetime.fromisoformat(day).replace(hour=WINDOW_START_HOUR, tzinfo=tz)
             - timedelta(days=WINDOW_START_DAYS_BEFORE))
    day_start = datetime.fromisoformat(day).replace(tzinfo=tz)
    return start <= now < day_start


def current_forecasts(lat, lon, start, end):
    """Freshest hourly forecast per local day -> {day: (tmax_c, tmin_c)} (complete days only)."""
    data = safe_http.get_json(openmeteo.PREVIOUS_RUNS_URL, params={
        "latitude": lat, "longitude": lon, "hourly": "temperature_2m", "timezone": "auto",
        "start_date": start, "end_date": end})
    by_day = {}
    for ts, v in zip(data["hourly"]["time"], data["hourly"]["temperature_2m"]):
        if v is not None:
            by_day.setdefault(ts[:10], []).append(v)
    return {d: (max(v), min(v)) for d, v in by_day.items() if len(v) >= 24}


def blend(w, mid, p_forecast):
    x = w[0] * _logit(mid) + w[1] * _logit(p_forecast) + w[2]
    return 1 / (1 + np.exp(-x))


def _market(m: pw.TempMarket) -> Market:
    return Market("polymarket", m.market_id, "", None, "Weather", None, "Yes", "No", True,
                  m.yes_token_id, m.created_at, None, f"{m.day}T12:00:00Z", None, True, False,
                  m.volume_usd, 0.0, None, None, None)


def run(conn, log=print, now=None, source=None) -> dict:
    now = now or datetime.now(timezone.utc)
    source = source or PolymarketSource()
    params, model = artifact.load_latest()
    version, w = params["version"], params["blend_weights"]
    run_id = f"wrun-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    log(f"Weather scoring with {version} (run {run_id})")
    money = ledger.bankroll(conn)
    event_cap = config.WEATHER_MAX_EVENT_FRACTION * money["bankroll"]
    scored = signals = 0
    bets = []
    events = pw.iter_events(False, (now - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
                            (now + timedelta(days=4)).strftime("%Y-%m-%dT23:59:59Z"))
    forecast_cache, failed = {}, set()
    for ev in events:
        raw_by_id = {str(r["id"]): r for r in ev.get("markets") or []}
        markets = pw.parse_event(ev)
        if not markets:
            continue
        m0 = markets[0]
        st = params["stations"].get(m0.station or "")
        if st is None or not st.get("tz"):
            continue
        tz = ZoneInfo(st["tz"])
        if not in_window(now, m0.day, tz):   # check timing before spending a request
            continue
        if m0.station not in forecast_cache and m0.station not in failed:
            try:
                forecast_cache[m0.station] = current_forecasts(
                    st["lat"], st["lon"], (now - timedelta(days=1)).strftime("%Y-%m-%d"),
                    (now + timedelta(days=3)).strftime("%Y-%m-%d"))
            except Exception as exc:
                failed.add(m0.station)
                log(f"  forecast unavailable for {m0.station}: {exc}")
        fc = forecast_cache.get(m0.station, {}).get(m0.day)
        if fc is None:
            continue
        tmax, tmin = fc
        day_end = datetime.fromisoformat(m0.day).replace(tzinfo=tz) + timedelta(days=1)
        lead = lead_for(now, day_end)
        spent_on_event = 0.0
        for m in markets:
            raw = raw_by_id.get(m.market_id, {})
            if not raw.get("acceptingOrders", True) or raw.get("closed"):
                continue
            if ledger.has_bet_on(conn, "polymarket", m.market_id):
                continue
            bid, ask = _float(raw.get("bestBid")), _float(raw.get("bestAsk"))
            if bid is None or ask is None or not 0 < bid <= ask < 1:
                continue
            scored += 1
            f_c = tmax if m.kind == "high" else tmin
            p_fc = model.prob(f_c, m.station, m.kind, lead, m.unit, m.lo, m.hi)
            p = float(blend(w, (bid + ask) / 2, p_fc))
            if abs(p - (bid + ask) / 2) <= config.EDGE_THRESHOLD:
                continue
            book = source.order_book(_market(m))
            if not book or not book.bids or not book.asks:
                continue
            bid, ask = book.bids[0][0], book.asks[0][0]
            mid = (bid + ask) / 2
            p = float(blend(w, mid, p_fc))  # re-blend with the live book midpoint
            if abs(p - mid) <= config.EDGE_THRESHOLD:
                continue
            signals += 1
            side = "YES" if p > mid else "NO"
            q = p if side == "YES" else 1 - p
            levels = book.asks if side == "YES" else no_side_levels(book.bids)
            room = min(money["available"], event_cap - spent_on_event)
            sized = size_bet(q, levels, money["bankroll"], room, config)
            if sized is None or sized["shares"] < book.min_order_shares:
                continue
            q_text = (raw.get("question") or "")
            fields = {
                "bet_id": f"bet-{uuid.uuid4().hex[:12]}", "run_id": run_id, "placed_at": utc_now(),
                "platform": "polymarket", "market_id": m.market_id, "question": q_text,
                "category": "Weather", "end_date": f"{m.day}T12:00:00Z", "side": side,
                "model_prob_yes": p, "market_mid": mid, "best_bid": bid, "best_ask": ask,
                "fill_price": sized["fill_price"], "slippage_per_share": config.SLIPPAGE_PER_SHARE,
                "shares": sized["shares"], "stake": sized["stake"], "kelly_full": sized["kelly_full"],
                "bankroll_at_bet": money["bankroll"], "model_version": version}
            ledger.append_bet(conn, fields)
            money["cash"] -= sized["stake"]
            money["available"] -= sized["stake"]
            spent_on_event += sized["stake"]
            bets.append(fields)
            log(f"  PAPER BET {side} ${sized['stake']:.2f} @ {sized['fill_price']:.3f} "
                f"(model {p:.1%} vs market {mid:.1%}, forecast {f_c:.1f}°C) — {q_text[:70]}")
    ledger.append_run(conn, {
        "run_id": run_id, "started_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "finished_at": utc_now(),
        "model_version": version, "markets_scored": scored, "signals": signals,
        "bets_placed": len(bets), "settings_json": json.dumps({
            "strategy": "weather", "EDGE_THRESHOLD": config.EDGE_THRESHOLD,
            "KELLY_MULTIPLIER": config.KELLY_MULTIPLIER,
            "SLIPPAGE_PER_SHARE": config.SLIPPAGE_PER_SHARE,
            "MAX_STAKE_FRACTION": config.MAX_STAKE_FRACTION,
            "WEATHER_MAX_EVENT_FRACTION": config.WEATHER_MAX_EVENT_FRACTION,
            "MAX_OPEN_EXPOSURE_FRACTION": config.MAX_OPEN_EXPOSURE_FRACTION,
            "code_commit": _git_commit()})})
    conn.commit()
    summary = {"markets_scored": scored, "signals": signals, "bets_placed": len(bets)}
    if failed:
        summary["stations_without_forecast"] = len(failed)
    log(f"Weather run finished: {summary}")
    return summary
