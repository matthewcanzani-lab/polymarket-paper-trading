"""Score open markets and log FAKE bets where the model disagrees with the market.

Each run:
  1. Pull open markets in the universe that end within SCORE_MAX_DAYS_TO_END days.
  2. For each, build the same features the model was trained on, as of NOW.
  3. Predict P(YES). Where |model - market midpoint| > EDGE_THRESHOLD, read the
     live order book, price a fake fill (walk the book + slippage), size it with
     quarter Kelly, and log it if it's still worth taking after costs.
  4. Append one `runs` row summarising the run (even when no bets are placed).

Paper trading only: the order book is READ to price a pretend fill. No order
is ever sent anywhere.
"""
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np

from pmodel import config, universe
from pmodel.db import utc_now
from pmodel.model import registry
from pmodel.model.train import _git_commit
from pmodel.model.features import features_at, static_info
from pmodel.sources.base import MarketSource
from pmodel.trading import ledger
from pmodel.trading.sizing import no_side_levels, size_bet

SETTINGS_KEYS = ("EDGE_THRESHOLD", "KELLY_MULTIPLIER", "SLIPPAGE_PER_SHARE", "MAX_STAKE_FRACTION",
                 "MIN_STAKE_USD", "SCORE_MAX_DAYS_TO_END", "MIN_VOLUME_USD", "BANKROLL_START_USD")


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def candidates(source: MarketSource, conn, now: datetime, log=print):
    raws = source.iter_markets(
        closed=False, end_date_min=_iso(now), volume_min=config.MIN_VOLUME_USD,
        end_date_max=_iso(now + timedelta(days=config.SCORE_MAX_DAYS_TO_END)))
    out, skipped = [], {"not_binary": 0, "not_in_universe": 0, "not_trading": 0, "already_bet": 0}
    for raw in raws:
        m = source.parse_market(raw)
        if m is None:
            skipped["not_binary"] += 1
        elif not universe.is_in_universe(m):
            skipped["not_in_universe"] += 1
        elif not source.is_accepting_orders(raw):
            skipped["not_trading"] += 1
        elif ledger.has_bet_on(conn, m.platform, m.market_id):
            skipped["already_bet"] += 1
        else:
            out.append(m)
    log(f"{len(out):,} markets to score (skipped: {skipped})")
    return out


def live_mid(market):
    """Current midpoint from the market listing, or the last price if no clean quote."""
    bid, ask = market.best_bid, market.best_ask
    if bid is not None and ask is not None and 0 <= bid <= ask <= 1:
        return (bid + ask) / 2
    return market.yes_price


def _features_now(source: MarketSource, market, t: int):
    hist = source.price_history(market)
    if not hist:
        return None
    step = source.history_step_seconds(market)
    # Price history can be hours old (hourly or 12-hourly buckets). Add the CURRENT
    # midpoint as the newest point, stamped so features_at treats it as known right
    # now, so the model judges the market as it is, not as it was. (Without this a
    # stale price looks like a huge "edge" whenever the market has since moved.)
    mid = live_mid(market)
    keep = [p for p in hist if p.ts < t - step]
    ts = np.array([p.ts for p in keep] + ([t - step] if mid is not None else []), dtype=np.int64)
    ps = np.array([p.price for p in keep] + ([mid] if mid is not None else []), dtype=np.float64)
    if ts.size == 0:
        return None
    trades = None
    if market.condition_id:
        th = source.trade_history(market)
        if th.buckets:
            trades = (np.array([b.hour_ts for b in th.buckets], dtype=np.int64),
                      np.array([b.usd for b in th.buckets]),
                      np.array([b.trades for b in th.buckets], dtype=np.float64))
    return features_at(static_info(market), ts, ps, t, step, trades)


def run(source: MarketSource, conn, workers=8, log=print, snapshot_path=None) -> dict:
    started = datetime.now(timezone.utc)
    model = registry.latest()
    run_id = f"run-{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    log(f"Scoring with model {model.version} (run {run_id})")
    markets = candidates(source, conn, started, log)
    t = int(time.time())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        feats = list(pool.map(lambda m: _features_now(source, m, t), markets))
    scored = [(m, f) for m, f in zip(markets, feats) if f is not None]
    log(f"Built features for {len(scored):,} markets")
    probs = model.predict([f for _, f in scored]) if scored else []

    money = ledger.bankroll(conn)
    signals, bets, snapshot = 0, [], []
    for (market, feat), p in zip(scored, probs):
        row = {"market_id": market.market_id, "question": market.question,
               "category": market.category, "end_date": market.end_date,
               "volume_usd": market.volume_usd, "last_price": feat["price"],
               "best_bid": market.best_bid, "best_ask": market.best_ask,
               "model_prob_yes": float(p), "status": "no edge"}
        snapshot.append(row)
        # Cheap pre-check on the last price before spending a request on the order book.
        if abs(p - feat["price"]) <= config.EDGE_THRESHOLD:
            continue
        book = source.order_book(market)
        if not book or not book.bids or not book.asks:
            row["status"] = "edge, but empty order book"
            continue
        bid, ask = book.bids[0][0], book.asks[0][0]
        mid = (bid + ask) / 2
        row.update(best_bid=bid, best_ask=ask)
        if abs(p - mid) <= config.EDGE_THRESHOLD:
            continue
        signals += 1
        side = "YES" if p > mid else "NO"
        q = p if side == "YES" else 1 - p
        levels = book.asks if side == "YES" else no_side_levels(book.bids)
        sized = size_bet(q, levels, money["bankroll"], money["cash"], config)
        if sized is None or sized["shares"] < book.min_order_shares:
            row["status"] = "edge, but not worth it after fill costs"
            continue
        row["status"] = f"PAPER BET {side}"
        fields = {
            "bet_id": f"bet-{uuid.uuid4().hex[:12]}", "run_id": run_id, "placed_at": utc_now(),
            "platform": market.platform, "market_id": market.market_id,
            "question": market.question, "category": market.category,
            "end_date": market.end_date, "side": side, "model_prob_yes": float(p),
            "market_mid": mid, "best_bid": bid, "best_ask": ask,
            "fill_price": sized["fill_price"], "slippage_per_share": config.SLIPPAGE_PER_SHARE,
            "shares": sized["shares"], "stake": sized["stake"], "kelly_full": sized["kelly_full"],
            "bankroll_at_bet": money["bankroll"], "model_version": model.version,
        }
        ledger.append_bet(conn, fields)
        money["cash"] -= sized["stake"]
        bets.append(fields)
        log(f"  PAPER BET {side} ${sized['stake']:.2f} @ {sized['fill_price']:.3f} "
            f"(model {p:.1%} vs market {mid:.1%}) — {market.question[:70]}")

    ledger.append_run(conn, {
        "run_id": run_id, "started_at": _iso(started), "finished_at": utc_now(),
        "model_version": model.version, "markets_scored": len(scored), "signals": signals,
        "bets_placed": len(bets),
        "settings_json": json.dumps({**{k: getattr(config, k) for k in SETTINGS_KEYS},
                                     "code_commit": _git_commit()}),
    })
    conn.commit()
    if snapshot_path:
        # Not part of the ledger (it's overwritten each run): feeds the dashboard's
        # Live Markets / Edge Finder pages.
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps({
            "run_id": run_id, "scored_at": _iso(started), "model_version": model.version,
            "edge_threshold": config.EDGE_THRESHOLD, "markets": snapshot}))
    summary = {"markets_scored": len(scored), "signals": signals, "bets_placed": len(bets)}
    log(f"Run finished: {summary}")
    return summary
