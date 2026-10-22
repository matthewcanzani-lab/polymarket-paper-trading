"""Settle fake bets whose markets have resolved.

Settlement never edits a bet: it appends a row to `settlements`.
Payout = shares x (what one share of the side we bought paid out):
  YES side: yes_payout;  NO side: 1 - yes_payout;  50-50 split: 0.5 either way.
"""
from pmodel.db import utc_now
from pmodel.sources.base import MarketSource
from pmodel.trading import ledger


def run(source: MarketSource, conn, log=print) -> dict:
    open_bets = ledger.open_bets(conn)
    by_platform = [b for b in open_bets if b["platform"] == source.platform]
    stats = {"open": len(by_platform), "settled": 0, "still_open": 0}
    ids = sorted({b["market_id"] for b in by_platform})
    raws = {}
    for i in range(0, len(ids), 100):
        for raw in source.markets_by_id(ids[i:i + 100]):
            raws[str(raw["id"])] = raw
    for bet in by_platform:
        raw = raws.get(bet["market_id"])
        yes = source.yes_payout(raw) if raw else None
        if yes is None:
            stats["still_open"] += 1
            continue
        per_share = yes if bet["side"] == "YES" else 1 - yes
        payout = bet["shares"] * per_share
        pnl = payout - bet["stake"]
        ledger.append_settlement(conn, {
            "bet_id": bet["bet_id"], "settled_at": utc_now(), "yes_payout": yes,
            "payout": payout, "pnl": pnl})
        stats["settled"] += 1
        log(f"  settled {bet['side']} on {bet['question'][:60]}: P&L ${pnl:+.2f}")
    conn.commit()
    log(f"Settlement: {stats}")
    return stats
