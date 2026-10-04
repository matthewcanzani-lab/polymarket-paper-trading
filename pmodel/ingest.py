"""Pull data from a MarketSource into the market database."""
import itertools

from pmodel import config, db
from pmodel.sources.base import MarketSource


def ingest_markets(source: MarketSource, conn, closed: bool, end_date_min=None,
                   max_markets=None, log=print) -> dict:
    """Fetch active (closed=False) or resolved (closed=True) markets and store them.

    Active markets also get a snapshot row (price/volume as seen right now).
    Closed markets also get an outcome row.
    """
    now = db.utc_now()
    stats = {"seen": 0, "stored": 0, "skipped_non_binary": 0, "resolved": 0, "ambiguous": 0}
    raws = source.iter_markets(closed=closed, end_date_min=end_date_min)
    if max_markets:
        raws = itertools.islice(raws, max_markets)
    for raw in raws:
        stats["seen"] += 1
        market = source.parse_market(raw)
        if market is None:
            stats["skipped_non_binary"] += 1
            continue
        db.upsert_market(conn, market, now)
        stats["stored"] += 1
        if closed:
            res = source.parse_resolution(raw)
            if res:
                db.upsert_outcome(conn, res, now)
                stats[res.status] += 1
        else:
            db.insert_snapshot(conn, market, now)
        if stats["seen"] % 500 == 0:
            conn.commit()
            log(f"  ...{stats['seen']} markets processed")
    conn.commit()
    return stats


def ingest_price_history(source: MarketSource, conn, min_volume=None, limit=None,
                         refetch=False, log=print) -> dict:
    """Fetch YES price history for resolved markets at or above the volume cutoff.

    Each resolved market only needs fetching once (its history can't change).
    """
    min_volume = config.MIN_VOLUME_USD if min_volume is None else min_volume
    sql = """
        SELECT m.* FROM markets m JOIN outcomes o USING (platform, market_id)
        WHERE m.platform = ? AND o.status = 'resolved' AND m.volume_usd >= ?
          AND m.yes_token_id IS NOT NULL
    """
    if not refetch:
        sql += " AND m.history_fetched_at IS NULL"
    sql += " ORDER BY m.volume_usd DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql, (source.platform, min_volume)).fetchall()
    log(f"Fetching price history for {len(rows)} resolved markets (volume >= ${min_volume:,.0f})")

    stats = {"markets": 0, "points": 0, "empty": 0}
    for i, row in enumerate(rows, 1):
        market = db.market_from_row(row)
        points = source.price_history(market)
        db.save_price_history(conn, market.platform, market.market_id, points, db.utc_now())
        stats["markets"] += 1
        stats["points"] += len(points)
        stats["empty"] += not points
        if i % 50 == 0:
            conn.commit()
            log(f"  ...{i}/{len(rows)}")
    conn.commit()
    return stats
