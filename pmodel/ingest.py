"""Pull data from a MarketSource into the market database."""
import itertools
from concurrent.futures import ThreadPoolExecutor
import time

from pmodel import db, universe
from pmodel.sources.base import MarketSource


def ingest_markets(source: MarketSource, conn, closed: bool, end_date_min=None,
                   volume_min=None, max_markets=None, log=print) -> dict:
    """Fetch active (closed=False) or resolved (closed=True) markets and store them.

    Active markets also get a snapshot row (price/volume as seen right now).
    Closed markets also get an outcome row.
    """
    now = db.utc_now()
    started = time.monotonic()
    stats = {"seen": 0, "stored": 0, "skipped_non_binary": 0, "resolved": 0, "ambiguous": 0}
    raws = source.iter_markets(closed=closed, end_date_min=end_date_min, volume_min=volume_min)
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
            rate = stats["seen"] / max(time.monotonic() - started, 1e-9)
            log(f"  ...{stats['seen']:,} markets processed ({rate:.0f}/s)")
    conn.commit()
    return stats


def ingest_price_history(source: MarketSource, conn, limit=None, refetch=False,
                         sample=False, workers=6, log=print) -> dict:
    """Fetch YES price history for resolved markets in the universe (see universe.py).

    Each resolved market only needs fetching once (its history can't change).
    Default order is newest first. With sample=True the order is a fixed
    pseudo-random shuffle (the same every run), so `limit` takes an unbiased,
    reproducible sample across categories and dates — and a later run with a
    bigger limit simply extends it.
    Downloads run on `workers` threads; all database writes stay on this thread.
    """
    where, params = universe.sql_filter("m")
    sql = f"""
        SELECT m.* FROM markets m JOIN outcomes o USING (platform, market_id)
        WHERE m.platform = ? AND o.status = 'resolved' AND m.yes_token_id IS NOT NULL
          AND {where}
    """
    if not refetch:
        sql += " AND m.history_fetched_at IS NULL"
    if sample:
        sql += " ORDER BY (CAST(m.market_id AS INTEGER) * 2654435761) % 1000003, m.market_id"
    else:
        sql += " ORDER BY m.end_date DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql, [source.platform, *params]).fetchall()
    log(f"Fetching price history for {len(rows):,} resolved markets in the universe")

    stats = {"markets": 0, "points": 0, "empty": 0}
    markets = [db.market_from_row(row) for row in rows]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (market, points) in enumerate(
                zip(markets, pool.map(source.price_history, markets)), 1):
            db.save_price_history(conn, market.platform, market.market_id, points, db.utc_now())
            stats["markets"] += 1
            stats["points"] += len(points)
            stats["empty"] += not points
            if i % 500 == 0:
                conn.commit()
                rate = i / (time.monotonic() - started)
                log(f"  ...{i:,}/{len(rows):,} ({rate:.1f} markets/s)")
    conn.commit()
    return stats


def backfill_condition_ids(source: MarketSource, conn, log=print) -> int:
    """Fill in condition IDs for markets that have price history but no condition ID."""
    ids = [r[0] for r in conn.execute(
        "SELECT market_id FROM markets WHERE platform=? AND condition_id IS NULL "
        "AND history_fetched_at IS NOT NULL", (source.platform,))]
    log(f"Looking up condition IDs for {len(ids):,} markets")
    done = 0
    for i in range(0, len(ids), 100):
        found = source.condition_ids(ids[i:i + 100])
        conn.executemany("UPDATE markets SET condition_id=? WHERE platform=? AND market_id=?",
                         [(cid, source.platform, mid) for mid, cid in found.items() if cid])
        done += len(found)
        if (i // 100) % 50 == 0:
            conn.commit()
            log(f"  ...{i + 100:,}")
    conn.commit()
    return done


def ingest_trades(source: MarketSource, conn, limit=None, workers=8, log=print) -> dict:
    """Hourly traded-dollar history for markets that already have price history."""
    sql = """SELECT * FROM markets WHERE platform=? AND history_fetched_at IS NOT NULL
             AND condition_id IS NOT NULL AND trades_fetched_at IS NULL
             ORDER BY (CAST(market_id AS INTEGER) * 2654435761) % 1000003, market_id"""
    if limit:
        sql += f" LIMIT {int(limit)}"
    markets = [db.market_from_row(r) for r in conn.execute(sql, (source.platform,))]
    log(f"Fetching trade history for {len(markets):,} markets")
    stats = {"markets": 0, "capped": 0, "no_trades": 0}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (m, hist) in enumerate(zip(markets, pool.map(source.trade_history, markets)), 1):
            db.save_trade_history(conn, m.platform, m.market_id, hist, db.utc_now())
            stats["markets"] += 1
            stats["capped"] += not hist.complete
            stats["no_trades"] += not hist.buckets
            if i % 500 == 0:
                conn.commit()
                log(f"  ...{i:,}/{len(markets):,} ({i / (time.monotonic() - started):.1f}/s)")
    conn.commit()
    return stats
