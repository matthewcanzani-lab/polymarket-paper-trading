"""Turn resolved markets + price history into a training table.

Each row is one "snapshot": a market seen at time t, the features available at
t (features.py), and what eventually happened (the label).

Snapshot times per market (all must fall while the market was still open):
  * 6h, 1d, 3d, 7d, 14d, 30d before the SCHEDULED end, and
  * 6h, 1d, 3d, 7d after the market was created.
The close time is used here only to drop snapshots after the market had
closed — the live scorer likewise only ever sees open markets. It is never a
feature. Each market's snapshots share a total weight of 1, so long markets
with many snapshots don't count more than short ones.

SURVIVORSHIP GUARD: we can only label markets that have resolved. Near the end
of the data, a long-dated market is only present if it ended EARLY (often
because the event happened), which would skew recent data toward YES. So a
market is included only if its SCHEDULED end (known in advance) is at least
RESOLUTION_LAG_DAYS before the date the outcomes were pulled.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from pmodel import universe
from pmodel.model.features import FEATURE_NAMES, features_at, iso_to_unix, static_info

H = 3600
RESOLUTION_LAG_DAYS = 7
BEFORE_END_H = (6, 24, 72, 168, 336, 720)
AFTER_OPEN_H = (6, 24, 72, 168)


def snapshot_times(created, end, closed):
    times = set()
    for h in BEFORE_END_H:
        if end:
            times.add(end - h * H)
    for h in AFTER_OPEN_H:
        if created:
            times.add(created + h * H)
    last_open = (closed or end) - H  # at least an hour before trading stopped
    first = (created or 0) + H
    return sorted(t for t in times if first <= t <= last_open)


def build(conn, source, platform="polymarket", log=print) -> pd.DataFrame:
    where, params = universe.sql_filter("m")
    pulled = conn.execute("SELECT MAX(recorded_at) FROM outcomes WHERE platform=?",
                          (platform,)).fetchone()[0]
    cutoff = datetime.strptime(pulled, "%Y-%m-%dT%H:%M:%SZ") - timedelta(days=RESOLUTION_LAG_DAYS)
    cutoff = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    log(f"Using markets scheduled to end by {cutoff} (outcomes pulled {pulled})")
    markets = conn.execute(
        f"""SELECT m.*, o.resolved_yes, o.resolved_at FROM markets m
            JOIN outcomes o USING (platform, market_id)
            WHERE m.platform = ? AND o.status = 'resolved' AND m.history_fetched_at IS NOT NULL
              AND m.end_date <= ? AND {where}""", [platform, cutoff, *params]).fetchall()
    log(f"Building snapshots for {len(markets):,} markets...")
    rows = []
    from pmodel.db import market_from_row
    for i, row in enumerate(markets, 1):
        hist = conn.execute(
            "SELECT ts, price FROM price_history WHERE platform=? AND market_id=? ORDER BY ts",
            (platform, row["market_id"])).fetchall()
        if not hist:
            continue
        ts = np.fromiter((h[0] for h in hist), dtype=np.int64, count=len(hist))
        ps = np.fromiter((h[1] for h in hist), dtype=np.float64, count=len(hist))
        step = source.history_step_seconds(market_from_row(row))
        info = static_info(row)
        created, end = iso_to_unix(row["created_at"]), iso_to_unix(row["end_date"])
        closed = iso_to_unix(row["closed_time"]) or iso_to_unix(row["resolved_at"])
        snaps = []
        for t in snapshot_times(created, end, closed):
            f = features_at(info, ts, ps, t, step)
            if f is not None:
                snaps.append((t, f))
        for t, f in snaps:
            f.update(market_id=row["market_id"], t=t, label=int(row["resolved_yes"]),
                     category=row["category"] or "unknown", weight=1.0 / len(snaps),
                     closed_ts=closed)
            rows.append(f)
        if i % 10_000 == 0:
            log(f"  ...{i:,} markets, {len(rows):,} snapshots")
    df = pd.DataFrame(rows)
    log(f"{len(df):,} snapshots from {df['market_id'].nunique() if len(df) else 0:,} markets")
    return df


__all__ = ["build", "snapshot_times", "FEATURE_NAMES"]
