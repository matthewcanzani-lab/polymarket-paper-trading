"""SQLite storage for market data (markets, snapshots, price history, outcomes).

Tables are keyed by (platform, market_id) so a second exchange can share them.

LOOK-AHEAD NOTE: `markets.volume_usd` is the volume *as of the last time we
pulled it*. For a resolved market that's its final lifetime volume — something
you could NOT have known while it was still trading. It's fine for choosing
which markets to study, but it must never be used as a model feature.
`market_snapshots` fixes this going forward: every pull records what each
active market looked like at that moment, so later we'll have true
point-in-time volume and prices.
"""
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from pmodel import config
from pmodel.sources.base import Market, PricePoint, Resolution

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    platform        TEXT NOT NULL,
    market_id       TEXT NOT NULL,
    question        TEXT NOT NULL,
    slug            TEXT,
    category        TEXT,
    event_title     TEXT,
    outcome_yes     TEXT NOT NULL,
    outcome_no      TEXT NOT NULL,
    is_yes_no       INTEGER NOT NULL,
    yes_token_id    TEXT,
    created_at      TEXT,
    start_date      TEXT,
    end_date        TEXT,
    closed_time     TEXT,
    active          INTEGER NOT NULL,
    closed          INTEGER NOT NULL,
    volume_usd      REAL NOT NULL,
    liquidity_usd   REAL NOT NULL,
    yes_price       REAL,
    best_bid        REAL,
    best_ask        REAL,
    tags            TEXT,             -- JSON list of the platform's tag labels
    first_seen_at   TEXT NOT NULL,
    last_updated_at TEXT NOT NULL,
    history_fetched_at TEXT,
    PRIMARY KEY (platform, market_id)
);

-- One row per active market per pull: what we could see at that moment.
CREATE TABLE IF NOT EXISTS market_snapshots (
    platform      TEXT NOT NULL,
    market_id     TEXT NOT NULL,
    snapshot_at   TEXT NOT NULL,
    yes_price     REAL,
    best_bid      REAL,
    best_ask      REAL,
    volume_usd    REAL,
    liquidity_usd REAL,
    PRIMARY KEY (platform, market_id, snapshot_at)
);

CREATE TABLE IF NOT EXISTS price_history (
    platform  TEXT NOT NULL,
    market_id TEXT NOT NULL,
    ts        INTEGER NOT NULL,   -- unix seconds UTC
    price     REAL NOT NULL,      -- YES price 0..1
    PRIMARY KEY (platform, market_id, ts)
);

CREATE TABLE IF NOT EXISTS outcomes (
    platform      TEXT NOT NULL,
    market_id     TEXT NOT NULL,
    resolved_yes  INTEGER,        -- 1 YES won, 0 NO won, NULL ambiguous/void
    status        TEXT NOT NULL,  -- 'resolved' | 'ambiguous'
    resolved_at   TEXT,
    recorded_at   TEXT NOT NULL,
    PRIMARY KEY (platform, market_id)
);

CREATE INDEX IF NOT EXISTS idx_markets_closed_volume ON markets (closed, volume_usd);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path: Path = None) -> sqlite3.Connection:
    path = Path(path or config.MARKET_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def upsert_market(conn, m: Market, now: str) -> None:
    row = asdict(m)
    row["is_yes_no"] = int(m.is_yes_no)
    row["active"] = int(m.active)
    row["closed"] = int(m.closed)
    row["tags"] = json.dumps(m.tags)
    row["now"] = now
    conn.execute(
        """
        INSERT INTO markets (platform, market_id, question, slug, category, event_title,
            outcome_yes, outcome_no, is_yes_no, yes_token_id, created_at, start_date,
            end_date, closed_time, active, closed, volume_usd, liquidity_usd, yes_price,
            best_bid, best_ask, tags, first_seen_at, last_updated_at)
        VALUES (:platform, :market_id, :question, :slug, :category, :event_title,
            :outcome_yes, :outcome_no, :is_yes_no, :yes_token_id, :created_at, :start_date,
            :end_date, :closed_time, :active, :closed, :volume_usd, :liquidity_usd, :yes_price,
            :best_bid, :best_ask, :tags, :now, :now)
        ON CONFLICT (platform, market_id) DO UPDATE SET
            question=excluded.question, slug=excluded.slug, category=excluded.category,
            event_title=excluded.event_title, outcome_yes=excluded.outcome_yes,
            outcome_no=excluded.outcome_no, is_yes_no=excluded.is_yes_no,
            yes_token_id=excluded.yes_token_id, created_at=excluded.created_at,
            start_date=excluded.start_date, end_date=excluded.end_date,
            closed_time=excluded.closed_time, active=excluded.active, closed=excluded.closed,
            volume_usd=excluded.volume_usd, liquidity_usd=excluded.liquidity_usd,
            yes_price=excluded.yes_price, best_bid=excluded.best_bid,
            best_ask=excluded.best_ask, tags=excluded.tags, last_updated_at=excluded.last_updated_at
        """,
        row,
    )


def insert_snapshot(conn, m: Market, now: str) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO market_snapshots VALUES (?,?,?,?,?,?,?,?)""",
        (m.platform, m.market_id, now, m.yes_price, m.best_bid, m.best_ask,
         m.volume_usd, m.liquidity_usd),
    )


def upsert_outcome(conn, r: Resolution, now: str) -> None:
    conn.execute(
        """
        INSERT INTO outcomes VALUES (?,?,?,?,?,?)
        ON CONFLICT (platform, market_id) DO UPDATE SET
            resolved_yes=excluded.resolved_yes, status=excluded.status,
            resolved_at=excluded.resolved_at, recorded_at=excluded.recorded_at
        """,
        (r.platform, r.market_id, r.resolved_yes, r.status, r.resolved_at, now),
    )


def save_price_history(conn, platform: str, market_id: str, points: list[PricePoint], now: str) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO price_history VALUES (?,?,?,?)",
        [(platform, market_id, p.ts, p.price) for p in points],
    )
    conn.execute(
        "UPDATE markets SET history_fetched_at=? WHERE platform=? AND market_id=?",
        (now, platform, market_id),
    )


def market_from_row(row) -> Market:
    fields = {k: row[k] for k in Market.__dataclass_fields__ if k != "tags"}
    for k in ("is_yes_no", "active", "closed"):
        fields[k] = bool(fields[k])
    return Market(**fields, tags=json.loads(row["tags"] or "[]"))
