"""The paper-trading ledger (data/predictions.db) — append-only and tamper-evident.

Why you can trust it:
  1. APPEND-ONLY. SQLite triggers reject every UPDATE and DELETE on the ledger
     tables. Settling a bet doesn't edit the bet; it adds a separate row in
     `settlements`.
  2. HASH CHAIN. Every row stores the hash of the row before it, plus its own
     hash over all its fields. Changing, deleting or reordering any past row
     (even by dropping the triggers) breaks the chain, and `verify` reports it.
  3. TIMESTAMPED OUTSIDE THIS COMPUTER. The file is committed to git and
     pushed, so GitHub holds dated copies showing each bet existed before its
     market resolved.

FAKE money only. Nothing in this module (or project) can place a real order.
"""
import hashlib
import json
import sqlite3
from pathlib import Path

from pmodel import config

GENESIS = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    seq INTEGER PRIMARY KEY,
    run_id TEXT UNIQUE NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    model_version TEXT,
    markets_scored INTEGER NOT NULL,
    signals INTEGER NOT NULL,
    bets_placed INTEGER NOT NULL,
    settings_json TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    row_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bets (
    seq INTEGER PRIMARY KEY,
    bet_id TEXT UNIQUE NOT NULL,
    run_id TEXT NOT NULL,
    placed_at TEXT NOT NULL,
    platform TEXT NOT NULL,
    market_id TEXT NOT NULL,
    question TEXT NOT NULL,
    category TEXT,
    end_date TEXT,
    side TEXT NOT NULL CHECK (side IN ('YES', 'NO')),
    model_prob_yes REAL NOT NULL,
    market_mid REAL NOT NULL,
    best_bid REAL,
    best_ask REAL,
    fill_price REAL NOT NULL,      -- average price paid per share, incl. slippage
    slippage_per_share REAL NOT NULL,
    shares REAL NOT NULL,
    stake REAL NOT NULL,
    kelly_full REAL NOT NULL,
    bankroll_at_bet REAL NOT NULL,
    model_version TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    row_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settlements (
    seq INTEGER PRIMARY KEY,
    bet_id TEXT UNIQUE NOT NULL REFERENCES bets(bet_id),
    settled_at TEXT NOT NULL,
    yes_payout REAL NOT NULL,      -- what one YES share paid: 1, 0, or 0.5 for a split
    payout REAL NOT NULL,
    pnl REAL NOT NULL,
    prev_hash TEXT NOT NULL,
    row_hash TEXT NOT NULL
);
"""

LEDGER_TABLES = ("runs", "bets", "settlements")


def _guard_triggers():
    sql = []
    for t in LEDGER_TABLES:
        for op in ("UPDATE", "DELETE"):
            sql.append(f"""CREATE TRIGGER IF NOT EXISTS {t}_no_{op.lower()}
                BEFORE {op} ON {t}
                BEGIN SELECT RAISE(ABORT, '{t} is append-only: {op} is not allowed'); END;""")
    return "\n".join(sql)


def connect(path: Path = None) -> sqlite3.Connection:
    path = Path(path or config.PREDICTIONS_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.executescript(_guard_triggers())
    return conn


def _hash(prev_hash: str, fields: dict) -> str:
    payload = prev_hash + json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _last_hash(conn, table) -> str:
    row = conn.execute(f"SELECT row_hash FROM {table} ORDER BY seq DESC LIMIT 1").fetchone()
    return row[0] if row else GENESIS


def _append(conn, table: str, fields: dict) -> str:
    prev = _last_hash(conn, table)
    row_hash = _hash(prev, fields)
    cols = list(fields) + ["prev_hash", "row_hash"]
    conn.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                 [fields[c] for c in fields] + [prev, row_hash])
    return row_hash


def append_run(conn, fields: dict) -> str:
    return _append(conn, "runs", fields)


def append_bet(conn, fields: dict) -> str:
    return _append(conn, "bets", fields)


def append_settlement(conn, fields: dict) -> str:
    return _append(conn, "settlements", fields)


def verify(conn) -> list[str]:
    """Recompute every hash. Returns a list of problems (empty = intact)."""
    problems = []
    for table in LEDGER_TABLES:
        prev = GENESIS
        rows = conn.execute(f"SELECT * FROM {table} ORDER BY seq").fetchall()
        for r in rows:
            fields = {k: r[k] for k in r.keys() if k not in ("seq", "prev_hash", "row_hash")}
            if r["prev_hash"] != prev:
                problems.append(f"{table} row {r['seq']}: chain broken (a row was removed or reordered)")
            if _hash(r["prev_hash"], fields) != r["row_hash"]:
                problems.append(f"{table} row {r['seq']}: contents changed after it was written")
            prev = r["row_hash"]
    triggers = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    for t in LEDGER_TABLES:
        for op in ("update", "delete"):
            if f"{t}_no_{op}" not in triggers:
                problems.append(f"protection trigger {t}_no_{op} is missing")
    return problems


def open_bets(conn):
    return conn.execute("""SELECT b.* FROM bets b LEFT JOIN settlements s USING (bet_id)
                           WHERE s.bet_id IS NULL ORDER BY b.seq""").fetchall()


def bankroll(conn) -> dict:
    """Fake bankroll: starting money + profit/loss on settled bets. Cash excludes open stakes."""
    realized = conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM settlements").fetchone()[0]
    open_stake = conn.execute(
        """SELECT COALESCE(SUM(b.stake), 0) FROM bets b
           LEFT JOIN settlements s USING (bet_id) WHERE s.bet_id IS NULL""").fetchone()[0]
    equity = config.BANKROLL_START_USD + realized
    # "available" = what new bets may use: cash, but never past the exposure cap.
    available = max(min(equity - open_stake,
                        config.MAX_OPEN_EXPOSURE_FRACTION * equity - open_stake), 0.0)
    return {"bankroll": equity, "open_stake": open_stake, "cash": equity - open_stake,
            "available": available, "realized_pnl": realized}


def has_bet_on(conn, platform: str, market_id: str) -> bool:
    return conn.execute("SELECT 1 FROM bets WHERE platform=? AND market_id=?",
                        (platform, market_id)).fetchone() is not None
