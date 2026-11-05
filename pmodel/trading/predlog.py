"""The prediction log (data/predlog/): EVERY market a model scores, not only the
ones it bets on, so the forward record keeps growing even when no bet clears
the rules.

  * predictions-YYYY-MM.csv: the first prediction each model version makes for
    each market (later re-scores of the same market by the same version are not
    logged, so every market counts once and can't be picked after the fact).
  * outcomes-YYYY-MM.csv: how each predicted market resolved.

Same protections as the bet ledger: rows are only ever appended, and every row
carries the hash of the row before it plus a hash of its own contents, so any
edit, deletion or reordering shows up in `verify`. Plain text (not SQLite)
because thousands of rows a day are added and git stores appended text
efficiently, while it re-stores a binary file in full on every commit.
"""
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pmodel import config
from pmodel.trading.ledger import GENESIS, _hash

DIR = config.DATA_DIR / "predlog"

PRED_FIELDS = ["logged_at", "run_id", "strategy", "model_version", "platform", "market_id",
               "question", "category", "end_date", "model_prob_yes", "market_mid",
               "raw_model_prob_yes"]
OUTCOME_FIELDS = ["resolved_at", "platform", "market_id", "yes_payout"]
KINDS = {"predictions": PRED_FIELDS, "outcomes": OUTCOME_FIELDS}


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(x):
    return "" if x is None else f"{float(x):.6f}"


def _files(kind, directory):
    return sorted(Path(directory).glob(f"{kind}-*.csv"))


def read(kind, directory=None) -> list[dict]:
    rows = []
    for path in _files(kind, directory or DIR):
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _append(kind, rows: list[dict], directory, last_hash) -> str:
    """Append rows (all values already strings) to this month's file, chaining hashes."""
    fields = KINDS[kind]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = rows[0][fields[0]][:7] if rows else _now()[:7]
    path = directory / f"{kind}-{stamp}.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["prev_hash", "row_hash"])
        if new:
            w.writeheader()
        for r in rows:
            body = {k: r[k] for k in fields}
            h = _hash(last_hash, body)
            w.writerow({**body, "prev_hash": last_hash, "row_hash": h})
            last_hash = h
    return last_hash


def _last_hash(existing):
    return existing[-1]["row_hash"] if existing else GENESIS


def log_predictions(scored: list[dict], strategy: str, version: str, run_id: str,
                    directory=None, now=None) -> int:
    """Log the first prediction `version` makes for each market in `scored`
    (dashboard snapshot rows: market_id, question, category, end_date,
    model_prob_yes, last_price = market midpoint, optional raw_model_prob_yes)."""
    directory = directory or DIR
    existing = read("predictions", directory)
    seen = {(r["model_version"], r["platform"], r["market_id"]) for r in existing}
    now = now or _now()
    rows = []
    for s in scored:
        key = (version, "polymarket", str(s["market_id"]))
        if key in seen or s.get("model_prob_yes") is None or s.get("last_price") is None:
            continue
        seen.add(key)
        rows.append({"logged_at": now, "run_id": run_id, "strategy": strategy,
                     "model_version": version, "platform": "polymarket",
                     "market_id": str(s["market_id"]), "question": s.get("question") or "",
                     "category": s.get("category") or "", "end_date": s.get("end_date") or "",
                     "model_prob_yes": _num(s["model_prob_yes"]),
                     "market_mid": _num(s["last_price"]),
                     "raw_model_prob_yes": _num(s.get("raw_model_prob_yes"))})
    if rows:
        _append("predictions", rows, directory, _last_hash(existing))
    return len(rows)


def resolve(source, directory=None, now=None, log=print) -> dict:
    """Record outcomes for predicted markets that have resolved (append-only)."""
    directory = directory or DIR
    now = now or _now()
    outcomes = read("outcomes", directory)
    done = {(r["platform"], r["market_id"]) for r in outcomes}
    due = sorted({r["market_id"] for r in read("predictions", directory)
                  if ("polymarket", r["market_id"]) not in done and r["end_date"]
                  and r["end_date"] <= now})
    raws = {}
    for i in range(0, len(due), 100):
        for raw in source.markets_by_id(due[i:i + 100]):
            raws[str(raw["id"])] = raw
    rows = []
    for mid in due:
        raw = raws.get(mid)
        yes = source.yes_payout(raw) if raw else None
        if yes is not None:
            rows.append({"resolved_at": now, "platform": "polymarket", "market_id": mid,
                         "yes_payout": _num(yes)})
    if rows:
        _append("outcomes", rows, directory, _last_hash(outcomes))
    stats = {"due": len(due), "resolved": len(rows)}
    log(f"Prediction log outcomes: {stats}")
    return stats


def verify(directory=None) -> list[str]:
    problems = []
    for kind, fields in KINDS.items():
        prev = GENESIS
        for i, r in enumerate(read(kind, directory or DIR), 1):
            if r["prev_hash"] != prev:
                problems.append(f"{kind} row {i}: chain broken (a row was removed or reordered)")
            if _hash(r["prev_hash"], {k: r[k] for k in fields}) != r["row_hash"]:
                problems.append(f"{kind} row {i}: contents changed after it was written")
            prev = r["row_hash"]
    return problems


def record(directory=None) -> list[dict]:
    """Forward accuracy per (strategy, model version) over resolved predictions."""
    directory = directory or DIR
    outcome = {r["market_id"]: float(r["yes_payout"]) for r in read("outcomes", directory)}
    groups = {}
    for r in read("predictions", directory):
        groups.setdefault((r["strategy"], r["model_version"]), []).append(r)
    out = []
    for (strategy, version), rows in groups.items():
        done = [r for r in rows if r["market_id"] in outcome]
        y = np.array([outcome[r["market_id"]] for r in done])
        brier = lambda col: (float(np.mean((np.array([float(r[col]) for r in done]) - y) ** 2))
                             if done and all(r[col] for r in done) else None)
        out.append({"strategy": strategy, "version": version, "logged": len(rows),
                    "resolved": len(done), "first": rows[0]["logged_at"],
                    "brier_model": brier("model_prob_yes"), "brier_market": brier("market_mid"),
                    "brier_raw": brier("raw_model_prob_yes")})
    out.sort(key=lambda g: g["first"])
    return out
