"""Command-line entry point. Run `python -m pmodel --help` to see commands."""
import argparse

from pmodel import config, db
from pmodel.sources.polymarket import PolymarketSource


def cmd_ingest_markets(args):
    source, conn = PolymarketSource(), db.connect()
    if args.which in ("active", "all"):
        print("Pulling ACTIVE markets...")
        print("  ", _fmt(_ingest(source, conn, False, None, args.min_volume, args.max)))
    if args.which in ("resolved", "all"):
        since = args.since or config.HISTORY_START_DATE
        print(f"Pulling RESOLVED markets that ended on/after {since}...")
        print("  ", _fmt(_ingest(source, conn, True, since, args.min_volume, args.max)))


def _ingest(source, conn, closed, end_min, volume_min, max_markets):
    from pmodel.ingest import ingest_markets
    if volume_min:
        print(f"  (only markets with volume >= ${volume_min:,.0f})")
    return ingest_markets(source, conn, closed=closed, end_date_min=end_min,
                          volume_min=volume_min, max_markets=max_markets)


def _fmt(stats):
    return ", ".join(f"{k}={v:,}" for k, v in stats.items())


def cmd_ingest_history(args):
    from pmodel.ingest import ingest_price_history
    stats = ingest_price_history(PolymarketSource(), db.connect(), limit=args.limit,
                                 refetch=args.refetch, sample=args.sample, workers=args.workers)
    print("  ", _fmt(stats))


def cmd_volume_report(args):
    from pmodel.analysis.volume import build_report, save_chart
    conn = db.connect()
    report = build_report(conn)
    print(report)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "volume_report.txt").write_text(report + "\n")
    chart = save_chart(conn, config.REPORTS_DIR / "volume_distribution.png", note=args.note)
    print(f"\nSaved {config.REPORTS_DIR / 'volume_report.txt'} and {chart}")


def cmd_ingest_trades(args):
    from pmodel.ingest import backfill_condition_ids, ingest_trades
    source, conn = PolymarketSource(), db.connect()
    backfill_condition_ids(source, conn)
    print("  ", _fmt(ingest_trades(source, conn, limit=args.limit, workers=args.workers)))


def cmd_train(args):
    from pmodel.model import dataset, train
    conn = db.connect()
    df = dataset.build(conn, PolymarketSource())
    train.run(df)


def _ledger(args):
    from pmodel.trading import ledger
    return ledger.connect(args.ledger)


def cmd_score(args):
    from pmodel.trading import scorer
    scorer.run(PolymarketSource(), _ledger(args), workers=args.workers)


def cmd_settle(args):
    from pmodel.trading import settle
    settle.run(PolymarketSource(), _ledger(args))


def cmd_ledger(args):
    from pmodel.trading import ledger
    conn = _ledger(args)
    problems = ledger.verify(conn)
    money = ledger.bankroll(conn)
    q = lambda sql: conn.execute(sql).fetchone()[0]
    print(f"Ledger: {args.ledger or config.PREDICTIONS_DB_PATH}")
    print(f"  integrity:   {'OK - no edits, deletions or reordering detected' if not problems else 'PROBLEMS:'}")
    for p in problems:
        print(f"    ! {p}")
    print(f"  runs:        {q('SELECT COUNT(*) FROM runs'):,}")
    print(f"  bets:        {q('SELECT COUNT(*) FROM bets'):,} "
          f"({len(ledger.open_bets(conn)):,} open, {q('SELECT COUNT(*) FROM settlements'):,} settled)")
    print(f"  bankroll:    ${money['bankroll']:,.2f} (realized P&L ${money['realized_pnl']:+,.2f}), "
          f"cash ${money['cash']:,.2f}, at risk ${money['open_stake']:,.2f}")
    for r in conn.execute("SELECT * FROM bets ORDER BY seq DESC LIMIT 10"):
        print(f"  {r['placed_at']} {r['side']:3} ${r['stake']:7.2f} @ {r['fill_price']:.3f} "
              f"model {r['model_prob_yes']:.0%} mkt {r['market_mid']:.0%}  {r['question'][:50]}")


def cmd_status(args):
    conn = db.connect()
    q = lambda sql: conn.execute(sql).fetchone()[0]
    print(f"Market DB: {config.MARKET_DB_PATH}")
    print(f"  markets:          {q('SELECT COUNT(*) FROM markets'):,}")
    print(f"  active:           {q('SELECT COUNT(*) FROM markets WHERE closed=0 AND active=1'):,}")
    resolved = q("SELECT COUNT(*) FROM outcomes WHERE status='resolved'")
    print(f"  resolved (clean): {resolved:,}")
    print(f"  w/ price history: {q('SELECT COUNT(*) FROM markets WHERE history_fetched_at IS NOT NULL'):,}")
    print(f"  price points:     {q('SELECT COUNT(*) FROM price_history'):,}")
    print(f"  snapshots:        {q('SELECT COUNT(*) FROM market_snapshots'):,}")
    print(f"MIN_VOLUME_USD = {config.MIN_VOLUME_USD:,.0f}")


def cmd_backup(args):
    from pmodel.backup import backup_predictions
    backup_predictions()


def main():
    p = argparse.ArgumentParser(prog="pmodel", description="Polymarket paper-trading model (paper only).")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest-markets", help="pull market listings + outcomes")
    s.add_argument("which", choices=["active", "resolved", "all"], nargs="?", default="all")
    s.add_argument("--max", type=int, help="stop after N markets (for a quick test)")
    s.add_argument("--since", help="resolved markets: only those ending on/after this date "
                                   "(default HISTORY_START_DATE); use a recent date for daily top-ups")
    s.add_argument("--min-volume", type=float, help="skip markets below this lifetime volume (server-side)")
    s.set_defaults(func=cmd_ingest_markets)

    s = sub.add_parser("ingest-history", help="pull price history for resolved markets in the universe")
    s.add_argument("--limit", type=int, help="only fetch N markets")
    s.add_argument("--sample", action="store_true",
                   help="fixed pseudo-random order instead of newest-first (use with --limit)")
    s.add_argument("--workers", type=int, default=6, help="parallel downloads (default 6)")
    s.add_argument("--refetch", action="store_true", help="re-download even if already fetched")
    s.set_defaults(func=cmd_ingest_history)

    s = sub.add_parser("volume-report", help="show volume distribution to pick a cutoff")
    s.add_argument("--note", help="caveat text to print on the chart")
    s.set_defaults(func=cmd_volume_report)
    s = sub.add_parser("ingest-trades", help="hourly traded-$ history for markets with price history")
    s.add_argument("--limit", type=int)
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(func=cmd_ingest_trades)

    sub.add_parser("train", help="train + backtest a new model version (saved under models/)").set_defaults(func=cmd_train)
    for name, func, helptext in (
            ("score", cmd_score, "score open markets and log PAPER bets"),
            ("settle", cmd_settle, "settle paper bets whose markets resolved"),
            ("ledger", cmd_ledger, "show the paper ledger and check it hasn't been tampered with")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--ledger", help="ledger file (default data/predictions.db)")
        if name == "score":
            s.add_argument("--workers", type=int, default=8)
        s.set_defaults(func=func)
    sub.add_parser("status", help="row counts in the market database").set_defaults(func=cmd_status)
    sub.add_parser("backup", help="copy the prediction log to BACKUP_DIR").set_defaults(func=cmd_backup)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
