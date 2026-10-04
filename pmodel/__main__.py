"""Command-line entry point. Run `python -m pmodel --help` to see commands."""
import argparse

from pmodel import config, db
from pmodel.sources.polymarket import PolymarketSource


def cmd_ingest_markets(args):
    source, conn = PolymarketSource(), db.connect()
    if args.which in ("active", "all"):
        print("Pulling ACTIVE markets...")
        print("  ", _fmt(_ingest(source, conn, False, None, args.max)))
    if args.which in ("resolved", "all"):
        print(f"Pulling RESOLVED markets that ended on/after {config.HISTORY_START_DATE}...")
        print("  ", _fmt(_ingest(source, conn, True, config.HISTORY_START_DATE, args.max)))


def _ingest(source, conn, closed, end_min, max_markets):
    from pmodel.ingest import ingest_markets
    return ingest_markets(source, conn, closed=closed, end_date_min=end_min, max_markets=max_markets)


def _fmt(stats):
    return ", ".join(f"{k}={v:,}" for k, v in stats.items())


def cmd_ingest_history(args):
    from pmodel.ingest import ingest_price_history
    stats = ingest_price_history(PolymarketSource(), db.connect(), min_volume=args.min_volume,
                                 limit=args.limit, refetch=args.refetch)
    print("  ", _fmt(stats))


def cmd_volume_report(args):
    from pmodel.analysis.volume import build_report, save_chart
    conn = db.connect()
    report = build_report(conn)
    print(report)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "volume_report.txt").write_text(report + "\n")
    chart = save_chart(conn, config.REPORTS_DIR / "volume_distribution.png")
    print(f"\nSaved {config.REPORTS_DIR / 'volume_report.txt'} and {chart}")


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
    s.set_defaults(func=cmd_ingest_markets)

    s = sub.add_parser("ingest-history", help="pull price history for resolved markets above the cutoff")
    s.add_argument("--min-volume", type=float, help="override MIN_VOLUME_USD for this run")
    s.add_argument("--limit", type=int, help="only fetch N markets (largest first)")
    s.add_argument("--refetch", action="store_true", help="re-download even if already fetched")
    s.set_defaults(func=cmd_ingest_history)

    sub.add_parser("volume-report", help="show volume distribution to pick a cutoff").set_defaults(func=cmd_volume_report)
    sub.add_parser("status", help="row counts in the market database").set_defaults(func=cmd_status)
    sub.add_parser("backup", help="copy the prediction log to BACKUP_DIR").set_defaults(func=cmd_backup)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
