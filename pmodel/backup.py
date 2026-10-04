"""Back up the prediction log (data/predictions.db).

Uses SQLite's own backup API, which makes a consistent copy even if another
process is writing at the same moment (a plain file copy might not).
Copies go to BACKUP_DIR (e.g. iCloud Drive). The git commit of
data/predictions.db is the second, independent backup.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pmodel import config

KEEP_LAST = 60  # dated copies to keep


def backup_predictions(src: Path = None, dest_dir: Path = None) -> Path | None:
    src = Path(src or config.PREDICTIONS_DB_PATH)
    dest_dir = dest_dir or config.BACKUP_DIR
    if dest_dir is None:
        raise SystemExit("BACKUP_DIR is not set. Add it to .env (see .env.example).")
    if not src.exists():
        print(f"No prediction log at {src} yet — nothing to back up.")
        return None
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    dest = dest_dir / f"predictions_{stamp}.db"
    source, target = sqlite3.connect(src), sqlite3.connect(dest)
    try:
        source.backup(target)
    finally:
        source.close()
        target.close()
    copies = sorted(dest_dir.glob("predictions_*.db"))
    for old in copies[:-KEEP_LAST]:
        old.unlink()
    print(f"Backed up {src.name} -> {dest}")
    return dest
