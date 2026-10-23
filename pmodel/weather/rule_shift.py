"""Measure how the 23 Oct 2026 rule change (hourly routine METAR reports instead
of NOAA "all times") shifts each station's resolved temperature.

shift = new-rule value - old-rule resolved value, over recent days that resolved
under the old rule. Added to the forecast bias for markets using the new rule.
"""
import numpy as np
import pandas as pd

from pmodel.sources import metar

SHRINK_N = 10


def measure(conn, start, end, log=print) -> dict:
    win = pd.read_sql("""SELECT w.station, w.kind, w.unit, w.lo, w.hi, w.day, f.tz
        FROM weather_markets w JOIN (SELECT DISTINCT station, tz FROM weather_forecasts) f
          USING (station)
        WHERE w.yes_payout = 1 AND w.lo > -1e8 AND w.hi < 1e8 AND w.day BETWEEN ? AND ?""",
                      conn, params=(start, end))
    temps, rows = {}, []
    for r in win.itertuples():
        key = (r.station, r.day)
        if key not in temps:
            try:
                temps[key] = metar.day_temps(r.station, r.day, r.tz)
            except Exception as exc:  # one bad station-day shouldn't stop the run
                log(f"  {r.station} {r.day}: {exc}")
                temps[key] = []
        t = temps[key]
        if len(t) < 18:  # too few reports to trust the day's high/low
            continue
        v = max(t) if r.kind == "high" else min(t)
        old = (r.lo + r.hi) / 2
        old_c = (old - 32) * 5 / 9 if r.unit == "F" else old
        rows.append({"station": r.station, "kind": r.kind, "shift_c": v - old_c})
    d = pd.DataFrame(rows)
    out = {"global": {}, "by_key": {}, "days": [start, end], "n": len(d)}
    for kind, g in d.groupby("kind"):
        out["global"][kind] = float(g.shift_c.mean())
    for (st, kind), g in d.groupby(["station", "kind"]):
        w = len(g) / (len(g) + SHRINK_N)
        out["by_key"][f"{st}|{kind}"] = float(w * g.shift_c.mean() + (1 - w) * out["global"][kind])
    return out
