"""Small inline-SVG charts. Colours come from CSS variables in the page, so the
charts follow light/dark mode. Each data mark has a <title> for hover tooltips."""
from datetime import datetime
from html import escape

W, H = 640, 300
PAD_L, PAD_R, PAD_T, PAD_B = 56, 16, 16, 40


def _svg(body, label):
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(label)}" '
            f'class="chart">{body}</svg>')


def _ts(iso):
    return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").timestamp()


def pnl_chart(points, start_bankroll):
    """Cumulative fake P&L after each settlement (a step line with a zero baseline)."""
    if not points:
        return '<p class="empty">No settled bets yet — the curve starts at the first settlement.</p>'
    xs = [_ts(t) for t, _ in points]
    ys = [v for _, v in points]
    x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
    lo, hi = min(0.0, *ys), max(0.0, *ys)
    if hi - lo < 1:
        hi, lo = hi + 1, lo - 1
    sx = lambda x: PAD_L + (x - x0) / (x1 - x0) * (W - PAD_L - PAD_R)
    sy = lambda y: PAD_T + (hi - y) / (hi - lo) * (H - PAD_T - PAD_B)
    path, prev_y = [f"M{sx(x0):.1f},{sy(0):.1f}"], 0.0
    for x, y in zip(xs, ys):
        path.append(f"H{sx(x):.1f}V{sy(y):.1f}")
        prev_y = y
    body = [f'<line class="axis" x1="{PAD_L}" x2="{W - PAD_R}" y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>',
            f'<text class="tick" x="{PAD_L - 6}" y="{sy(0) + 4:.1f}" text-anchor="end">$0</text>',
            f'<text class="tick" x="{PAD_L - 6}" y="{sy(hi) + 4:.1f}" text-anchor="end">${hi:+,.0f}</text>',
            f'<text class="tick" x="{PAD_L - 6}" y="{sy(lo) + 4:.1f}" text-anchor="end">${lo:+,.0f}</text>',
            f'<path class="line" d="{"".join(path)}"/>']
    for (t, v), x in zip(points, xs):
        body.append(f'<circle class="dot" cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="4">'
                    f'<title>{t[:10]}: cumulative P&amp;L ${v:+,.2f} '
                    f'(bankroll ${start_bankroll + v:,.2f})</title></circle>')
    body += [f'<text class="tick" x="{PAD_L}" y="{H - 12}">{points[0][0][:10]}</text>',
             f'<text class="tick" x="{W - PAD_R}" y="{H - 12}" text-anchor="end">{points[-1][0][:10]}</text>']
    return _svg("".join(body), "Cumulative paper profit and loss")


def calibration_chart(rows, forecast_key="forecast", actual_key="actual", n_key="n", compact=False):
    """Forecast vs how often it happened, per 10-point band. Diagonal = perfect.
    compact=True draws a square chart with the axis titles below/left (no wide margin)."""
    if not rows:
        return '<p class="empty">Not enough settled bets for a calibration chart yet.</p>'
    if compact:
        return _calibration_compact(rows, forecast_key, actual_key, n_key)
    side = min(W - PAD_L - PAD_R, H - PAD_T - PAD_B)
    sx = lambda v: PAD_L + v * side
    sy = lambda v: PAD_T + (1 - v) * side
    body = [f'<rect class="frame" x="{PAD_L}" y="{PAD_T}" width="{side}" height="{side}"/>',
            f'<line class="diag" x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}"/>']
    for v in (0, 0.5, 1):
        body.append(f'<text class="tick" x="{sx(v):.1f}" y="{PAD_T + side + 16}" '
                    f'text-anchor="middle">{v:.0%}</text>')
        body.append(f'<text class="tick" x="{PAD_L - 6}" y="{sy(v) + 4:.1f}" '
                    f'text-anchor="end">{v:.0%}</text>')
    pts = sorted(rows, key=lambda r: r[forecast_key])
    body.append('<polyline class="line" points="' + " ".join(
        f"{sx(r[forecast_key]):.1f},{sy(r[actual_key]):.1f}" for r in pts) + '"/>')
    for r in pts:
        body.append(f'<circle class="dot" cx="{sx(r[forecast_key]):.1f}" '
                    f'cy="{sy(r[actual_key]):.1f}" r="5"><title>Said {r[forecast_key]:.0%}, '
                    f'happened {r[actual_key]:.0%} ({r[n_key]:,.0f})</title></circle>')
    body.append(f'<text class="tick" x="{sx(0.5):.1f}" y="{H - 4}" text-anchor="middle">'
                f'forecast</text>')
    body.append(f'<text class="tick" x="{sx(1) + 8:.1f}" y="{sy(0.5):.1f}">how often it happened</text>')
    return _svg("".join(body), "Calibration: forecast versus outcome")


def _calibration_compact(rows, fk, ak, nk):
    size, pl, pr, pt = 300, 62, 22, 12
    side = size - pl - pr
    height = pt + side + 44
    sx = lambda v: pl + v * side
    sy = lambda v: pt + (1 - v) * side
    body = [f'<rect class="frame" x="{pl}" y="{pt}" width="{side}" height="{side}"/>',
            f'<line class="diag" x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}"/>']
    for v in (0, 0.5, 1):
        body.append(f'<text class="tick" x="{sx(v):.1f}" y="{pt + side + 16}" text-anchor="middle">{v:.0%}</text>')
        body.append(f'<text class="tick" x="{pl - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{v:.0%}</text>')
    pts = sorted(rows, key=lambda r: r[fk])
    body.append('<polyline class="line" points="' + " ".join(
        f"{sx(r[fk]):.1f},{sy(r[ak]):.1f}" for r in pts) + '"/>')
    for r in pts:
        body.append(f'<circle class="dot" cx="{sx(r[fk]):.1f}" cy="{sy(r[ak]):.1f}" r="5"><title>Said '
                    f'{r[fk]:.0%}, happened {r[ak]:.0%} ({r[nk]:,.0f} bets)</title></circle>')
    body.append(f'<text class="tick" x="{sx(0.5):.1f}" y="{pt + side + 36}" text-anchor="middle">'
                f'what the model said</text>')
    body.append(f'<text class="tick" transform="translate(13 {sy(0.5):.1f}) rotate(-90)" '
                f'text-anchor="middle">what happened</text>')
    return (f'<svg viewBox="0 0 {size} {height}" role="img" class="chart" '
            f'aria-label="Calibration: forecast versus outcome">{"".join(body)}</svg>')
