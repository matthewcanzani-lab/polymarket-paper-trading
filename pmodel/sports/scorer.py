"""Sports strategy, PREDICTIONS ONLY: compare Polymarket game markets with
sportsbooks' de-vigged odds (Pinnacle when it quotes the game, else the median
bookmaker) and log every matched game to the prediction log before kickoff.

No paper bets are placed. Whether bookmaker odds beat Polymarket's price is
measured first, on every matched game; betting rules come only after that
record exists (and would be a new, labeled version).

Games are matched on league, both team names and a start time within
MAX_START_GAP; anything ambiguous is skipped rather than guessed.
"""
import json
import re
import uuid
from datetime import datetime, timedelta, timezone

from pmodel import config
from pmodel.db import utc_now
from pmodel.model.train import _git_commit
from pmodel.sources import odds_api
from pmodel.sources.polymarket import PolymarketSource, _float, _maybe_json_list
from pmodel.trading import ledger, predlog

VERSION = "sports-v20261105-odds"
LOOKAHEAD = timedelta(days=3)
MAX_START_GAP = timedelta(hours=3)
_STOP = {"fc", "cf", "sc", "afc", "club", "saudi", "the", "cd", "ac", "ssc", "sv", "bk", "w",
         "de", "1", "calcio"}
_WIN = re.compile(r"^Will (.+?) win on ")
_DRAW = re.compile(r" end in a draw\?$")


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _tokens(name: str) -> frozenset:
    words = re.sub(r"[^a-z0-9 ]", " ", name.lower().replace("-", " ")).split()
    return frozenset(w for w in words if w not in _STOP)


def same_team(a: str, b: str) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta and tb) and (ta <= tb or tb <= ta)


def _parse_time(text):
    """Gamma's gameStartTime looks like '2026-11-06 00:00:00+00'."""
    if not text:
        return None
    text = re.sub(r"\+00$", "+00:00", text.strip().replace(" ", "T").replace("Z", "+00:00"))
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


def describe(raw: dict) -> dict | None:
    """-> {league, teams (a, b), target (a team name or 'Draw'), start} or None."""
    if raw.get("sportsMarketType") != "moneyline":
        return None
    league = (raw.get("slug") or "").split("-")[0]
    title = ((raw.get("events") or [{}])[0] or {}).get("title") or raw.get("question") or ""
    teams = re.split(r" vs\.? ", title.split(":")[-1].strip())
    outcomes = _maybe_json_list(raw.get("outcomes")) or []
    start = _parse_time(raw.get("gameStartTime"))
    if league not in odds_api.LEAGUES or len(teams) != 2 or start is None or len(outcomes) != 2:
        return None
    q = raw.get("question") or ""
    if [str(o).lower() for o in outcomes] != ["yes", "no"]:
        target = str(outcomes[0])                       # "Stars vs. Blue Jackets": P(Stars)
    elif _DRAW.search(q):
        target = "Draw"
    elif _WIN.search(q):
        target = _WIN.search(q).group(1)                 # "Will Everton FC win on ...?"
    else:
        return None
    return {"league": league, "teams": (teams[0].strip(), teams[1].strip()), "target": target,
            "start": start}


def match(desc: dict, events: list[dict]):
    """The odds event for this game and the bookmaker outcome name for the target."""
    a, b = desc["teams"]
    found = []
    for ev in events:
        home, away = ev.get("home_team", ""), ev.get("away_team", "")
        paired = ((same_team(a, home) and same_team(b, away))
                  or (same_team(a, away) and same_team(b, home)))
        if paired and abs(odds_api.commence(ev) - desc["start"]) <= MAX_START_GAP:
            found.append(ev)
    if len(found) != 1:
        return None, None                                # none, or ambiguous: skip
    ev = found[0]
    if desc["target"] == "Draw":
        return ev, "Draw"
    names = [n for n in (ev["home_team"], ev["away_team"]) if same_team(desc["target"], n)]
    return (ev, names[0]) if len(names) == 1 else (None, None)


def candidates(source, now):
    out = []
    for raw in source.iter_markets(closed=False, end_date_min=_iso(now),
                                   end_date_max=_iso(now + LOOKAHEAD),
                                   volume_min=config.MIN_VOLUME_USD):
        desc = describe(raw)
        bid, ask = _float(raw.get("bestBid")), _float(raw.get("bestAsk"))
        if (desc and desc["start"] > now and source.is_accepting_orders(raw)
                and bid is not None and ask is not None and 0 < bid <= ask < 1):
            out.append((raw, desc, (bid + ask) / 2))
    return out


def run(conn, log=print, now=None, source=None, fetch=None, predlog_dir=None) -> dict:
    now = now or datetime.now(timezone.utc)
    source = source or PolymarketSource()
    fetch = fetch or odds_api.fetch
    run_id = f"srun-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    cands = candidates(source, now)
    by_league = {}
    for c in cands:
        by_league.setdefault(c[1]["league"], []).append(c)
    leagues = sorted(by_league, key=lambda lg: -len(by_league[lg]))[:config.ODDS_MAX_SPORTS_PER_RUN]
    log(f"Sports odds ({VERSION}, run {run_id}): {len(cands)} pre-game markets in "
        f"{len(by_league)} leagues; fetching {leagues}")
    rows, skipped, credits = [], {"no odds": 0, "no match": 0}, None
    for lg in leagues:
        if credits is not None and credits < config.ODDS_MIN_CREDITS_LEFT:
            log(f"  stopping: only {credits} odds credits left this month")
            break
        try:
            events, credits = fetch(odds_api.LEAGUES[lg])
        except odds_api.OddsUnavailable as exc:
            log(f"  {lg}: odds unavailable ({exc})")
            skipped["no odds"] += len(by_league[lg])
            if "no ODDS_API_KEY" in str(exc):
                break
            continue
        for raw, desc, mid in by_league[lg]:
            ev, name = match(desc, events)
            fair = odds_api.consensus(ev) if ev else None
            if not fair or name not in fair[0]:
                skipped["no match"] += 1
                continue
            rows.append({"market_id": str(raw["id"]), "question": raw.get("question") or "",
                         "category": "Sports", "end_date": raw.get("endDate") or "",
                         "model_prob_yes": fair[0][name], "last_price": mid})
    logged = predlog.log_predictions(rows, "sports", VERSION, run_id, predlog_dir) \
        if predlog_dir and rows else 0
    ledger.append_run(conn, {
        "run_id": run_id, "started_at": _iso(now), "finished_at": utc_now(),
        "model_version": VERSION, "markets_scored": len(rows), "signals": 0, "bets_placed": 0,
        "settings_json": json.dumps({"strategy": "sports (predictions only, no bets)",
                                     "leagues": leagues, "odds_region": odds_api.REGION,
                                     "credits_left": credits, "code_commit": _git_commit()})})
    conn.commit()
    summary = {"candidates": len(cands), "matched": len(rows), "predictions_logged": logged,
               "skipped": skipped, "odds_credits_left": credits}
    log(f"Sports run finished: {summary}")
    return summary
