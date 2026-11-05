"""Sportsbook odds from the-odds-api.com (read-only; GET only via safe_http).

Only moneyline ("h2h") odds, one region, so each league costs 1 credit per
fetch. The API key is a secret: read from config (the ODDS_API_KEY environment
variable) and scrubbed from any error message so it can't end up in a log.
"""
from datetime import datetime, timezone
from statistics import median

from pmodel import config, safe_http

BASE = "https://api.the-odds-api.com/v4/sports"
REGION = "eu"          # includes Pinnacle, the sharpest widely-quoted bookmaker
SHARP_BOOK = "pinnacle"

# Polymarket slug prefix -> the-odds-api sport key.
LEAGUES = {
    "nfl": "americanfootball_nfl", "cfb": "americanfootball_ncaaf",
    "nba": "basketball_nba", "cbb": "basketball_ncaab", "nhl": "icehockey_nhl",
    "mlb": "baseball_mlb", "ufc": "mma_mixed_martial_arts",
    "epl": "soccer_epl", "lal": "soccer_spain_la_liga", "sea": "soccer_italy_serie_a",
    "bun": "soccer_germany_bundesliga", "fl1": "soccer_france_ligue_one",
    "ucl": "soccer_uefa_champs_league", "uel": "soccer_uefa_europa_league",
    "mls": "soccer_usa_mls", "spl": "soccer_saudi_arabia_pro_league",
    "elc": "soccer_efl_champ", "ere": "soccer_netherlands_eredivisie",
    "por": "soccer_portugal_primeira_liga", "mex": "soccer_mexico_ligamx",
    "bra": "soccer_brazil_campeonato",
}


class OddsUnavailable(Exception):
    pass


def _scrub(text: str) -> str:
    return text.replace(config.ODDS_API_KEY, "***") if config.ODDS_API_KEY else text


def fetch(sport_key: str) -> tuple[list[dict], int | None]:
    """Upcoming/in-play events with moneyline odds -> (events, credits remaining)."""
    if not config.ODDS_API_KEY:
        raise OddsUnavailable("no ODDS_API_KEY set")
    try:
        data, headers = safe_http.get_json(
            f"{BASE}/{sport_key}/odds",
            params={"apiKey": config.ODDS_API_KEY, "regions": REGION, "markets": "h2h",
                    "oddsFormat": "decimal", "dateFormat": "iso"},
            with_headers=True)
    except Exception as exc:  # never let the key reach a log via the URL in the message
        raise OddsUnavailable(_scrub(f"{type(exc).__name__}: {exc}")) from None
    left = headers.get("x-requests-remaining")
    return data, int(float(left)) if left not in (None, "") else None


def fair_probs(bookmaker: dict) -> dict | None:
    """De-vigged probabilities {outcome name: p} from one bookmaker's h2h prices."""
    market = next((m for m in bookmaker.get("markets", []) if m.get("key") == "h2h"), None)
    if not market:
        return None
    inv = {o["name"]: 1 / float(o["price"]) for o in market["outcomes"] if float(o["price"]) > 1}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()} if total else None


def consensus(event: dict) -> tuple[dict, str] | None:
    """Pinnacle's fair line if quoted, else the median fair line across bookmakers."""
    books = {b["key"]: fair_probs(b) for b in event.get("bookmakers", [])}
    books = {k: v for k, v in books.items() if v}
    if not books:
        return None
    if SHARP_BOOK in books:
        return books[SHARP_BOOK], SHARP_BOOK
    names = set.intersection(*(set(v) for v in books.values()))
    if not names:
        return None
    probs = {n: median(v[n] for v in books.values()) for n in names}
    total = sum(probs.values())
    return {n: p / total for n, p in probs.items()}, f"median of {len(books)} books"


def commence(event: dict) -> datetime:
    return datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00")).astimezone(timezone.utc)
