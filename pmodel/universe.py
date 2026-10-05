"""Which markets the model studies and bets on — one definition, used everywhere.

A market is in the universe when:
  * its lifetime volume is at least MIN_VOLUME_USD ($10k, chosen from the Phase 1
    volume report: bid-ask spreads stop shrinking much above it, and ~1,000 open
    markets above it resolve in any 30-day window), and
  * it is NOT one of the rapid-fire crypto "Up or Down" coin-flip markets
    (5-minute / 15-minute / hourly / daily series). They are ~170k of the ~190k
    crypto markets above $10k and would swamp the training data and the bet log.

Platform-neutral: the exclusions match on `recurrence` (any platform can fill it)
and on question wording, both configurable in config.py.
"""
from pmodel import config


def sql_filter(alias="m"):
    """SQL WHERE fragment + params selecting in-universe markets."""
    a = f"{alias}." if alias else ""
    clauses = [f"{a}volume_usd >= ?"]
    params = [config.MIN_VOLUME_USD]
    if config.EXCLUDE_RECURRENCES:
        marks = ",".join("?" * len(config.EXCLUDE_RECURRENCES))
        clauses.append(f"COALESCE({a}recurrence, '') NOT IN ({marks})")
        params += list(config.EXCLUDE_RECURRENCES)
    for pattern in config.EXCLUDE_QUESTION_PATTERNS:
        clauses.append(f"{a}question NOT LIKE ?")
        params.append(f"%{pattern}%")
    return " AND ".join(clauses), params


def is_in_universe(market) -> bool:
    if market.volume_usd < config.MIN_VOLUME_USD:
        return False
    if market.recurrence and market.recurrence in config.EXCLUDE_RECURRENCES:
        return False
    q = market.question.lower()
    return not any(p.lower() in q for p in config.EXCLUDE_QUESTION_PATTERNS)
