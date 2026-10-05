"""The ONLY way this project talks to the internet.

Safety guarantee for paper trading: this module can only make HTTP GET requests
to an explicit allowlist of public, read-only data hosts. There is deliberately
no POST/PUT/DELETE function, so nothing in this codebase can submit an order,
even by accident. A test (tests/test_safety.py) fails if anyone adds one.
"""
import sys
import threading
import time
from urllib.parse import urlparse

import requests

from pmodel import config

ALLOWED_HOSTS = {
    "gamma-api.polymarket.com",  # market listings and metadata (public)
    "clob.polymarket.com",       # price history (public, read-only endpoint)
    "data-api.polymarket.com",   # public trade history (read-only)
}

_local = threading.local()  # one Session per thread (Sessions aren't guaranteed thread-safe)


def _session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers["User-Agent"] = "polymarket-model-v2 (paper trading research)"
    return _local.session


class HostNotAllowed(Exception):
    pass


def get_json(url, params=None, retries=4, timeout=30):
    """GET a URL and return parsed JSON, retrying politely on rate limits/errors."""
    host = urlparse(url).hostname
    if host not in ALLOWED_HOSTS:
        raise HostNotAllowed(f"{host} is not on the read-only allowlist")

    delay = 2.0
    for attempt in range(retries + 1):
        time.sleep(config.REQUEST_PAUSE_SEC)
        try:
            resp = _session().get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            problem = type(exc).__name__
        else:
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == retries:
                    resp.raise_for_status()
                problem = f"HTTP {resp.status_code}"
            else:
                resp.raise_for_status()
                return resp.json()
        print(f"  [retry] {host}: {problem}, waiting {delay:.0f}s", file=sys.stderr, flush=True)
        time.sleep(delay)
        delay *= 2
