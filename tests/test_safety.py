"""Guards for the PAPER-TRADING-ONLY rule. If one of these fails, stop and ask why."""
import re
from pathlib import Path

import pytest

from pmodel import safe_http

SRC = Path(__file__).resolve().parent.parent / "pmodel"

FORBIDDEN = [
    r"\.post\(", r"\.put\(", r"\.delete\(", r"\.patch\(",       # any write request
    r"py_clob_client", r"web3", r"eth_account", r"private_key",  # wallet / order libraries
    r"PRIVATE_KEY", r"mnemonic", r"create_order", r"post_order",
]


def test_no_order_or_wallet_code_anywhere():
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        for pattern in FORBIDDEN:
            if re.search(pattern, text):
                offenders.append(f"{path.name}: {pattern}")
    assert not offenders, offenders


def test_http_only_reaches_allowlisted_hosts():
    with pytest.raises(safe_http.HostNotAllowed):
        safe_http.get_json("https://example.com/anything")


def test_requirements_have_no_trading_libraries():
    reqs = (SRC.parent / "requirements.txt").read_text().lower()
    for lib in ("py-clob-client", "web3", "eth-account"):
        assert lib not in reqs
