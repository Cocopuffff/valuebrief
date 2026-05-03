"""
base.py
~~~~~~~
Shared Pydantic base model and common validators for the schemas package.
"""

from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict
import re


class StrictBaseModel(BaseModel):
    """Base model with strict validation: forbid extra fields, strip whitespace."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


_TICKER_RE = re.compile(r"^[A-Z0-9]{1,10}(\.[A-Z]{1,4})?$")


def validate_ticker(v: str) -> str:
    """Canonical ticker validator: strip, uppercase, validate, and normalise.

    Raises ``ValueError`` with the original input on invalid tickers.
    """
    if not isinstance(v, str):
        raise ValueError(f"Invalid ticker: {v!r}")
    cleaned = v.strip().upper()
    if not _TICKER_RE.fullmatch(cleaned):
        raise ValueError(f"Invalid ticker: {v!r}")
    return cleaned


Ticker = Annotated[str, AfterValidator(validate_ticker)]


def validate_tickers(raw_tickers: list[str]) -> list[str]:
    """Validate and normalise a list of ticker symbols.

    Returns a list of normalised (stripped, uppercased) tickers.
    Raises ``ValueError`` with a collected message if any ticker is invalid.
    """
    invalid: list[str] = []
    normalised: list[str] = []
    for t in raw_tickers:
        try:
            normalised.append(validate_ticker(t))
        except ValueError:
            invalid.append(repr(t))
    if invalid:
        raise ValueError(f"Invalid ticker(s): {', '.join(invalid)}")
    return normalised


_BLOCK_ID_RE = re.compile(r"^block-[a-f0-9]{8}$")


def validate_block_id(v: str) -> str:
    """Validate a block ID: 'block-' followed by 8 hex chars."""
    if not _BLOCK_ID_RE.match(v):
        raise ValueError(f"Invalid block ID: {v}")
    return v
