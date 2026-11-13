"""HTTP fetching with explicit retry semantics.

404 means "not published (yet)" and is NOT retried; connection errors, 429 and 5xx are retried with
exponential backoff; other 4xx fail immediately. No response body is ever executed or evaluated.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import requests


class NotYetPublished(Exception):
    """The server answered 404 for this URL."""


class FetchError(Exception):
    """The fetch failed after retries (or with a non-retryable status)."""


@dataclass
class Fetched:
    body: bytes
    last_modified: datetime | None
    url: str


def get_bytes(url: str, *, session=None, headers=None, params=None, timeout: float = 30.0, retries: int = 3,
              backoff: float = 2.0, sleep=time.sleep) -> Fetched:
    get = (session or requests).get
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code == 404:
                raise NotYetPublished(url)
            if r.status_code == 429 or r.status_code >= 500:
                raise FetchError(f"{url}: HTTP {r.status_code}")
            if r.status_code >= 400:
                raise FetchError(f"{url}: HTTP {r.status_code} (not retried)") from None
            lm = r.headers.get("Last-Modified") if getattr(r, "headers", None) else None
            return Fetched(r.content, parsedate_to_datetime(lm).astimezone(UTC) if lm else None, url)
        except NotYetPublished:
            raise
        except requests.RequestException as e:
            last = FetchError(f"{url}: {e}")
        except FetchError as e:
            last = e
            if "not retried" in str(e):
                raise
        if attempt < retries:
            sleep(backoff * 2 ** (attempt - 1))
    raise FetchError(f"{url}: failed after {retries} attempts: {last}")
