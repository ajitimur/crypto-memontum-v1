"""Listing the `data.binance.vision` bucket.

The bucket is S3-listable, and that listing is the only survivorship-free
enumeration of what Binance spot ever traded. Per ADR-0008 the Universe is built
from it and explicitly **not** from `exchangeInfo` — nor from Binance's own
`fetch-all-trading-pairs.sh`, which wraps `exchangeInfo` and therefore returns
only symbols still listed today.

A listing page is bucket metadata rather than data: S3 publishes no `.CHECKSUM`
beside it, so there is nothing here to verify. Every *data* file the listing
points at is fetched through `crypto_momentum.data.fetch.fetch_archive_file`,
which refuses a payload whose SHA256 does not match the published one.

Parsing is a pure function of bytes and is tested against recorded pages. Only
`walk_listing` reaches the network, and its opener is injected.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from typing import NamedTuple
from urllib.parse import urlencode

from crypto_momentum.data.fetch import UrlOpener, download

# The bucket is served from its own S3 endpoint; the `data.binance.vision` CDN
# host in front of it serves objects but not the ListBucket XML.
LISTING_ENDPOINT = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
KLINES_PREFIX = "data/spot/monthly/klines/"

# S3 caps a page at 1000 keys and reports `IsTruncated` when it does.
MAX_KEYS_PER_PAGE = 1000

_S3_NAMESPACE = "{http://s3.amazonaws.com/doc/2006-03-01/}"
# A stuck marker would loop forever against a paginated bucket. The real klines
# prefix is a few thousand entries, so anything beyond this is a bug.
_MAX_PAGES = 10_000


class MalformedListing(Exception):
    """Bytes that are not a readable S3 ListBucketResult."""


class ListingPage(NamedTuple):
    """One page of a bucket listing.

    `prefixes` are the directory-like `CommonPrefixes` a `delimiter=/` listing
    collapses; `keys` are the objects directly under the requested prefix.
    """

    prefixes: tuple[str, ...]
    keys: tuple[str, ...]
    is_truncated: bool
    next_marker: str | None


class Listing(NamedTuple):
    """Every page of one prefix, concatenated in the bucket's own key order."""

    prefix: str
    prefixes: tuple[str, ...]
    keys: tuple[str, ...]


def listing_url(
    prefix: str, *, marker: str | None = None, max_keys: int = MAX_KEYS_PER_PAGE
) -> str:
    """Address one page of the bucket listing under `prefix`.

    `delimiter=/` is always set, so a listing of `.../klines/` returns one
    `CommonPrefix` per symbol rather than every object beneath it.
    """
    query: list[tuple[str, str]] = [("delimiter", "/"), ("prefix", prefix)]
    query.append(("max-keys", str(max_keys)))
    if marker is not None:
        query.append(("marker", marker))
    return f"{LISTING_ENDPOINT}?{urlencode(query)}"


def parse_listing_page(payload: bytes) -> ListingPage:
    """Parse one `ListBucketResult` document."""
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as error:
        raise MalformedListing(f"could not parse bucket listing: {error}") from error
    if root.tag != f"{_S3_NAMESPACE}ListBucketResult":
        raise MalformedListing(f"expected a ListBucketResult, got {root.tag!r}")

    prefixes = tuple(
        _required_text(element, "Prefix")
        for element in root.findall(f"{_S3_NAMESPACE}CommonPrefixes")
    )
    keys = tuple(
        _required_text(element, "Key")
        for element in root.findall(f"{_S3_NAMESPACE}Contents")
    )
    # Every listing here sets `delimiter=/`, and S3 supplies NextMarker whenever
    # it does. A truncated page without one is a response we do not understand,
    # and `walk_listing` refuses it rather than guessing a marker: guessing from
    # the last key silently skips the CommonPrefixes that sort after it.
    is_truncated = _text(root, "IsTruncated") == "true"
    next_marker = _text(root, "NextMarker") or None
    return ListingPage(
        prefixes=prefixes, keys=keys, is_truncated=is_truncated, next_marker=next_marker
    )


def walk_listing(
    prefix: str,
    *,
    open_url: UrlOpener | None = None,
    max_keys: int = MAX_KEYS_PER_PAGE,
) -> Listing:
    """Page through the whole of `prefix` and return everything under it.

    The only function in this module that opens a socket. `open_url` is injected
    so every test above it runs against recorded pages.
    """
    opener = open_url or download
    prefixes: list[str] = []
    keys: list[str] = []
    marker: str | None = None
    for _ in range(_MAX_PAGES):
        page = parse_listing_page(opener(listing_url(prefix, marker=marker, max_keys=max_keys)))
        prefixes.extend(page.prefixes)
        keys.extend(page.keys)
        if not page.is_truncated:
            return Listing(prefix=prefix, prefixes=tuple(prefixes), keys=tuple(keys))
        if page.next_marker is None or page.next_marker == marker:
            raise MalformedListing(
                f"listing of {prefix!r} says it is truncated but does not advance "
                f"past marker {marker!r}"
            )
        marker = page.next_marker
    raise MalformedListing(f"listing of {prefix!r} did not terminate within {_MAX_PAGES} pages")


def _required_text(element: ElementTree.Element, tag: str) -> str:
    value = _text(element, tag)
    if value is None:
        raise MalformedListing(f"listing entry is missing <{tag}>")
    return value


def _text(element: ElementTree.Element, tag: str) -> str | None:
    found = element.find(f"{_S3_NAMESPACE}{tag}")
    if found is None or found.text is None:
        return None
    return found.text.strip()
