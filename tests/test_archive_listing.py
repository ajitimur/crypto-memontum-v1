"""Bucket listing pages parse, and a truncated listing pages through to the end.

Every page here was listed from the live bucket once and committed. The opener
is injected, so no test in this module reaches the network.
"""

import pytest

from crypto_momentum.data.archive_listing import (
    MalformedListing,
    listing_url,
    parse_listing_page,
    walk_listing,
)
from crypto_momentum.data.fetch import ArchiveUnavailable

SR_PREFIX = "data/spot/monthly/klines/SR"
SR_PAGES = ("klines-SR-page1.xml", "klines-SR-page2.xml", "klines-SR-page3.xml")
# The recorded SR pages were listed two keys at a time to force pagination.
SR_MAX_KEYS = 2


def paged_opener(recorded_listing_page, prefix, page_names, max_keys):
    """Serve the recorded pages under `prefix`, chained by their own markers."""
    served = {}
    marker = None
    for name in page_names:
        payload = recorded_listing_page(name)
        served[listing_url(prefix, marker=marker, max_keys=max_keys)] = payload
        marker = parse_listing_page(payload).next_marker

    def open_url(url: str) -> bytes:
        if url not in served:
            raise ArchiveUnavailable(f"no recorded page for {url}")
        return served[url]

    return open_url


def single_page_opener(recorded_listing_page, prefix, name, max_keys=1000):
    return paged_opener(recorded_listing_page, prefix, (name,), max_keys)


def test_a_delimited_page_reports_its_directories_and_its_marker(recorded_listing_page):
    page = parse_listing_page(recorded_listing_page("klines-SR-page1.xml"))

    assert page.prefixes == (
        "data/spot/monthly/klines/SRMBIDR/",
        "data/spot/monthly/klines/SRMBNB/",
    )
    assert page.keys == ()
    assert page.is_truncated
    assert page.next_marker == "data/spot/monthly/klines/SRMBNB/"


def test_a_final_page_reports_no_marker(recorded_listing_page):
    page = parse_listing_page(recorded_listing_page("klines-SR-page3.xml"))

    assert not page.is_truncated
    assert page.next_marker is None


def test_a_file_listing_reports_object_keys(recorded_listing_page):
    page = parse_listing_page(recorded_listing_page("SRMUSDT-1d.xml"))

    assert page.prefixes == ()
    assert page.keys[0] == "data/spot/monthly/klines/SRMUSDT/1d/SRMUSDT-1d-2020-08.zip"
    assert len(page.keys) == 56


def test_a_truncated_listing_is_paged_through_to_the_end(recorded_listing_page):
    listing = walk_listing(
        SR_PREFIX,
        open_url=paged_opener(recorded_listing_page, SR_PREFIX, SR_PAGES, SR_MAX_KEYS),
        max_keys=SR_MAX_KEYS,
    )

    assert listing.prefixes == (
        "data/spot/monthly/klines/SRMBIDR/",
        "data/spot/monthly/klines/SRMBNB/",
        "data/spot/monthly/klines/SRMBTC/",
        "data/spot/monthly/klines/SRMBUSD/",
        "data/spot/monthly/klines/SRMUSDT/",
    )


def test_a_listing_that_never_advances_its_marker_fails_rather_than_looping():
    stuck = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        b"<IsTruncated>true</IsTruncated><NextMarker></NextMarker>"
        b"</ListBucketResult>"
    )

    with pytest.raises(MalformedListing, match="does not advance"):
        walk_listing("data/spot/monthly/klines/", open_url=lambda url: stuck)


def test_bytes_that_are_not_a_bucket_listing_fail_loudly():
    with pytest.raises(MalformedListing):
        parse_listing_page(b"<html>404</html>")


def test_a_well_formed_document_that_is_not_a_listing_is_rejected():
    with pytest.raises(MalformedListing, match="ListBucketResult"):
        parse_listing_page(b"<Error><Code>AccessDenied</Code></Error>")


def test_the_listing_url_always_carries_a_delimiter():
    url = listing_url("data/spot/monthly/klines/", marker="data/spot/monthly/klines/BTCUSDT/")

    assert "delimiter=%2F" in url
    assert "prefix=data%2Fspot%2Fmonthly%2Fklines%2F" in url
    assert "marker=data%2Fspot%2Fmonthly%2Fklines%2FBTCUSDT%2F" in url
