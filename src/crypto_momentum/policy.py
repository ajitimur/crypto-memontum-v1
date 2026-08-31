"""`policy/` — the hand-maintained artifacts the Universe policy is applied from.

Two files, both inert TOML for the same reason `config.py` is: a research input
that could execute code is a research input that could smuggle in lookahead.

The reading lives here rather than in `sim/universe_policy.py` because the
simulation core reaches for no filesystem — see the invariant asserted in
`tests/test_simulation.py`. This module is the door: bytes in, a validated,
digested `ExclusionList` or `VenueListing` out, and the pure policy layer takes
it from there.

Each file is digested on the way in. The version a result quotes says which list
was meant; the SHA256 says which list was actually read, and the two disagree
exactly when someone edited a list without bumping its version.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

from crypto_momentum.sim.universe_policy import (
    ExclusionList,
    PolicyError,
    VenueListing,
)

POLICY_DIRNAME = "policy"
EXCLUSIONS_FILENAME = "exclusions-v1.toml"
TOKOCRYPTO_LISTING_FILENAME = "tokocrypto-listing-v1.toml"


def policy_root(repo_root: Path | str) -> Path:
    """Where the versioned policy artifacts live, relative to a repo checkout."""
    return Path(repo_root) / POLICY_DIRNAME


def load_exclusion_list(path: Path | str) -> ExclusionList:
    """Read a versioned exclusion list from its TOML file, digest and all."""
    path = Path(path)
    document, digest = _load_toml(path, "exclusion list")
    return ExclusionList.from_document(document, sha256=digest, path=str(path))


def load_venue_listing(path: Path | str) -> VenueListing:
    """Read a dated venue listing from its TOML file."""
    path = Path(path)
    document, digest = _load_toml(path, "venue listing")
    return VenueListing.from_document(document, sha256=digest, path=str(path))


def _load_toml(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise PolicyError(f"cannot read {label} {path}: {error}") from error
    try:
        document = tomllib.loads(payload.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise PolicyError(f"{path} is not valid TOML: {error}") from error
    return document, hashlib.sha256(payload).hexdigest()
