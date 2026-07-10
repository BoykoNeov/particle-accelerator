"""Runtime feature switches for optional accsim addons.

The pure-Python **baseline** is always on and never gated: the accelerator
optics/tracking core (Stages 0-6) and the toy event generator
(``accsim.events``), which depend only on numpy/scipy/matplotlib. Anything past
that baseline — an *addon / expansion / module / component* that pulls an
external tool (Docker/Pythia/Delphes), a heavy dependency, or an optional
extension — sits behind a runtime switch defined here, **default OFF**, per the
project working agreement (see ``docs/CONVENTIONS.md`` -> *Feature switches*).

Two switch surfaces read **one** source of truth (this registry):

* **In-package callers** guard their heavy entry point with :func:`require`,
  which raises :class:`AddonDisabledError` (carrying the instruction to enable
  it) when the addon is off. This is the switch for future in-package additions
  (e.g. a Delphes or LHAPDF step called from inside ``accsim``). Call it
  *before* importing the optional dependency, so "off" fails cleanly instead of
  crashing on a missing import.
* **Standalone scripts / CI** flip the same flag via the
  ``ACCSIM_ENABLE_<NAME>`` environment variable (e.g.
  ``ACCSIM_ENABLE_PYTHIA=1``). Running a standalone pipeline script *is* the
  opt-in, so its gate is deliberately light.

**Precedence:** a programmatic override (:func:`enable` / :func:`disable` /
:func:`enabled`) beats the environment; with no override the env var decides;
absent both, the addon is OFF.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = [
    "KNOWN_ADDONS",
    "AddonDisabledError",
    "UnknownAddonError",
    "env_var_for",
    "is_enabled",
    "enable",
    "disable",
    "reset",
    "enabled",
    "require",
]

# The fixed set of known addon switches. Add a name here when — and only when —
# real gated code lands behind it; do not scaffold empty flags for additions
# that do not exist yet (one feature per change).
KNOWN_ADDONS: frozenset[str] = frozenset({"pythia"})

# Programmatic overrides: name -> bool. A name absent here falls back to the
# environment. This is process-global state; tests reset it via an autouse
# fixture (see tests/conftest.py) so switches never leak between tests.
_overrides: dict[str, bool] = {}

# Values (case-insensitive, whitespace-stripped) that read as "on" in the env.
_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})


class UnknownAddonError(KeyError):
    """Raised when an addon name is not in :data:`KNOWN_ADDONS` (typo guard)."""


class AddonDisabledError(RuntimeError):
    """Raised by :func:`require` when a gated addon is off."""


def _check_known(name: str) -> None:
    if name not in KNOWN_ADDONS:
        raise UnknownAddonError(f"unknown addon {name!r}; known addons: {sorted(KNOWN_ADDONS)}")


def env_var_for(name: str) -> str:
    """The env var that toggles addon ``name`` (``ACCSIM_ENABLE_<NAME>``)."""
    _check_known(name)
    return f"ACCSIM_ENABLE_{name.upper()}"


def is_enabled(name: str) -> bool:
    """Whether addon ``name`` is currently on (override > env var > OFF)."""
    _check_known(name)
    if name in _overrides:
        return _overrides[name]
    return os.environ.get(env_var_for(name), "").strip().lower() in _TRUTHY


def enable(name: str) -> None:
    """Programmatically turn addon ``name`` on (beats the environment)."""
    _check_known(name)
    _overrides[name] = True


def disable(name: str) -> None:
    """Programmatically turn addon ``name`` off (beats the environment)."""
    _check_known(name)
    _overrides[name] = False


def reset(name: str | None = None) -> None:
    """Clear programmatic overrides (all, or just ``name``), reverting to env.

    Primarily for test isolation; the suite's autouse fixture calls ``reset()``
    around every test so no test's :func:`enable` leaks into the next.
    """
    if name is None:
        _overrides.clear()
    else:
        _check_known(name)
        _overrides.pop(name, None)


@contextmanager
def enabled(name: str, on: bool = True) -> Iterator[None]:
    """Scoped override: force addon ``name`` on (or off) inside the block.

    The prior override state — including *no* override — is restored on exit,
    even on exception. This is the primary API precisely because it guarantees
    the flag never leaks past the ``with`` block.
    """
    _check_known(name)
    had_prior = name in _overrides
    prior = _overrides.get(name, False)
    _overrides[name] = on
    try:
        yield
    finally:
        if had_prior:
            _overrides[name] = prior
        else:
            _overrides.pop(name, None)


def require(name: str) -> None:
    """Raise :class:`AddonDisabledError` unless addon ``name`` is enabled.

    Call at the top of a gated in-package entry point, *before* importing the
    heavy/optional dependency.
    """
    if not is_enabled(name):
        raise AddonDisabledError(
            f"addon {name!r} is an optional accsim addon and is OFF by default. "
            f"Enable it with accsim.features.enable({name!r}), the "
            f"`with accsim.features.enabled({name!r})` context manager, or by "
            f"setting {env_var_for(name)}=1 in the environment."
        )
