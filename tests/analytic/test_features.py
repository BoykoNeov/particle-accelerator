"""Runtime feature switches for optional addons (accsim.features).

This is not a physics quantity, so there is no closed form to derive — the
gates are behavioral, matching the project rule that *everything past the
pure-Python toy baseline is an opt-in runtime switch, default OFF*:

1. every known addon defaults OFF, and the baseline imports with all OFF;
2. ``require`` raises when off and passes when on;
3. the context manager restores prior state (including *no* override) on exit;
4. precedence: a programmatic override beats the environment variable;
5. unknown addon names are rejected (typo guard).
"""

from __future__ import annotations

import importlib

import pytest

from accsim import features

ADDON = "pythia"  # a representative gated addon (see KNOWN_ADDONS for the full set)


def test_all_known_addons_default_off() -> None:
    for name in features.KNOWN_ADDONS:
        assert features.is_enabled(name) is False


def test_baseline_imports_with_all_addons_off() -> None:
    # The pure-Python baseline — accelerator core + toy generator — must be
    # importable and usable with every switch off. Nothing baseline is gated.
    assert all(not features.is_enabled(n) for n in features.KNOWN_ADDONS)
    accsim = importlib.import_module("accsim")
    events = importlib.import_module("accsim.events")
    assert accsim.__version__  # core imported
    assert events is not None  # toy generator imported, ungated


def test_require_raises_when_off_passes_when_on() -> None:
    with pytest.raises(features.AddonDisabledError):
        features.require(ADDON)
    features.enable(ADDON)
    features.require(ADDON)  # no raise
    features.disable(ADDON)
    with pytest.raises(features.AddonDisabledError):
        features.require(ADDON)


def test_disabled_error_message_names_the_env_var() -> None:
    with pytest.raises(features.AddonDisabledError) as exc:
        features.require(ADDON)
    assert features.env_var_for(ADDON) in str(exc.value)


def test_context_manager_restores_no_prior_override() -> None:
    assert ADDON not in features._overrides
    with features.enabled(ADDON):
        assert features.is_enabled(ADDON) is True
    # Restored to "no override" (not merely to False).
    assert ADDON not in features._overrides
    assert features.is_enabled(ADDON) is False


def test_context_manager_restores_prior_override_and_survives_exception() -> None:
    features.disable(ADDON)  # prior override present, = False
    with pytest.raises(RuntimeError):
        with features.enabled(ADDON):
            assert features.is_enabled(ADDON) is True
            raise RuntimeError("boom")
    # Prior override (False) restored even though the block raised.
    assert features._overrides.get(ADDON) is False
    assert features.is_enabled(ADDON) is False


def test_context_manager_can_force_off() -> None:
    features.enable(ADDON)
    with features.enabled(ADDON, on=False):
        assert features.is_enabled(ADDON) is False
    assert features.is_enabled(ADDON) is True


def test_env_var_toggles_when_no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    var = features.env_var_for(ADDON)
    for truthy in ("1", "true", "YES", " On "):
        monkeypatch.setenv(var, truthy)
        assert features.is_enabled(ADDON) is True
    for falsey in ("0", "false", "", "nope"):
        monkeypatch.setenv(var, falsey)
        assert features.is_enabled(ADDON) is False


def test_override_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    var = features.env_var_for(ADDON)
    monkeypatch.setenv(var, "1")  # env says ON
    features.disable(ADDON)  # override says OFF
    assert features.is_enabled(ADDON) is False  # override wins
    features.reset(ADDON)  # drop override -> env decides again
    assert features.is_enabled(ADDON) is True


def test_unknown_addon_is_rejected() -> None:
    for fn in (features.is_enabled, features.enable, features.disable, features.require):
        with pytest.raises(features.UnknownAddonError):
            fn("lhapdf")  # not a known addon — no code behind it (typo guard)
