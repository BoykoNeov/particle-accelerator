r"""Scenario files: a lattice, its reference particle and how to read it, as JSON.

The scenario format is the seam between the browser **editor** (``editor/index.html``,
which composes a machine element by element and shows its linear optics live) and
the package: the editor writes a ``.accsim.json`` file and :func:`load_scenario`
turns it into a :class:`~accsim.lattice.Lattice` that every function in the package
accepts. The reverse direction, :func:`scenario_from_lattice`, lets a lattice built
in Python be opened in the editor.

The format is deliberately flat and typed by name, so a file is readable and
diffable without the package::

    {
      "format": "accsim-scenario/1",
      "name": "FODO cell",
      "periodic": true,
      "reference": {"species": "proton", "energy_mode": "total", "energy_eV": 1e10},
      "elements": [
        {"type": "Quadrupole", "name": "QF", "length": 0.5, "k1": 0.6},
        {"type": "Drift", "name": "D", "length": 1.0},
        {"type": "Dipole", "name": "B", "length": 2.0, "angle": 0.1309},
        ...
      ]
    }

Every ``type`` is the name of an element class in :mod:`accsim.elements`, and every
other key of an element record is a constructor argument of that class, in that
class's units (SI: m, rad, eV, V, Hz). Two conveniences exist because the editor
needs them and a hand-written file wants them:

- an ``RFCavity`` may carry ``"harmonic": h`` instead of ``"frequency"``; the
  frequency is then ``h beta0 c / C`` with ``C`` the total length of the
  ``elements`` list, exactly :meth:`~accsim.elements.rfcavity.RFCavity.from_harmonic`;
- the reference particle may be named by ``species`` (``electron``, ``positron``,
  ``proton``) and by ``energy_mode`` (``total``, ``kinetic``, ``momentum``); a
  ``custom`` species gives ``mass_eV`` and ``charge`` explicitly.

Optional keys the package does not consume but round-trips: ``initial_twiss`` (the
entrance Twiss of a non-periodic *line*, which the editor propagates from) and
``beam`` (emittances and momentum spread, for the editor's envelope plot).

**The editor's optics is a port of this package, not a second opinion.** Its
JavaScript core (``editor/accsim-optics.js``) reproduces each element's matrix and
the Twiss / tune / chromaticity / momentum-compaction / survey code entry for entry,
and ``tests/analytic/test_scenario.py`` holds the two against each other on every
bundled preset (under Node, skipped when Node is absent). A number the editor shows
is therefore the number accsim computes, to 1e-9 — and any change to an element's
map here must be mirrored there, which that test is what enforces.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any

from .elements import (
    Aperture,
    Collimator,
    Corrector,
    Dipole,
    Drift,
    Element,
    Octupole,
    Quadrupole,
    RFCavity,
    Sextupole,
    SkewQuadrupole,
    Solenoid,
    ThinOctupole,
    ThinQuadrupole,
    ThinSextupole,
    ThinSkewQuadrupole,
    Wiggler,
)
from .lattice import Lattice
from .reference import (
    CLIGHT,
    ELECTRON_ANOMALOUS_MOMENT,
    ELECTRON_MASS_EV,
    PROTON_ANOMALOUS_MOMENT,
    PROTON_MASS_EV,
    ReferenceParticle,
)

SCENARIO_FORMAT = "accsim-scenario/1"

#: ``type`` name -> (class, constructor fields in order). The field names are both
#: the constructor keywords and the attribute names the instance keeps, which is
#: what makes :func:`element_to_dict` the exact inverse of :func:`element_from_dict`.
_REGISTRY: dict[str, tuple[type[Element], tuple[str, ...]]] = {
    "Drift": (Drift, ("length",)),
    "Quadrupole": (Quadrupole, ("length", "k1")),
    "ThinQuadrupole": (ThinQuadrupole, ("k1l",)),
    "Dipole": (Dipole, ("length", "angle", "k1", "e1", "e2")),
    "Sextupole": (Sextupole, ("length", "k2")),
    "ThinSextupole": (ThinSextupole, ("k2l",)),
    "Octupole": (Octupole, ("length", "k3")),
    "ThinOctupole": (ThinOctupole, ("k3l",)),
    "SkewQuadrupole": (SkewQuadrupole, ("length", "k1s")),
    "ThinSkewQuadrupole": (ThinSkewQuadrupole, ("k1sl",)),
    "Solenoid": (Solenoid, ("length", "ks")),
    "RFCavity": (RFCavity, ("voltage", "frequency", "phi_s")),
    "Corrector": (Corrector, ("kick_x", "kick_y")),
    "Wiggler": (Wiggler, ("period", "h0", "periods")),
    "Aperture": (Aperture, ("shape", "half_x", "half_y", "length")),
    "Collimator": (Collimator, ("shape", "half_x", "half_y", "length")),
}
#: Types whose constructor takes the misalignment keywords ``dx``, ``dy``, ``roll``.
_ALIGNABLE = frozenset(t for t in _REGISTRY if t not in ("RFCavity", "Aperture", "Collimator"))
_ALIGN_FIELDS = ("dx", "dy", "roll")
#: Boolean flags that are constructor keywords too.
_FLAGS: dict[str, tuple[str, ...]] = {"Dipole": ("fringe",)}

_SPECIES: dict[str, tuple[float, float, float | None]] = {
    "electron": (ELECTRON_MASS_EV, -1.0, ELECTRON_ANOMALOUS_MOMENT),
    "positron": (ELECTRON_MASS_EV, +1.0, ELECTRON_ANOMALOUS_MOMENT),
    "proton": (PROTON_MASS_EV, +1.0, PROTON_ANOMALOUS_MOMENT),
}


class ScenarioError(ValueError):
    """A scenario file that cannot be turned into a lattice."""


# --- reference particle -----------------------------------------------------------
def reference_from_dict(d: Mapping[str, Any]) -> ReferenceParticle:
    """Build the reference particle from the scenario's ``reference`` record."""
    species = str(d.get("species", "custom"))
    if species in _SPECIES:
        mass, charge, g = _SPECIES[species]
    elif species == "custom":
        mass, charge, g = float(d["mass_eV"]), float(d.get("charge", 1.0)), None
    else:
        raise ScenarioError(f"unknown species {species!r}")
    if "mass_eV" in d:
        mass = float(d["mass_eV"])
    if "charge" in d:
        charge = float(d["charge"])
    if "anomalous_moment" in d:
        g = None if d["anomalous_moment"] is None else float(d["anomalous_moment"])
    mode = str(d.get("energy_mode", "total"))
    energy = float(d["energy_eV"])
    if mode == "total":
        return ReferenceParticle.from_total_energy(mass, energy, charge, g)
    if mode == "kinetic":
        return ReferenceParticle.from_kinetic_energy(mass, energy, charge, g)
    if mode == "momentum":
        return ReferenceParticle.from_momentum(mass, energy, charge, g)
    raise ScenarioError(f"unknown energy_mode {mode!r}")


def reference_to_dict(ref: ReferenceParticle) -> dict[str, Any]:
    """The inverse of :func:`reference_from_dict`; names the species when it can."""
    for name, (mass, charge, _) in _SPECIES.items():
        if ref.mass_eV == mass and ref.charge == charge:
            return {"species": name, "energy_mode": "total", "energy_eV": ref.total_energy_eV}
    out: dict[str, Any] = {
        "species": "custom",
        "mass_eV": ref.mass_eV,
        "charge": ref.charge,
        "energy_mode": "total",
        "energy_eV": ref.total_energy_eV,
    }
    if ref.anomalous_moment is not None:
        out["anomalous_moment"] = ref.anomalous_moment
    return out


# --- elements ---------------------------------------------------------------------
def element_from_dict(
    d: Mapping[str, Any],
    *,
    circumference: float | None = None,
    ref: ReferenceParticle | None = None,
) -> Element:
    """One element record -> one :class:`Element`.

    ``circumference`` and ``ref`` are needed only by an ``RFCavity`` given through a
    ``harmonic`` number; :func:`load_scenario` supplies them.
    """
    kind = d.get("type")
    if kind not in _REGISTRY:
        raise ScenarioError(f"unknown element type {kind!r}")
    cls, fields = _REGISTRY[kind]
    kwargs: dict[str, Any] = {}
    for f in fields:
        if f in d and d[f] is not None and d[f] != "":
            kwargs[f] = d[f] if f == "shape" else _num(d[f], f, kind)
    if kind == "RFCavity" and d.get("harmonic") not in (None, ""):
        h = int(d["harmonic"])
        if circumference is None or ref is None:
            raise ScenarioError("an RFCavity given by harmonic needs the lattice circumference")
        if h != d["harmonic"] or h <= 0:
            raise ScenarioError(f"harmonic must be a positive integer, got {d['harmonic']!r}")
        kwargs["frequency"] = h * ref.beta0 * CLIGHT / circumference
    if kind == "Wiggler":
        kwargs["periods"] = int(kwargs["periods"])
    if kind in _ALIGNABLE:
        for f in _ALIGN_FIELDS:
            if d.get(f) not in (None, "", 0, 0.0):
                kwargs[f] = _num(d[f], f, kind)
    if kind == "Dipole" and kwargs.get("angle") and (kwargs.get("dx") or kwargs.get("dy")):
        # accsim refuses this only when the kick is asked for; refuse it at load time so a
        # scenario that loads is one every optics call accepts.
        raise ScenarioError(
            f"Dipole {d.get('name')!r}: a bending dipole cannot be displaced (dx/dy) — "
            "accsim has no rigid-body model for a translated curved body; steer with a Corrector"
        )
    for f in _FLAGS.get(kind, ()):
        if f in d:
            kwargs[f] = bool(d[f])
    name = d.get("name")
    try:
        return cls(name=name, **kwargs)  # type: ignore[call-arg]
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise ScenarioError(f"{kind} {name!r}: {exc}") from exc


def element_to_dict(elem: Element) -> dict[str, Any]:
    """One :class:`Element` -> one element record (the exact inverse of the above)."""
    kind = type(elem).__name__
    if kind not in _REGISTRY:
        raise ScenarioError(f"{kind} has no scenario representation")
    _, fields = _REGISTRY[kind]
    out: dict[str, Any] = {"type": kind}
    if elem.name is not None:
        out["name"] = elem.name
    for f in fields:
        v = getattr(elem, f)
        if v is not None:
            out[f] = v
    if kind in _ALIGNABLE:
        for f in _ALIGN_FIELDS:
            v = getattr(elem, f)
            if v != 0.0:
                out[f] = v
    for f in _FLAGS.get(kind, ()):
        if getattr(elem, f):
            out[f] = True
    return out


def _num(v: Any, field_name: str, kind: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError) as exc:
        raise ScenarioError(f"{kind}.{field_name} must be a number, got {v!r}") from exc
    if not math.isfinite(x):
        raise ScenarioError(f"{kind}.{field_name} must be finite, got {v!r}")
    return x


def _length_of(d: Mapping[str, Any]) -> float:
    if d.get("type") == "Wiggler":
        return float(d.get("period", 0.0)) * float(d.get("periods", 0))
    return float(d.get("length", 0.0) or 0.0)


# --- the scenario -----------------------------------------------------------------
@dataclass
class Scenario:
    """A loaded scenario: the lattice plus the file's metadata."""

    lattice: Lattice
    name: str = "untitled"
    description: str = ""
    periodic: bool = True
    initial_twiss: dict[str, float] = field(default_factory=dict)
    beam: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = scenario_from_lattice(self.lattice, name=self.name, periodic=self.periodic)
        if self.description:
            d["description"] = self.description
        if self.initial_twiss:
            d["initial_twiss"] = dict(self.initial_twiss)
        if self.beam:
            d["beam"] = dict(self.beam)
        return d


def load_scenario(source: Mapping[str, Any] | str | PathLike[str]) -> Scenario:
    """Read a scenario from a dict, a JSON string, or a path to a JSON file."""
    if isinstance(source, Mapping):
        data = dict(source)
    else:
        text = source if isinstance(source, str) and source.lstrip().startswith("{") else None
        if text is None:
            text = Path(source).read_text(encoding="utf-8")
        data = json.loads(text)
    fmt = data.get("format", SCENARIO_FORMAT)
    if fmt != SCENARIO_FORMAT:
        raise ScenarioError(f"unsupported scenario format {fmt!r} (want {SCENARIO_FORMAT!r})")
    if "reference" not in data:
        raise ScenarioError("scenario has no 'reference' record")
    ref = reference_from_dict(data["reference"])
    records = list(data.get("elements", []))
    circumference = sum(_length_of(r) for r in records)
    elements = [element_from_dict(r, circumference=circumference, ref=ref) for r in records]
    return Scenario(
        lattice=Lattice(elements, ref),
        name=str(data.get("name", "untitled")),
        description=str(data.get("description", "")),
        periodic=bool(data.get("periodic", True)),
        initial_twiss=dict(data.get("initial_twiss", {}) or {}),
        beam=dict(data.get("beam", {}) or {}),
    )


def scenario_from_lattice(
    lattice: Lattice, *, name: str = "untitled", periodic: bool = True
) -> dict[str, Any]:
    """A lattice built in Python -> a scenario dict the editor can open."""
    return {
        "format": SCENARIO_FORMAT,
        "name": name,
        "periodic": periodic,
        "reference": reference_to_dict(lattice.ref),
        "elements": [element_to_dict(e) for e in lattice.elements],
    }


def dump_scenario(
    scenario: Scenario | Lattice,
    path: str | PathLike[str] | None = None,
    *,
    name: str = "untitled",
    periodic: bool = True,
) -> str:
    """Serialise to JSON text; write it to ``path`` as well when one is given."""
    if isinstance(scenario, Lattice):
        data = scenario_from_lattice(scenario, name=name, periodic=periodic)
    else:
        data = scenario.to_dict()
    text = json.dumps(data, indent=2)
    if path is not None:
        Path(path).write_text(text + "\n", encoding="utf-8")
    return text


__all__ = [
    "SCENARIO_FORMAT",
    "Scenario",
    "ScenarioError",
    "dump_scenario",
    "element_from_dict",
    "element_to_dict",
    "load_scenario",
    "reference_from_dict",
    "reference_to_dict",
    "scenario_from_lattice",
]
