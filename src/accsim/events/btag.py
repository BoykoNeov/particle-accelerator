r"""b-tagging efficiency measured against a Delphes card (milestone E2).

Delphes does not *simulate* a b-tagging algorithm. Its ``BTagging`` module is a
**parametrisation**: for each jet it looks up an efficiency formula chosen by the
jet's associated parton flavour, evaluates it at that jet's kinematics, and sets
a bit with that probability. The card therefore *is* the closed form this
milestone gates against -- there is a known right answer for the tagging rate of
every jet, written down in the card, and a correct analysis must recover it.

This module is the host-side half of that check (baseline: numpy only, no
Docker, no ROOT). It

* **parses** the ``BTagging`` modules out of a Delphes card -- resolving ``source``
  includes, because the multi-working-point cards keep their formulas in
  separate files -- into :class:`BTagWorkingPoint` records;
* **evaluates** the card's TCL efficiency expressions over numpy arrays of jet
  kinematics; and
* **compares** a measured tagging rate against the configured one as a pull.

Formula provenance: parsed, never transcribed
---------------------------------------------
The efficiency expressions are read out of the card file itself -- the same file
copied out of the Delphes image and handed to ``DelphesHepMC3`` -- and evaluated
here. They are **never** retyped into Python. A transcribed formula is a
remembered constant wearing a disguise: it would drift silently the day the card
changes, and a typo in it would be invisible because both sides of the
comparison would then share it. Parsing keeps the reference and the simulation
reading from one source.

What this gate does and does not establish
------------------------------------------
Stated plainly, because it is the weakest analytic gate in this repo and it
would be easy to over-claim.

**It is a round-trip / consistency gate, not a symbolic derivation.** Unlike
Robinson's theorem or ``sigma = 4 pi alpha^2 / 3s``, there is no independent
physics closed form here; the reference is a fit parametrisation the card
happens to encode (the CMS card's cites arXiv:1211.4462). What is being proven
is that the measured rate reproduces the configured one -- i.e. that the
extraction, the flavour labelling, the binning and the efficiency estimator are
all right.

**The shipped flavour label is not independent of the tagger.** Delphes'
``BTagging`` keys on exactly the ``Jet.Flavor`` that ``JetFlavorAssociation``
writes. So histogramming ``Jet.Flavor`` against the tag bit and recovering the
card is, on its own, a closed loop: it cannot detect a wrong flavour *definition*,
only a wrong flavour *handling*. Breaking that loop needs a truth label derived
some other way -- a dR match of the jet to the generator-level heavy quarks --
which is why the pipeline emits the generator b/c quarks alongside the jets and
cross-checks the two labellings. See ``pipelines/pp_ttbar_btag/``.

The bin-average subtlety
------------------------
The configured efficiency varies across a p_T bin, and the jet spectrum inside
that bin is steeply falling, so the bin is *not* populated at its centre. The
quantity a measured rate converges to is the **mean of the formula over the jets
actually in the bin**,

    eps_expected = (1/N) sum_i f(pt_i, eta_i)

not ``f(bin_centre)``. Comparing against the bin centre is a real and quiet bias
-- it survives every "looks about right" plot inspection -- and the analytic
suite asserts it is measurably wrong on a falling spectrum while the jet-wise
mean closes. This is also what lets the same code gate a smooth ``tanh`` card and
a piecewise step-function card with no special-casing: step edges inside a bin
are averaged correctly by construction.

Why the pull uses the *expected* variance
-----------------------------------------
The binomial error is quoted as ``sqrt(eps_exp (1 - eps_exp) / N)`` -- the
variance under the hypothesis being tested -- rather than the observed
``sqrt(eps_obs (1 - eps_obs) / N)``. Two reasons: it is the correct null
variance for a goodness-of-fit pull, and the observed form degenerates to a zero
error (hence an infinite pull) whenever a bin happens to tag none or all of its
jets, which is common for the light-mistag formulas at ~1%.

**Units:** p_T and energy in GeV, ``eta``/``phi`` dimensionless -- the Delphes
convention, since the formulas are the card's.

TCL expression semantics honoured here
--------------------------------------
The card's expressions are evaluated with Delphes' reading of them, not
Python's:

* a comparison yields the number ``1``/``0``, so ``(pt > 20.0) * (0.15)`` is
  arithmetic -- the step-function cards are built entirely out of this;
* ``&&`` / ``||`` bind *looser* than the comparisons around them. They are
  translated to Python's ``and``/``or`` (which have the same loose precedence)
  and then evaluated element-wise, rather than to ``&``/``|``, which bind
  tighter than comparison and would silently reassociate the expression;
* ``^`` is exponentiation. This is Delphes' own formula parser (``TFormula``
  semantics), *not* stock TCL, where ``^`` is bitwise xor.

Evaluation is done by walking a parsed :mod:`ast` with an explicit node
whitelist -- no ``eval`` of card text.
"""

from __future__ import annotations

import ast
import pathlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "BOTTOM_FLAVOR",
    "CHARM_FLAVOR",
    "LIGHT_FLAVOR",
    "BTagWorkingPoint",
    "CardFormulaError",
    "EfficiencyPoint",
    "efficiency_vs_pt",
    "evaluate_card_formula",
    "measure_efficiency",
    "parse_btagging_working_points",
    "roc_points",
]

# Flavour keys as Delphes uses them: the |PDG| code of the hardest parton
# associated to the jet. Key 0 is the card's *default* formula -- it catches
# light quarks and gluons, i.e. it is the mistag rate.
LIGHT_FLAVOR: Final[int] = 0
CHARM_FLAVOR: Final[int] = 4
BOTTOM_FLAVOR: Final[int] = 5


# --------------------------------------------------------------------------
# The card's expression language
# --------------------------------------------------------------------------


def _reduce2(ufunc: np.ufunc) -> object:
    """Turn a binary ufunc into the card's variadic ``min``/``max``."""

    def _call(*args: ArrayLike) -> NDArray[np.float64]:
        if not args:
            raise ValueError("min/max needs at least one argument")
        out = np.asarray(args[0], dtype=float)
        for a in args[1:]:
            out = ufunc(out, np.asarray(a, dtype=float))
        return out

    return _call


# Functions a Delphes formula may call. Deliberately a closed list: anything
# outside it is a parse error, not a silent fallback to some Python builtin.
_FUNCS: Final[dict[str, object]] = {
    "abs": np.abs,
    "acos": np.arccos,
    "asin": np.arcsin,
    "atan": np.arctan,
    "ceil": np.ceil,
    "cos": np.cos,
    "cosh": np.cosh,
    "exp": np.exp,
    "floor": np.floor,
    "log": np.log,
    "log10": np.log10,
    "max": _reduce2(np.maximum),
    "min": _reduce2(np.minimum),
    "pow": np.power,
    "sin": np.sin,
    "sinh": np.sinh,
    "sqrt": np.sqrt,
    "tan": np.tan,
    "tanh": np.tanh,
}

# Jet kinematic variables a BTagging formula may reference.
_VARIABLES: Final[tuple[str, ...]] = ("pt", "eta", "phi", "energy")

# `!` used as logical negation, but never the `!=` operator, and never the
# second character of `<=` / `>=` / `==` (which cannot precede a `!` anyway --
# the lookbehind is belt-and-braces against a malformed card).
_BANG = re.compile(r"(?<![=!<>])!(?!=)")


def _to_python_source(expr: str) -> str:
    """Rewrite a Delphes/TCL formula into equivalent Python *source*.

    Only the surface syntax is changed here; the semantics (comparisons as 0/1,
    element-wise logic) are applied by the evaluator below.
    """
    # TCL line continuations: a trailing backslash joins the next line.
    src = re.sub(r"\\\s*\n", " ", expr)
    src = src.replace("\n", " ").replace("\t", " ")
    # `^` is exponentiation in Delphes' formula parser (TFormula), not xor.
    src = src.replace("^", "**")
    # Map the logical operators onto Python's, which share TCL's loose
    # precedence relative to comparison. Order matters: rewrite the two-character
    # operators before the one-character `!`.
    src = src.replace("&&", " and ").replace("||", " or ")
    src = _BANG.sub(" not ", src)
    return src.strip()


_ALLOWED_BINOPS: Final[dict[type, object]] = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
    ast.Mod: np.mod,
}

_ALLOWED_CMPOPS: Final[dict[type, object]] = {
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
}


class CardFormulaError(ValueError):
    """A card expression could not be parsed or evaluated."""


def _eval_node(node: ast.AST, variables: Mapping[str, NDArray[np.float64]]) -> NDArray[np.float64]:
    """Evaluate one whitelisted AST node to a float array."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, variables)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CardFormulaError(f"non-numeric constant in formula: {node.value!r}")
        return np.asarray(float(node.value))

    if isinstance(node, ast.Name):
        try:
            return variables[node.id]
        except KeyError:
            known = ", ".join(sorted(variables))
            raise CardFormulaError(
                f"formula references unknown variable {node.id!r} (known: {known})"
            ) from None

    if isinstance(node, ast.BinOp):
        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise CardFormulaError(
                f"operator {type(node.op).__name__} not allowed in a card formula"
            )
        return op(_eval_node(node.left, variables), _eval_node(node.right, variables))  # type: ignore[operator]

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, variables)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return np.negative(operand)
        if isinstance(node.op, ast.Not):
            return (operand == 0).astype(float)
        raise CardFormulaError(f"unary {type(node.op).__name__} not allowed in a card formula")

    if isinstance(node, ast.BoolOp):
        # TCL `&&` / `||`: operands are truthy-if-nonzero, result is 1/0.
        parts = [_eval_node(v, variables) != 0 for v in node.values]
        combine = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        out = parts[0]
        for p in parts[1:]:
            out = combine(out, p)
        return np.asarray(out, dtype=float)

    if isinstance(node, ast.Compare):
        # A comparison evaluates to the *number* 1 or 0, which is what makes the
        # step-function cards arithmetic. Chained comparisons are AND-folded.
        left = _eval_node(node.left, variables)
        result: NDArray[np.bool_] | None = None
        for op_node, right_node in zip(node.ops, node.comparators, strict=True):
            op = _ALLOWED_CMPOPS.get(type(op_node))
            if op is None:
                raise CardFormulaError(
                    f"comparison {type(op_node).__name__} not allowed in a card formula"
                )
            right = _eval_node(right_node, variables)
            this = np.asarray(op(left, right), dtype=bool)  # type: ignore[operator]
            result = this if result is None else np.logical_and(result, this)
            left = right
        assert result is not None
        return np.asarray(result, dtype=float)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise CardFormulaError("only plain function calls are allowed in a card formula")
        func = _FUNCS.get(node.func.id)
        if func is None:
            raise CardFormulaError(f"unknown function {node.func.id!r} in a card formula")
        if node.keywords:
            raise CardFormulaError("keyword arguments are not allowed in a card formula")
        return func(*[_eval_node(a, variables) for a in node.args])  # type: ignore[operator]

    raise CardFormulaError(f"{type(node).__name__} is not allowed in a card formula")


def evaluate_card_formula(
    expr: str,
    *,
    pt: ArrayLike,
    eta: ArrayLike = 0.0,
    phi: ArrayLike = 0.0,
    energy: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Evaluate a Delphes efficiency formula over jet kinematics.

    Parameters
    ----------
    expr:
        The formula text exactly as it appears in the card (line continuations
        and all).
    pt, eta, phi, energy:
        Jet kinematics, broadcast against each other. ``energy`` defaults to
        ``pt * cosh(eta)`` (the massless value) so that a card referencing it
        still evaluates; no BTagging formula in the shipped cards does.

    Returns
    -------
    The formula's value, broadcast to the shape of the inputs.
    """
    pt_a = np.asarray(pt, dtype=float)
    eta_a = np.asarray(eta, dtype=float)
    phi_a = np.asarray(phi, dtype=float)
    energy_a = pt_a * np.cosh(eta_a) if energy is None else np.asarray(energy, dtype=float)
    variables = {"pt": pt_a, "eta": eta_a, "phi": phi_a, "energy": energy_a}

    source = _to_python_source(expr)
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - malformed card
        raise CardFormulaError(f"could not parse card formula {expr!r}: {exc}") from exc
    value = _eval_node(tree, variables)
    return np.broadcast_to(np.asarray(value, dtype=float), np.broadcast(pt_a, eta_a).shape).copy()


# --------------------------------------------------------------------------
# Working points
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BTagWorkingPoint:
    """One ``module BTagging`` block: a name, an output bit, and the formulas.

    ``formulas`` maps ``|PDG|`` flavour code to the card's expression for that
    flavour. Key ``0`` is the card's default -- the mistag rate applied to any
    flavour without its own formula (light quarks and gluons).
    """

    name: str
    bit_number: int
    formulas: Mapping[int, str] = field(default_factory=dict)

    def formula_for(self, flavor: int) -> str:
        """The expression the card applies to ``flavor`` (default if unlisted)."""
        key = abs(int(flavor))
        if key in self.formulas:
            return self.formulas[key]
        try:
            return self.formulas[LIGHT_FLAVOR]
        except KeyError:
            raise CardFormulaError(
                f"working point {self.name!r} has no formula for flavour {flavor} "
                "and no default (flavour 0) formula"
            ) from None

    def efficiency(
        self,
        flavor: ArrayLike,
        pt: ArrayLike,
        eta: ArrayLike = 0.0,
    ) -> NDArray[np.float64]:
        """The configured tagging probability of each jet.

        Jets are grouped by ``|flavor|`` so each formula is evaluated once over
        its own subset -- vectorised, and it keeps a per-flavour formula from
        ever being applied to the wrong jets.
        """
        flav = np.abs(np.asarray(flavor, dtype=int))
        pt_a, eta_a = np.broadcast_arrays(np.asarray(pt, dtype=float), np.asarray(eta, dtype=float))
        if flav.shape != pt_a.shape:
            flav = np.broadcast_to(flav, pt_a.shape)
        out = np.empty(pt_a.shape, dtype=float)
        for key in np.unique(flav):
            sel = flav == key
            out[sel] = evaluate_card_formula(
                self.formula_for(int(key)), pt=pt_a[sel], eta=eta_a[sel]
            )
        return out

    def tagged(self, btag_bits: ArrayLike) -> NDArray[np.bool_]:
        """Decode this working point's bit out of Delphes' ``Jet.BTag`` mask.

        ``BTag`` is a bit *mask*, not a boolean: a card with three working
        points packs Loose/Medium/Tight into bits 0/1/2 of the same integer, so
        ``BTag == 1`` would mean "loose but not medium" rather than "loose".
        """
        bits = np.asarray(btag_bits, dtype=np.int64)
        return ((bits >> int(self.bit_number)) & 1).astype(bool)


# --------------------------------------------------------------------------
# Card parsing
# --------------------------------------------------------------------------

_COMMENT_LINE = re.compile(r"^\s*#")
_SOURCE_LINE = re.compile(r"^\s*source\s+(\S+)\s*$")
_MODULE_HEAD = re.compile(r"\bmodule\s+BTagging\s+(\w+)\s*\{")
_BIT_NUMBER = re.compile(r"\bset\s+BitNumber\s+(\d+)")


def _resolve_sources(path: pathlib.Path, _seen: frozenset[pathlib.Path] = frozenset()) -> str:
    """Read a card, inlining ``source`` includes, and drop whole-line comments.

    Includes resolve relative to the *including* file's directory, which is how
    the Delphes cards are laid out (``CMS_PhaseII_0PU.tcl`` sits beside the
    ``btagLoose.tcl`` it sources). Comment lines are dropped **before** the
    module scan so the cards' commented-out ``# add EfficiencyFormula ...``
    template lines are never mistaken for configuration.
    """
    path = path.resolve()
    if path in _seen:  # pragma: no cover - malformed card
        raise CardFormulaError(f"circular `source` include at {path}")
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if _COMMENT_LINE.match(raw):
            continue
        inc = _SOURCE_LINE.match(raw)
        if inc:
            child = (path.parent / inc.group(1)).resolve()
            if child.is_file():
                lines.append(_resolve_sources(child, _seen | {path}))
                continue
            # A card may source something unrelated to b-tagging that is not
            # shipped; that must not break parsing of the blocks we do want.
            lines.append(f"# unresolved source: {inc.group(1)}")
            continue
        lines.append(raw)
    return "\n".join(lines)


def _brace_block(text: str, open_index: int) -> tuple[str, int]:
    """Return the text inside the brace at ``open_index`` and the index after it.

    A plain ``text.index('}')`` would stop at the first closing brace, which for
    ``add EfficiencyFormula {5} {...}`` is the *flavour* key -- so the block must
    be scanned with a depth counter.
    """
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : i], i + 1
    raise CardFormulaError("unbalanced braces in card")


def _parse_efficiency_formulas(block: str) -> dict[int, str]:
    """Pull every ``add EfficiencyFormula {code} {expr}`` out of a module block."""
    formulas: dict[int, str] = {}
    pos = 0
    needle = "add EfficiencyFormula"
    while True:
        found = block.find(needle, pos)
        if found < 0:
            return formulas
        cursor = found + len(needle)
        key_open = block.index("{", cursor)
        key_text, cursor = _brace_block(block, key_open)
        expr_open = block.index("{", cursor)
        expr_text, pos = _brace_block(block, expr_open)
        formulas[abs(int(key_text.strip()))] = expr_text.strip()


def parse_btagging_working_points(
    card: str | pathlib.Path,
) -> tuple[BTagWorkingPoint, ...]:
    """Parse every ``module BTagging`` block of a Delphes card.

    Returns the working points in **bit-number order**, which for the
    multi-working-point cards is loose -> medium -> tight (loosest first).
    """
    text = _resolve_sources(pathlib.Path(card))
    points: list[BTagWorkingPoint] = []
    for head in _MODULE_HEAD.finditer(text):
        name = head.group(1)
        block, _ = _brace_block(text, text.index("{", head.start()))
        bit = _BIT_NUMBER.search(block)
        if bit is None:  # pragma: no cover - malformed card
            raise CardFormulaError(f"module BTagging {name} has no `set BitNumber`")
        formulas = _parse_efficiency_formulas(block)
        if not formulas:  # pragma: no cover - malformed card
            raise CardFormulaError(f"module BTagging {name} has no EfficiencyFormula")
        points.append(BTagWorkingPoint(name=name, bit_number=int(bit.group(1)), formulas=formulas))
    if not points:
        raise CardFormulaError(f"no `module BTagging` block found in {card}")
    return tuple(sorted(points, key=lambda p: p.bit_number))


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EfficiencyPoint:
    """A measured tagging rate beside the rate the card configured."""

    n_jets: int
    n_tagged: int
    measured: float
    expected: float
    error: float
    pt_low: float = float("nan")
    pt_high: float = float("nan")

    @property
    def pull(self) -> float:
        """``(measured - expected) / error``; ``nan`` for an empty/degenerate bin."""
        if not np.isfinite(self.error) or self.error <= 0.0:
            return float("nan")
        return (self.measured - self.expected) / self.error

    @property
    def expected_tags(self) -> float:
        """How many tagged jets the card predicts here."""
        return self.n_jets * self.expected

    @property
    def gaussian_valid(self) -> bool:
        """Whether the pull is meaningfully Gaussian in this bin.

        The binomial variance ``N p (1-p)`` is what sets the pull's scale, and
        the Gaussian approximation to it needs that quantity to be comfortably
        above 1 -- **not** merely a lot of jets. The distinction bites exactly
        where the tight working points live: a bin can hold thousands of jets
        and still expect only ~1 tag at a 0.1% mistag rate, and there the count
        is Poisson, the achievable pulls are discrete, and their mean square is
        not 1. Averaging such bins into a chi-square inflates it and invites the
        wrong diagnosis -- a threshold nudge instead of a fix.

        Gating on jet count alone would let those bins in; this gates on the
        variance itself, which is the quantity the approximation is about.
        """
        return bool(self.n_jets * self.expected * (1.0 - self.expected) >= 10.0)


def measure_efficiency(
    tagged: ArrayLike,
    expected: ArrayLike,
    *,
    pt_low: float = float("nan"),
    pt_high: float = float("nan"),
) -> EfficiencyPoint:
    """Compare a set of tag decisions against the card's configured rate.

    ``expected`` is the *per-jet* configured efficiency of the same jets; the
    reference is its mean over them (see the module docstring on why this, and
    not the formula at a bin centre). The error is the binomial width under that
    expectation.
    """
    tag = np.asarray(tagged, dtype=bool).ravel()
    exp = np.asarray(expected, dtype=float).ravel()
    if exp.shape != tag.shape:
        exp = np.broadcast_to(exp, tag.shape)
    n = int(tag.size)
    if n == 0:
        return EfficiencyPoint(0, 0, float("nan"), float("nan"), float("nan"), pt_low, pt_high)
    n_tag = int(np.count_nonzero(tag))
    mean_exp = float(np.mean(exp))
    error = float(np.sqrt(mean_exp * (1.0 - mean_exp) / n))
    return EfficiencyPoint(n, n_tag, n_tag / n, mean_exp, error, pt_low, pt_high)


def efficiency_vs_pt(
    wp: BTagWorkingPoint,
    *,
    flavor: ArrayLike,
    pt: ArrayLike,
    eta: ArrayLike,
    btag_bits: ArrayLike,
    pt_edges: Sequence[float],
    select_flavor: int,
) -> list[EfficiencyPoint]:
    """Measured-vs-configured tagging rate in p_T bins, for one jet flavour.

    ``select_flavor`` picks the jets by their truth ``|flavor|``; pass
    :data:`LIGHT_FLAVOR` for the mistag rate, which selects every jet whose
    flavour the card has no dedicated formula for.
    """
    flav = np.abs(np.asarray(flavor, dtype=int)).ravel()
    pt_a = np.asarray(pt, dtype=float).ravel()
    eta_a = np.broadcast_to(np.asarray(eta, dtype=float).ravel(), pt_a.shape)
    tag = wp.tagged(np.asarray(btag_bits).ravel())

    want = abs(int(select_flavor))
    if want == LIGHT_FLAVOR:
        specific = {k for k in wp.formulas if k != LIGHT_FLAVOR}
        keep = ~np.isin(flav, sorted(specific))
    else:
        keep = flav == want

    expected = wp.efficiency(flav, pt_a, eta_a)
    out: list[EfficiencyPoint] = []
    for low, high in zip(pt_edges[:-1], pt_edges[1:], strict=True):
        sel = keep & (pt_a >= low) & (pt_a < high)
        out.append(
            measure_efficiency(tag[sel], expected[sel], pt_low=float(low), pt_high=float(high))
        )
    return out


def roc_points(
    working_points: Iterable[BTagWorkingPoint],
    *,
    flavor: ArrayLike,
    pt: ArrayLike,
    eta: ArrayLike,
    btag_bits: ArrayLike,
    signal_flavor: int = BOTTOM_FLAVOR,
) -> list[tuple[str, EfficiencyPoint, EfficiencyPoint]]:
    """The (mistag, efficiency) operating points of a multi-working-point card.

    One entry per working point: ``(name, background_point, signal_point)``,
    where the background is the card's default-formula (light/gluon) jets. This
    is an *operating-point* ROC -- the discrete points a card actually offers --
    not a continuous discriminant sweep, which Delphes does not expose (it stores
    a decision bit, never a discriminant value).
    """
    edges = [float(np.min(np.asarray(pt, dtype=float))), float("inf")]
    out: list[tuple[str, EfficiencyPoint, EfficiencyPoint]] = []
    for wp in working_points:
        sig = efficiency_vs_pt(
            wp,
            flavor=flavor,
            pt=pt,
            eta=eta,
            btag_bits=btag_bits,
            pt_edges=edges,
            select_flavor=signal_flavor,
        )[0]
        bkg = efficiency_vs_pt(
            wp,
            flavor=flavor,
            pt=pt,
            eta=eta,
            btag_bits=btag_bits,
            pt_edges=edges,
            select_flavor=LIGHT_FLAVOR,
        )[0]
        out.append((wp.name, bkg, sig))
    return out
