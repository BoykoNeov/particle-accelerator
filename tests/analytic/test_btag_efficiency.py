r"""Analytic gate for the b-tagging efficiency extraction (milestone E2).

The reference is the Delphes card: it *configures* a tagging probability for
every jet, so there is a known right answer and a correct analysis must recover
it. These gates are layered so that a wrong card model and a wrong efficiency
estimator cannot cancel:

1. **Parsing** -- a hand-written card, whose formulas the test therefore knows
   exactly, round-trips through the parser (including a ``source`` include, the
   nested ``{flavour} {expr}`` braces, and commented-out template lines).
2. **Evaluation** -- the expression evaluator is checked against **sympy**, an
   independent authority that shares no code with it, on the real CMS formulas;
   and the TCL semantics that a naive translation gets wrong (comparison-as-0/1,
   ``&&`` precedence) are pinned with expressions that *discriminate*.
3. **Round-trip** -- synthetic jets tagged by a Bernoulli draw at the card's own
   efficiency are measured back, asserted as a **unit-width pull distribution**
   rather than "within error", since "within error" is trivially bought by
   inflating the error.
4. **Traps** -- the two mistakes that reproduce a plausible plot: flavour-blind
   pooling, and comparing against the formula at a bin centre. Both are asserted
   to be *measurably wrong* on the same input on which the correct treatment
   closes, so the gate cannot be passed by a method that does not discriminate.
5. **Anchors** -- the physics orderings ``eps_b > eps_c > eps_light`` and
   ``eps_loose > eps_medium > eps_tight``, which no correct card can violate.
"""

from __future__ import annotations

import ast
import json
import pathlib

import numpy as np
import pytest
import sympy as sp

from accsim.events.btag import (
    BOTTOM_FLAVOR,
    CHARM_FLAVOR,
    LIGHT_FLAVOR,
    BTagWorkingPoint,
    CardFormulaError,
    efficiency_vs_pt,
    evaluate_card_formula,
    measure_efficiency,
    parse_btagging_working_points,
    roc_points,
)

# The real CMS card's formulas, verbatim (arXiv:1211.4462). These are test
# *input*, not a transcribed reference: both the evaluator under test and the
# independent sympy evaluator are handed this same string, so a typo here cannot
# make the comparison vacuous -- it would still compare two evaluators.
CMS_LIGHT = "0.01+0.000038*pt"
CMS_CHARM = "0.25*tanh(0.018*pt)*(1/(1+ 0.0013*pt))"
CMS_BOTTOM = "0.85*tanh(0.0025*pt)*(25.0/(1+0.063*pt))"


# ---------------------------------------------------------------------------
# 1. Parsing
# ---------------------------------------------------------------------------


def _write_card(tmp_path: pathlib.Path) -> pathlib.Path:
    """A miniature multi-working-point card mimicking CMS_PhaseII's layout.

    Deliberately exercises every structural feature of the real cards: a
    ``source`` include *inside* a module block, the nested ``{code} {expr}``
    braces, a commented-out template line, out-of-order bit numbers, and a
    piecewise step formula alongside a smooth one.
    """
    (tmp_path / "btag_tight.tcl").write_text(
        "  add EfficiencyFormula {0} {0.001}\n"
        "  add EfficiencyFormula {5} {(pt <= 30.0) * (0.20) + \\\n"
        "                             (pt > 30.0 && pt <= 100.0) * (0.45) + \\\n"
        "                             (pt > 100.0) * (0.30)\n"
        "  }\n",
        encoding="utf-8",
    )
    card = tmp_path / "mini_card.tcl"
    card.write_text(
        "# a comment line that must be ignored\n"
        "module BTagging BTaggingTight {\n"
        "  set JetInputArray JetEnergyScale/jets\n"
        "  set BitNumber 2\n"
        "  source btag_tight.tcl\n"
        "}\n"
        "\n"
        "module BTagging BTaggingLoose {\n"
        "  set BitNumber 0\n"
        "  # add EfficiencyFormula {99} {1.0}\n"
        f"  add EfficiencyFormula {{0}} {{{CMS_LIGHT}}}\n"
        f"  add EfficiencyFormula {{4}} {{{CMS_CHARM}}}\n"
        f"  add EfficiencyFormula {{5}} {{{CMS_BOTTOM}}}\n"
        "}\n",
        encoding="utf-8",
    )
    return card


def test_parser_recovers_the_written_card(tmp_path: pathlib.Path) -> None:
    """Every formula, bit number and name comes back exactly as written."""
    wps = parse_btagging_working_points(_write_card(tmp_path))

    # Returned loosest-first, by bit number -- NOT in file order (Tight is first
    # in the file), which is what makes this an ordering assertion.
    assert [w.name for w in wps] == ["BTaggingLoose", "BTaggingTight"]
    assert [w.bit_number for w in wps] == [0, 2]

    loose, tight = wps
    assert loose.formulas[LIGHT_FLAVOR] == CMS_LIGHT
    assert loose.formulas[CHARM_FLAVOR] == CMS_CHARM
    assert loose.formulas[BOTTOM_FLAVOR] == CMS_BOTTOM

    # The commented-out template must not have been parsed as configuration.
    assert 99 not in loose.formulas

    # The `source`d block was inlined: Tight's formulas live in the other file.
    assert set(tight.formulas) == {LIGHT_FLAVOR, BOTTOM_FLAVOR}
    assert tight.formulas[LIGHT_FLAVOR] == "0.001"


def test_parser_survives_an_unresolvable_source(tmp_path: pathlib.Path) -> None:
    """A missing include unrelated to b-tagging must not break the b-tag parse."""
    card = tmp_path / "c.tcl"
    card.write_text(
        "module MomentumSmearing X {\n  source not_shipped.tcl\n}\n"
        "module BTagging B {\n  set BitNumber 0\n"
        "  add EfficiencyFormula {0} {0.02}\n}\n",
        encoding="utf-8",
    )
    (wp,) = parse_btagging_working_points(card)
    assert wp.formulas[LIGHT_FLAVOR] == "0.02"


def test_parser_rejects_a_card_with_no_btagging(tmp_path: pathlib.Path) -> None:
    card = tmp_path / "c.tcl"
    card.write_text("module Efficiency E {\n  set X 1\n}\n", encoding="utf-8")
    with pytest.raises(CardFormulaError, match="no `module BTagging`"):
        parse_btagging_working_points(card)


# ---------------------------------------------------------------------------
# 2. Evaluation -- against sympy, and on the semantics a naive port gets wrong
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("expr", [CMS_LIGHT, CMS_CHARM, CMS_BOTTOM])
def test_evaluator_matches_sympy(expr: str) -> None:
    """The smooth CMS formulas agree with an independent symbolic evaluator.

    sympy shares no code with the ast walker under test, so this pins the
    arithmetic, the operator precedence and the ``tanh`` mapping at once.
    """
    pt_sym = sp.Symbol("pt", positive=True)
    reference = sp.lambdify(pt_sym, sp.sympify(expr, locals={"pt": pt_sym}), "numpy")

    pt = np.geomspace(15.0, 3000.0, 200)
    assert np.allclose(evaluate_card_formula(expr, pt=pt), reference(pt), rtol=0, atol=1e-14)


def test_evaluator_matches_delphes_own_evaluator() -> None:
    """The authority gate: agree with **Delphes' own** formula evaluator.

    sympy covers the smooth ``delphes_card_CMS.tcl`` expressions, but the card the
    pipeline actually runs (``CMS_PhaseII_0PU``) uses long piecewise step
    functions with eta splits and open-ended top bins -- a different animal, and
    the one where a chained-comparison or precedence slip would hide.

    The reference values were produced by ``DelphesFormula`` (a ``TFormula``
    subclass -- literally the class Delphes' ``BTagging`` module evaluates with)
    running inside the Delphes image, then frozen into
    ``data/delphes_formula_reference.json`` so this runs in CI without Docker.
    The grid deliberately lands **on** the card's step edges (pt 20/30/100/1000,
    |eta| 1.8/2.4/3.4), where an off-by-one between ``<`` and ``<=``, or a gap
    between adjacent ranges, would show up as a wrong value rather than a wrong
    shape.

    Agreement is required to be **exact**: both sides evaluate the same
    arithmetic in IEEE double precision, so anything above zero is a semantic
    difference, not a rounding one.
    """
    doc = json.loads(
        (pathlib.Path(__file__).parent / "data" / "delphes_formula_reference.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(doc["formulas"]) == 9  # 3 working points x {light, c, b}

    worst = 0.0
    for entry in doc["formulas"]:
        pts = np.array(entry["points"], dtype=float)
        got = evaluate_card_formula(entry["formula"], pt=pts[:, 0], eta=pts[:, 1])
        worst = max(worst, float(np.max(np.abs(got - pts[:, 2]))))
        # A formula that evaluated to a constant everywhere would match trivially
        # if the reference were also constant; require the reference to vary.
        assert pts[:, 2].max() > pts[:, 2].min(), f"{entry['label']} is constant"
    assert worst == 0.0, f"disagrees with DelphesFormula by {worst:.3e}"


def test_comparison_evaluates_to_one_or_zero() -> None:
    """TCL comparisons are *numbers*: the step-function cards are arithmetic.

    If a comparison returned a Python bool that then OR-ed under ``+``, the
    two-term sum below would collapse to 1.0 instead of the step values.
    """
    expr = "(pt <= 30.0) * (0.20) + (pt > 30.0) * (0.45)"
    got = evaluate_card_formula(expr, pt=np.array([10.0, 30.0, 30.001, 100.0]))
    assert np.allclose(got, [0.20, 0.20, 0.45, 0.45])


def test_every_comparison_operator_is_exact_on_its_boundary() -> None:
    """All six comparisons, checked *at* the boundary value.

    The shipped cards happen to use only ``<``/``<=``/``>``, so the frozen
    DelphesFormula reference cannot exercise ``>=``, ``==`` or ``!=`` -- a
    swapped operator in the evaluator's table would go unnoticed there. Each is
    therefore pinned directly, on the one input that distinguishes it from its
    neighbour: the boundary itself.
    """
    at = np.array([29.0, 30.0, 31.0])
    assert np.allclose(evaluate_card_formula("pt < 30.0", pt=at), [1.0, 0.0, 0.0])
    assert np.allclose(evaluate_card_formula("pt <= 30.0", pt=at), [1.0, 1.0, 0.0])
    assert np.allclose(evaluate_card_formula("pt > 30.0", pt=at), [0.0, 0.0, 1.0])
    assert np.allclose(evaluate_card_formula("pt >= 30.0", pt=at), [0.0, 1.0, 1.0])
    assert np.allclose(evaluate_card_formula("pt == 30.0", pt=at), [0.0, 1.0, 0.0])
    assert np.allclose(evaluate_card_formula("pt != 30.0", pt=at), [1.0, 0.0, 1.0])


def test_bare_comparisons_add_arithmetically_not_as_logical_or() -> None:
    """``1 + 1 == 2``, not ``1``.

    Every term in the shipped cards is multiplied by a float before being
    summed, which *hides* this: ``bool * float`` is already a float. But numpy's
    ``bool + bool`` is logical OR, so if a comparison returned a bool array a
    card summing two bare conditions would silently saturate at 1. Pinning the
    documented TCL semantics directly, rather than relying on the shipped cards
    happening not to trip over it.
    """
    got = evaluate_card_formula("(pt > 10.0) + (pt > 20.0)", pt=np.array([5.0, 15.0, 25.0]))
    assert np.allclose(got, [0.0, 1.0, 2.0])


def test_logical_and_binds_looser_than_comparison() -> None:
    """``&&`` must not be ported to ``&``, which binds *tighter* than comparison.

    For ``pt > 30 && pt <= 100``, a ``&``-translation parses as
    ``pt > (30 & pt) <= 100`` and gives a different answer. This expression is
    chosen so the two readings actually disagree, rather than merely being
    written differently.
    """
    pt = np.array([10.0, 50.0, 200.0])
    # Unparenthesised, so precedence is the only thing deciding the answer.
    got = evaluate_card_formula("(pt > 30.0 && pt <= 100.0) * 1.0", pt=pt)
    assert np.allclose(got, [0.0, 1.0, 0.0])
    assert np.allclose(evaluate_card_formula("pt > 30.0 && pt <= 100.0", pt=pt), [0.0, 1.0, 0.0])

    # Show the rejected translation really does reassociate, by comparing the
    # PARSE TREES rather than numbers -- a numeric comparison can agree by
    # coincidence on a particular sample, a tree shape cannot.
    and_tree = ast.parse("pt > 30.0 and pt <= 100.0", mode="eval").body
    amp_tree = ast.parse("pt > 30.0 & pt <= 100.0", mode="eval").body
    assert isinstance(and_tree, ast.BoolOp), "`and` must give (a>b) AND (c<=d)"
    # `&` binds tighter than comparison, so Python reads ONE chained comparison
    # `pt > (30.0 & pt) <= 100.0` -- a different expression entirely.
    assert isinstance(amp_tree, ast.Compare)
    assert len(amp_tree.ops) == 2
    assert isinstance(amp_tree.comparators[0], ast.BinOp)


def test_caret_is_exponentiation_not_xor() -> None:
    """Delphes' formula parser reads ``^`` as pow (TFormula), unlike stock TCL."""
    assert np.allclose(evaluate_card_formula("pt^2", pt=np.array([3.0])), [9.0])


def test_evaluator_refuses_anything_outside_the_whitelist() -> None:
    """Card text is walked, never ``eval``-ed: unknown names/calls are errors."""
    with pytest.raises(CardFormulaError, match="unknown function"):
        evaluate_card_formula("os(pt)", pt=np.array([1.0]))
    with pytest.raises(CardFormulaError, match="unknown variable"):
        evaluate_card_formula("nonsense * pt", pt=np.array([1.0]))


def test_bitmask_decoding_separates_working_points() -> None:
    """``Jet.BTag`` is a mask: ``== 1`` would mean "loose but not medium"."""
    loose = BTagWorkingPoint("L", 0, {LIGHT_FLAVOR: "0.1"})
    medium = BTagWorkingPoint("M", 1, {LIGHT_FLAVOR: "0.01"})
    tight = BTagWorkingPoint("T", 2, {LIGHT_FLAVOR: "0.001"})

    bits = np.array([0b000, 0b001, 0b011, 0b111, 0b100])
    assert list(loose.tagged(bits)) == [False, True, True, True, False]
    assert list(medium.tagged(bits)) == [False, False, True, True, False]
    assert list(tight.tagged(bits)) == [False, False, False, True, True]


# ---------------------------------------------------------------------------
# 3. Round-trip: recover the card from Bernoulli-tagged synthetic jets
# ---------------------------------------------------------------------------


def _cms_working_point() -> BTagWorkingPoint:
    return BTagWorkingPoint(
        name="BTagging",
        bit_number=0,
        formulas={LIGHT_FLAVOR: CMS_LIGHT, CHARM_FLAVOR: CMS_CHARM, BOTTOM_FLAVOR: CMS_BOTTOM},
    )


def _falling_spectrum(rng: np.random.Generator, n: int, *, pt_min: float = 20.0) -> np.ndarray:
    """A steeply falling jet p_T spectrum -- the realistic case, and the one in
    which a bin is *not* populated at its centre."""
    return pt_min * (1.0 + rng.exponential(scale=0.6, size=n)) ** 3


def _simulate(
    wp: BTagWorkingPoint, rng: np.random.Generator, flavor: np.ndarray, pt: np.ndarray
) -> np.ndarray:
    """Tag each jet with a Bernoulli draw at the card's configured probability --
    exactly what Delphes' BTagging module does."""
    eta = np.zeros_like(pt)
    return (rng.random(pt.shape) < wp.efficiency(flavor, pt, eta)).astype(np.int64)


@pytest.mark.parametrize("flavour", [BOTTOM_FLAVOR, CHARM_FLAVOR, LIGHT_FLAVOR])
def test_measured_efficiency_pull_is_unit_width(flavour: int) -> None:
    """The round-trip gate, asserted as a pull distribution.

    "Agrees within its error" is satisfiable by inflating the error, so the
    assertion is on the *width*: over independent pseudo-experiments the pull
    must be a unit Gaussian. Both a biased extraction (mean shifts) and a
    mis-stated error (width moves) fail.
    """
    wp = _cms_working_point()
    edges = [30.0, 60.0, 120.0, 400.0]
    pulls: list[float] = []

    for seed in range(60):
        rng = np.random.default_rng(20260720 + seed)
        pt = _falling_spectrum(rng, 8000)
        flavor = np.full(pt.size, flavour)
        bits = _simulate(wp, rng, flavor, pt)
        for point in efficiency_vs_pt(
            wp,
            flavor=flavor,
            pt=pt,
            eta=np.zeros_like(pt),
            btag_bits=bits,
            pt_edges=edges,
            select_flavor=flavour,
        ):
            if point.n_jets > 200:
                pulls.append(point.pull)

    p = np.array(pulls)
    assert np.all(np.isfinite(p))
    assert p.size > 100
    # Unbiased, unit width. Tolerances are ~3 sigma of the sample mean/std at
    # this sample size, not free parameters.
    assert abs(p.mean()) < 0.25, f"biased extraction: mean pull {p.mean():.3f}"
    assert 0.80 < p.std(ddof=1) < 1.20, f"mis-stated error: pull width {p.std(ddof=1):.3f}"


def test_zero_tag_bin_still_has_a_usable_error() -> None:
    """The error is the variance under the *expectation*, not under the observation.

    A ~0.1% mistag rate in a modest bin routinely tags nothing. The observed
    binomial width is then exactly zero -- an infinite pull, from a bin that is
    in perfect agreement. Quoting the expected width instead keeps the bin
    finite and correctly weighted, which is the whole reason for the choice.
    """
    wp = BTagWorkingPoint("T", 0, {LIGHT_FLAVOR: "0.001"})
    n = 500
    point = measure_efficiency(
        np.zeros(n, dtype=bool), wp.efficiency(np.zeros(n, int), np.full(n, 50.0))
    )

    assert point.n_tagged == 0
    assert point.measured == 0.0
    assert point.error > 0.0, "a zero-tag bin must still carry an error"
    assert np.isfinite(point.pull), "a zero-tag bin must not produce an infinite pull"
    # 0 out of 500 at p=0.001 is an unremarkable outcome, not a discrepancy.
    assert abs(point.pull) < 1.0


def test_gaussian_validity_gates_on_variance_not_jet_count() -> None:
    """A bin is Gaussian when ``N p (1-p)`` is large -- not when ``N`` is.

    The two criteria come apart exactly where the tight working points live. A
    bin can hold thousands of jets and still expect a handful of tags at a 0.1%
    mistag rate; its count is Poisson and its pull is not Gaussian, so folding it
    into a chi-square inflates the chi-square and invites the wrong diagnosis --
    loosening a threshold instead of fixing the statistic.
    """
    # Plenty of jets, ~3 expected tags: NOT Gaussian, despite N = 3000.
    poissonish = measure_efficiency(np.zeros(3000, dtype=bool), np.full(3000, 0.001))
    assert poissonish.n_jets == 3000
    assert poissonish.expected_tags < 5.0
    assert not poissonish.gaussian_valid

    # Same jet count at a realistic b-tag efficiency: hundreds of tags, fine.
    assert measure_efficiency(np.zeros(3000, dtype=bool), np.full(3000, 0.5)).gaussian_valid

    # And FEW jets at a high efficiency is fine -- it is the variance that counts.
    # This is the case a jet-count floor would wrongly reject.
    small = measure_efficiency(np.zeros(40, dtype=bool), np.full(40, 0.5))
    assert small.n_jets < 50
    assert small.gaussian_valid


# ---------------------------------------------------------------------------
# 4. Traps -- the two mistakes that still produce a convincing plot
# ---------------------------------------------------------------------------


def test_light_selection_takes_only_default_formula_jets() -> None:
    """The mistag sample must be the jets the card has *no* formula for.

    Worth asserting on its own: because ``expected`` is built per jet from the
    true flavour, a selection that wrongly swept in b and c jets would move
    measured and expected *together* and still close as a pull. Only the
    composition reveals it -- so this gate checks what was selected, not how
    well it agreed.
    """
    wp = _cms_working_point()
    rng = np.random.default_rng(99)
    # Gluons (21) and strange (3) have no dedicated formula, so they are mistag
    # jets exactly like the light quarks -- the documented default-formula rule.
    flavor = np.array([BOTTOM_FLAVOR] * 400 + [CHARM_FLAVOR] * 300 + [21] * 500 + [3] * 200)
    pt = rng.uniform(40.0, 60.0, flavor.size)
    eta = np.zeros_like(pt)
    bits = _simulate(wp, rng, flavor, pt)

    kwargs = {"flavor": flavor, "pt": pt, "eta": eta, "btag_bits": bits, "pt_edges": [0.0, 1e9]}
    light = efficiency_vs_pt(wp, select_flavor=LIGHT_FLAVOR, **kwargs)[0]
    bottom = efficiency_vs_pt(wp, select_flavor=BOTTOM_FLAVOR, **kwargs)[0]

    assert light.n_jets == 700, "gluon and strange jets are mistag jets; b and c are not"
    assert bottom.n_jets == 400
    # And the two populations are nowhere near each other -- contamination would show.
    assert light.expected < 0.05 < 0.5 < bottom.expected
    assert abs(light.pull) < 3.0
    assert abs(bottom.pull) < 3.0


def test_flavour_blind_pooling_is_measurably_wrong() -> None:
    """Pooling flavours must fail where the flavour-resolved measurement closes.

    Delphes' BTagging keys on the jet's flavour, so a measurement that ignores
    the truth label reports a mixture rate and attributes it to one flavour. On
    a mixed sample that is off by many sigma -- and it is exactly the failure
    that plots perfectly well.
    """
    wp = _cms_working_point()
    rng = np.random.default_rng(20260720)
    n = 40_000
    pt = _falling_spectrum(rng, n)
    # A realistic ttbar-like mixture: some b, some c, mostly light.
    flavor = rng.choice([BOTTOM_FLAVOR, CHARM_FLAVOR, LIGHT_FLAVOR], size=n, p=[0.25, 0.15, 0.60])
    bits = _simulate(wp, rng, flavor, pt)
    eta = np.zeros_like(pt)
    edges = [30.0, 1e9]

    resolved = efficiency_vs_pt(
        wp,
        flavor=flavor,
        pt=pt,
        eta=eta,
        btag_bits=bits,
        pt_edges=edges,
        select_flavor=BOTTOM_FLAVOR,
    )[0]
    assert abs(resolved.pull) < 3.0, "the correct, flavour-resolved measurement must close"

    # The flavour-blind mistake: measure the tag rate of ALL jets, compare it
    # against the b expectation.
    b_expected = float(np.mean(wp.efficiency(np.full(n, BOTTOM_FLAVOR), pt, eta)))
    blind = measure_efficiency(bits.astype(bool), np.full(n, b_expected))
    assert abs(blind.pull) > 20.0, (
        "flavour-blind pooling must be caught; it agreed to "
        f"{blind.pull:.1f} sigma, so this gate would not discriminate"
    )


def test_bin_centre_expectation_is_biased_on_a_falling_spectrum() -> None:
    """The expectation must be the jet-wise mean, not the formula at bin centre.

    Inside a wide bin the spectrum piles up at the low edge while the efficiency
    still varies, so ``f(bin_centre)`` is not what the rate converges to. The
    correct treatment closes and the bin-centre one is many sigma out -- if it
    were not, the choice would be cosmetic and this gate would be vacuous.
    """
    wp = _cms_working_point()
    rng = np.random.default_rng(4242)
    n = 200_000
    pt = _falling_spectrum(rng, n)
    low, high = 30.0, 100.0
    pt = pt[(pt >= low) & (pt < high)]
    flavor = np.full(pt.size, BOTTOM_FLAVOR)
    eta = np.zeros_like(pt)
    bits = _simulate(wp, rng, flavor, pt)

    correct = efficiency_vs_pt(
        wp,
        flavor=flavor,
        pt=pt,
        eta=eta,
        btag_bits=bits,
        pt_edges=[low, high],
        select_flavor=BOTTOM_FLAVOR,
    )[0]
    assert abs(correct.pull) < 3.0, f"jet-wise mean must close, got {correct.pull:.2f} sigma"

    centre = float(evaluate_card_formula(CMS_BOTTOM, pt=0.5 * (low + high)))
    biased = measure_efficiency(bits.astype(bool), np.full(pt.size, centre))
    assert abs(biased.pull) > 10.0, (
        f"bin-centre expectation must be measurably wrong, got {biased.pull:.1f} sigma"
    )
    # And the bias is a real physics-sized shift, not a hair.
    assert abs(centre - correct.expected) > 0.02


# ---------------------------------------------------------------------------
# 5. Anchors -- orderings no correct card can violate
# ---------------------------------------------------------------------------


def test_flavour_ordering_holds_across_the_spectrum() -> None:
    """``eps_b > eps_c > eps_light`` at every p_T -- a tagger that fails this is
    not a b-tagger."""
    wp = _cms_working_point()
    pt = np.geomspace(20.0, 1000.0, 60)
    eps_b = wp.efficiency(np.full(pt.size, BOTTOM_FLAVOR), pt)
    eps_c = wp.efficiency(np.full(pt.size, CHARM_FLAVOR), pt)
    eps_l = wp.efficiency(np.full(pt.size, LIGHT_FLAVOR), pt)
    assert np.all(eps_b > eps_c)
    assert np.all(eps_c > eps_l)


def test_working_point_ordering_and_roc_are_monotonic() -> None:
    """A tighter working point must buy purity with efficiency, in both.

    This is the multi-working-point card's own consistency: the ROC points must
    advance monotonically in *both* coordinates, which a mis-decoded bitmask
    (e.g. reading ``BTag == 1``) would break.
    """
    loose = BTagWorkingPoint("L", 0, {LIGHT_FLAVOR: "0.1", BOTTOM_FLAVOR: "0.85*tanh(0.0025*pt)"})
    medium = BTagWorkingPoint("M", 1, {LIGHT_FLAVOR: "0.01", BOTTOM_FLAVOR: "0.65*tanh(0.0025*pt)"})
    tight = BTagWorkingPoint("T", 2, {LIGHT_FLAVOR: "0.001", BOTTOM_FLAVOR: "0.45*tanh(0.0025*pt)"})
    wps = [loose, medium, tight]

    rng = np.random.default_rng(7)
    n = 60_000
    pt = _falling_spectrum(rng, n)
    flavor = rng.choice([BOTTOM_FLAVOR, LIGHT_FLAVOR], size=n, p=[0.5, 0.5])
    eta = np.zeros_like(pt)

    # Nested tagging: a tight-tagged jet is also medium- and loose-tagged, which
    # is how a real discriminant threshold behaves.
    u = rng.random(n)
    bits = np.zeros(n, dtype=np.int64)
    for wp in wps:
        bits |= ((u < wp.efficiency(flavor, pt, eta)).astype(np.int64)) << wp.bit_number

    points = roc_points(wps, flavor=flavor, pt=pt, eta=eta, btag_bits=bits)
    assert [name for name, _, _ in points] == ["L", "M", "T"]

    effs = [sig.measured for _, _, sig in points]
    mistags = [bkg.measured for _, bkg, _ in points]
    assert effs[0] > effs[1] > effs[2], f"b-efficiency not monotonic: {effs}"
    assert mistags[0] > mistags[1] > mistags[2], f"mistag not monotonic: {mistags}"

    # Each point reproduces its own card formula.
    for _, bkg, sig in points:
        assert abs(sig.pull) < 4.0
        assert abs(bkg.pull) < 4.0
