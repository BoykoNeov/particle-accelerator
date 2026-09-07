/*
 * teach.js — the explanation layer of the accsim scenario editor.
 *
 * Content only: no physics is computed here. Every number this file talks about
 * comes from `accsim-optics.js`, and every claim below is traceable to a map in
 * that file, to a line of `docs/CONVENTIONS.md`, or to text the editor already
 * showed. Where accsim's linear core genuinely cannot see something (an
 * octupole's detuning, an aperture clipping a particle) the text says so rather
 * than implying the page is drawing it.
 *
 * Three rungs, each written for a different reader:
 *   novice     — no symbols, no jargon, an analogy where one is honest
 *   student    — the working definition, the units, what changes it
 *   physicist  — the exact convention, the formula, where it lives in accsim
 *
 * The level is a UI preference (stored beside the tab under STORE_KEY); it is
 * never part of a scenario, so it does not travel in an export or a share link.
 */
(function (root) {
  "use strict";

  const LEVELS = [
    { id: "off", label: "Teaching off" },
    { id: "novice", label: "Novice" },
    { id: "student", label: "Student" },
    { id: "physicist", label: "Physicist" },
  ];
  const KEY = { novice: "n", student: "s", physicist: "p" };

  // ------------------------------------------------------------------ elements
  // `b` is the one-line palette caption per level; n/s/p are the hover panels.
  const ELEMENTS = {
    Dipole: {
      t: "Dipole — the bending magnet",
      b: { n: "steers the beam round the corner", s: "sector bend, optional gradient and pole faces", p: "sector body + hard-edge faces; edges are optics-active" },
      n: "The magnet that turns the beam. Its field pushes every particle sideways by the same angle, so the beamline changes direction. A ring is nothing more than enough dipoles that the turns add up to a full circle.",
      s: "A sector bend of length L through angle θ: curvature h = θ/L, bending radius ρ = L/θ. On its own it focuses horizontally and acts as a drift vertically. It is also where dispersion is born — a particle with the wrong momentum bends by a different amount, so it leaves on a different path.",
      p: "Body is focusingBlock(h² + k₁) in x and focusingBlock(−k₁) in y, with r₁₆ = h·c₁, r₂₆ = h·s₁ and the ζ row fixed by symplecticity. A pole face adds px += h·tan(e)·x and py −= h·tan(e)·y, so a positive face angle defocuses horizontally and focuses vertically; e₁ = e₂ = θ/2 is the rectangular bend.",
      try: "set the angle to zero and watch the floor plan straighten out.",
      src: "accsim.elements.dipole — sector body, edges, and the F2 chromaticity terms.",
    },
    Quadrupole: {
      t: "Quadrupole — the lens",
      b: { n: "squeezes the beam one way, spreads it the other", s: "thick, k₁ > 0 focuses horizontally", p: "focusingBlock(±k₁, L); chromaticity ∝ ∮β k₁ ds" },
      n: "A lens for the beam. It squeezes the beam in one direction and lets it spread in the other — no magnet can squeeze both at once. Machines get round that by alternating squeeze and spread lenses, which holds the beam together on average.",
      s: "Thick quadrupole of strength k₁ [1/m²]. k₁ > 0 focuses horizontally and defocuses vertically; a short one acts like a lens of focal length 1/(k₁L). How much it matters depends on β where it sits — the same magnet is far more powerful at a β peak.",
      p: "focusingBlock(k₁, L) in x, focusingBlock(−k₁, L) in y, ζ–δ carrying L/γ₀². Contributes −(1/4π)∮β_x k₁ ds and +(1/4π)∮β_y k₁ ds to the chromaticity.",
      try: "flip the sign of k₁ and watch the two β curves trade places.",
    },
    ThinQuadrupole: {
      t: "Thin quadrupole",
      b: { n: "the same lens, squeezed into a single point", s: "integrated strength k₁L", p: "M[px][x] = −k₁L, M[py][y] = +k₁L" },
      n: "The same lens as a quadrupole, but treated as though it had no length at all — one instantaneous nudge at a single point. It keeps a sketch simple.",
      s: "Carries the integrated strength k₁L [1/m] and no length: px −= k₁L·x, py += k₁L·y. Focal length 1/(k₁L).",
      p: "The L → 0 limit of Quadrupole at fixed k₁L: identity but for M[px][x] = −k₁L and M[py][y] = +k₁L. Its chromaticity entry is the same integral collapsed to a point, ∓(1/4π)β k₁L.",
    },
    Sextupole: {
      t: "Sextupole — the chromatic corrector",
      b: { n: "fixes the lenses for off-energy particles", s: "thick, chromaticity correction", p: "linear map is a drift; reaches ξ by feed-down ∝ k₂βD" },
      n: "A correction magnet. Particles that are slightly off in energy get focused wrongly by the ordinary lenses. Put a sextupole where those particles have drifted off to one side and it pushes exactly them, by exactly the missing amount. Its push grows with the square of how far off-centre a particle is, so it ignores the ones in the middle.",
      s: "Strength k₂ [1/m³]. Its kick is quadratic in position, so it has no linear map at all — to this page a sextupole is a drift. It reaches the chromaticity through feed-down on the dispersion, and the shift is proportional to k₂·β·D. A sextupole where the dispersion is zero does nothing for chromaticity.",
      p: "Linear map = drift. ξ_x += (1/4π)∮k₂ β_x D_x ds and ξ_y −= (1/4π)∮k₂ β_y D_x ds (sextupoleFeeddown, entry for entry with accsim.twiss). The nonlinear kick itself, its resonance driving terms and its amplitude detuning are tracking/normal-form quantities in Python.",
      try: "watch the chromaticity tile, then drag the sextupole into a stretch where the dispersion curve sits at zero — it stops doing anything.",
    },
    ThinSextupole: {
      t: "Thin sextupole",
      b: { n: "the chromatic corrector as a single point", s: "integrated strength k₂L", p: "identity matrix; ξ shift ±(1/4π)β k₂L D_x" },
      n: "A sextupole treated as one instantaneous push instead of a magnet with length. Same job: fix the focusing for off-energy particles.",
      s: "Integrated strength k₂L [1/m²], no length. Its matrix is the identity, and like the thick one it only reaches the chromaticity where the dispersion is non-zero.",
      p: "Identity matrix. ξ_x += (1/4π)β_x k₂L D_x, ξ_y −= (1/4π)β_y k₂L D_x.",
    },
    Octupole: {
      t: "Octupole",
      b: { n: "pushes only the far-out particles", s: "thick, amplitude detuning (not visible here)", p: "linear map is a drift; dQ/dJ is a normal-form quantity" },
      n: "A magnet whose push grows with the cube of the distance from the centre. It barely touches particles near the middle and shoves the ones far out. That is used to stop a whole beam drifting into a resonance together — the outer particles wobble at a slightly different rate from the inner ones.",
      s: "Strength k₃ [1/m⁴]. It has no linear map, so nothing on this page moves when you change it — not β, not the tunes, not the chromaticity. Its real effect is the tune depending on oscillation amplitude, which needs tracking or normal form.",
      p: "Linear map = drift. The detuning dQ/dJ is an accsim.taylor / normal-form quantity (roadmap axis O), deliberately outside this editor's linear core; PTC's anhx is dQ/d(2J), half of accsim's dQ/dJ.",
    },
    ThinOctupole: {
      t: "Thin octupole",
      b: { n: "the same, as a single point", s: "integrated strength k₃L (not visible here)", p: "identity matrix; nonlinear only" },
      n: "An octupole with no length. As on this page nothing linear happens, the picture will not change when you edit it.",
      s: "Integrated strength k₃L [1/m³]. Identity matrix: the editor's linear optics cannot see it.",
      p: "Identity matrix; the cubic kick lands on three quantities at three powers of the orbit in accsim's tracking, none of them linear.",
    },
    SkewQuadrupole: {
      t: "Skew quadrupole — the coupler",
      b: { n: "tilts sideways motion into up-and-down", s: "couples the planes", p: "half-sum/half-difference of focusingBlock(±k₁ₛ)" },
      n: "A quadrupole rotated by 45°. Instead of squeezing left–right or up–down, it tips the two directions into each other: kick the beam sideways and it starts moving up and down too. Usually unwanted, sometimes deliberate.",
      s: "Strength k₁ₛ [1/m²]. It couples x and y, and once the planes are coupled the ordinary uncoupled Twiss functions are no longer the right description — so the editor stops drawing β and dispersion and leaves you the geometry, the matrices and the orbit. accsim's coupled_twiss (Edwards–Teng) handles it in Python.",
      p: "Half-sum and half-difference of focusingBlock(±k₁ₛ, L) on the diagonal and off-diagonal 2×2 blocks. The editor flags coupling when the off-diagonal norm of the one-turn map exceeds 1e-9; normal-mode tunes, ΔQ_min and the ε_y sharing are the G1/G2 milestones in Python.",
    },
    ThinSkewQuadrupole: {
      t: "Thin skew quadrupole",
      b: { n: "the coupler as a single point", s: "integrated k₁ₛL", p: "M[px][y] = M[py][x] = +k₁ₛL" },
      n: "A skew quadrupole with no length: one instantaneous nudge that mixes sideways and up-and-down motion.",
      s: "Integrated strength k₁ₛL [1/m]: px += k₁ₛL·y, py += k₁ₛL·x. It couples the planes, so the Optics tab stops.",
      p: "M[px][y] = M[py][x] = +k₁ₛL, identity elsewhere.",
    },
    Solenoid: {
      t: "Solenoid",
      b: { n: "a coil that makes the beam corkscrew", s: "field along the beam", p: "rotating-frame map with K = kₛ/2" },
      n: "A coil wrapped around the beam pipe with its field pointing along the beam. Everything going through it rotates — the beam corkscrews. The big detectors at colliders sit inside one, which is why their optics is such a nuisance.",
      s: "Strength kₛ [rad/m]; the beam is rotated by kₛL/2 on the way through, and it focuses in both planes at once. Because it rotates, it couples x and y, so as with a skew quadrupole the editor stops drawing the uncoupled optics.",
      p: "Standard rotating-frame map with K = kₛ/2. kₛ is charge-free in both accsim and xtrack (the sign lives in the strength, not the species); its vector potential is what the spin map needs.",
    },
    Wiggler: {
      t: "Wiggler",
      b: { n: "a row of magnets the beam snakes through", s: "periodic field, focuses vertically", p: "x-block is a drift; y-block focuses at h₀²/2" },
      n: "A row of short magnets with alternating poles. The beam snakes through it and radiates light — for a light source, that light is the whole point of the machine. The snake path is slightly longer than a straight line, which shows up in the timing.",
      s: "Its length is period × number of periods; h₀ is the peak curvature, so the peak field is h₀·Bρ. It focuses vertically and nowhere else, at strength h₀²/2. That focusing is not an end effect — it is there all the way along, and does not slice away.",
      p: "x-block is a drift, y-block is focusingBlock(h₀²/2, L). ζ–δ picks up L/γ₀² + ¼Lθ²(2 + 1/γ₀²) with θ = h₀/k, plus a constant kick −¼Lθ²; the R56 beats a drift's by K²/2 and does not grow with energy. No reference code has this element — see the roadmap's axis T.",
      try: "load the 'Wiggler ring' preset and follow the vertical β curve through it; the horizontal one does not notice.",
    },
    Drift: {
      t: "Drift — empty pipe",
      b: { n: "empty pipe: the beam just spreads", s: "field-free space", p: "M[x][px] = L, M[ζ][δ] = L/γ₀²" },
      n: "Empty pipe. Nothing steers the beam here, so it carries on and slowly spreads out. Most of a real machine is this — the magnets are the exception.",
      s: "Field-free length L: x += L·px. The beam always gets wider in a drift; β grows quadratically as you move away from a waist, which is why long straights need lenses on both sides.",
      p: "M[x][px] = M[y][py] = L and M[ζ][δ] = L/γ₀² — the δ = Δp/p₀ convention, not L/(β₀²γ₀²). The exact drift is not the paraxial one: it detunes with no magnets present at all.",
      src: "docs/CONVENTIONS.md — the drift R56 entry, derived symbolically.",
    },
    RFCavity: {
      t: "RF cavity",
      b: { n: "the accelerating gap, and the beam's clock", s: "longitudinal focusing", p: "M[δ][ζ] = −qVk_rf cos φₛ / (β₀²E₀)" },
      n: "The accelerating gap — the only thing in the ring that adds energy. It is also the beam's clock: a particle arriving early gets a slightly smaller push and a late one a slightly bigger push, so particles bunch up around the right arrival time instead of smearing all the way round.",
      s: "Voltage V at harmonic h (frequency f = h·β₀c/C) with synchronous phase φₛ. Its map only touches the longitudinal plane — δ gets a kick proportional to how far the particle sits from the bunch centre. Combined with the momentum compaction, that gives the synchrotron tune Qₛ.",
      p: "M[δ][ζ] = −qV·k_rf·cos φₛ / (β₀²E₀) with k_rf = 2πf/(β₀c); everything else is the identity. Qₛ comes from the lumped-cavity ½Tr(M_s) = 1 − ½r₆₅ηC. accsim's cavity carries no −sin φₛ offset in this map.",
    },
    Corrector: {
      t: "Corrector",
      b: { n: "a small steering nudge", s: "constant steering kick", p: "affine kick on px, py; identity matrix" },
      n: "A small steering magnet. It nudges the beam sideways by a fixed angle, whatever the particle was doing. This is what operators actually turn when a beam is off centre.",
      s: "Zero length, identity matrix, and a constant kick (px += kick_x, py += kick_y). On a ring, the orbit that comes back to itself with these kicks in it is the closed orbit — the Orbit tab.",
      p: "Enters as the element's affine kick, never its matrix; the closed orbit is the fixed point (I − M)⁻¹k of the affine one-turn map. Element.kick is where a misalignment's constant part lands too.",
      try: "switch to the Orbit tab and change kick x — the whole ring's orbit moves, not just the bit after the corrector.",
    },
    Aperture: {
      t: "Aperture",
      b: { n: "how big the hole is", s: "beam-pipe limit (geometry only here)", p: "drift matrix; clipping is a tracking-time test" },
      n: "The size of the hole the beam has to fit through. Nothing on this page stops a particle — this is a note about the geometry. The actual scraping happens when accsim tracks particles in Python.",
      s: "A half-width in x and y with a circular, elliptical or rectangular shape. Its linear map is a drift, so it changes no curve here; what it is for is comparing its half-width against the rms beam size on the Beam size tab.",
      p: "Drift matrix of the stated length. The shape test is applied per-particle during tracking in accsim, not in the linear map.",
    },
    Collimator: {
      t: "Collimator",
      b: { n: "jaws that scrape the beam's edge", s: "jaws, with a length", p: "drift matrix; clipping is a tracking-time test" },
      n: "A pair of jaws deliberately closed in on the beam to scrape off the outermost particles, so they are lost somewhere chosen rather than somewhere expensive.",
      s: "Same as an aperture but with a real length. Its linear map is a drift; the scraping is a tracking-time test in accsim, so no curve on this page will move when you close the jaws.",
      p: "Drift matrix of the stated length; the shape test lives in accsim's tracking.",
    },
  };

  // -------------------------------------------------------------------- tips
  const TIPS = {
    // ---- summary tiles
    "tile:length": {
      t: "Circumference",
      n: "How far the beam travels in one lap. For a beamline that is not a loop, it is just the distance from one end to the other.",
      s: "The sum of every element's length. Everything longitudinal hangs off it: the revolution time, the RF frequency for a given harmonic, and the integrals that become the momentum compaction.",
      p: "Σ elementLength over the resolved sequence; a wiggler's length is period × periods, and thin elements contribute nothing.",
    },
    "tile:closure": {
      t: "Bend closure",
      n: "A ring has to close on itself. Add up every bending angle: one full turn means the machine really is a loop. Anything else and the beam comes back pointing the wrong way.",
      s: "Σθ / 2π. Exactly one turn is a closed ring. When it is not, the Floor plan tab shows the leftover gap in metres — that is the picture of what is wrong.",
      p: "The angle sum is only half of closure — the survey's translation has to close too, and an angle sum of 2π does not guarantee it. The floor plan reports the residual gap; three different wrong walkers all still close the angle.",
      src: "roadmap axis R — survey and ring closure.",
    },
    "tile:stability": {
      t: "Stability",
      n: "Does the beam survive lap after lap? Green means a particle that starts near the middle stays near the middle for ever. Red means the wobble grows every lap until the particle is gone.",
      s: "Per plane, the machine is stable when |½ Tr M| < 1 for that 2×2 block of the one-turn matrix — the two numbers under the chip. At exactly 1 the focusing is critical; beyond it there is no matched β at all.",
      p: "halfTrace of the one-turn map; cos μ = ½Tr M, so |½Tr| ≥ 1 means the phase advance is not real and matchPeriodic throws 'unstable'.",
    },
    "tile:tunes": {
      t: "Tunes",
      n: "As it goes round, the beam wobbles from side to side and up and down. The tune counts how many complete wobbles happen in one lap — 6.3 means six and a third.",
      s: "Q = μ/2π accumulated all the way round (for a line, the same number is the total phase advance in units of 2π). The fractional part is what decides resonances; the integer part rarely matters.",
      p: "Taken from the accumulated Twiss phase rather than acos(½Tr), so it carries the integer part. Its momentum derivative dQ/dδ is the chromaticity tile.",
    },
    "tile:workingpoint": {
      t: "Working point",
      n: "A map of where your machine sits, with the danger lines drawn on it. Land on a line and the wobble gets a push in the same direction every lap, so it grows until the beam is lost. The dot is you.",
      s: "The unit square of fractional tunes with the resonance lines m·Qx + n·Qy = p up to order 3. Thick lines are the low orders — the dangerous ones. Machines are deliberately parked in a clear patch between them.",
      p: "Orders 1–3 only, clipped to the unit square. Whether a given line is actually driven, and how strongly, is a resonance-driving-term question — accsim's normal-form axis O, not this diagram.",
    },
    "tile:chroma": {
      t: "Chromaticity",
      n: "Particles with slightly the wrong energy wobble at a slightly different rate from the rest. This number says how much. Left to itself it comes out negative, and being too negative costs you the beam; sextupoles pull it back towards zero.",
      s: "ξ = dQ/dδ — a particle with δ = 1e-3 has its tune shifted by ξ × 1e-3. 'Natural' is what the quadrupoles and dipoles give on their own; the headline number adds the sextupoles' feed-down on top.",
      p: "chromaticity = naturalChromaticity + sextupoleFeeddown, both by β-integration; the dipole terms include the curvature and pole-face pieces (milestone F2). It is the un-normalised derivative dQ/dδ, never Q'/Q — a stray Q is the classic error here.",
      src: "docs/CONVENTIONS.md — natural chromaticity; cross-checked against xtrack's dqx.",
    },
    "tile:alphac": {
      t: "Momentum compaction",
      n: "A particle carrying more momentum takes a slightly different route round the ring. This says whether that route is longer or shorter, and by how much.",
      s: "α_c is the fractional change in path length per unit δ. The number that actually matters for the RF is the slip factor η = α_c − 1/γ₀²: it decides whether a higher-momentum particle arrives early or late. γₜ is the energy where the two effects cancel and η passes through zero.",
      p: "Computed by the exact identity route: α_c = 1/γ₀² − (M[ζ][x]D_x + M[ζ][px]D′_x + M[ζ][δ])/C on the matched dispersion. η > 0 is 'above transition' in this convention, and γₜ = 1/√α_c only exists for α_c > 0.",
    },
    "tile:qs": {
      t: "Synchrotron tune",
      n: "Bunches slosh slowly back and forth in energy and arrival time. This counts how many of those slow sloshes happen per lap — a tiny number, so one slosh takes hundreds of laps.",
      s: "The small-amplitude synchrotron tune. It needs an RF cavity and a non-zero slip factor; if the longitudinal focusing has the wrong sign or is far too strong there is no stable bucket at all and the tile says so.",
      p: "½Tr(M_s) = 1 − ½r₆₅ηC with r₆₅ summed over the cavities, Qₛ = acos(½Tr)/2π. The lumped-cavity formula throws 'rf-unstable' when |½Tr| ≥ 1.",
    },
    "tile:beta": {
      t: "Beta at the start",
      n: "How wide the beam is allowed to swing here, sideways and up-and-down. A bigger number means a fatter beam at this point.",
      s: "β at s = 0 for a ring — the periodic solution, the one shape that repeats lap after lap — or at the exit for a line, together with the dispersion there. The physical size is √(εβ + (σ_δD)²), which the Beam size tab draws.",
      p: "matchPeriodic on the one-turn map (the Courant–Snyder fixed point), or the propagated entrance Twiss for a line.",
    },
    "tile:beam": {
      t: "The beam",
      n: "What is going round the machine: which particle, and how much energy each one carries.",
      s: "Total energy, the Lorentz factor γ, and the magnetic rigidity Bρ. Rigidity is the practical one: a field B over a length L bends by BL/Bρ, so the same magnet bends a stiffer beam less.",
      p: "Bρ = p/(c|q|) with p in eV/c, γ₀ = E/m, β₀ = √(1 − 1/γ₀²). The strengths k₁, k₂, k₃ are already normalised by Bρ, which is why changing the energy alone does not move the optics — see the taper milestones for what actually does.",
    },
    "tile:problems": {
      t: "Problems",
      n: "Something in the sequence cannot be built as written. The message names the element and what is wrong with it; fix that one and the rest of the page comes back.",
      s: "Per-element validation, the same rules accsim's Python constructors enforce — so a scenario this editor accepts is one accsim will load.",
      p: "validateElement per record, run before any matrix is built; the message carries the element's index so the sequence list can highlight it.",
    },

    // ---- machine settings
    "m:ringline": {
      t: "Ring or line",
      n: "A ring is a closed loop the beam goes round again and again. A line is a one-way pipe from one machine to another. The questions you can ask are different: only a ring has tunes and a closed orbit.",
      s: "Ring means periodic: the editor looks for the one β that repeats turn after turn, and reports tunes, chromaticity, compaction and the closed orbit. Line means the beam enters with the Twiss you type and the editor just propagates it to the exit.",
      p: "periodic=true takes matchPeriodic of the one-turn map; false propagates initial_twiss and reports the accumulated phase advance instead of a tune.",
    },
    "m:species": {
      t: "Particle",
      n: "What is going round. Electrons are light and radiate strongly; protons are heavy and do not.",
      s: "Sets the rest mass and the charge, which together with the energy fix γ, β and the rigidity Bρ.",
      p: "Only mass and charge; the sign of the charge enters the RF slope and the rigidity, not the geometric strengths k₁, k₂, kₛ.",
    },
    "m:energy": {
      t: "Energy",
      n: "How much energy each particle carries. More energy means a stiffer beam that the same magnet bends less.",
      s: "Given as total, kinetic or momentum×c — the readout below shows all of them plus γ, β₀ and the rigidity Bρ. Because the strengths here are normalised strengths, changing the energy alone does not move the optics; it changes what field a magnet needs to deliver them.",
      p: "reference() converts the chosen mode to total energy; γ₀ = E/m, β₀ = √(1 − 1/γ₀²), Bρ = p/(c|q|). What does move the optics with energy is the sag of a real machine — accsim's tapering milestones.",
    },
    "m:twiss0": {
      t: "Entrance Twiss",
      n: "For a beamline there is no lap to repeat, so you have to say what shape the beam is in when it arrives. These numbers are that shape.",
      s: "β and α per plane plus the dispersion and its slope, propagated from the entrance. α is the tilt of the beam's ellipse: negative α means it is still converging, zero is a waist, positive means it is spreading.",
      p: "Seeds propagateTwiss for the non-periodic case; the same numbers a ring would have got from matchPeriodic.",
    },
    "m:emit": {
      t: "Emittance",
      n: "How much room the beam takes up as a bunch — a fat, messy beam has a big emittance and a tightly-packed one a small emittance. It is a property of the beam you put in, not of the magnets.",
      s: "The rms beam size is √(εβ + (σ_δD)²), so ε sets the scale and β sets the shape along the machine. It only affects the Beam size tab; it changes no optics.",
      p: "Used purely for the envelope. Where it comes from in a real ring — the balance of radiation damping against quantum excitation — is accsim's radiation axis in Python.",
    },
    "m:sigma_delta": {
      t: "Momentum spread",
      n: "Not every particle has exactly the right energy. This says how spread out they are — a few parts in a thousand is typical.",
      s: "The rms of δ = Δp/p₀. It matters where the dispersion is large: the second term of √(εβ + (σ_δD)²) is what makes the beam fat inside the arcs.",
      p: "Enters only the envelope. Together with the chromaticity it also sets the tune spread of the bunch, ΔQ ≈ ξ·σ_δ.",
    },

    // ---- element parameters
    "p:name": {
      t: "Name",
      n: "A label for this magnet so you can find it again in the list. It has no effect on the physics.",
      s: "Carried through into the exported scenario and the generated Python, where it becomes the element's name= argument.",
      p: "Free text; the editor auto-names by type prefix (B, Q, S, O, QS, SOL, W, D, RF, C, AP, COL) and never uses it in a computation.",
    },
    "p:length": {
      t: "Length",
      n: "How long the magnet or the gap is, along the beam.",
      s: "In metres, along s. For a magnet, strength × length is what the beam actually feels; for a drift it is the distance over which the beam spreads.",
      p: "Sets the element's contribution to the circumference and, for a thick element, the exponent in its exact map. The editor slices thick elements for plotting only — every element here slices exactly.",
    },
    "p:angle": {
      t: "Bend angle",
      n: "How far this magnet turns the beam. All the bend angles in a ring have to add up to one full circle.",
      s: "The geometric bend angle θ in radians. With the length it fixes the curvature h = θ/L and the radius ρ = L/θ; the aside under the box shows it in mrad and degrees, and the header tile checks the sum against 2π.",
      p: "θ is the design angle, not a field. h = θ/L drives the horizontal focusing h², the dispersion generation r₁₆ = h·c₁, and the ζ–x coupling; the required field is h·Bρ.",
    },
    "p:k1": {
      t: "Quadrupole strength k₁",
      n: "How hard this lens squeezes. Positive squeezes side-to-side and lets the beam spread up-and-down; negative does the opposite. Zero and the magnet does nothing.",
      s: "k₁ [1/m²] is the field gradient normalised by the rigidity. k₁ > 0 focuses horizontally, k₁ < 0 vertically; a thin lens of the same integrated strength has focal length 1/(k₁L). Its effect on the tune is roughly β·k₁L/4π, so where it sits matters as much as how strong it is.",
      p: "focusingBlock(k₁) in x and focusingBlock(−k₁) in y. On a dipole, k₁ makes it a combined-function magnet: the horizontal focusing becomes h² + k₁ and a Maxwell curvature–sextupole term appears in the chromaticity.",
    },
    "p:k1l": {
      t: "Integrated strength k₁L",
      n: "How hard this point-like lens squeezes. Positive squeezes side-to-side.",
      s: "k₁L [1/m], strength times length rolled into one number: px −= k₁L·x, py += k₁L·y. Focal length 1/(k₁L).",
      p: "The thin-lens limit at fixed k₁L; identical optics to a short thick quadrupole of the same integral to first order in L.",
    },
    "p:k2": {
      t: "Sextupole strength k₂",
      n: "How hard the correction magnet pushes. Its push grows with the square of how far off-centre a particle is, so raising this barely touches the beam core.",
      s: "k₂ [1/m³]. It changes nothing linear — β, the tunes and the floor plan are all blind to it. Watch the chromaticity tile instead, and only where the dispersion is non-zero.",
      p: "The linear map stays a drift; ξ_x += (1/4π)∮k₂β_xD_x ds and ξ_y −= (1/4π)∮k₂β_yD_x ds. The quadratic kick, its feed-down on a displaced orbit and its resonance driving terms are Python-side.",
    },
    "p:k2l": {
      t: "Integrated strength k₂L",
      n: "The correction magnet's strength rolled into one number, as a single push.",
      s: "k₂L [1/m²]. Only the chromaticity tile responds, and only where the dispersion is non-zero.",
      p: "ξ_x += (1/4π)β_x k₂L D_x, ξ_y −= (1/4π)β_y k₂L D_x.",
    },
    "p:k3": {
      t: "Octupole strength k₃",
      n: "How hard the magnet shoves the far-out particles. Nothing on this page will move when you change it — the effect only shows up when particles are actually tracked round.",
      s: "k₃ [1/m⁴]. There is no linear map, so the editor is genuinely blind to it. What it does — make the tune depend on oscillation amplitude — needs tracking or normal form in Python.",
      p: "Linear map = drift. dQ/dJ from accsim's normal form (axis O); note that a sextupole detunes linearly in action too, so only the k₃ scaling separates the two.",
    },
    "p:k3l": {
      t: "Integrated strength k₃L",
      n: "The octupole's strength as a single push. Nothing on this page responds to it.",
      s: "k₃L [1/m³]; the matrix is the identity, so no curve here changes.",
      p: "Identity matrix; nonlinear only.",
    },
    "p:k1s": {
      t: "Skew strength k₁ₛ",
      n: "How hard this tilted lens mixes side-to-side motion into up-and-down motion.",
      s: "k₁ₛ [1/m²]. Any non-zero value couples the planes, and the editor then refuses to draw uncoupled β and dispersion because they would be wrong.",
      p: "Sets the off-diagonal blocks. The coupling strength that matters physically is the driving term |C⁻|, whose ΔQ_min and ε_y sharing accsim computes in Python (milestone G1).",
    },
    "p:k1sl": {
      t: "Integrated skew strength",
      n: "The tilted lens as a single push that mixes the two directions.",
      s: "k₁ₛL [1/m]: px += k₁ₛL·y, py += k₁ₛL·x. It couples the planes, so the Optics tab stops.",
      p: "M[px][y] = M[py][x] = +k₁ₛL.",
    },
    "p:ks": {
      t: "Solenoid strength kₛ",
      n: "How fast the beam corkscrews inside the coil.",
      s: "kₛ [rad/m]; the beam rotates by kₛL/2 through the magnet, and it focuses in both planes at once. It couples the planes, so the Optics tab stops.",
      p: "K = kₛ/2 in the rotating-frame map; charge-free in both accsim and xtrack.",
    },
    "p:h0": {
      t: "Peak curvature h₀",
      n: "How hard the wiggler's poles bend the beam at their strongest point — that sets how sharply it snakes.",
      s: "h₀ [1/m] is the peak curvature, so the peak field is h₀·Bρ (the aside shows it). The vertical focusing goes as h₀²/2, so it does not care about the sign.",
      p: "y-block focusingBlock(h₀²/2, L); θ = h₀/k with k = 2π/period drives the ζ terms. Momentum enters squared here.",
    },
    "p:period": {
      t: "Wiggler period",
      n: "How long one full snake wiggle is. Shorter periods mean a tighter snake.",
      s: "The spatial period [m]. Together with the number of periods it fixes the magnet's length; it also sets θ = h₀·period/2π, which controls the extra path length.",
      p: "k = 2π/period; the vertical focusing h₀²/2 is independent of it, the ζ terms are not.",
    },
    "p:periods": {
      t: "Number of periods",
      n: "How many wiggles there are. Period × this is the total length of the magnet.",
      s: "Count of full periods; the element's length is period × periods, which is what enters the circumference.",
      p: "Integer count; the map is exact for any total length, so this simply scales L.",
    },
    "p:voltage": {
      t: "RF voltage",
      n: "How big a push the accelerating gap gives. More voltage means the bunch is held together more tightly in time.",
      s: "Peak voltage V. The longitudinal focusing is proportional to V·cos φₛ, so Qₛ grows roughly as √V.",
      p: "Enters r₆₅ = −qV·k_rf·cos φₛ/(β₀²E₀); Qₛ = acos(1 − ½r₆₅ηC)/2π.",
    },
    "p:phi_s": {
      t: "Synchronous phase",
      n: "Where on the RF wave the ideal particle sits. That choice decides whether the bunch is being accelerated or just held together.",
      s: "φₛ in radians. The focusing goes as cos φₛ, so at φₛ = 0 you get the strongest bunching and no net acceleration; the sign of cos φₛ also has to match the sign of the slip factor or there is no bucket at all.",
      p: "Only cos φₛ enters this linear map — accsim's cavity carries no −sin φₛ offset here; the energy gain itself is a tracking quantity.",
    },
    "p:harmonic": {
      t: "Harmonic number",
      n: "How many bunches would fit round the ring, evenly spaced. Choosing it is how you pick the RF frequency.",
      s: "An integer h; the frequency follows from the circumference, f = h·β₀c/C, and the aside shows the result. Change the ring's length and the frequency follows automatically.",
      p: "f = h·β₀c/C recomputed from the resolved circumference each time; switching to 'frequency' freezes the current value instead.",
    },
    "p:frequency": {
      t: "RF frequency",
      n: "How fast the accelerating voltage swings back and forth.",
      s: "In hertz, fixed rather than tied to the ring's length. Useful for a line, or when you want the frequency to stay put while you edit the lattice.",
      p: "Used directly in k_rf = 2πf/(β₀c); no consistency check against the circumference is applied in this mode.",
    },
    "p:kick": {
      t: "Steering kick",
      n: "How far this little magnet nudges the beam sideways, in angle. A ten-thousandth of a radian is a typical size.",
      s: "A constant angular kick in radians, added to px or py regardless of where the particle is. On a ring it moves the whole closed orbit, not only the part downstream.",
      p: "The element's affine kick; the closed orbit is (I − M)⁻¹k. Orbit correction against a measured response converges linearly, not quadratically, once sextupole feed-down is in play.",
    },
    "p:half": {
      t: "Half-aperture",
      n: "Half the width of the hole, measured from the centre line. Compare it with the beam size to see whether the beam fits.",
      s: "half_x and half_y in metres, interpreted by the shape. Nothing on this page clips: put the Beam size tab beside it and compare with the σ curve.",
      p: "Geometry only; the per-particle test runs in accsim's tracking.",
    },
    "p:shape": {
      t: "Aperture shape",
      n: "Whether the hole is a circle, an oval or a rectangle.",
      s: "Circular uses half_x as the radius; elliptical and rectangular use both half-widths.",
      p: "Passed through to accsim's Aperture/Collimator constructor unchanged; it never enters the linear map.",
    },
    "p:align": {
      t: "Misalignment",
      n: "Real magnets are never exactly where the drawing says. Shift one sideways and the beam no longer goes through its middle, so the magnet steers it — that is where a bent orbit comes from.",
      s: "dx, dy in metres and roll in radians. A displaced quadrupole acts as a corrector of strength k₁L·dx, so the Orbit tab responds while the optics does not; a rolled quadrupole couples the planes instead.",
      p: "The offset enters as the affine kick (I − M)d, and roll conjugates the body by s_rotation. A rolled or displaced bend needs the curved rigid-body geometry rather than a rotation, so the editor refuses those.",
      try: "put 0.0005 m of dx on a quadrupole and switch to the Orbit tab.",
    },
    "p:e1": {
      t: "Entry pole face",
      n: "The angle the magnet's end face is cut at. Cutting it changes how the magnet focuses, without changing how much it bends.",
      s: "e₁ in radians. A positive face angle defocuses horizontally and focuses vertically — that is exactly how a rectangular bend (e₁ = e₂ = θ/2) trades its horizontal focusing for vertical focusing. The buttons below set the two standard cases.",
      p: "Hard-edge kick px += h·tan(e₁)·x, py −= h·tan(e₁)·y. Edges are optics-active: they move β, tune, dispersion and the chromaticity, and their ±(1/4π)β h tan e terms are in F2.",
    },
    "p:e2": {
      t: "Exit pole face",
      n: "The same cut, at the far end of the magnet.",
      s: "e₂ in radians, applied on the way out. A rectangular bend has both faces at half the bend angle.",
      p: "Same hard-edge kick applied after the body; mirroring a block in the editor swaps e₁ and e₂ for you.",
    },
    "p:fringe": {
      t: "Fringe field",
      n: "Real magnets do not stop dead at their ends — the field trails off. Switching this on tells accsim to include that when it tracks particles.",
      s: "Hard-edge fringe, tracking only; the linear map is blind to it. β, the dispersion and the natural chromaticity come out bit-identical with it on, so nothing on this page will move.",
      p: "A second-order effect: the gap is 12 map entries, not 5, and ζ's share is cubic, so only symplecticity gates it. accsim.taylor sees it; the linear core cannot.",
    },

    // ---- views
    "view:synoptic": {
      t: "The lattice strip",
      n: "The machine drawn end to end, with one block per magnet. Blocks above the line focus one way, blocks below the other. Click one to edit it, drag it to move it.",
      s: "The sequence laid out against s. Height and side encode the sign of the leading strength; dipoles are full-height blocks, thin elements are narrow ticks. Everything below lines up with it in s.",
      p: "Drawn from the resolved element list, so a wiggler shows its true length and a harmonic cavity its resolved frequency.",
    },
    "panel:beta": {
      t: "β functions",
      sub: { n: "how wide the beam is allowed to swing, along the machine", s: "the envelope shape; size is √(εβ + (σ_δD)²)" },
      n: "How wide the beam swings as it moves along, in the two directions. A tall peak is a place where the beam is fat and easy to lose; a low, flat stretch is a place where it is small. This is not the beam's size in metres — it is the shape the size follows.",
      s: "β_x and β_y in metres. The physical rms size is √(εβ + (σ_δD)²), so β says where the beam is fat and the emittance says how fat. On a ring this is the periodic solution: the one shape that repeats lap after lap.",
      p: "propagateTwiss from matchPeriodic (ring) or the entrance Twiss (line), on the plotting slicing. β has units of metres but is not a length of the beam; the ellipse's area is the emittance.",
      try: "hover anywhere on the curve — the readout gives you α and the phase advance there too.",
    },
    "panel:disp": {
      t: "Dispersion",
      sub: { n: "how far off-course a particle with the wrong momentum rides", s: "x offset per unit δ; created in the bends" },
      n: "Particles with slightly the wrong momentum ride to one side of the ideal path. This curve says how far off, per unit of momentum error. It is created in the bending magnets and nowhere else.",
      s: "D_x in metres: a particle at δ rides at x = D·δ. Bends generate it, quadrupoles shape it, and where it is zero a sextupole can no longer correct chromaticity — which is why light-source cells work so hard to close it back to zero in the straights.",
      p: "The matched periodic dispersion from the one-turn map, propagated with the dispersive kick per element. D_y is only drawn when something makes it non-zero; note accsim is blind to orbit-angle-driven vertical dispersion.",
    },
    "panel:sigma": {
      t: "rms beam size",
      sub: { n: "the actual width of the beam, in millimetres", s: "√(εβ + (σ_δD)²) per plane" },
      n: "The actual width of the beam in millimetres, which is what has to fit down the pipe. It comes from two things: the shape the magnets impose, and how messy the beam was to start with.",
      s: "σ = √(εβ + (σ_δD)²) per plane, from the emittances and momentum spread in the inspector. The second term is why the beam is fattest in the arcs, where the dispersion is large.",
      p: "Envelope only — no radiation equilibrium, no coupling-driven ε_y sharing; both are Python-side (axis B, milestone G1).",
    },
    "panel:orbit": {
      t: "Closed orbit",
      sub: { n: "the path the beam actually takes, versus the ideal one", s: "the fixed point of the affine one-turn map" },
      n: "The path the beam actually takes when the magnets are not perfectly placed or a steering magnet is on — as against the ideal line through the middle. Every particle oscillates about this, not about the centre.",
      s: "The orbit that closes on itself after one lap, given every constant kick in the ring. A single corrector moves it everywhere, not just downstream, and the response is largest where β is largest.",
      p: "The 4D fixed point (I − M)⁻¹k of the affine one-turn map, at δ = 0. A 6D solve is a different question — the closed orbit is where the energy sag is centred.",
    },
    "view:floor": {
      t: "Floor plan",
      n: "The machine seen from above, as it would be drawn on the floor of the building. The line is the beam's path; the colours are the magnets.",
      s: "The planar survey: each bend turns the frame, everything else goes straight. For a ring it also reports the closure gap — how far the end of the sequence lands from its start.",
      p: "Yaw-only survey in (X, Z); the closure gap is the translation residual, which the angle sum alone does not guarantee.",
    },
    "view:matrix": {
      t: "Matrices",
      n: "Every element can be written as a table of numbers that turns 'where the particle is now' into 'where it is after'. Multiply them all together and you have the whole machine in one table.",
      s: "6×6 linear maps on (x, px, y, py, ζ, δ). The top-left 4×4 is the transverse motion; the ζ and δ rows are the longitudinal one. A one-turn matrix with an off-diagonal 2×2 block between the x and y quarters means the planes are coupled.",
      p: "Xsuite/MAD-X ordering, momenta normalised to P₀, δ = Δp/p₀, ζ = s − β₀ct. Elements compose right to left. Every matrix here is held against the Python package to 1e-9 by tests/analytic/test_scenario.py.",
    },

    // ---- tabs
    "tab:optics": {
      t: "Optics tab",
      n: "The two pictures that say how the magnets shape the beam: how wide it swings, and how far off-course an off-momentum particle rides.",
      s: "β functions and dispersion against s, from the periodic solution (ring) or the entrance Twiss (line).",
      p: "propagateTwiss on the plotting slicing; hover gives α and the accumulated phase.",
    },
    "tab:envelope": {
      t: "Beam size tab",
      n: "The same information turned into the beam's actual width in millimetres, which is what has to fit down the pipe.",
      s: "σ = √(εβ + (σ_δD)²) per plane, using the emittances in the inspector.",
      p: "Envelope only; no radiation equilibrium.",
    },
    "tab:orbit": {
      t: "Orbit tab",
      n: "Where the beam really goes when a magnet is out of place or a steering magnet is switched on.",
      s: "The closed orbit for a ring, or the trajectory for a line. It stays flat until something kicks it.",
      p: "The affine fixed point at δ = 0; correctors and misalignments are the only sources.",
    },
    "tab:floor": {
      t: "Floor plan tab",
      n: "The machine drawn as it would sit on the floor, seen from above.",
      s: "The planar survey and, for a ring, the closure gap.",
      p: "Yaw-only survey; the translation residual is the half of closure the angle sum does not check.",
    },
    "tab:matrix": {
      t: "Matrices tab",
      n: "The machine written out as tables of numbers, for when you want to see the arithmetic itself.",
      s: "The one-turn (or end-to-end) matrix, plus the matrix of whatever you have selected.",
      p: "6×6 on (x, px, y, py, ζ, δ); select up to three elements to see theirs.",
    },
    "tab:learn": {
      t: "Learn tab",
      n: "A short guided path through this editor, written for whatever level you have picked in the header.",
      s: "Level-graded cards; the buttons load presets or switch tabs, and the two that change your lattice say so first and are undoable.",
      p: "Content only — nothing here computes anything the other tabs do not.",
    },

    // ---- error and empty states, levelled
    "err:unstable": {
      t: "No periodic solution",
      n: "The focusing is too strong (or too weak) for the beam to come back to the same shape each lap, so there is no steady beam to draw. Soften the lenses, or space them differently, and it will come back. The two numbers in the Stability tile tell you which direction is at fault.",
      s: "The one-turn map has |½ Tr M| ≥ 1 in a plane, so no matched β exists there. Weaken the focusing or change the cell length; the ½Tr readouts say which plane.",
      p: "matchPeriodic threw 'unstable': cos μ = ½Tr M is outside [−1, 1]. Nothing downstream of the match (tunes, chromaticity, compaction) can be defined.",
    },
    "err:coupled": {
      t: "The planes are coupled",
      n: "Something in the machine tips side-to-side motion into up-and-down motion — a tilted lens, a solenoid, or a rotated magnet. Once that happens, 'how wide sideways' and 'how wide vertically' are no longer separate questions, and this editor does not draw them. The geometry, the matrices and the orbit are still right.",
      s: "A skew quadrupole, a solenoid or a rolled magnet mixes x and y, and the uncoupled Courant–Snyder optics would be wrong. accsim's coupled_twiss handles it in Python; the editor shows the geometry, the matrices and the orbit only.",
      p: "The off-diagonal norm of the one-turn map exceeds 1e-9. Edwards–Teng (milestone G2) is the Python route; normal-mode tunes, ΔQ_min and the ε_y sharing come with it.",
    },
    "err:resonant": {
      t: "Integer tune",
      n: "The wobble happens a whole number of times per lap, so every lap repeats the last one's error and nothing settles down. Nudge a lens strength slightly and it will resolve.",
      s: "The dispersion cannot close on an integer tune — the periodic solution is singular there.",
      p: "The (I − M) solve for the matched dispersion is singular at an integer tune; the same condition is the first-order resonance line on the tune diagram.",
    },
    "empty:orbit-ring": {
      t: "The orbit is the design axis",
      n: "Nothing is pushing the beam off centre, so it runs exactly down the middle. Add a steering magnet, or nudge a magnet sideways with dx, and this picture will come alive.",
      s: "Nothing steers it: add a corrector, or displace a magnet (dx, dy) to see the orbit move.",
      p: "The affine kick vector is zero, so the fixed point is the origin and no orbit is computed.",
    },
    "empty:orbit-line": {
      t: "The trajectory is the design axis",
      n: "Nothing pushes the beam off the centre line, so it goes straight down it. Add a steering magnet to see it bend away.",
      s: "Add a corrector kick to see it move.",
      p: "Zero affine kick; the trajectory starts and stays at the origin.",
    },
  };

  const ALIAS = {
    "p:dx": "p:align", "p:dy": "p:align", "p:roll": "p:align",
    "p:kick_x": "p:kick", "p:kick_y": "p:kick",
    "p:half_x": "p:half", "p:half_y": "p:half",
    "p:emit_x_nm": "m:emit", "p:emit_y_nm": "m:emit", "p:sigma_delta_pm": "m:sigma_delta",
    "p:beta_x": "m:twiss0", "p:beta_y": "m:twiss0", "p:alpha_x": "m:twiss0", "p:alpha_y": "m:twiss0",
    "p:disp_x": "m:twiss0", "p:disp_px": "m:twiss0",
    "p:mass_eV": "m:species", "p:charge": "m:species",
    "p:energy_GeV": "m:energy", "p:energy_mode": "m:energy", "p:species": "m:species",
  };

  function entry(key) {
    if (!key) return null;
    const k = ALIAS[key] || key;
    if (k.slice(0, 3) === "el:") return ELEMENTS[k.slice(3)] || null;
    return TIPS[k] || null;
  }

  // The caller escapes `head` into a data- attribute, but the HTML parser decodes it
  // again on the way back out of dataset, and it can carry an element name that came
  // from an imported scenario or a share link. It is escaped here, where it is used.
  function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

  /** The hover panel for a key, or null when there is nothing to say at this level. */
  function tip(key, level, head) {
    if (level === "off") return null;
    const e = entry(key);
    if (!e) return null;
    const body = e[KEY[level]];
    if (!body) return null;
    let h = `<h4>${head ? esc(head) : e.t || ""}</h4><p>${body}</p>`;
    if (e.try && level !== "physicist") h += `<p class="try"><b>Try it —</b> ${e.try}</p>`;
    if (e.src && level === "physicist") h += `<p class="src">${e.src}</p>`;
    return h;
  }

  /** The one-line palette caption, or null to keep the editor's own terse hint. */
  function brief(type, level) {
    if (level === "off") return null;
    const e = ELEMENTS[type];
    return e && e.b ? e.b[KEY[level]] || null : null;
  }

  /** A short caption printed under a plot panel's title (novice and student only). */
  function panelSub(key, level) {
    if (level === "off" || level === "physicist") return null;
    const e = TIPS[key];
    return e && e.sub ? e.sub[KEY[level]] || null : null;
  }

  /** The levelled replacement for the editor's own error / empty-state prose. */
  function message(key, level, fallback) {
    const e = entry(key);
    if (!e || level === "off") return fallback;
    const body = e[KEY[level]];
    if (!body) return fallback;
    return `<b>${e.t}.</b><br>${body}`;
  }

  // ------------------------------------------------------- the live guide strip
  /**
   * One paragraph that reads the machine actually on screen. `r` is the analyse()
   * result, `sc` the scenario, `h` a bag of helpers from the editor (fmt only).
   * Returns null when there is nothing worth saying.
   */
  function guide(level, r, sc, h) {
    if (level === "off" || !r) return null;
    const fmt = h.fmt;
    const n = sc.elements.length;
    if (!n) {
      return { title: "Start here", html: level === "novice"
        ? "The machine is empty. Click <b>Dipole</b> in the palette on the left to bend the beam, then <b>Quadrupole</b> to focus it — or pick a ready-made machine from <b>Presets</b> at the top and take it apart."
        : "Empty sequence. Add elements from the palette, or load a preset to start from a working lattice." };
      }
    if (r.errors.length) {
      return { title: "Fix this first", html: level === "novice"
        ? `One element is not valid, so nothing can be computed: <b>${r.errors[0]}</b>. Click it in the sequence list on the right — the bad box is marked.`
        : `Validation failed: ${r.errors[0]}` };
    }
    const ring = r.periodic;
    const bits = [];
    if (level === "novice") {
      const bends = sc.elements.filter((e) => e.type === "Dipole").length;
      // Skew quadrupoles couple the planes rather than focusing, so they are not
      // "lenses" in the sentence below.
      const quads = sc.elements.filter((e) => e.type === "Quadrupole" || e.type === "ThinQuadrupole").length;
      bits.push(ring
        ? `This is a <b>ring</b> ${fmt(r.length, 4)} m around: the beam comes back to the start and goes again.`
        : `This is a <b>beamline</b> ${fmt(r.length, 4)} m long: the beam goes through once and leaves.`);
      if (bends) bits.push(`${bends} bending magnet${bends > 1 ? "s" : ""} steer${bends > 1 ? "" : "s"} it and ${quads} lens${quads === 1 ? "" : "es"} keep${quads === 1 ? "s" : ""} it from spreading out.`);
      if (r.opticsError) {
        bits.push(r.opticsError.code === "coupled"
          ? "Something in it mixes sideways and vertical motion, so the beam-shape curves are switched off — the floor plan and the orbit still work."
          : "Right now the focusing does not settle into a repeating shape, so there are no curves to draw. Try weakening a lens.");
      } else if (r.twiss) {
        let bmax = 0, bs = 0;
        for (const p of r.twiss) { const b = Math.max(p.beta_x, p.beta_y); if (b > bmax) { bmax = b; bs = p.s; } }
        bits.push(`The beam swings widest about ${fmt(bs, 3)} m along. Hover any curve to read it off, and click a magnet to change it.`);
      }
    } else if (level === "student") {
      bits.push(ring ? `Ring, C = ${fmt(r.length, 5)} m, ${n} elements.` : `Line, L = ${fmt(r.length, 5)} m, ${n} elements.`);
      if (r.opticsError) bits.push(`No matched optics: <b>${r.opticsError.code}</b> — ${r.opticsError.message}.`);
      else if (r.twiss) {
        let bx = 0, by = 0, dmin = Infinity, dmax = -Infinity;
        for (const p of r.twiss) { if (p.beta_x > bx) bx = p.beta_x; if (p.beta_y > by) by = p.beta_y; if (p.disp_x < dmin) dmin = p.disp_x; if (p.disp_x > dmax) dmax = p.disp_x; }
        bits.push(`β peaks at ${fmt(bx, 4)} m horizontally and ${fmt(by, 4)} m vertically; the dispersion runs ${fmt(dmin, 3)} to ${fmt(dmax, 3)} m.`);
        if (ring && r.chromaticity && r.naturalChromaticity) {
          const corrected = Math.abs(r.chromaticity[0] - r.naturalChromaticity[0]) > 1e-9 || Math.abs(r.chromaticity[1] - r.naturalChromaticity[1]) > 1e-9;
          bits.push(corrected
            ? `The sextupoles move the chromaticity from (${fmt(r.naturalChromaticity[0], 3)}, ${fmt(r.naturalChromaticity[1], 3)}) to (${fmt(r.chromaticity[0], 3)}, ${fmt(r.chromaticity[1], 3)}).`
            : `No sextupole is doing anything: the chromaticity is still the natural (${fmt(r.naturalChromaticity[0], 3)}, ${fmt(r.naturalChromaticity[1], 3)}).`);
        }
        if (ring && r.eta != null) bits.push(`η = ${fmt(r.eta, 3)}, so a higher-momentum particle arrives ${r.eta > 0 ? "later" : "earlier"} each lap${r.gamma_t ? ` (γₜ = ${fmt(r.gamma_t, 4)})` : ""}.`);
      }
      if (!r.hasKicks && ring) bits.push("Nothing kicks the beam, so the closed orbit is the design axis.");
    } else {
      bits.push(`${ring ? "periodic" : "line"} · ${n} elements · C = ${fmt(r.length, 8)} m · ½Tr = (${fmt(r.halfTrace.x, 6)}, ${fmt(r.halfTrace.y, 6)})`);
      if (ring) {
        const res = r.totalAngle - 2 * Math.PI;
        bits.push(`Σθ − 2π = ${fmt(Math.abs(r.totalAngle) < 1e-12 ? 0 : res, 3)} rad`);
      }
      if (r.opticsError) bits.push(`opticsError: ${r.opticsError.code} — ${r.opticsError.message}`);
      else if (r.tunes) {
        bits.push(`Q = (${fmt(r.tunes[0], 6)}, ${fmt(r.tunes[1], 6)})`);
        if (r.chromaticity) bits.push(`ξ = (${fmt(r.chromaticity[0], 5)}, ${fmt(r.chromaticity[1], 5)}), natural (${fmt(r.naturalChromaticity[0], 5)}, ${fmt(r.naturalChromaticity[1], 5)})`);
        if (r.alpha_c != null) bits.push(`α_c = ${fmt(r.alpha_c, 5)}, η = ${fmt(r.eta, 5)}`);
        if (r.Qs != null) bits.push(`Q_s = ${fmt(r.Qs, 5)}`);
      }
      if (r.coupled) bits.push("coupled: |off-diagonal| > 1e-9");
      if (r.warnings && r.warnings.length) bits.push(`warnings: ${r.warnings.join("; ")}`);
      return { title: "Readout", html: `<span class="mono">${bits.join(" · ")}</span>` };
    }
    return { title: level === "novice" ? "What you are looking at" : "Reading this machine", html: bits.join(" ") };
  }

  // ------------------------------------------------------------- the Learn tab
  // `act` values are handled by the editor: preset, tab, level, flipquad, zerosext.
  const LESSONS = {
    novice: [
      { t: "What a machine is made of",
        h: "An accelerator is a pipe with magnets round it. Two kinds do almost all the work. <b>Bending magnets</b> steer the beam — put enough of them in a circle and the beam comes back to where it started. <b>Lenses</b> stop the beam spreading out sideways, the way a lens stops light spreading. Everything else is refinement.<br><br>The strip across the middle of this page is the machine drawn end to end, one block per magnet. The curves under it are what the beam does.",
        a: [{ act: "preset", arg: "fodo-cell", label: "Load the simplest machine" }, { act: "tab", arg: "optics", label: "Show me the curves" }] },
      { t: "Why lenses come in pairs",
        h: "There is no magnet that squeezes a beam in both directions at once — squeeze it sideways and it spreads vertically, and the other way round. The trick every accelerator uses is to alternate them: squeeze, drift, spread, drift, over and over. On average the beam stays together in both directions.<br><br>That pattern is called a FODO cell, and it is the first preset in the list.",
        a: [{ act: "preset", arg: "fodo-cell", label: "Load a FODO cell" }, { act: "flipquad", label: "Break it: flip the first lens", note: "changes k₁ → −k₁ on the first quadrupole. Ctrl+Z puts it back." }] },
      { t: "The two curves",
        h: "The blue and orange curves on the Optics tab say how wide the beam is allowed to swing at each point — blue side to side, orange up and down. Where a curve peaks the beam is fat and easy to lose; where it dips the beam is small.<br><br>They are not the beam's width in millimetres. For that, switch to the Beam size tab, which combines these curves with how messy your beam is.",
        a: [{ act: "tab", arg: "envelope", label: "Show me the real width" }] },
      { t: "Off-energy particles",
        h: "No two particles have exactly the same energy, and a bending magnet turns a slightly faster particle a little less. So the off-energy ones ride to one side of the ideal path. The <b>Dispersion</b> curve says how far off, and it only exists where there are bends.<br><br>Those particles are also focused slightly wrongly by the lenses. Fixing that is what the green sextupole magnets are for.",
        a: [{ act: "preset", arg: "dba-cell", label: "See a cell built to control it" }] },
      { t: "When it goes wrong",
        h: "Push a lens too hard and the beam never settles into a repeating shape — the page says so instead of drawing a curve, because there is genuinely nothing to draw. That is not a bug in the machine you built; it is the machine telling you it will lose the beam.<br><br>Everything you change here is undoable with Ctrl+Z, so break things freely.",
        a: [{ act: "level", arg: "student", label: "Step up to Student" }] },
    ],
    student: [
      { t: "The periodic solution",
        h: "On a ring, the interesting β is not one you choose — it is the one shape that reproduces itself after a full lap. The editor finds it by matching the one-turn matrix, and it exists only when |½ Tr M| < 1 in both planes. That single condition is the whole of linear stability.<br><br>For a line there is no lap to repeat, so you type the entrance Twiss yourself and the editor propagates it.",
        a: [{ act: "tab", arg: "matrix", label: "Show me the one-turn matrix" }] },
      { t: "Where a magnet's power comes from",
        h: "A quadrupole's effect on the tune goes roughly as β·k₁L/4π — so the same magnet is far more powerful at a β peak than at a β minimum. That is why lattice designers care so much about <i>where</i> things sit, and why a single misaligned quadrupole at a peak can dominate the orbit.<br><br>Try dragging a quadrupole from a low-β stretch to a high-β one and watch the tunes move.",
        a: [{ act: "preset", arg: "fodo-cell", label: "Load a FODO cell to try it on" }] },
      { t: "Chromaticity and the sextupole trick",
        h: "ξ = dQ/dδ: off-momentum particles have a different tune, and left alone ξ is negative — the quadrupoles focus a low-momentum particle too hard. The fix is a magnet whose strength grows with position, placed where off-momentum particles are already displaced. That is a sextupole sitting in dispersion.<br><br>The shift is proportional to k₂·β·D, so a sextupole where D = 0 buys you nothing. Watch the chromaticity tile while you move one.",
        a: [{ act: "preset", arg: "dba-cell", label: "Load a corrected cell" }, { act: "zerosext", label: "Turn its sextupoles off", note: "sets k₂ (or k₂L) to zero on every sextupole. Ctrl+Z puts them back." }] },
      { t: "The longitudinal plane",
        h: "The slip factor η = α_c − 1/γ₀² decides whether a higher-momentum particle arrives early or late. Below transition it arrives early, above transition late, and at γₜ the two effects cancel and the RF loses its grip. Add a cavity and the editor reports the synchrotron tune — normally a few thousandths, so one longitudinal oscillation takes hundreds of laps.",
        a: [{ act: "preset", arg: "electron-ring", label: "Load a ring with a cavity" }] },
      { t: "Orbits and errors",
        h: "The design axis is a fiction: magnets are never exactly where the drawing says. Displace a quadrupole by dx and it acts as a corrector of strength k₁L·dx, and the closed orbit moves — everywhere in the ring, not just downstream of the error. Correcting it is a matter of finding kicks that cancel that, which is what the response matrix in accsim is for.",
        a: [{ act: "preset", arg: "kicked-ring", label: "Load a ring with an error in it" }, { act: "tab", arg: "orbit", label: "Show me the orbit" }] },
      { t: "What this page cannot see",
        h: "Everything here is <i>linear</i>. A sextupole shows up only through chromaticity; an octupole not at all; an aperture never clips anything; the fringe-field switch moves no curve. Those effects are real and accsim computes them — by tracking particles, by building the second-order map, or by normal form — but in Python, not here.<br><br>Export the machine as Python and the same numbers come out of the package, plus everything this page leaves out.",
        a: [{ act: "level", arg: "physicist", label: "Step up to Physicist" }] },
    ],
    physicist: [
      { t: "What this core is",
        h: "editor/accsim-optics.js is a <i>port</i> of accsim's linear half, not a second implementation: element matrices and kicks, matchPeriodic, propagateTwiss, naturalChromaticity + sextupoleFeeddown, the exact-identity momentum compaction, the lumped-cavity Qₛ, the 4D affine closed orbit and the planar survey. tests/analytic/test_scenario.py runs it under Node and holds every one of those against the Python package at rtol 1e-9 on all six presets. A disagreement is a bug on one side, never a tolerance to loosen." },
      { t: "Conventions worth checking before you trust a number",
        h: "(x, px, y, py, ζ, δ) in Xsuite/MAD-X ordering; px,py normalised to P₀; δ = Δp/p₀ (momentum, not energy); ζ = s − β₀ct with ahead ⇒ ζ > 0. A drift's R56 is L/γ₀², not L/(β₀²γ₀²). Chromaticity is the un-normalised dQ/dδ. η = α_c − 1/γ₀², so η > 0 is above transition. Elements compose right to left." },
      { t: "Where the linear core stops",
        h: "Sextupole and octupole bodies are drifts here; their kicks, feed-down, RDTs and amplitude detuning are accsim.taylor and the normal-form axis. Coupled optics (Edwards–Teng) is Python-only, which is why the editor refuses to draw β when the off-diagonal norm exceeds 1e-9. Rolled and displaced bends are refused outright — a rolled bend needs the curved rigid-body geometry, not a rotation. Radiation, spin, tapering and the 6D closed orbit are all outside this file." },
      { t: "The wiggler, and why it has no arbiter",
        h: "No reference code in this project's stack has the element, so its gates are internal: the vertical focusing is h₀²/2 and does not slice away, the R56 beats a drift's by K²/2 and is energy-invariant, and ζ carries a term symplecticity demanded. Momentum enters squared. That makes the wiggler preset a good test of whether you believe the map or the picture.",
        a: [{ act: "preset", arg: "wiggler-ring", label: "Load the wiggler ring" }] },
      { t: "Round-tripping",
        h: "Export JSON gives an accsim-scenario/1 document that ac.load_scenario(path).lattice reads directly; ac.dump_scenario goes the other way, so any Lattice you build in Python opens here. Export Python gives the same machine as constructor calls — useful when you want the parts of accsim this page cannot reach.",
        a: [{ act: "export", arg: "python", label: "Show me the Python" }] },
    ],
  };

  function lessons(level) { return LESSONS[level] || []; }

  root.AccsimTeach = { LEVELS, tip, brief, panelSub, message, guide, lessons, entry };
})(typeof self !== "undefined" ? self : this);
