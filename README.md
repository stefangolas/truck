# truck — the certified B-rep kernel

`truck` is an open-source shape processing kernel in Rust by
[ricosjp](https://github.com/ricosjp/truck) (Apache-2.0). This repository is a
**permanent architectural fork** whose core crates have been rewritten into a
certified boundary-representation kernel: every geometric answer is a typed
outcome carrying the evidence that produced it, and every refusal states why
the kernel could not certify an answer instead of returning an approximation
silently. Upstream compatibility is not a design constraint — a hand-back into
an upstream truck type would erase the proof, and the proofs are the product.

The kernel is developed through an automated packet/worker/verify loop driven
by the sibling `look` repository (where it is vendored as `vendor/truck/`).
This fork receives bulk sync commits from that tree and hosts the kernel-side
design documentation (mirrored snapshot under [`docs/`](docs/), deep-dive in
[`OVERVIEW.md`](OVERVIEW.md)).

## The one-sentence model

Every fallible kernel operation returns
`Outcome<T> = Result<Certified<T>, Refusal>`: either your geometry **plus the
certificate that produced it**, or a **typed refusal** naming the precise
reason nothing was produced. Nothing panics on bad geometry, uncertainty is
never dressed up as success, and a refusal is data you can match on — a design
signal, not an error string.

## Core design principles

These are the load-bearing rules. Each is stated the way the repo states it,
with its enforcement point.

**1. Every pipeline stage is a fallible constructor.**
Output types represent *stronger* states and carry evidence for the
obligations discharged. Correctness is established incrementally over an
explicit chain — STEP entities → resolved topology → converted geometry with
retained identity → certified curve-on-surface → quotient-resolved trim loops
→ conforming triangulation — never inferred from the final render. A smooth
blob can be the result of any of a dozen invalid intermediates.
(`docs/MATHEMATICAL_FOUNDATION.md` §0; house rule **H-2**: every fallible
operation returns `Outcome<T>`, never `Option` — "`None` is not a diagnosis".)

**2. Certificates carry their method.**
Every certificate records *how* the value was computed —
`Method ∈ {Exact, Interval, Float, None}`. A value computed in floats is
never recorded `Exact`. There is deliberately no convenience constructor that
stamps a method onto raw numbers, and no `From<f64>`: a certified answer
cannot be manufactured casually. (**H-6**; `truck-base/src/evidence.rs`.)

**3. Three-valued verdicts. Uncertainty is a state, not a failure.**
`CERTIFIED_WITHIN_TOLERANCE | FAILED | INCONCLUSIVE`. An audit that cannot
decide returns `Inconclusive` and stays `Inconclusive` — no conversion path
from uncertainty to success exists anywhere in the tree.

**4. Knowledge is an order, and contradiction is typed.**
Propositions carry `Truth = Unknown | True | False | Both` under a knowledge
ordering; composing evidence that assigns conflicting truth to one proposition
returns `Contradictory(witness)` — a distinct refusal arm, not a panic or a
picked side. (`truck-base/src/evidence.rs`.)

**5. Four failure layers are kept apart.**
A stage is not a face. A failure of the machine is not a fact about the
geometry. A missing mesher is not an invalid face. `Unresolved` is a statement
about the evidence, `Unsupported` is a proved statement about the input
against a declared envelope, `Inconsistent` is a proved contradiction in the
source, and `OperationalFailure` is about this run only — and there is no
`From<OperationalFailure>` conversion into an `Unsupported` outcome.
(`truck-certified/src/formal/outcome.rs`.)

**6. No panics on data.**
No `unwrap`, `expect`, `panic!`, `todo!`, or range-indexing on any path
reachable from untrusted geometry, enforced by crate-level
`#![deny(clippy::unwrap_used, ...)]`. Grandfathered moved code is marked, and
new unwraps under an existing allow are forbidden. (**H-1**.)

**7. No anonymous epsilon. Topology identity never uses tolerance.**
Every length comparison goes through `ToleranceCtx`; a literal `1e-6` in a
predicate is a defect. Dimensionless comparisons may use literals and must
name the quantity. If two faces are meant to share an edge they reuse the same
`Edge` identity — geometric coincidence never implies topological identity.
(**H-3**; authored-incidence doctrine.)

**8. Budgets, never bare loops.**
Any iteration whose count depends on geometry takes a `&mut Budget` and
returns `NumericallyUnresolved { spent, witness }` on exhaustion. There is no
non-termination detector — there is a budget, and that is why epistemic
closure survives. A hard-coded `for _ in 0..16` is a defect. (**H-5**.)

**9. Canonical carriers, analytic preservation.**
The certified solver family proves its theorems over a declared carrier set
𝒢 (Line/Circle; Plane/Cylinder/Cone/Sphere/Torus; placed and extruded
analytic forms). Input outside 𝒢 refuses `NonCanonicalCarrier` *before any
work* rather than degrading to splines — and an operation whose exact result
is an analytic carrier emits that carrier, never an approximation of it.

**10. Enclosure soundness: over-estimation is fine, under-estimation is the bug.**
`enclose(box) ⊇ { f(p) : p ∈ box }` for every carrier and box, with
outward-rounded interval arithmetic (never fast-math or FMA contraction).
Under-estimation is a silent-wrong-answer bug that invalidates every
certificate built on top of it. (BG-ENC-001/002/003.)

**11. Search in floats, certify exactly (SFC).**
A float heuristic may *search*; the *certificate* is an exact or interval
predicate computed independently of the search. `FloatHint<T>` carries no
evidence status and cannot be constructed into an evidence position — the
discipline is unviolatable by construction. Every float heuristic in the tree
decomposes into (float search, exact verify) or is refused at review.

**12. Authority is minted only by named introduction rules.**
Evidence-bearing constructors are `pub(super)` and not re-exported; outside a
subtree there is exactly one way to obtain each certified artifact. New
evidence kinds are booked by adding a row to the certificate mapping table —
a spec edit, never a worker widening a type on its own judgement.
(`docs/CERTIFICATE_MAPPING.md`.)

**13. Lower bounds stay lower bounds. Bound direction is conservative.**
Certified quantities are typed as what they are (`LfsLowerBound`, never a bare
`lfs`), and gates have the form `q < c · lfs_lower`, so substituting a lower
bound can refuse an instance the true value would admit — and can never admit
one it would refuse. Refusals are epistemic: a refusal asserts the bound could
not be certified, not that the feature is small. Certificates carry
`@establishes` / `@does-not-establish` annotations; a type that would overclaim
is deliberately not created.

**14. Diagnosis is not repair.**
Diagnostics are analysis-only: `ManifoldDiagnostics` never mutates its input
and no repair is offered. Where a repair is legitimately the product, it is a
named, bounded, explicit operation — never a silent normalization.

**15. Determinism is a contract, down to bit-reproducibility.**
Certified paths perform no transcendental calls (`sin`, `cos`, `atan2`, `exp`,
`log`, `powf`, `sqrt`); frame bases are built by Gram–Schmidt in fixed index
order rather than SVD because SVD is not cross-platform bit-reproducible;
hash-map iteration order never defines observable output; ties break to the
lowest index. Identical ordered input produces byte-identical certificates.

**16. Honest error composition.**
Forward error composes through a modulus contract (`Lipschitz | Holder | Pole
| Unbounded`) with subadditivity *read off the shape, never declared by a
caller*. Composing a non-subadditive operand refuses rather than publishing a
bound that may under-report — and a near-degenerate cell publishes an honest
`Pole` instead of `Unbounded`, because "an honest non-subadditive bound beats
no bound at all". (BG-EVD-004.)

**17. Policy is separate from detection.**
A detector establishes a fact; product policy decides what to do. Gates are
labelled by standing — **[DERIVED]** (follows from a stated theorem),
**[PROVISIONAL SUFFICIENT GATE]** (may not support a `ProvenConstruction`
claim), or **[POLICY]** (a choice) — and the label travels in provenance.

**18. The margin sweep is the testable form of epistemic closure.**
Sweep an item's margin parameter logarithmically toward zero and assert the
outcome degrades monotonically:
`Proven → CertifiedEquivalent → NumericallyUnresolved → UnsupportedEnvelope`
— never skipping to a wrong-but-confident answer. A gated item without a
margin sweep is not done. (BG-TEST-SWEEP.)

## Essential theorems

The mathematical guarantees the kernel is built on. Statements are given as
the repo states them; each carries its implementation point.

### Numerics: how uncertainty is contained

**Krawczyk existence and uniqueness (BG-NUM-003).**
For `K = m − Y·F(m) + (I − Y·F'(box))·(box − m)`:
`K ⊆ interior(box)` proves a unique root in the box (existence **and**
uniqueness); `K ∩ box = ∅` proves no root; otherwise bisect under budget.
Three sharp edges the implementation codifies: the center term is a *point*
evaluation (an interval center decorrelates the linear part against the
contraction term and certifies nothing); uniqueness requires *strict*
inclusion (`K ⊆ box` non-strict proves existence only); and validity holds
for *any* invertible preconditioner Y — Y's quality affects tightness only.
A vanishing midpoint derivative says nothing about the box, so the operator
bisects rather than refuses. Enforced in `truck-certified/src/formal/
bezier_isect.rs` (2-D) and `truck-certified/src/ssi.rs` (square 3×3 systems;
the pseudo-inverse-preconditioned rectangular route is explicitly rejected
because the theorem does not transfer).

**Three-state root isolation, and "multiple roots refuse, never empty".**
Every box classifies as Subdivide / Excluded (Bernstein range excludes zero) /
Root (Krawczyk strict inclusion). Multiple or tangential roots must return
`NumericallyUnresolved(RootNotIsolated)`, **not** an empty list — reporting
"no root" for a tangential double root is precisely the silent-wrong-answer
class this system exists to prevent. (BG-NUM-002.)

**Exact predicates.**
Decisions that admit topology are exact sign predicates over the `f64`
inputs, computed with Shewchuk expansion arithmetic (`formal/exact.rs`) — an
`Expansion` sums with zero rounding error to the exact value, so its `sign`
decides the exact sign of that value. **No tolerance establishes topology.**
There is exactly one `Expansion` implementation in the workspace.

**Bernstein hull enclosure.**
For a patch in Bézier form, the convex hull of control points is a
conservative range enclosure of the patch and its derivative patches (order 2)
over any compact subbox — hulls are enclosures for *polynomial* quantities
only; rational quantities are bounded via homogeneous numerator/denominator
with directed-rounded division, and only with certified-positive weights
(the projection of a hull is not the hull of the projection otherwise).
Interval de Casteljau is arranged so the linear case evaluates to the exact
endpoint range. (`truck-certified/src/hull.rs`.)

**Certified elementary functions.**
In-tree interval `sin`/`cos`/... with three theorem-backed obligations:
inclusion monotonicity; truncation error *bounded* (the alternating Taylor
series' first omitted term), never estimated; exact argument reduction where
the choice of k cannot make the answer wrong, only wide. Built in-tree
because a C big-float dependency would sit between the kernel and every
certified evaluation. (BG-ENC-005.)

### Geometry: what the solvers prove

**Theorem 3 (exclusion = hull separation).**
`0 ∉ conv{c_αβ} ⇔ conv{P¹} ∩ conv{P²} = ∅` (Minkowski sum identity) on every
subbox, with separating directions inherited downward monotonically. This
powers the per-carrier span BVH (CFP-005; `truck-certified/src/bvh.rs` in the
look tree — pending the next fork sync): one exact
Theorem-3 sign row — `min_α λ·P¹_α > max_β λ·P²_β` decided by `Expansion`
predicates — prunes every descendant span pair, with no re-verification. The
float search (GJK hint) changes *which* pairs are enumerated, never what a
pair's answer is.

**Theorem 4 (implicit reduction, analytic × spline).**
Contact between a recognized analytic carrier (plane or quadric) and a
spline patch becomes ONE scalar equation `h = g∘S` on the spline chart in
exact Bernstein form, with the 2×2 `∇h = 0` system feeding the landed
Krawczyk directly. The elevation trap is packet-normative: `∂h/∂u` and
`∂h/∂v` must be degree-elevated to a common bidegree before pairing, or the
hull test is invalid. (`truck-evidence/src/contact/implicit2d.rs` in the
look tree — pending the next fork sync.)

**Gauss-map cone certificates.**
When two carriers' gradient-direction cones over a box are disjoint, no point
of the shared zero set can be tangent — the branch is transversal by proof
(rank deficiency ⇔ parallel normals), **loop-free** (every branch meets the
box boundary, closing seed completeness), and carries a **fixed continuation
axis** with certified margin. Three verdicts, one vocabulary:
`LoopFree | TransversalByProof | AxisFixed{axis, margin}`, each on the
`Certified`/`Interval` vocabulary. (`truck-evidence/src/contact/gff.rs`.)

**Theorem 2.1 (crossing-angle identity).**
For the certified intersection engine, `σ_min(DF̂) = √(1 − |n_A · n_B|`
*exactly*, with `‖DF̂‖ = √2` exactly: conditioning depends only on the
crossing angle of the two surface normals, never on NURBS parameterization
speed. A kernel conditioned on raw `DF` escalates on knot placement rather
than on geometry. (`CERTIFIED_INTERACTION_ENGINE_SPEC.md` — look-tree docs
snapshot, pending the next fork sync.)

**Exact analytic degeneracy (BG-ANA).**
Two cylinders are tangent iff `|d_axes| = r₀ ± r₁`; a plane is tangent to a
sphere iff `dist(centre, plane) = r`; coaxiality and coincidence are exact
conditions on carrier parameters. Every analytic pair returns `Proven` with
`Method::Exact`, or a typed classification of the degenerate position — never
a float-certified result. Outcomes switch cleanly transverse → tangent →
disjoint under margin sweeps, with no band of wrong-but-confident answers.

**Stratified reach.**
The certified quantity is a per-stratum lower bound
`lfs_σ(x) = min(ρ̲(σ), dist̲(x, non-incident strata), ϱ̲_wedge(x))` — because
the global reach of a mechanical B-rep is **zero** (it collapses at every
sharp edge), so any code path using a global reach is a defect. A knife edge
or crack drives the wedge term to zero: faces whose bound is 0 route to a
certified collapse, not to a certificate. (BG-FID-001; Federer's closed-form
reach decomposition is demoted to motivation for trimmed patches until its
open lemma lands.)

**The isotopy lemma and the one-sheet condition.**
Two-sided Hausdorff closeness plus an almost-tangency bound plus boundary
correspondence give a proper local homeomorphism — hence a *covering of some
degree*, not a homeomorphism. The double-covered circle
`X' = (R + ε·cos(t/2))·e(t), t ∈ [0, 4π]` passes every metric check and voids
every certificate above it. Degree-one covering per component (via Krawczyk
fibre isolation) or fibrewise uniqueness on a certified partition must be
discharged separately. (`docs/FORMAL_SYSTEM_BREP_GENERATION.md` §6.2.)

### Topology and material: what makes a body

**The nine B-rep invariants (BG-INV-101…109).**
Coedge pairing, vertex link = single cycle, Euler–Poincaré (necessary only —
a pinch point satisfies it, so it never substitutes for the link test),
same-parameter on every edge use, domain–boundary correspondence,
representation in 𝒢, tolerance monotonicity, shell nesting (a forest, decided
by certified winding — never `count >= 1`), and wedge non-degeneracy. Every
checker returns `Outcome<()>`: a violated invariant is `Contradictory`, an
undecidable one is a typed refusal — a checker returning
`NumericallyUnresolved` on a healthy input is a defect.
(`truck-topology/src/invariants/`.)

**Watertightness by construction (index identity).**
If every mesh boundary vertex's position index is a pure function of
(entity identity, sample ordinal) — never of coordinates — and incident faces
satisfy `I(A, E) == reverse(I(B, E))` as integer sequences, the emitted mesh
is edge-watertight *by construction* and positional welding is never invoked.
Closedness is a property of index bookkeeping derived from authored
incidence, verified by a winding audit whose failure is `FAILED`, never a
warning. (`truck-geometry/src/constructive/mod.rs`.)

**Membership by propagation; material-state booleans.**
For a connected face of A cut by Ξ = f ∩ ∂B: χ_B is locally constant on
f∖Ξ and the dual adjacency graph is connected — **one certified seed per
face** determines all fragments. Flip parity comes from contact order (odd
flips, even doesn't), decided by sector signs; a tangential arc is even-order
by construction and never flips — which removes the commonest
inverted-material defect. Coincident fragments are decided by the material
state 4-tuple, no case enumeration. Material parity is the boundary's
winding number mod 2: an edge the boundary crossed twice separates nothing.

**Cut-open existence and deck arithmetic.**
Every compact connected orientable two-manifold with boundary admits a finite
cut graph whose cut-open fundamental domain is a planar polygonal schema —
implemented as the certified quotient/deck machinery, with ambient periods
resolved to a five-state lattice and certified integer deck arithmetic
(rank-0/1/2: a torus carries a verified rank-two deck group; only a regular
ring torus is certified — spindle and horn tori refuse).
(`docs/FORMAL_SYSTEM_STEP_INGESTION.md`, `truck-certified/src/domain/`.)

### Composition: how guarantees add up

**Epistemic closure (§22).**
Under the ordering obligations, totality, and mutual exclusivity of the
outcome classifier, every input terminates within its ledger and returns a
well-formed outcome with exactly one terminal classification: no path ends
unclassified, and no path returns a construction whose preconditions were not
established. Proof by induction over the dispatch DAG.

**The composition theorem and the modulus contract.**
If every step's error satisfies `ε_{i+1} ≤ ω_i(ε_i) + τ_i` with `ε_i < 𝔪_i`,
the combinatorial result equals the exact result and the geometric error obeys
the nested recurrence — the fundamental bound. The split form is a *corollary*
requiring certified subadditivity (tangency is Hölder-½, so this system
contains nonlinear moduli by design), and the recurrence is not
order-symmetric. (`docs/FORMAL_SYSTEM_BREP_GENERATION.md` §18;
`truck-base/src/evidence.rs` `Modulus`.)

**STEP ingestion symbolic closure (Theorem 1).**
The atlas procedure terminates and produces exactly one outcome
`Valid | Inconsistent | Ambiguous | Unsupported | Unresolved`; valid outputs
are gauge-invariant; every valid semantic configuration inside the declared
envelope has a representation in the atlas language; no input reaches an
untyped "unknown configuration" state. Proved from finite combinatorics
(Lemmas 1–10); what is *assumed* (numerical embedding, solver completeness)
is separated explicitly. Corollary: corpus growth cannot invalidate closure
by itself.

**The refusal taxonomy.**
`Refusal`: `Empty` · `UnsupportedEnvelope(ChartDegenerate | ReachTooSmall |
NonCanonicalCarrier | NonPositiveNurbsWeight | ContactReductionDeferred |
ConstructRefused)` · `NumericallyUnresolved { spent, witness }` ·
`CompositionMarginExhausted` · `InputOutsideBackwardBudget` ·
`Contradictory(witness)` · `Collapsed(knife-edge | apex-vanishing,
certificate)` · `ForwardToleranceExceeded { bound, allowed }`.
A `Collapsed` answer is certified — it is just not a realisation. The full
algebra (`Certificate`, `PropMap`, `Budget`, `Margin`, `Modulus`) lives in
`truck-base/src/evidence.rs`.

## The house rules (H-1 … H-8)

| Rule | Law |
|---|---|
| H-1 | No panics on data: no `unwrap`/`expect`/`panic!`/`todo!`/range-indexing reachable from untrusted geometry |
| H-2 | Every fallible operation returns `Outcome<T>` — never `Option`; `None` is not a diagnosis |
| H-3 | No absolute constants in predicates; every length comparison goes through `ToleranceCtx` |
| H-4 | No `cfg!(debug_assertions)`-dependent semantics — validity checks run always or are a named `_unchecked` variant |
| H-5 | Budget or bound, never a bare loop; geometry-dependent iteration spends from `&mut Budget` |
| H-6 | Certificates carry their method; a float-computed value is never recorded `Exact` |
| H-7 | Tests are three-layer: named witnesses, property tests, and a margin sweep where a margin parameter exists |
| H-8 | Anchors are symbols (file + enclosing symbol + pattern + expected count), never line numbers |

## Crates

| Crate | Role in the fork |
|---|---|
| `truck-base` | Evidence algebra (`Outcome`, `Certified`, `Refusal`, `Certificate`, `Method`, `Budget`, `Modulus`), tolerance types |
| `truck-geotrait` | Geometric traits; `Outcome`-returning `IncludeCurve`; `MetricSpace` migration |
| `truck-geometry` | NURBS/B-splines, analytic `specifieds`, `decorators` (`IntersectionCurve`, `Offset`, `RbfSurface`, `CoonsSurface`), `constructive/` spine/frame recipes, the canonical carrier model and structural recognizer |
| `truck-topology` | Vertex/edge/wire/face/shell/solid + the nine invariant checkers and `ManifoldDiagnostics` |
| `truck-polymesh` | `PolygonMesh` and polygon structures |
| `truck-meshalgo` | Certified tessellation, edge-sample ledger; realization-evidence and validity integration |
| `truck-modeling` | CAD facade: extrude, revolve, fillet, chamfer, split, section, booleans, placement |
| `truck-shapeops` | The boundary rewrite: split → classify → decide → assemble; legacy transversal booleans |
| `truck-stepio` | STEP ingestion through certified encoders with provenance |
| `truck-evidence` | Certified interval enclosures, exact predicates, the Krawczyk operator, the contact layer, stratified reach |
| `truck-certified` | Certified SSI, span BVH, Bernstein hulls, the `formal/` face-realization routes, the certificate-calculus engine |
| `truck-assembly`, `truck-derivers` | Upstream crates, largely untouched |

## Agentic CAD API surface

The entry point for building and modifying solids programmatically is
`truck_shapeops::facade` — build123d-shaped operations over a certified
kernel. Rust-only; the Python binding (pyo3) is booked but deliberately
deferred.

### How modeling works

1. **Sketch a closed profile** on the z = 0 plane out of line and circle
   pieces.
2. **Arrange it.** One call decides which regions of the plane are material
   (interiors, holes, nesting) — you never hand-build faces or sew shells.
3. **Lift it into 3-D**: extrude by a height, extrude along a vector, or
   revolve about the z-axis. You get back a closed, manifold solid —
   acceptance is checked before the solid exists.
4. **Modify**: place it (`translate` / `rotate` / `scale` / `mirror`), soften
   it (`fillet` / `chamfer`), combine it with others (add, subtract,
   intersect), or slice it (`section` / `split`).
5. **Measure or ship**: bounding box, topology iteration, certified
   tessellation, and export to STL / OBJ / VTK / STEP.

### What every call gives you back

Each fallible operation returns one of two things. Either your geometry
**plus a certificate** — how the answer was computed (exactly, with interval
proofs, or in floats), which invariants hold, and what compute it consumed —
or a **typed refusal** naming the precise reason nothing was produced: the
input used an unsupported curve/surface type, the compute budget ran out, the
contact was degenerate, and so on. You hand heavy numerics a small budget so
a hard problem fails fast instead of hanging. Nothing panics on bad geometry,
and a result the kernel could not verify is never dressed up as success.

In Rust terms that is `Outcome<T> = Result<Certified<T>, Refusal>`: take the
shape from `.value`, and match on the `Refusal` arms when you want retry or
fallback logic. Refusals are data, not error strings.

### The operations

| Intent | Operations (all in `truck_shapeops::facade`) |
|---|---|
| Create | `make_face`, `make_hull`, `extrude`, `extrude_vector`, `revolve` |
| Place | `translate`, `rotate`, `scale`, `mirror`, `mirror_about_plane` |
| Feature | `fillet`, `chamfer` |
| Combine | `boolean_op` (add / subtract / intersect), `section`, `split` |
| Query | `bounding_box` |

The names track build123d deliberately, so a build123d-shaped agent program
maps almost one-to-one. Anything the kernel cannot yet support is listed
under *Expressiveness envelope* below — and refuses with a typed reason
instead of approximating.

### End-to-end example

The shape below is kept compiling and passing as
[`truck-shapeops/tests/readme_surface_check.rs`](truck-shapeops/tests/readme_surface_check.rs).

```rust
use truck_base::cgmath64::{Point3, Vector3};
use truck_base::evidence::Budget;
use truck_geometry::arrange::arrange;
use truck_geometry::canonical::{Curve, Surface};
use truck_geometry::prelude::*;
use truck_shapeops::facade::{self, Mode};
use truck_topology::Solid;

// A CCW square profile on z = 0 (canonical carriers only).
let profile = vec![
    Curve::Line(Line(Point3::new(0.0, 0.0, 0.0), Point3::new(4.0, 0.0, 0.0))),
    Curve::Line(Line(Point3::new(4.0, 0.0, 0.0), Point3::new(4.0, 4.0, 0.0))),
    Curve::Line(Line(Point3::new(4.0, 4.0, 0.0), Point3::new(0.0, 4.0, 0.0))),
    Curve::Line(Line(Point3::new(0.0, 4.0, 0.0), Point3::new(0.0, 0.0, 0.0))),
];

// The 2-D arrangement decides material regions (winding numbers);
// extrusion lifts them into a closed, manifold solid. Note `.value`:
// `arrange` also returns a Certified<Arrangement>, not a bare one.
let arrangement = arrange(&profile, None).map_err(|e| format!("{e:?}"))?.value;
let block: Solid<Point3, Curve, Surface> =
    facade::extrude(&profile, &arrangement, 2.0).map_err(|e| format!("{e:?}"))?.value;

// Fallible ops take an explicit budget and hand back evidence.
let mut budget = Budget::new(1000, 1000, 1000);
let moved = facade::translate(&block, Vector3::new(1.0, 0.0, 0.0))
    .map_err(|e| format!("{e:?}"))?
    .value;
let _bb = facade::bounding_box(&moved, &mut budget).map_err(|e| format!("{e:?}"))?.value;

// Booleans: a smaller solid fully inside the block's footprint, punched out.
let inner = vec![ /* a 1x1 CCW square at (1,1) */ ];
let hole: Solid<Point3, Curve, Surface> =
    facade::extrude(&inner, &arrange(&inner, None)?.value, 2.0)?.value;
let _cut = facade::boolean_op(&block, Mode::Subtract, &hole, &mut budget)?.value;

// Refusals are first-class: match them instead of stringifying errors.
match facade::revolve(&profile, &arrangement, std::f64::consts::TAU) {
    Ok(certified) => { /* certified.value: Solid */ }
    Err(Refusal::UnsupportedEnvelope(case)) => { /* e.g. NonCanonicalCarrier */ }
    Err(e) => { /* inspect e: Refusal — see truck_base::evidence */ }
}
```

Two ergonomic facts agents trip on:

- **`Refusal` does not implement `std::error::Error`.** `?` into
  `anyhow::Result` or `Box<dyn Error>` will not compile. Match on the
  refusal, or map it into your error type (`map_err(|e| format!("{e:?}"))`
  is the convention in this repo's tests).
- **`truck_geometry::prelude` shadows `Result`.** It re-exports
  `errors::Result<T> = Result<T, Error>`, so a glob import replaces
  `std::result::Result` in that scope (and `?` then wants a
  `geometry::errors::Error`). Use `std::result::Result` explicitly in
  signatures of prelude-globbing modules.
- **Degenerate booleans refuse rather than guess.** Subtracting a solid
  from itself, or cutting along a coincident boundary plane, returns
  `UnsupportedEnvelope(ContactReductionDeferred)` or
  `Contradictory(..)` — construct inputs with strictly interior contact
  or accept the refusal.

### Beyond the facade

- `truck_modeling::{extrude, revolve, until, spine_sweep, facet_sweep}` —
  `extrude_until` (build123d `extrude(until=)`), taper, spine sweeps, and the
  direct facet-realization backend with winding audits.
- `truck_shapeops::boolean` — the composable boolean pipeline is public:
  `contact()` → `split_fragments()` → `classify_fragments()` →
  `fragment_decision()` → `boolean()`.
- `truck_evidence` — certified interval enclosures, exact analytic
  surface-pair intersections, the contact layer, and Krawczyk root proofs.
- `truck_certified` — certified parametric maps, quotient-domain/atlas
  substrate, and the formal face-realization routes (gated by
  `TRUCK_FORMAL_RECOVERY`).
- `truck_meshalgo::tessellation` — certified tessellation with per-face
  diagnostics; failures emit machine-readable `FaceDiagnosticRecord`s.
- Output: `truck-polymesh` (STL/OBJ/serde JSON), `truck-meshalgo` VTK,
  `truck-stepio` STEP.

## Expressiveness envelope

The kernel's contract is **certified construction or typed refusal — never a
silent approximation.** That makes the boundary of what it can build a
first-class part of the API, not a footnote. Three states per capability:

**Certified today** (every op carries evidence or refuses with a typed reason):

- Primitives (`cuboid`), `make_face`, `make_hull`
- Extrusion: by height, along a vector, with taper, `extrude_until`
  (build to a target plane), `project_profile`
- Revolution about the z-axis (line/circle profiles)
- Spine sweeps: `SpineFrameRecipe` (spine × profile law × frame law:
  fixed-plane, architectural-up, parallel-transport, radial) and Coons4 patches
- Booleans: union / difference / intersection of single-shell solids of
  canonical carriers (plus the public composable pipeline:
  contact → split → classify → decide → assemble)
- Fillets on plane-plane edges; circular-rim fillets; chamfers on straight edges
- Section / split by plane; mirror / rotate / scale / translate; bounding box
- Certified tessellation with per-face machine-readable diagnostics

**Compose it yourself** (facade/agent-layer recipes over the certified
primitives — no kernel work needed):

- Patterns: map + boolean-union loops
- Ribs and bosses: profile + `extrude_until` + boolean composition
- Steps and pockets: multiple arrangements + stacked extrusions

**Typed refusal today** (the op refuses with the matching `Refusal` arm and
the booking for its future program exists — but no construction spec is
landed):

- Fillets/chamfers beyond plane-plane edges and circular rims
  (topology-changing/face-consuming fillets are deferred by decision in
  `BUILD123D_COVERAGE_PLAN.md`)
- Loft / multi-section sweeps (substrate booked: `certified_map` clients in
  `CERTIFIED_PHASE1_BOOKING.md`)
- Multi-shell or non-canonical boolean inputs (booked as the RW-MULTISHELL
  fold in `SOLVER_FAMILY_PLAN.md`; v1 refuses typed)
- Shell / offset / thicken (post-hoc 3-D offsets are out by doctrine; a
  2-D-arrangement-offset wall recipe is the intended strictly-better form,
  not yet landed)
- Post-hoc face drafting, general (non-spine) sweeps, `ExtrudedCurve`
  emission

This boundary is a design surface, not an omission: outside it the kernel
would have to emit uncertified answers the way general-purpose kernels do.
The intended workflow for agents is to treat a refusal as a design signal —
reformulate with the certified verbs — and the refusal census over realistic
generated workloads is the instrument that decides which deferred program
books next.

## Sync model

The canonical kernel source is the `look` repository's `vendor/truck/` tree,
where all kernel changes land through the packet/worker/verify loop
(`loop/ORCHESTRATOR.md` in the look repo). This fork repository receives bulk
sync commits from that tree and hosts the kernel-side design documentation;
do not edit kernel code here and expect it to survive the next sync.

## License

Apache License 2.0, inherited from upstream. See `LICENSE` files in each crate.
