# truck — the certified B-rep kernel fork

`truck` is an open-source shape processing kernel in Rust by
[ricosjp](https://github.com/ricosjp/truck) (Apache-2.0). This repository is a
permanent architectural fork whose core crates have been rewritten into a
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

## Design principle

Every fallible kernel operation returns
`Outcome<T> = Result<Certified<T>, Refusal>`: either the geometry together
with the certificate that produced it, or a typed refusal naming the reason
no answer was certified.

## Main theorems

**Krawczyk existence and uniqueness.**
`K = m − Y·F(m) + (I − Y·F'(box))·(box − m)`. `K ⊆ interior(box)` proves a
unique root in the box. `K ∩ box = ∅` proves no root in the box. Any other
result bisects under budget. Validity holds for any invertible preconditioner
`Y`; `Y` affects tightness only. Used in: root isolation
(`formal/bezier_isect.rs` 2-D, `ssi.rs` square 3×3) and branch certification
throughout the contact funnel.

**Three-state root isolation.**
Every box classifies as Subdivide, Excluded (Bernstein range excludes zero),
or Root (Krawczyk strict inclusion). Multiple or tangential roots return
`NumericallyUnresolved`, never an empty list. Used in: contact curve
extraction; the empty-list case is the silent-wrong-answer class the
discipline exists to prevent.

**Enclosure soundness (BG-ENC-001/002/003).**
`enclose(box) ⊇ { f(p) : p ∈ box }` for every carrier and box. Interval
arithmetic rounds outward. Under-estimation invalidates every certificate
built on it. Used in: every certified interval evaluation in the kernel.

**Exact predicates.**
An `Expansion` (Shewchuk arithmetic) is a non-overlapping `f64` component
list that sums with zero rounding error to the exact value; its `sign`
decides the exact sign of that value. No tolerance establishes topology.
Used in: analytic line–line, line–circle, circle–circle intersection
decisions, and the span BVH sign rows.

**Bernstein hull enclosure.**
For a patch in Bézier form, the convex hull of control points is a
conservative range enclosure of the patch and its derivative patches over any
compact subbox. Hulls enclose polynomial quantities only; rational quantities
require certified-positive weights and directed-rounded division. Used in:
exclusion tests, the `∇h = 0` critical-point systems, and span BVH leaf
bounds.

**Theorem 3 (exclusion = hull separation).**
`0 ∉ conv{c_αβ} ⇔ conv{P¹} ∩ conv{P²} = ∅` on every subbox. Separating
directions inherit downward monotonically. One exact sign row
(`min_α λ·P¹_α > max_β λ·P²_β`, decided by `Expansion` predicates) prunes
every descendant span pair. Used in: the per-carrier span BVH broadphase; a
float search (GJK) selects candidate directions and is never evidence.

**Theorem 4 (implicit reduction, analytic × spline).**
Contact between a recognized analytic carrier (plane or quadric) and a spline
patch is the zero set of `h = g∘S` on the spline chart, in exact Bernstein
form. Critical points are the 2×2 system `∇h = 0` for the Krawczyk operator.
Partial derivatives are degree-elevated to a common bidegree before pairing.
Used in: the spline×analytic contact stage of the funnel
(`contact/implicit2d.rs`).

**Gauss-map cone certificates.**
Disjoint gradient-direction cones over a box prove: no tangency inside the
box (rank deficiency ⇔ parallel normals), loop-freedom (every branch meets
the box boundary), and a fixed continuation axis with certified margin.
Verdicts: `LoopFree | TransversalByProof | AxisFixed`. Used in: contact
funnel screening; boundary-crossing seeds are complete on `LoopFree` cells.

**Crossing-angle identity (Theorem 2.1).**
`σ_min(DF̂) = √(1 − |n_A · n_B|)` exactly, with `‖DF̂‖ = √2` exactly.
Conditioning depends only on the crossing angle of the two surface normals,
never on NURBS parameterization speed. Used in: solver conditioning and
escalation thresholds.

**Exact analytic degeneracy (BG-ANA).**
Two cylinders are tangent iff `|d_axes| = r₀ ± r₁`; a plane is tangent to a
sphere iff `dist(centre, plane) = r`; coaxiality and coincidence are exact
conditions on carrier parameters. Analytic pairs return `Proven` with
`Method::Exact` or a typed degenerate classification, never a float-certified
result. Used in: analytic pair dispatch; the margin sweep must switch
transverse → tangent → disjoint with no wrong-but-confident band.

**Stratified reach (BG-FID-001).**
The certified quantity is the per-stratum lower bound
`lfs_σ = min(ρ̲(σ), dist̲(x, non-incident strata), ϱ̲_wedge(x))`. The global
reach of a mechanical B-rep is zero at every sharp edge, so global reach is
never used. Gates of the form `q < c · lfs_σ̲` can only refuse. Used in:
conditioning gates, isotopy margins, clustering radii.

**Isotopy lemma and the one-sheet condition.**
Two-sided Hausdorff closeness, an almost-tangency bound, and boundary
correspondence prove a covering of some degree, not a homeomorphism. The
double-covered circle `(R + ε·cos(t/2))·e(t)`, `t ∈ [0, 4π]`, passes every
metric condition. Degree-one covering per component (Krawczyk fibre
isolation) or fibrewise uniqueness on a certified partition discharges it.
Used in: sweep and offset embedding certificates.

**The nine B-rep invariants (BG-INV-101…109).**
Coedge pairing, vertex link = single cycle, Euler–Poincaré (necessary only),
same-parameter, domain–boundary correspondence, representation in 𝒢 within
τ_rep, tolerance monotonicity, shell nesting (forest, by certified winding),
wedge non-degeneracy. A violated invariant is `Contradictory`; an
undecidable one is a typed refusal. Used in: topology validity
(`truck-topology/src/invariants/`).

**Watertightness by index identity.**
A mesh position index is a pure function of (entity identity, sample
ordinal), never of coordinates. Incident faces satisfy
`I(A, E) == reverse(I(B, E))` as integer sequences. The emitted mesh is
edge-watertight by construction; positional welding is never invoked. A
failed winding audit is `FAILED`, never a warning. Used in: the direct facet
realization backend (`truck-geometry/src/constructive/`).

**Membership by propagation.**
For a connected face of A cut by Ξ = f ∩ ∂B, χ_B is locally constant on
f∖Ξ and the dual adjacency graph is connected. One certified seed per face
determines all fragments. Flip parity is contact order; odd flips, even does
not; tangential arcs are even-order and never flip. Coincident fragments are
decided by the material state 4-tuple. Material parity is the boundary's
winding number mod 2. Used in: boolean fragment classification and
orientation.

**Cut-open existence.**
Every compact connected orientable two-manifold with boundary admits a
finite cut graph whose cut-open domain is a planar polygonal schema. Used
in: the quotient/deck fundamental-domain machinery and certified integer
deck arithmetic over ambient periods.

**Epistemic closure (§22).**
Under the ordering obligations, totality, and mutual exclusivity of the
outcome classifier, every input terminates within its ledger and returns
exactly one terminal outcome. No path ends unclassified; no path returns a
construction whose preconditions were not established. Used in: the dispatch
DAG as a whole.

**The composition theorem and the modulus contract.**
If `ε_{i+1} ≤ ω_i(ε_i) + τ_i` with `ε_i < 𝔪_i` at every step, the
combinatorial result equals the exact result and the geometric error obeys
the nested recurrence. Subadditivity is read off the modulus shape, never
declared; composing a non-subadditive operand refuses. A near-degenerate
cell publishes an honest `Pole` instead of `Unbounded`. Used in: forward
error bounds across operation chains (`Modulus` in `truck-base/src/
evidence.rs`).

**STEP ingestion symbolic closure (Theorem 1).**
The atlas procedure terminates and produces exactly one outcome:
`Valid | Inconsistent | Ambiguous | Unsupported | Unresolved`. Valid outputs
are gauge-invariant. Proved from finite combinatorics (Lemmas 1–10); the
assumed parts (numerical embedding, solver completeness) are separated
explicitly. Used in: STEP face ingestion (`truck-stepio`).

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
