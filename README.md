# truck — the certified B-rep kernel fork

`truck` is an open-source shape processing kernel in Rust by
[ricosjp](https://github.com/ricosjp/truck) (Apache-2.0). This repository is a
fork whose core crates have been rewritten into a **certified
boundary-representation kernel**: every geometric answer is a typed outcome
carrying the evidence that produced it, and every refusal states why the
kernel could not certify an answer instead of returning an approximation
silently.

**Start with [`OVERVIEW.md`](OVERVIEW.md)** — what the fork is, the
certification model, the capability map by crate, program status, and where
the deep specifications live. The mirrored documentation snapshot is under
[`docs/`](docs/).

## Status

The rewrite is developed through an automated packet/worker/verify loop
driven by the `look` repository (the kernel is vendored there as
`vendor/truck/`). Landed to date: the base kernel program, the audit
remediation, the analytic solver family, the build123d coverage program, and
the constructive-geometry contract with its spine/frame/profile, diagnostics,
ledger, and Coons surface packets. The certified-kernel promotion plan is
booked in [`CERTIFIED-KERNEL-PLAN.md`](CERTIFIED-KERNEL-PLAN.md).

## Crates

| Crate | Role in the fork |
|---|---|
| `truck-base` | Evidence algebra (`Outcome`, `Certified`, `Refusal`, `Certificate`, `Method`, `Budget`), tolerance types |
| `truck-geotrait` | Geometric traits; `Outcome`-returning `IncludeCurve`; `MetricSpace` migration |
| `truck-geometry` | NURBS/B-splines, analytic `specifieds`, `decorators` (`IntersectionCurve`, `Offset`, `RbfSurface`, `CoonsSurface`), `constructive/` recipes |
| `truck-topology` | Vertex/edge/wire/face/shell/solid + substrate diagnostics and `ManifoldDiagnostics` |
| `truck-polymesh` | `PolygonMesh` and polygon structures |
| `truck-meshalgo` | Certified tessellation, edge-sample ledger; `formal/` (Krawczyk, analytic support schemas) and `domain/` (quotient/deck) substrate |
| `truck-modeling` | CAD facade: extrude, revolve, fillet, chamfer, split, section, booleans, placement |
| `truck-shapeops` | The boundary rewrite: split → classify → decide → assemble; legacy transversal booleans |
| `truck-stepio` | STEP ingestion through certified encoders with provenance |
| `truck-evidence` | Reference implementation of the evidence pattern (P-6) |
| `truck-assembly`, `truck-derivers` | Upstream crates, largely untouched |

## Agentic CAD API surface

This is the concrete entry surface for building and modifying solids
programmatically (including from agent harnesses). It is Rust-only; the
Python binding (pyo3) is booked but deliberately deferred.

### The certified contract

Every fallible geometric operation returns evidence, never a bare value and
never an approximation:

```rust
type Outcome<T> = Result<Certified<T>, Refusal>;
```

- `Certified<T>` carries the value plus a `Certificate` (the evidence tuple:
  properties, method `Exact | Interval | Float | None`, remaining `Budget`,
  margin, modulus).
- `Refusal` is a typed, matchable vocabulary (`Empty`,
  `UnsupportedEnvelope(..)`, `NumericallyUnresolved { spent, witness }`,
  `NonCanonicalCarrier`, `Collapsed { .. }`, `ForwardToleranceExceeded`,
  ...) — treat refusals as control flow, not errors. Nothing panics on bad
  geometry; house rules deny `unwrap`/`expect`/`panic`/`indexing_slicing`.
- Fallible numerics thread an explicit `Budget { subdiv, newton, depth }`
  so a caller can bound work per call; exhaustion is a typed refusal, not a
  hang.
- Verdicts are three-valued where honesty matters
  (`CertifiedWithinTolerance | Failed | Inconclusive`) — an inconclusive
  result is never promoted to success.

Geometry must use the **canonical carriers** (`truck_geometry::canonical`):
`Curve` = Line / Circle / BSpline / NURBS / Intersection / SpineFrame, and
`Surface` = Plane / Cylinder / Sphere / Cone / Torus / Placed. Certified
operations refuse `NonCanonicalCarrier` on anything else.

### The build123d-shaped facade: `truck_shapeops::facade`

The intended modeling entry point — build123d-named operations that compose
landed certified primitives and add zero new geometry:

| Op | Signature sketch |
|---|---|
| `make_face` | `(&[Curve]) -> Outcome<Vec<Face>>` — one face per material region |
| `make_hull` | `(&[Point3]) -> Outcome<Face>` — exact 2-D convex hull |
| `extrude` | `(&[Curve], &Arrangement, height) -> Outcome<Solid>` |
| `extrude_vector` | `(&[Curve], &Arrangement, dir, both: bool)` |
| `revolve` | `(&[Curve], &Arrangement, angle)` — about the z-axis |
| `boolean_op` | `(&Solid, Mode::{Add, Subtract, Intersect}, &Solid, &mut Budget)` |
| `fillet` / `chamfer` | `(&Solid, specs, &mut Budget)` — `BlendSpec::Straight(..)` / `BlendSpec::Circular(..)` |
| `section` / `split` | `(&Solid, &Plane, &mut Budget)` — cut faces / `(plus, minus)` halves |
| `mirror` / `mirror_about_plane` / `rotate` / `scale` / `translate` | `(&Solid, ..) -> Outcome<Solid>` |
| `bounding_box` | `(&Solid, &mut Budget) -> Outcome<BoundingBox<Point3>>` |

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

### Known limits (typed, not silent)

Booleans require single-shell solids of canonical carriers; fillets cover
plane-plane edges and circular rims; `mirror` (solid) is axis-aligned planes
(`mirror_about_plane` is general); revolve is z-axis with line/circle
profiles; no shell/offset/thicken, patterns, or sketch solver. Unsupported
requests refuse with the matching `Refusal` arm.

## Sync model

The canonical kernel source is the `look` repository's `vendor/truck/` tree,
where all kernel changes land through the packet/worker/verify loop
(`loop/ORCHESTRATOR.md` in the look repo). This fork repository receives bulk
sync commits from that tree (latest: `ce43777d`, look rev `a28c095`) and
hosts the kernel-side design documentation; do not edit kernel code here and
expect it to survive the next sync.

## License

Apache License 2.0, inherited from upstream. See `LICENSE` files in each crate.
