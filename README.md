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
under *Known limits* below — and refuses with a typed reason instead of
approximating.

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

### Expressiveness envelope

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

### The contract underneath

For readers who want the formal view — everything above is a thin table over
these rules (full details in [`OVERVIEW.md`](OVERVIEW.md) and
`truck-base/src/evidence.rs`):

- `Outcome<T> = Result<Certified<T>, Refusal>`; `Certified<T>` = value +
  certificate (properties, method `Exact | Interval | Float | None`,
  remaining budget, margin, modulus).
- Certified operations accept only **canonical carriers**
  (`truck_geometry::canonical`): curves are Line / Circle / BSpline / NURBS /
  Intersection / SpineFrame; surfaces are Plane / Cylinder / Sphere / Cone /
  Torus / Placed. Anything else refuses `NonCanonicalCarrier` before any
  contact work starts.
- Fallible numerics thread `Budget { subdiv, newton, depth }`; exhaustion is
  `NumericallyUnresolved { spent, witness }`, not a hang.
- Verdicts are three-valued (`CertifiedWithinTolerance | Failed |
  Inconclusive`); the house rules deny `unwrap`/`expect`/`panic`/
  `indexing_slicing` in kernel code.

## Sync model

The canonical kernel source is the `look` repository's `vendor/truck/` tree,
where all kernel changes land through the packet/worker/verify loop
(`loop/ORCHESTRATOR.md` in the look repo). This fork repository receives bulk
sync commits from that tree (latest: `ce43777d`, look rev `a28c095`) and
hosts the kernel-side design documentation; do not edit kernel code here and
expect it to survive the next sync.

## License

Apache License 2.0, inherited from upstream. See `LICENSE` files in each crate.
