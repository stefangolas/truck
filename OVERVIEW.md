# The certified B-rep kernel — what this fork is

This repository is a fork of [ricosjp/truck](https://github.com/ricosjp/truck)
whose core crates have been rewritten into a **certified boundary-representation
kernel**: every geometric answer is a typed outcome carrying the evidence that
produced it, and every refusal says why the kernel could not certify an answer
rather than returning an approximation silently.

The consuming application is `look` (a sibling repository), a native
GLB/STL/STEP screenshot tool; the fork is vendored there as `vendor/truck/` and
is developed through an automated packet/worker/verify loop (see "Where the
depth lives" below).

Scale of the rewrite to date: **+74,262 / −1,555 lines** across 205 files over
the vendored base (re-derive with
`git diff --shortstat da72cd5..HEAD -- vendor/truck` in the look repo).

## The model

1. **Typed outcomes, not exceptions.** Public geometric entry points return
   `Outcome<T>` — either `Certified<T>` (a value plus the `Certificate` that
   produced it) or a typed `Refusal` (`Empty`,
   `UnsupportedEnvelope(EnvelopeCase)`, `NumericallyUnresolved { spent,
   witness }`, `Contradictory`, `Collapsed`, …). Geometric predicates never
   panic and never return bare `bool`s.
2. **Certificates carry their method (H-6).** Every certificate records *how*
   the value was computed — `Method ∈ {Exact, Interval, Float, None}`. A value
   computed in floats is never recorded `Exact`.
3. **Three-valued verdicts.** `CERTIFIED_WITHIN_TOLERANCE | FAILED |
   INCONCLUSIVE`. Uncertainty surfaces as `INCONCLUSIVE`, never silently as
   success.
4. **Interval/certified numerics where it matters.** Krawczyk root isolation
   (the workspace's only operator: `formal/bezier_isect.rs`), outward-rounded
   interval evaluation, Bernstein subdivision, Shewchuk `Expansion` exact
   predicates, and hull enclosures of Bézier forms.
5. **No parallel validation universes.** New evidence composes with the
   existing carriers (`MeshedShellOutcome`, `FaceValidityCertificate`,
   provenance vocabulary). The unified booking table for new evidence kinds is
   the look repo's `docs/CERTIFICATE_MAPPING.md` (sections A–C); additions are
   spec edits, never worker discretion.
6. **Tolerance discipline.** `DirectTolerance` and a ratchet over tolerance
   sites; every comparison site is classified and counted, and the count only
   moves through reviewed packets.

## Capability map by crate

| Crate | What it is now |
|---|---|
| `truck-base` | The evidence algebra (`Outcome`, `Certified`, `Refusal`, `Certificate`, `Method`, `Budget`), tolerance types, cgmath re-exports. The algebra moved here (BG-S0-001) so trait-level code can return `Outcome` without dependency cycles. |
| `truck-geotrait` | `ParametricCurve`/`ParametricSurface` and friends, with `IncludeCurve` returning `Outcome<bool>`; the `MetricSpace` generic-bound migration. |
| `truck-geometry` | Knot vectors, B-splines/NURBS; analytic `specifieds` (cylinder, cone, sphere, …) returning certified outcomes; `decorators` (`IntersectionCurve`, `Offset`, rolling-ball `RbfSurface`, the new `CoonsSurface`); `constructive/` — the spine/frame/profile recipe contract of the constructive-geometry program (spine trait, frame laws: fixed-plane, architectural-up, radial, parallel-transport). |
| `truck-topology` | Vertex/edge/wire/face/shell/solid, plus the substrate diagnostics: `shell_condition`, `face_adjacency`, `singular_vertices`, `connected_components`, and the landed `ManifoldDiagnostics` aggregate with BFS `orientation_parity` (analysis only — never silent repair). |
| `truck-polymesh` | `PolygonMesh` and polygon data structures (upstream, consumed by the facet backend and tessellation). |
| `truck-meshalgo` | Certified tessellation: per-face constrained triangulation, `MeshedShellOutcome`, `FaceValidityCertificate`, face diagnosis/validity; the edge-sample ledger (`triangulation_with_ledger`, index-identity without welding). Plus the certified substrate: `tessellation/formal/` (37 modules behind `TRUCK_FORMAL_RECOVERY` — Krawczyk `bezier_isect`, analytic support identification for cylinder/cone/torus, planar slice/holes, envelope bounds, span/contact/branch incidence) and `tessellation/domain/` (cut-open fundamental domains: plan, quotient, canonical, deck). |
| `truck-modeling` | The CAD facade (extrude, revolve, fillet, chamfer, split, section, booleans, `make_face`/`make_hull`, placement ops) landed by the build123d coverage program; the constructive facet backend lands here next. |
| `truck-shapeops` | The boundary rewrite — split → classify → decide → assemble (`split.rs`, `classify.rs`, `assemble.rs`, `section.rs`) with typed-refusal envelope boundaries — and the legacy transversal Boolean path it is replacing. |
| `truck-stepio` | STEP ingestion through certified encoders (line, circle, ellipse, cylinder, cone, sphere, torus, B-spline/NURBS) with per-entity provenance and typed refusals. |
| `truck-evidence` | The reference implementation of the evidence pattern (build-spec item P-6); the algebra itself lives in `truck-base` and is re-exported here. |
| `truck-assembly`, `truck-derivers` | Upstream crates, largely untouched. |

The upstream render crates (`truck-platform`, `truck-rendimpl`, `truck-js`)
are not part of the vendored kernel; rendering lives in the `look` executable.

## Program status

- **Landed:** the base kernel loop (76/76), audit BG-AUDIT-001 (17/17), the
  analytic solver family, the build123d coverage program (P1–P12), the
  constructive-geometry contract and its first seven packets (contract,
  recipe/profile, analytic frame laws, parallel transport, edge-sample ledger,
  manifold diagnostics, Coons surface).
- **In flight:** the direct facet realization backend (shared-topology
  `PolygonMesh` emission, no sewing/welding), then certificate integration
  (CG-007) and the `SpineFrameSurface` topology constructor (CG-009).
- **Next program:** the certified-kernel plan (`CERTIFIED-KERNEL-PLAN.md`) —
  promotion of `formal/`+`domain/` into a `truck-certified` crate and the ten
  certified operations, sequenced against the constructive-geometry program by
  the unified mapping and the truck-meshalgo exclusivity rule.

## Where the depth lives

**As of 2026-08-31 the complete look documentation set is mirrored in this
repository under [`docs/`](docs/)** (all 74 files: the mathematical
foundation, build spec, formal systems, certificate mapping, wave/audit
records, defect index, benchmarks, images). Until references are reconciled
there are two copies; treat the **look repo's `docs/` as the working set** —
the autobuild loop reads and updates it — and this mirror as a convenience
snapshot. Sync is manual for now; it is pure mathematics and changes only
through reviewed packets, so drift is slow and detectable.

The load-bearing documents, whichever copy you read:

- `docs/MATHEMATICAL_FOUNDATION.md` — the mathematical bedrock and contract
  registry (topology/identity, conversion/provenance, periodic/quotient,
  arrangement, CDT, shell contracts; Parts I–VIII).
- `docs/GENERATION_KERNEL_BUILD_SPEC.md` — the build specification: house
  rules (H-1…), stages, the certified evaluation interface, the analytic
  solver track, global test obligations, and the landed item register
  (BG-* entries with their theorems and refusals).
- `docs/FORMAL_SYSTEM_BREP_GENERATION.md`,
  `docs/FORMAL_SYSTEM_STEP_INGESTION.md` — the two formal systems.
- `docs/CERTIFICATE_MAPPING.md` — the unified certificate/evidence mapping
  shared with the certified-kernel program.
- `docs/KERNEL_AUTOBUILD_LOOP.md`, and in the look repo `loop/ORCHESTRATOR.md`
  and `loop/STATE.md` — the autobuild loop that produces the kernel and its
  current state.

Native to this repository (not mirrored):

- `CONSTRUCTIVE-GEOMETRY-BUILD-SPEC.md` — the constructive geometry design
  (recipes, facet backend, TR-MESH-001, validation doctrine §9, phases).
- `CERTIFIED-KERNEL-PLAN.md` — the ten certified operations, contract freezes
  (F1–F3, X1–X2), phases and gates.
- `GEN-001.md`, `NIST_RECOVERY_HANDOFF*.md` — the generic substrate landing
  and the NIST corpus recovery record.
