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

## License

Apache License 2.0, inherited from upstream. See `LICENSE` files in each crate.
