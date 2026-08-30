# Truck Constructive Geometry Kernel — Build Specification

**Status:** Reference plan for the next implementation phase.
**Supersedes:** the prior direct-BREP draft specification (which itself superseded the original eight-packet kernel proposal).
**Scope:** TR-SWP-001, TR-SWP-002 (facet + parametric modes), TR-MESH-001, TR-TOP-001 (reduced), TR-VAL-001 (integration), TR-GEO-001. TR-GEO-002, TR-DIR-001, TR-NRB-001, TR-INS-001 deferred with explicit promotion gates.

---

## 0. Purpose and governing thesis

This document defines the next implementation phase for Truck's constructive-geometry path.

The governing thesis:

```text
authored topology
→ shared boundary geometry
→ direct realization
→ separated topological/geometric certification
→ topology-preserving tessellation
```

A procedural client often knows BREP incidence before geometric realization begins. The kernel should preserve that knowledge rather than discarding it and recovering adjacency through geometric proximity, sewing, or Boolean reconstruction.

The Exeter benchmark does **not** replace this design. It provides empirical evidence about which parts Truck already handles well and which missing abstractions are worth promoting into the kernel. The benchmark established that complex procedural geometry can already be built using existing Truck primitives by:

```text
solve geometry analytically
→ sample explicit points
→ construct explicitly shared vertices/edges
→ emit planar faces
→ assemble Shell/Solid
```

and that ribs and vault webs can be built without generic pipe-shell, filling, sewing, or healing operations.

Therefore the goal of this phase is **not to invent another BREP kernel on top of Truck**. The goal is to make generic constructive operations reusable while preserving the extremely fast direct-facet path demonstrated by the benchmark.

---

## 1. Normative design principles

The words **MUST**, **SHOULD**, and **MAY** are normative.

### 1.1 Truck topology remains authoritative

Truck's existing `Vertex`, `Edge`, `Wire`, `Face`, `Shell`, `Solid` remain the sole topological representation. No parallel half-edge graph or application-specific topology kernel SHALL be introduced.

Truck already provides identity-based incidence, orientation-independent edge identity, shell classification, connectivity, boundary extraction, singular-vertex detection, face adjacency, and unique-edge tessellation infrastructure.

### 1.2 Incidence is authored, not inferred

If two faces are intended to share an edge, they MUST reuse the same Truck `Edge` identity. Geometric coincidence MUST NOT imply topological identity. Two distinct edges MAY occupy identical geometric loci. No coordinate hash-consing belongs in the constructive path.

### 1.3 Analytic/constructive fast paths are first-class

The kernel MUST NOT force procedural clients through smooth-surface solvers when direct constructive evaluation is sufficient. The following path is a first-class output mode:

```text
analytic/parametric recipe
→ explicit samples
→ explicit facets
→ exact shared topology
```

It MUST NOT be regarded merely as an interim approximation on the way to smooth BREP. For rendering, visualization, collision, large procedural assemblies, and many simulation preprocessors, the facet realization may be the preferred representation.

### 1.4 Geometry recipe and realization backend are distinct

A constructive operation SHOULD describe **what geometry exists** independently from how it is realized. Example:

$$X(s,v)=C(s)+T(s)P(s,v)$$

describes profile transport along a framed spine. It may be realized as:

```text
FACET BACKEND
recipe → sampled rings → planar facets → PolygonMesh
```

or:

```text
PARAMETRIC BACKEND
recipe → SpineFrameSurface → trimmed BREP faces
```

Both consume the same design semantics. See §8 for the output-doctrine consequences.

### 1.5 Topological validity and geometric validity remain separate

A combinatorially closed shell can still have folded surfaces, self-intersections, collapsed patches, incorrect curve/surface realization, or zero volume:

$$\text{TopologyValidity} \neq \text{GeometryValidity}.$$

New diagnostics MUST integrate with the fork's existing certification/provenance system rather than establishing a second incompatible validation vocabulary.

### 1.6 Domain-specific generators remain client-side

Truck MUST NOT know about starcuts, tiercerons, muqarnas, tracery, architectural bays, cooling-network design, rocket-nozzle design, turbine-blade design, period solvers, or ornamental grammars. Clients decide topology and design intent. Truck realizes generic geometry.

### 1.7 Determinism (normative summary)

For identical ordered input and tolerance, the following MUST be stable: topology counts, orientation results, spine sample parameters, facet diagonal choices, mesh edge subdivisions, validation verdicts. Parallel computation MUST write into deterministic indexed slots. Floating-point reductions affecting certification MUST use deterministic reduction order. Hash-map iteration order MUST NOT define observable output ordering.

### 1.8 Tolerance doctrine (normative summary)

Topology identity NEVER uses tolerance. Numerical geometry does. New code SHOULD consume the fork's existing tolerance/evidence context where possible rather than creating an unrelated global constant or duplicate tolerance hierarchy. Tolerance-bearing operations include curve endpoint checks, surface boundary checks, Jacobian lower bounds, sampling, approximation, and NURBS fitting.

---

## 2. Packet disposition

| Packet | Status for next phase | Change from original |
| --- | --- | --- |
| `TR-DIR-001` | **DEFER / MINIMIZE** | Native Truck handles already provide authored topology (§3) |
| `TR-TOP-001` | **REDUCED** | Aggregate/extend existing diagnostics only (§4) |
| `TR-GEO-001` | **KEEP** | Generic Coons4 remains valuable (§5) |
| `TR-GEO-002` | **DEFER** | Triangular transfinite patch not justified yet (§6) |
| `TR-SWP-001` | **HIGHEST PRIORITY** | Strongly validated by Exeter and non-architectural cases (§7) |
| `TR-SWP-002` | **HIGHEST PRIORITY** | Explicit facet + parametric realization modes (§8) |
| `TR-VAL-001` | **KEEP / INTEGRATE** | Reuse existing certificate/evidence infrastructure (§9) |
| `TR-MESH-001` | **KEEP / NARROW** | Edge-first sampling exists; preserve IDs through output assembly (§10) |
| `TR-NRB-001` | **FOLLOW-ON** | Required for smooth STEP interoperability (§11) |
| `TR-INS-001` | **OPTIONAL** | Profile separately after core work (§12) |

---

## 3. TR-DIR-001 — Direct construction facade

### Status

**Deferred from the critical path.**

### 3.1 Finding

The original proposal specified an explicit-key `DirectBrepBuilder<VK, EK, FK>` over Truck topology. The audit and the Exeter implementation demonstrate that Truck's existing handles already allow callers to author incidence directly. Existing constructors and identity semantics already implement the hard part of the proposed behavior. A keyed registry is therefore currently an **ergonomic abstraction**, not a missing kernel capability.

### 3.2 Next-phase rule

Do NOT implement a large `DirectBrepBuilder` packet during the core phase. Clients MAY maintain `HashMap<VertexKey, Vertex>`, `HashMap<EdgeKey, Edge>`, `HashMap<FaceKey, Face>` locally.

### 3.3 Private grid registries (amendment)

Realization backends still need keyed entity caches internally. The facet backend's structured grid (vertex `(i,j)` created exactly once, §8A.4) **is** a keyed registry with `(i,j)` as the key. Therefore:

- FAC MUST implement one private, internal grid/entity registry — not a public builder API, not a second ad-hoc registry per call site.
- The same internal registry pattern MUST be reused (not re-implemented) when TR-SWP-002-BREP lands, so the codebase does not accumulate divergent hand-rolled registries.
- This internal type is a kernel implementation detail; it MUST NOT be promoted to public API by the promotion gates of §22 without its own justification.

### 3.4 Promotion criterion

Reconsider a kernel-level public builder only if at least two unrelated clients independently reproduce substantial boilerplate involving logical key management, orientation handling, debug provenance, or typed construction failures, and a shared abstraction clearly reduces complexity.

### 3.5 Forbidden behavior (applies to any future builder)

Any future builder MUST NOT merge by coordinate, infer adjacency, silently sew, or silently orient geometry.

---

## 4. TR-TOP-001 — Strong manifold diagnostics

### Status

**Reduced but retained.**

### 4.1 Existing substrate

Truck already exposes substantial topology analysis, including `shell_condition()`, `connected_components()`, `extract_boundaries()`, `singular_vertices()`, and `face_adjacency()`. Its shell classification already identifies invalid high-degree edge incidence rather than merely algebraically canceling orientations. Do NOT duplicate these algorithms.

### 4.2 Required deliverable

Provide a thin diagnostic aggregation layer:

```rust
pub struct ManifoldDiagnostics {
    pub shell_condition: ShellCondition,
    pub connected_components: usize,
    pub boundary_edges: Vec<EdgeID>,
    pub irregular_edges: Vec<EdgeDiagnostic>,
    pub singular_vertices: Vec<VertexDiagnostic>,
    pub orientation_conflicts: Vec<OrientationDiagnostic>,
}
```

Exact names are non-normative. The purpose is **actionable explanation**, not a second topology system.

### 4.3 Vertex-link refinement

If existing `singular_vertices()` cannot distinguish all relevant local topologies, add explicit vertex-link classification. For a closed 2-manifold, the link $\operatorname{lk}(v)$ must form one cycle. For a manifold with boundary, it must form one path. Detect: two sheets touching only at one vertex; multiple disconnected local fans; branching local fan.

### 4.4 Orientation

Provide orientation analysis. Automatic mutation is optional. Preferred API: `shell.orientation_diagnostics()`, which returns a consistent parity assignment or the conflicting edges/faces. A separate explicit operation MAY apply the computed assignment. No silent orientation repair.

### 4.5 Outward orientation

For a consistently oriented closed triangulated shell:

$$V=\frac16\sum_{\triangle(a,b,c)} a\cdot(b\times c).$$

Use the sign of $V$ to determine global inward/outward orientation. Do not use global-centroid normal tests (valid concave solids are not guaranteed star-shaped). Reuse existing `CalcVolume` where possible.

### 4.6 Acceptance fixtures

```text
closed cube
cube missing face
one reversed cube face
three faces sharing one edge
four faces sharing one edge
two shells touching only at a vertex
two disconnected shells
concave valid solid
```

---

## 5. TR-GEO-001 — Coons4 boundary patch

### Status

**Retained, but removed from the rendering critical path.** Exeter demonstrates that complex web geometry does not require Coons surfaces for fast visualization; its webs can be built from shared sampled boundaries and planar facets. That finding changes priority, not generic usefulness.

### 5.1 Purpose

Provide a four-boundary transfinite surface:

$$S(u,v) = (1-v)c_0(u)+vc_1(u)+(1-u)d_0(v)+ud_1(v)-B(u,v)$$

with:

$$B(u,v)=(1-u)(1-v)P_{00}+u(1-v)P_{10}+(1-u)vP_{01}+uvP_{11}.$$

The boundary-correctness property follows by exact pairwise cancellation against the bilinear corner term (e.g. $S(u,0)=c_0(u)$ since $d_0(0)=P_{00}$, $d_1(0)=P_{10}$, and $B(u,0)=(1-u)P_{00}+uP_{10}$), and is verified numerically in tests. The patch boundary is the input data, not an approximation.

### 5.2 Required properties

The surface MUST reproduce all four supplied boundary curves exactly under their declared parameter correspondence ($S(u,0)=c_0(u)$, $S(u,1)=c_1(u)$, $S(0,v)=d_0(v)$, $S(1,v)=d_1(v)$). The constructor MUST validate corner consistency. It MUST NOT silently guess arbitrary boundary orientation. A convenience constructor MAY attempt finite legal reversals and return the chosen correspondence explicitly.

### 5.3 Required traits

Implement the appropriate fork equivalents of:

```text
ParametricSurface
ParametricSurface3D
BoundedSurface
ParameterDivision2D
SearchParameter<D2>
Invertible
Transformed<Matrix4>
IncludeCurve
```

where compatible. First derivatives MUST be analytic. Second derivatives SHOULD be analytic where all boundary curves support them.

### 5.4 Regularity

Do not assume a valid boundary implies a valid interior. Expose $J(u,v)=S_u\times S_v$. Certification must be able to demonstrate or refute $\|J(u,v)\|>\tau_J$. Folded Coons surfaces are construction-valid but geometry-invalid.

### 5.5 Generic acceptance fixtures

```text
planar rectangle
bilinear warped quad
single curved boundary
four compatible curved boundaries
incompatible corners
reversed-boundary fixture
known folded patch
```

No cathedral fixture in the kernel acceptance suite.

---

## 6. TR-GEO-002 — Triangular transfinite surface

### Status

**Deferred.** Do not implement during this phase.

### 6.1 Reason

Exeter demonstrated that triangular curved regions can be represented adequately by directly faceted geometry. Meanwhile Truck's core parametric surface abstraction is rectangular-domain oriented, making a genuine triangular parametric surface require special handling for domain mapping, collapsed parameter boundaries, `SearchParameter`, `ParameterDivision`, and Jacobian interpretation near the collapsed boundary. This was identified as the primary trait-system design risk in the original proposal.

### 6.2 Promotion gate

Implement only if another independent client demonstrates a real requirement for a smooth, true three-sided parametric surface. Until then use planar triangles, direct faceted curved regions, quad decomposition, or existing smooth alternatives.

---

## 7. TR-SWP-001 — Spine/frame constructive geometry

### Status

**Highest priority.** This is the strongest abstraction identified by both the original design and the Exeter benchmark. The Exeter client independently implemented the sampled form of $X(s,v)=C(s)+T(s)P(s,v)$ to construct its molded ribs. The same abstraction applies naturally to ducts, rails, moldings, coolant passages, rocket cooling channels, internal turbine passages, seals, and structural profiles.

### 7.1 Separation of recipe from representation

Implement a geometry recipe independent of realization backend. Conceptually:

```rust
pub struct SpineFrameRecipe<S, P, F> {
    spine: S,
    profile_law: P,
    frame_law: F,
}
```

Core evaluator: $X(s,v)=C(s)+T(s)P(s,v)$. Required conceptual API:

```rust
position(s, v) -> Point3
frame(s)       -> Frame3
profile(s, v)  -> Point2
```

Expose derivatives where available.

### 7.2 Spine smoothness contract (amendment)

**MVP spines MUST be $C^1$ continuous** on the evaluated parameter interval. Piecewise-linear spines have undefined tangents at corner vertices; with only the MVP frame laws this yields silent garbage or panics.

- The recipe MUST typed-refuse (not clamp, not silently smooth) a spine declared or detected as non-$C^1$ under the MVP laws, with an actionable error naming the offending parameter.
- A minimal explicit corner policy (`Miter` only) MAY be added as an extension; `Bevel`/`Round` MAY follow. Any corner policy MUST be explicit caller input — never implicit.
- Detection may be declaration-based (spine type) or sampling-based (tangent discontinuity beyond tolerance); the chosen mechanism MUST be documented and deterministic.

### 7.3 Frame laws

Required MVP:

```rust
FixedPlane { normal }
ArchitecturalUp { up }
ParallelTransport { initial_normal }
RadialAboutAxis { origin, axis }
```

Do not use implicit Frenet framing as the generic default.

**FixedPlane.** For planar spines: $t=C'/\|C'\|$, $b=n_{\text{plane}}$, $n=b\times t$. Reject zero tangent. Preferred for planar arcs and similar ribs/rails.

**ArchitecturalUp.** Given preferred vector $u$: $b=\dfrac{u\times t}{\|u\times t\|}$, $n=t\times b$. Reject the singular case $u\parallel t$ unless the caller supplies an explicit fallback policy.

**ParallelTransport.** Implement a rotation-minimizing (Bishop) frame. Requirements: stable at zero curvature; stable through inflections; deterministic from initial normal; minimal accumulated twist. The implementation SHOULD use a well-defined discrete parallel-transport construction (double-reflection method) rather than approximated Frenet frames.

**RadialAboutAxis.** For rotation around a fixed axis, derive the frame analytically. Rotated copies MUST remain equivariant under the declared rotation, modulo floating-point evaluation.

### 7.4 Profile laws

Required:

```rust
Constant(Profile2D)
Scale { profile, scale_law }
LinearCorrespondence { start, end }
```

A profile correspondence MUST be explicit. Do not attempt to infer correspondence between arbitrary profile topologies. Arbitrary split/merge profile topology is out of scope.

### 7.5 Failures

Typed refusal for:

```text
zero tangent
non-C1 spine (per §7.2)
non-finite frame
invalid supplied normal
profile correspondence mismatch
profile collapse
non-finite output
```

Global self-intersection is not this packet's responsibility (see §9.6 for the reduced global check).

### 7.6 Acceptance fixtures

Kernel tests:

```text
straight rectangular sweep
90-degree curved duct
tapered rectangular duct
circular pipe bend
S-shaped rail
variable-radius coolant passage
annular/radial sweep
polyline-spine refusal fixture (typed error, per §7.2)
```

External regression: one Exeter rib. The Exeter fixture is a benchmark/regression, not normative kernel semantics.

---

## 8. TR-SWP-002 — Spine/profile realization

### Status

**Highest priority.** This packet has **two realization modes**, sharing TR-SWP-001 geometry.

### 8.0 Realization output doctrine (amendment)

The two modes have different contracts regarding BREP topology:

- The **parametric mode** preserves the classic invariant: *tessellation density MUST NOT change BREP topology* (§8B.2).
- The **facet mode** intentionally violates that invariant if it emits BREP: an $m\times k$ grid becomes $m\cdot k$ planar faces. A faceted BREP shell is expensive to later smooth, heal, or use in booleans.

Therefore:

1. **`PolygonMesh` with exact shared indices is the primary, contractual output of the facet backend.**
2. Faceted `Shell`/`Solid` emission is an explicit opt-in secondary target, documented with its topology-count consequence, intended for cases where BREP consumption (e.g. downstream booleans) is explicitly required.
3. The index-identity convention that makes both facet output and TR-MESH-001 watertight (sample once, share integer indices by identity, never by coordinates) is defined **once**, in Phase 0 (§20), and both modes MUST conform to it.

---

## 8A. TR-SWP-002-FAC — direct facet realization

### Status

**Mandatory first backend.** The Exeter benchmark demonstrates that this representation can provide extremely low construction and rendering cost while retaining exact combinatorial closure.

### 8A.1 Input

```text
SpineFrameRecipe
sampling policy
profile topology
cap policy
output target
```

Output target (per §8.0):

```text
PolygonMesh                 (primary, contractual)
Truck Shell/Solid           (explicit opt-in; m·k planar faces)
```

### 8A.2 Structured sampling

For spine parameters $s_0,\ldots,s_m$ and profile vertices $p_0,\ldots,p_{k-1}$, evaluate $x_{ij}=X(s_i,p_j)$. The resulting structured grid determines topology directly.

### 8A.3 Identity rule

Grid vertex `(i,j)` is created exactly once (via the private grid registry, §3.3). Adjacent faces reuse that identity. Internal grid edges are similarly created once and traversed oppositely by adjacent faces. No geometric sewing.

### 8A.4 Face generation

For the grid cell `(i,j)–(i+1,j)–(i+1,j+1)–(i,j+1)` emit one quad if planarity is explicitly certified and the target supports it safely; otherwise two triangles. Diagonal choice MUST be deterministic. Do not choose diagonals opportunistically based on unstable floating-point comparisons.

### 8A.5 Sampling policies

Provide at least:

```rust
UniformCount
CustomParameters
```

Strongly preferred:

```rust
ChordTolerance
AngularTolerance
```

Sampling affects geometric approximation only. It MUST NOT alter recipe semantics.

### 8A.6 Caps

Closed planar profile start/end rings may be capped using existing planar face/triangulation support. Arbitrary nonplanar cap solving is out of scope.

### 8A.7 Performance contract

The fast path MUST NOT invoke surface fitting, Newton iteration, sewing, healing, Boolean operations, or generic surface/surface intersection. The hot loop should be dominated by curve evaluation, frame evaluation, profile transform, and index emission.

### 8A.8 Global sanity checks on facet output (amendment)

The facet backend can emit geometrically self-intersecting sweeps (tight bends, scale-through-zero) that pass every local check. The BVH remains deferred (§9.6), but the facet output MUST pass a cheap mesh-level sanity audit before certification:

```text
signed-volume sign sanity (§4.5)
twin-triangle winding audit: every interior mesh edge referenced by exactly two
  triangles with opposite winding (composable with the §10.4 invariant check)
optional deeper check via existing mesh collision analyzers (analyzers/collision.rs)
```

The audit's verdict participates in the tri-state doctrine of §9.7. A failed winding audit is `FAILED`, not a warning.

### 8A.9 Exeter regression criterion

The extracted kernel implementation MUST be benchmarked against the existing local cathedral implementation. Measure:

```text
construction wall time
allocation count if available
vertex count
face count
closure
signed volume
geometry deviation
```

The kernel extraction SHOULD achieve performance parity or improvement. A substantial regression requires justification before merge.

---

## 8B. TR-SWP-002-BREP — parametric realization

### Status

Second stage of this packet. Not required to preserve the proven rendering fast path.

### 8B.1 Surface structure

For a profile with $k$ edges, construct approximately $k$ side BREP faces plus optional caps, rather than one face per spine sample. Each side face realizes the same $X(s,v)=C(s)+T(s)P(s,v)$ continuously. Tessellation density MUST NOT change BREP topology.

### 8B.2 Shared longitudinal edges

Trajectory of profile vertex $p_j$: $E_j(s)=X(s,p_j)$ must be represented once and shared by the two adjacent profile-side faces. No sewing.

### 8B.3 Internal registry reuse

The side-face construction needs exactly the keyed-entity-cache pattern of §3.3. It MUST reuse the FAC-internal grid registry pattern rather than introducing a new one.

### 8B.4 Integration

If implemented as a new `truck_modeling::Surface` variant, explicitly audit:

```text
enum forwarding
trait derivation
transforms
inversion
parameter search
tessellation
serialization
STEP behavior
shapeops compatibility
```

The fork's closed `Surface` enum means adding a surface is a cross-cutting change even if trait forwarding is largely mechanical.

---

## 9. TR-VAL-001 — constructive realization certification

### Status

**Retained but redesigned as an integration packet.** Do not introduce an independent validation universe if the fork's existing certificate/evidence infrastructure can represent the evidence.

### 9.1 Doctrine

Validation asks: *does this geometric realization satisfy the topology and recipe already declared?* It does NOT infer topology.

### 9.2 Existing certification vocabulary

New checks SHOULD emit/compose with existing:

```text
MeshedShellOutcome
FaceValidityCertificate
existing provenance/evidence structures
existing tolerance context
```

rather than introducing parallel top-level concepts with overlapping semantics.

### 9.3 Required local evidence

For every constructive sweep:

```text
finite spine evaluation
nonzero tangent
valid frame
finite profile evaluation
noncollapsed profile
valid face orientation
closed structured topology
```

For parametric surfaces additionally:

```text
boundary agreement
Jacobian regularity
parameter-domain validity
```

### 9.4 Shared-edge evidence

If two faces declare the same edge, validate the edge against both surface realizations where applicable. Failure SHOULD identify:

```text
EdgeID
FaceID A
FaceID B
error against A
error against B
```

rather than emitting only a generic invalid-face result.

### 9.5 Jacobian criterion

For a parametric surface: $J=S_u\times S_v$. Require evidence that $\|J\|>\tau_J$ over the certified evaluation domain. Adaptive subdivision SHOULD reuse the fork's existing interval/certification machinery where available.

### 9.6 Global self-intersection (reduced)

A new generic BVH/global surface-intersection subsystem is **not required for this phase**. The Exeter benchmark demonstrates that lack of such machinery does not block useful complex constructive geometry. Global geometric sanity for facet output is covered by the mesh-level audit of §8A.8. A full BVH is promoted only when an independent benchmark demonstrates it is required (§22 gates).

### 9.7 Verdict doctrine (retained at reduced scope)

Certification verdicts remain three-valued:

```text
CERTIFIED_WITHIN_TOLERANCE
FAILED
INCONCLUSIVE
```

Uncertainty MUST surface as `INCONCLUSIVE`, never silently as success. A missing or inconclusive global check MUST be visible in the outcome, not omitted.

---

## 10. TR-MESH-001 — topology-preserving mesh assembly

### Status

**Retained and sharply narrowed.** This packet should not rewrite Truck tessellation.

### 10.1 Existing behavior

The audit established that existing shell tessellation already groups by unique `EdgeID`, samples each unique edge once, uses rayon, and runs per-face constrained triangulation. The missing step is preserving those shared edge samples into the final global `PolygonMesh` position index space.

### 10.2 Target pipeline

```text
EXISTING (conceptual):
EdgeID → one sampled polyline → Face A local positions → Face B local positions → positional welding

TARGET:
EdgeID → one sampled polyline → global mesh position IDs
                                        ↙            ↘
                                     Face A          Face B
```

### 10.3 Parallel entry point, not in-place surgery (amendment)

The ledger can only reach the output assembly if the code path exposes it, and `triangulation.rs` is very large. The low-risk shape is:

- add a **parallel entry point** (e.g. `triangulation_with_ledger`) that reuses the existing unique-edge sampling and per-face CDT internals and returns `(EdgeSampleLedger, per-face local-index triangulations)`;
- perform global index assembly **outside** the mature system;
- existing entry points remain bit-identical in behavior.

Do not modify the existing tessellation's semantics to achieve this.

### 10.4 Edge sample ledger

```rust
EdgeSampleLedger {
    edge_id,
    parameters,
    position_indices,
}
```

A reversed topological edge consumes the same integer sequence in reverse. This ledger type and the FAC grid registry (§3.3) are two consumers of the **single index-identity convention** frozen in Phase 0 (§20). They MUST share that convention, not define divergent ones.

### 10.5 Watertightness invariant

For incident faces $A,B$ sharing edge $E$:

$$I(A,E)=\operatorname{reverse}(I(B,E))$$

where $I$ is the mesh position-index sequence along that edge. This must hold as integer identity, not coordinate proximity.

**Watertightness argument.** If the shell is combinatorially closed (every edge degree exactly 2 with opposite traversal signs) and each mesh boundary vertex is assigned a global position index determined solely by `EdgeID` and sample ordinal, then every mesh edge is referenced by exactly two triangles with opposite winding. The emitted mesh is edge-watertight **by construction**; positional welding (`put_together_same_attrs`) is never invoked. Closedness is a property of index bookkeeping derived from authored incidence.

### 10.6 Attribute model

Position indices MAY be globally shared while normal indices, UV indices, and other face-varying attributes remain separate. This fits the existing OBJ-like `PolygonMesh` attribute organization.

### 10.7 Parallelism

Preserve the existing high-level shape:

```text
parallel unique-edge sampling
barrier/freeze ledger
parallel face interior work
parallel per-face CDT
deterministic output assembly
```

Do not rewrite the mature triangulation system simply to alter output indexing.

### 10.8 Acceptance criterion

For every closed test shell:

```text
tessellate
→ no positional welding
→ polygon mesh reports closed
```

Shared boundaries must use identical global position indices.

---

## 11. TR-NRB-001 — recipe/surface to NURBS

### Status

**Follow-on interoperability packet.** Not required for fast GLB rendering. Required if new constructive surfaces must export as conventional smooth STEP geometry. In this fork, "NURBS-compatible" specifically means fitting into the enum/`StepSurface` machinery, not merely producing B-splines; estimate after the exact interoperability contract is frozen.

### 11.1 Inputs

At minimum: `SpineFrameSurface`, `Coons4`.

### 11.2 Requirements

Approximation must preserve declared boundary curves, shared-edge consistency, requested positional tolerance, and orientation. Neighboring surfaces MUST NOT independently refit different versions of a shared edge.

### 11.3 Metrics

Report maximum positional deviation, maximum shared-boundary deviation, normal deviation where requested, degree, and control-point count. If requested tolerance cannot be achieved, refuse rather than silently degrading.

---

## 12. TR-INS-001 — prototype/instance preservation

### Status

Optional and independently benchmarked. Do not couple this to the constructive-surface work. Candidate representation:

```rust
Instance { prototype: ShapeRef, transform: Matrix4 }
```

Promote only if profiling demonstrates meaningful improvements for memory, GLB size, GPU upload, tessellation reuse, or assembly export.

---

## 13. Runtime complexity targets

| Operation | Target |
| --- | ---: |
| Direct structured facet generation | $O(mk)$ |
| Manifold diagnostics | $O(V+E+F)$ |
| Orientation analysis | $O(F+E)$ |
| Unique-edge tessellation | $O(S)$ |
| Face triangulation | existing behavior |
| Local constructive certification | $O(S)$ or adaptive equivalent |
| Mesh-level global sanity audit (§8A.8) | $O(\text{mesh edges} + \text{triangles})$; pairwise check optional |
| Boolean operations in required packets | **0** |

where $m$ = spine sample count, $k$ = profile complexity, $S$ = generated sample count.

---

## 14. Generic verification corpus

Kernel acceptance tests MUST remain non-architectural. Required fixtures:

```text
cube
holed prism
multi-hole plate
straight rectangular duct
tapered rectangular duct
90° curved duct
S-shaped swept member
annular/radial sweep
variable-radius coolant passage
ribbed panel
Coons warped quad shell
large repeated sweep assembly
```

---

## 15. Mutation tests

Every valid fixture should have targeted invalid variants. Each mutation MUST trigger the intended existing/new certification failure.

### 15.1 Topological mutations

```text
duplicate shared edge rather than reuse it
remove one face
reverse one face
attach third face to an edge
split one required shared vertex
join intentionally separate coincident vertices
```

### 15.2 Sweep mutations

```text
zero tangent
non-C1 spine (§7.2)
invalid initial normal
ArchitecturalUp singularity
profile scale through zero
invalid profile correspondence
NaN parameter law
```

### 15.3 Surface mutations

```text
incompatible Coons corners
boundary curve off surface
folded Coons patch
collapsed Jacobian
```

---

## 16. Benchmarks

### 16.1 Kernel microbenchmarks

```text
24-point profile × 32 spine stations
24-point profile × 128 spine stations
100 independent curved sweeps
1,000 independent curved sweeps
1,000 planar panels
10,000 direct planar faces
large closed cellular shell
```

Measure:

```text
construction time
tessellation time
allocations if measurable
peak memory
topology count
mesh vertex count
mesh triangle count
need for welding (target: always no)
validation time
```

### 16.2 External adversarial benchmark: Exeter

Exeter remains outside the normative kernel fixture suite. It is used to detect performance regression, allocation regression, topology-count explosion, volume drift, geometric drift, and closure regression. The benchmark implementation already demonstrates that the relevant cathedral geometry can be expressed largely as scalar geometry, explicit frames, sampled profiles, shared edges, planar faces, and combinatorial closure. After `TR-SWP-001`/`TR-SWP-002-FAC` land, the cathedral client should delete its local generic spine/profile transport implementation and consume the Truck abstraction. No cathedral-specific logic should move with it.

---

## 17. Cross-domain extraction tests

A proposed generic operation MUST be demonstrated on at least two geometrically unrelated uses where practical. For `SpineFrameRecipe`, examples: Exeter rib + curved rectangular duct + coolant passage. The implementation is generic only if the same kernel API handles all of them without domain flags. This is the principal defense against accidental architectural specialization.

---

## 18. Implementation sequence

### Phase 0 — Contract freeze

Before large implementation begins:

1. freeze `SpineFrameRecipe` concepts;
2. choose frame-law interfaces (including the §7.2 spine smoothness contract);
3. choose profile-law representation;
4. specify facet sampling interface and **the single index-identity convention** (§8.0, §10.4) shared by FAC grid output and the TR-MESH-001 ledger;
5. **freeze the certificate field-level mapping** (amendment): a concrete table of which existing certificate type (`MeshedShellOutcome`, `FaceValidityCertificate`, provenance/evidence structures) carries which new evidence — spine/frame validity, profile collapse, Jacobian bounds, shared-edge pair errors, mesh winding audit — and where a new evidence variant must be added. Phase C MUST NOT begin against an unfrozen mapping;
6. create generic fixture corpus;
7. create Exeter performance baseline.

No new generalized builder, BVH, or triangular-domain system.

### Phase A — constructive sweep core

- **A1 — TR-SWP-001:** `SpineFrameRecipe`; `FixedPlane`, `ArchitecturalUp`, `RadialAboutAxis`, `ParallelTransport`; `Constant`, `Scale`, `LinearCorrespondence` profiles; spine smoothness refusals.
- **A2 — TR-SWP-002-FAC:** structured sampling; shared indexed grid (private registry, §3.3); deterministic triangulation; planar caps; **`PolygonMesh` emission (primary)**; opt-in faceted Shell/Solid emission (§8.0); §8A.8 sanity audit.

**Exit gate.** All generic sweep fixtures pass. Exeter rib switches from local implementation to Truck implementation without meaningful performance regression. This is the first major merge gate.

### Phase B — mesh topology preservation

- **B1 — TR-MESH-001:** parallel `triangulation_with_ledger` entry point (§10.3); global index assembly outside the mature system; ledger exposed per §10.4.

**Exit gate.** Closed smooth Truck BREP → tessellation → closed `PolygonMesh` without `put_together_same_attrs` or equivalent positional welding.

### Phase C — topology and certification integration

- **C1 — reduced TR-TOP-001:** add only missing explicit diagnostics (§4).
- **C2 — TR-VAL-001:** integrate sweep/surface evidence into the existing certification system per the Phase 0 mapping; tri-state verdicts (§9.7).

**Exit gate.** Known-invalid constructive geometries fail for specific reasons without a parallel validation taxonomy.

### Phase D — smooth constructive surfaces

- **D1 — TR-SWP-002-BREP:** resolution-independent `SpineFrameSurface` (§8B), reusing the internal registry pattern.
- **D2 — TR-GEO-001:** Coons4 (§5). May proceed independently where crate boundaries permit.

**Exit gate.** A curved duct and warped boundary patch exist as true parametric Truck BREP and retessellate at multiple resolutions without changing BREP topology.

### Phase E — interoperability

- **E1 — TR-NRB-001:** boundary-preserving NURBS conversion (§11).

**Exit gate.** Constructive smooth surfaces export through the standard STEP-facing geometry representation within declared tolerance.

---

## 19. Explicitly deferred work

The following MUST NOT enter the critical path without new evidence:

```text
TR-GEO-002 triangular transfinite patch
large generic DirectBrepBuilder (public API)
new full manifold kernel
new global BVH/self-intersection subsystem
general planar arrangement solver
automatic proximity sewing
architectural period solver
recursive grammar system
generic Boolean replacement
```

Each requires its own independent justification via §20.

---

## 20. Promotion doctrine

A client-side algorithm is promoted into Truck only if it satisfies at least one of these conditions.

- **A. Independent reinvention.** Two unrelated clients independently need substantially the same mathematical operation.
- **B. Lost kernel information.** Truck already knows a fact internally but loses it downstream. `TR-MESH-001` is the canonical example.
- **C. Representation unlock.** The operation creates a general geometric representation otherwise unavailable or disproportionately awkward.
- **D. Client becomes a mini-kernel.** A client must implement generic frame transport, surface construction, topology generation, or validity mechanics unrelated to its own domain.

### Non-promotion doctrine

Repeated use inside one client is insufficient. The following remain client-side even if they produce enormous geometry: starcut construction, tierceron topology, boss placement, pier fitting, architectural repetition, muqarnas rules, rocket channel branching policy, turbine cooling circuit design, heat-exchanger channel layout. The kernel realizes these decisions; it does not make them.

---

## 21. Expected core code footprint

The benchmark and audit justify reducing the core effort substantially from the original proposal. Approximate new production code before smooth/NURBS phases:

```text
TR-SWP-001              600–1,000
TR-SWP-002-FAC          500–900
TR-MESH-001             400–800
TR-TOP-001 reduced      200–500
TR-VAL-001 integration  300–700
```

Approximate core: **~2,000–3,900 LOC production**, plus tests and benchmarks. Treat the upper bound as the floor of the honest range: discrete parallel transport plus its stability tests are the bulk of the real work.

Smooth follow-on:

```text
SpineFrameSurface       ~500–900
Coons4                  ~500–900
```

`TR-NRB-001` is estimated separately after the exact B-spline/NURBS interoperability contract is frozen. LOC estimates are planning ranges, not acceptance criteria.

---

## 22. Definition of completion

The next kernel phase is successful when all of the following hold.

### Constructive sweep

One generic API can construct a curved duct, coolant passage, molded curved member, and the external Exeter rib.

### Fast facet backend

Those shapes can be emitted directly as exact shared-topology faceted geometry — `PolygonMesh` as the contractual target — without sewing, healing, surface fitting, or Boolean operations.

### Global sanity

Facet output passes the §8A.8 mesh-level audit (winding, volume sign), with inconclusive global evidence surfaced as `INCONCLUSIVE`.

### Topology-preserving tessellation

Native/smooth Truck BREP tessellates with shared boundary position indices derived from `EdgeID`, not coordinate welding.

### Certification

Constructive geometry participates in the fork's existing evidence/certificate framework per the Phase 0 frozen mapping.

### Smooth optional realization

When requested, the same spine/profile recipe can become resolution-independent parametric BREP without rewriting client design logic.

### No domain contamination

No public kernel API knows what an Exeter vault, rocket nozzle, turbine blade, or muqarnas cell is.

---

## 23. Final architectural contract

```text
CLIENT
────────────────────────────────────
design graph
feature topology
symmetry
periodicity
engineering rules
profile choice
spine choice
boundary ownership
semantic interfaces

                 │
                 ▼

TRUCK CONSTRUCTIVE GEOMETRY
────────────────────────────────────
explicit topology
frame transport
profile transport
facet realization
parametric realization
boundary patching
manifold diagnostics
certification
topology-preserving tessellation
standard export
```

The governing principle:

$$\boxed{\text{The client decides the construction; Truck realizes it without rediscovering it.}}$$

The Exeter benchmark validates the fast end of this architecture. The next phase generalizes only the pieces that are genuinely kernel-level.

---

## Appendix A — Amendments relative to the prior draft

1. **Realization output doctrine (§8.0, §8A.1).** `PolygonMesh` with exact shared indices is the contractual primary output of the facet backend; faceted BREP is opt-in with its $m\cdot k$-face topology-count consequence documented.
2. **Spine smoothness contract (§7.2).** MVP spines must be $C^1$; non-$C^1$ spines typed-refuse. Optional explicit Miter corner policy; no implicit smoothing.
3. **TR-MESH-001 as a parallel entry point (§10.3).** New `triangulation_with_ledger`-style function reusing existing internals; existing entry points bit-identical; global assembly outside the mature system.
4. **Single index-identity convention (§8.0, §10.4, Phase 0 item 4).** FAC grid output and the tessellation ledger share one frozen convention for identity-by-index.
5. **Private grid registry acknowledged and reused (§3.3, §8B.3).** FAC's structured grid is a keyed entity cache; one internal implementation, reused by the BREP mode; not public API.
6. **Reduced global checks retained (§8A.8, §9.6, §9.7).** Mesh-level winding/volume-sign audit mandatory on facet output; optional reuse of existing collision analyzers; BVH still deferred; tri-state verdict doctrine retained.
7. **Phase 0 certificate mapping frozen at field level (Phase 0 item 5).** Which existing certificate types carry which new evidence is a Phase 0 deliverable, not a Phase C conversation.
