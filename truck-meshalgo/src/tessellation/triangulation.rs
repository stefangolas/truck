#![allow(clippy::many_single_char_names)]
// PR 4A.1 adds sidecar diagnostic types and outcome entry points that are not
// yet consumed by callers. Silence their dead-code until the census examples
// wire them up; remove this `allow` when the diagnostic API is finalized.
#![allow(dead_code, unused)]

use super::diagnosis;
use super::diagnosis::ObservedClosure;
use super::domain::lattice::AxisPeriodStatus;
use super::domain::lattice::CertifiedLattice;
use super::formal;
use super::source_evidence::{
    BoundId, EdgeUseId, ErasedOrientationMechanism, OrientationEvidence, OrientationOrigin,
    SourceBoundInput, SourceEdgeOrientationEvidence, SourceEdgeUseInput, SourceEvidenceError,
    SourceFaceInput, SourceFaceOrientationEvidence, SourceVertexKey,
};
use super::*;
use crate::filters::NormalFilters;
use crate::Point2;
use array_macro::array;
use handles::{FixedUndirectedEdgeHandle, FixedVertexHandle};
use itertools::Itertools;
use rustc_hash::FxHashMap as HashMap;
use serde::Serialize;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

type SPoint2 = spade::Point2<f64>;
type Cdt = ConstrainedDelaunayTriangulation<SPoint2>;
std::thread_local! {
    /// Optional document-local source face id, declared face index, and
    /// parameter-space periodic rank for probes.
    static PROBE_FACE_CONTEXT: std::cell::Cell<(Option<u64>, usize, u8)> =
        const { std::cell::Cell::new((None, usize::MAX, 0)) };
}

type MeshedShell = Shell<Point3, PolylineCurve, Option<PolygonMesh>>;
type MeshedCShell = CompressedShell<Point3, PolylineCurve, Option<PolygonMesh>>;

pub(super) trait SP<S>:
    Fn(&S, Point3, Option<(f64, f64)>) -> Option<(f64, f64)> + Parallelizable
{
}
impl<S, F> SP<S> for F where
    F: Fn(&S, Point3, Option<(f64, f64)>) -> Option<(f64, f64)> + Parallelizable
{
}

/// Print taxonomy summary.
pub fn print_taxonomy_summary() {}

pub(super) fn by_search_parameter<S>(
    surface: &S,
    point: Point3,
    hint: Option<(f64, f64)>,
) -> Option<(f64, f64)>
where
    S: MeshableSurface,
{
    surface
        .search_parameter(point, hint, 100)
        .or_else(|| surface.search_parameter(point, None, 100))
}

pub(super) fn by_search_nearest_parameter<S>(
    surface: &S,
    point: Point3,
    hint: Option<(f64, f64)>,
) -> Option<(f64, f64)>
where
    S: RobustMeshableSurface,
{
    surface
        .search_parameter(point, hint, 100)
        .or_else(|| surface.search_parameter(point, None, 100))
        .or_else(|| surface.search_nearest_parameter(point, hint, 100))
        .or_else(|| surface.search_nearest_parameter(point, None, 100))
        // Last, so it is reached only where every existing attempt returned
        // `None` — which is exactly the population that becomes
        // `BoundaryProjectionFailed`. A face that projects today projects
        // through the identical chain and gets the identical parameter.
        .or_else(|| by_structural_seeds(surface, point, hint))
}

/// Retry the parameter inverse from the starts the surface's own structure
/// suggests.
///
/// The chain above fails as a *numerical* matter, not a geometric one: it runs
/// a Newton iteration from a single start — a caller's hint, or the best cell
/// of a uniform presearch grid — and a single start is not enough on a
/// piecewise surface whose pieces the grid does not see. `search_parameter_seeds`
/// supplies one start per knot span, so every polynomial piece gets its own
/// attempt. Only the initialisation changes; the iteration is the same one.
///
/// This returns a parameter, not a verdict. A returned parameter is still
/// subject to the caller's incidence check — a nearest point is not an
/// incidence — so nothing is admitted here that the pipeline would not have
/// admitted from any other start.
fn by_structural_seeds<S>(surface: &S, point: Point3, hint: Option<(f64, f64)>) -> Option<(f64, f64)>
where
    S: MeshableSurface,
{
    if !diagnosis::spline_seed_recovery_enabled() {
        return None;
    }
    let seeds = surface.search_parameter_seeds();
    if seeds.is_empty() {
        return None;
    }
    let mut best: Option<((f64, f64), f64, f64)> = None;
    for seed in seeds {
        let Some(uv) = surface.search_parameter(point, seed, 100) else {
            continue;
        };
        let residual = surface.subs(uv.0, uv.1).distance(point);
        // Distance from the hint, in parameter space. Among starts that
        // converge equally well this is what keeps the boundary walk monotone:
        // a spline can carry the same 3D point in more than one span, and
        // taking whichever one happened to converge first would step the
        // traversal across the domain.
        let drift = match hint {
            Some((u0, v0)) => (uv.0 - u0).hypot(uv.1 - v0),
            None => 0.0,
        };
        let better = match best {
            None => true,
            Some((_, best_residual, best_drift)) => {
                if residual < best_residual * SEED_RESIDUAL_TIE
                    && best_residual < residual * SEED_RESIDUAL_TIE
                {
                    drift < best_drift
                } else {
                    residual < best_residual
                }
            }
        };
        if better {
            best = Some((uv, residual, drift));
        }
    }
    best.map(|(uv, _, _)| uv)
}

/// Within this factor two seeds' residuals are the same answer, and the choice
/// between them is made on traversal continuity instead.
///
/// Both are converged solutions of the same equation; their residuals differ
/// only by where the iteration stopped. Comparing them exactly would let
/// floating-point noise decide which span the boundary walk continues in.
const SEED_RESIDUAL_TIE: f64 = 1.0 + 1.0e-6;

/// Tessellates faces
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn shell_tessellation<'a, C, S>(
    shell: &Shell<Point3, C, S>,
    tol: f64,
    sp: impl SP<S>,
    lattice_of: impl Fn(&S) -> CertifiedLattice + Parallelizable,
) -> MeshedShell
where
    C: PolylineableCurve + 'a,
    S: PreMeshableSurface + 'a,
{
    let vmap: HashMap<_, _> = shell
        .vertex_par_iter()
        .map(|v| (v.id(), v.mapped(Point3::clone)))
        .collect();
    let eset: HashMap<_, _> = shell.edge_par_iter().map(move |e| (e.id(), e)).collect();
    let edge_map: HashMap<_, _> = eset
        .into_par_iter()
        .map(move |(id, edge)| {
            let v0 = vmap.get(&edge.absolute_front().id()).unwrap();
            let v1 = vmap.get(&edge.absolute_back().id()).unwrap();
            let curve = edge.curve();
            let poly = PolylineCurve::from_curve(&curve, curve.range_tuple(), tol);
            (id, Edge::debug_new(v0, v1, poly))
        })
        .collect();
    let create_edge = |edge: &Edge<Point3, C>| -> Edge<_, _> {
        let new_edge = edge_map.get(&edge.id()).unwrap();
        match edge.orientation() {
            true => new_edge.clone(),
            false => new_edge.inverse(),
        }
    };
    let create_boundary =
        |wire: &Wire<Point3, C>| -> Wire<_, _> { wire.edge_iter().map(create_edge).collect() };
    let create_face = move |face: &Face<Point3, C, S>| -> Face<_, _, _> {
        let wires: Vec<_> = face
            .absolute_boundaries()
            .iter()
            .map(create_boundary)
            .collect();
        let lattice = lattice_of(&face.surface());
        shell_create_polygon(
            &face.surface(),
            wires,
            face.orientation(),
            tol,
            &sp,
            &lattice,
        )
    };
    shell.face_par_iter().map(create_face).collect()
}

/// Tessellates faces
#[cfg(any(target_arch = "wasm32", test))]
pub(super) fn shell_tessellation_single_thread<'a, C, S>(
    shell: &'a Shell<Point3, C, S>,
    tol: f64,
    sp: impl SP<S>,
    lattice_of: impl Fn(&S) -> CertifiedLattice + Parallelizable,
) -> MeshedShell
where
    C: PolylineableCurve + 'a,
    S: PreMeshableSurface + 'a,
{
    use truck_base::entry_map::FxEntryMap as EntryMap;
    use truck_topology::Vertex as TVertex;
    let mut vmap = EntryMap::new(
        move |v: &TVertex<Point3>| v.id(),
        move |v| v.mapped(Point3::clone),
    );
    let mut edge_map = EntryMap::new(
        move |edge: &'a Edge<Point3, C>| edge.id(),
        move |edge| {
            let vf = edge.absolute_front();
            let v0 = vmap.entry_or_insert(vf).clone();
            let vb = edge.absolute_back();
            let v1 = vmap.entry_or_insert(vb).clone();
            let curve = edge.curve();
            let poly = PolylineCurve::from_curve(&curve, curve.range_tuple(), tol);
            Edge::debug_new(&v0, &v1, poly)
        },
    );
    let mut create_edge = move |edge: &'a Edge<Point3, C>| -> Edge<_, _> {
        let new_edge = edge_map.entry_or_insert(edge);
        match edge.orientation() {
            true => new_edge.clone(),
            false => new_edge.inverse(),
        }
    };
    let mut create_boundary = move |wire: &'a Wire<Point3, C>| -> Wire<_, _> {
        wire.edge_iter().map(&mut create_edge).collect()
    };
    let create_face = move |face: &'a Face<Point3, C, S>| -> Face<_, _, _> {
        let wires: Vec<_> = face
            .absolute_boundaries()
            .iter()
            .map(&mut create_boundary)
            .collect();
        let lattice = lattice_of(&face.surface());
        shell_create_polygon(
            &face.surface(),
            wires,
            face.orientation(),
            tol,
            &sp,
            &lattice,
        )
    };
    shell.face_iter().map(create_face).collect()
}

/// A meshed shell together with why each face that failed did so.
///
/// **G8.** The failure vector is positionally aligned with `shell.faces`:
/// `face_failures[i]` explains `shell.faces[i]`, and is `None` exactly when
/// that face tessellated. It is returned *with* the shell rather than emitted
/// or logged, so a caller cannot consume the mesh while ignoring the reason —
/// which is what made the previous empty-mesh convention lossy.
///
/// Each face's own identity remains on `CompressedFace::provenance`, so a
/// failure can be reported against a source entity rather than an index.
#[derive(Clone, Debug)]
pub struct MeshedShellOutcome {
    /// The meshed shell, shaped exactly as the legacy path produced it.
    pub shell: MeshedCShell,
    /// Why each face failed, positionally aligned with `shell.faces`.
    pub face_failures: Vec<Option<TessellationFailure>>,
    /// DIAG-001: structured diagnostic record for each failed face, positionally
    /// aligned with `shell.faces`. Populated only when `TRUCK_FACE_DIAG_JSONL`
    /// is set; all `None` otherwise.
    pub face_diagnoses: Vec<Option<diagnosis::FailedFaceDiagnosis>>,
    /// DIAG-001: what the formal cylinder-band fallback did on each face,
    /// positionally aligned with `shell.faces`. `None` for every face the
    /// fallback was not eligible for, which includes every face when the gate
    /// is closed.
    pub band_attempts: Vec<Option<CylinderBandAttempt>>,
    /// What the formal conical essential-band route did on each face,
    /// positionally aligned with `shell.faces`.
    ///
    /// A second vector rather than a widened first one, because the two routes
    /// have genuinely different vocabularies: the cylinder's exit set names a
    /// nonconformant-source repair this cell has none of, and this cell's names
    /// nappe and apex obligations the cylinder has none of. Flattening them
    /// would force a reconciliation to guess which cell a shared tag came from.
    /// Both are `None` on any face neither route was eligible for.
    pub cone_band_attempts: Vec<Option<ConeBandAttempt>>,
    /// What the torus annulus route did on each face, positionally aligned
    /// with `shell.faces`.
    ///
    /// A third vector rather than a widened first two, for the same reason:
    /// the torus route has its own vocabulary of typed exits
    /// ([`formal::TorusAnnulusExit`]) and its own conformance tag
    /// ([`formal::torus_cell::ConformanceTag`]), neither of which is
    /// reconcilable with the cylinder or cone cell's exit types. `None` on
    /// any face the torus route was not eligible for.
    pub torus_band_attempts: Vec<Option<TorusAnnulusAttempt>>,
}

/// What the cylinder-band fallback did on one eligible face.
///
/// Deliberately not a new taxonomy: the unrecovered arm carries
/// [`formal::cylinder_band::BandExit`] unchanged, which already names the stage
/// the attempt left from. "Attempted" is `Some(_)`; "recovered" is the first
/// variant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CylinderBandAttempt {
    /// `run_cylinder_band` returned a validated annular mesh, and that mesh
    /// replaced the preserved legacy failure.
    Recovered {
        /// Triangles in the validated annulus.
        triangles: usize,
        /// Whether the source that produced it was conformant, or was
        /// repaired by a named nonconformant normalization. A recovery from a
        /// malformed file is still a recovery, and is still not a clean read;
        /// carrying the distinction here is what keeps a census able to say
        /// which it was.
        conformance: formal::cylinder_band::SourceConformance,
    },
    /// `run_cylinder_band` returned a typed exit, and the original legacy
    /// failure was preserved unchanged.
    Refused(formal::cylinder_band::BandExit),
}

/// What the conical essential-band route did on one eligible face.
///
/// The same shape as [`CylinderBandAttempt`], and deliberately not the same
/// type: the unrecovered arm carries [`formal::cone_band::ConicalBandExit`]
/// unchanged, which names this cell's own obligations — same nappe, apex
/// exclusion, carrier order — and the recovered arm carries what the *source*
/// declared about outer-bound standing rather than a conformance repair,
/// because this cell has no repair to report.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConeBandAttempt {
    /// `run_conical_essential_band` returned a validated annular mesh, and
    /// that mesh replaced the preserved legacy failure.
    Recovered {
        /// Triangles in the validated annulus.
        triangles: usize,
        /// Which nappe the band was certified on.
        nappe: formal::cone::Nappe,
        /// What the source declared about outer-bound standing. Retained as
        /// provenance; the material region did not come from it.
        standing: formal::cone_band::ConicalSourceStanding,
    },
    /// `run_conical_essential_band` returned a typed exit, and the original
    /// legacy failure was preserved unchanged.
    Refused(formal::cone_band::ConicalBandExit),
}

/// What the torus annulus route did on one eligible face.
///
/// The same shape as [`CylinderBandAttempt`] and [`ConeBandAttempt`], and
/// deliberately a separate type: the unrecovered arm carries
/// [`formal::TorusAnnulusExit`], which names this route's own typed
/// distinctions (on-torus, winding, homology, material authority, realization),
/// and the recovered arm carries the [`formal::ConformanceTag`] that records
/// whether the certified cell relied on a malformed-source normalization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TorusAnnulusAttempt {
    /// The torus annulus was certified and realized, and the validated mesh
    /// replaced the preserved legacy failure.
    Recovered {
        /// Triangles in the validated annulus.
        triangles: usize,
        /// Whether the source that produced it was well-formed, or carried
        /// the double-outer-bound malformation. A recovery from a malformed
        /// source is still a recovery, and is still not a clean read.
        conformance: formal::torus_cell::ConformanceTag,
    },
    /// The torus annulus route returned a typed exit, and the original legacy
    /// failure was preserved unchanged.
    Refused(formal::TorusAnnulusExit),
}

/// Tessellates faces, discarding why any of them failed.
///
/// Legacy shape. Prefer [`cshell_tessellation_with_outcomes`].
pub(super) fn cshell_tessellation<'a, C, S>(
    shell: &CompressedShell<Point3, C, S>,
    tol: f64,
    sp: impl SP<S>,
    lattice_of: impl Fn(&S) -> CertifiedLattice + Parallelizable,
) -> MeshedCShell
where
    C: PolylineableCurve + 'a,
    S: PreMeshableSurface + 'a,
{
    cshell_tessellation_with_outcomes(
        shell,
        tol,
        sp,
        lattice_of,
        |_| {
            formal::SupportSurfaceSchema::not_structurally_identified(
                formal::SchemaIdentificationFailure::NoStructuralReader {
                    representation: "legacy_entry_point_reads_no_schema",
                },
            )
        },
        |_| {
            formal::CurveSchema::not_structurally_identified(
                formal::CurveSchemaFailure::NoStructuralReader {
                    representation: "legacy_entry_point_reads_no_schema",
                },
            )
        },
    )
    .shell
}

/// Tessellates faces, preserving why each face that failed did so.
pub(super) fn cshell_tessellation_with_outcomes<'a, C, S>(
    shell: &CompressedShell<Point3, C, S>,
    tol: f64,
    sp: impl SP<S>,
    lattice_of: impl Fn(&S) -> CertifiedLattice + Parallelizable,
    schema_of: impl Fn(&S) -> formal::SupportSurfaceSchema + Parallelizable,
    curve_schema_of: impl Fn(&C) -> formal::CurveSchema + Parallelizable,
) -> MeshedShellOutcome
where
    C: PolylineableCurve + 'a,
    S: PreMeshableSurface + 'a,
{
    cshell_tessellation_with_outcomes_and_cylinder(
        shell,
        tol,
        sp,
        lattice_of,
        schema_of,
        curve_schema_of,
        |_: &S| -> std::result::Result<formal::CertifiedEmbeddedCylinder, &'static str> {
            Err("cylinder_evidence_not_provided")
        },
        |_: &C| {
            formal::CurveSchema::not_structurally_identified(
                formal::CurveSchemaFailure::NoStructuralReader {
                    representation: "cylinder_evidence_not_provided",
                },
            )
        },
        |_: &C| None,
    )
}

/// [`cshell_tessellation_with_outcomes_and_cylinder`], additionally threading
/// the conical-surface adapter the conical essential-band route needs.
///
/// A fourth entry point rather than a widened third one, so every caller with
/// no conical evidence to offer keeps compiling and keeps behaving identically:
/// the delegation below supplies a `cone_of` that refuses every surface, which
/// makes the cone route unreachable and the cylinder entry point's output what
/// it was.
///
/// Only one new closure is needed. The two curve readers are surface-agnostic —
/// they classify a `Curve3D` into a [`formal::SourceCurveFamily`] and know
/// nothing about what the face is trimmed from — so the cone route reads its
/// complete source circles through the same two the cylinder route does.
#[allow(clippy::too_many_arguments)]
pub(super) fn cshell_tessellation_with_outcomes_and_cone<'a, C, S>(
    shell: &CompressedShell<Point3, C, S>,
    tol: f64,
    sp: impl SP<S>,
    lattice_of: impl Fn(&S) -> CertifiedLattice + Parallelizable,
    schema_of: impl Fn(&S) -> formal::SupportSurfaceSchema + Parallelizable,
    curve_schema_of: impl Fn(&C) -> formal::CurveSchema + Parallelizable,
    cylinder_of: impl Fn(&S) -> std::result::Result<formal::CertifiedEmbeddedCylinder, &'static str>
        + Parallelizable,
    cylinder_curve_schema_of: impl Fn(&C) -> formal::CurveSchema + Parallelizable,
    cylinder_curve_family_of: impl Fn(&C) -> Option<formal::SourceCurveFamily> + Parallelizable,
    cone_of: impl Fn(&S) -> std::result::Result<formal::CertifiedEmbeddedCone, &'static str>
        + Parallelizable,
) -> MeshedShellOutcome
where
    C: PolylineableCurve + 'a,
    S: PreMeshableSurface + 'a,
{
    cshell_tessellation_inner(
        shell,
        tol,
        sp,
        lattice_of,
        schema_of,
        curve_schema_of,
        cylinder_of,
        cylinder_curve_schema_of,
        cylinder_curve_family_of,
        cone_of,
        |_: &S| -> std::result::Result<formal::CertifiedEmbeddedTorus, &'static str> {
            Err("torus_evidence_not_provided")
        },
    )
}

/// [`cshell_tessellation_with_outcomes_and_cone`], additionally threading
/// the torus-surface adapter the torus annulus route needs.
///
/// A sixth entry point rather than a widened fifth one, for the same reason
/// every previous one was added: a caller with no torus evidence to offer
/// keeps compiling against the cone form, and that form's output is unchanged
/// by this route's existence because it supplies a `torus_of` that refuses
/// every surface.
///
/// Only one new closure. The torus route reads its complete source circles
/// through the same two curve readers the cylinder and cone routes do; what
/// differs is what the cell then requires of the circle it was handed —
/// on-torus membership and `Z²` winding, not constant-coordinate or nappe
/// obligations.
#[allow(clippy::too_many_arguments)]
pub(super) fn cshell_tessellation_with_outcomes_and_torus<'a, C, S>(
    shell: &CompressedShell<Point3, C, S>,
    tol: f64,
    sp: impl SP<S>,
    lattice_of: impl Fn(&S) -> CertifiedLattice + Parallelizable,
    schema_of: impl Fn(&S) -> formal::SupportSurfaceSchema + Parallelizable,
    curve_schema_of: impl Fn(&C) -> formal::CurveSchema + Parallelizable,
    cylinder_of: impl Fn(&S) -> std::result::Result<formal::CertifiedEmbeddedCylinder, &'static str>
        + Parallelizable,
    cylinder_curve_schema_of: impl Fn(&C) -> formal::CurveSchema + Parallelizable,
    cylinder_curve_family_of: impl Fn(&C) -> Option<formal::SourceCurveFamily> + Parallelizable,
    cone_of: impl Fn(&S) -> std::result::Result<formal::CertifiedEmbeddedCone, &'static str>
        + Parallelizable,
    torus_of: impl Fn(&S) -> std::result::Result<formal::CertifiedEmbeddedTorus, &'static str>
        + Parallelizable,
) -> MeshedShellOutcome
where
    C: PolylineableCurve + 'a,
    S: PreMeshableSurface + 'a,
{
    cshell_tessellation_inner(
        shell,
        tol,
        sp,
        lattice_of,
        schema_of,
        curve_schema_of,
        cylinder_of,
        cylinder_curve_schema_of,
        cylinder_curve_family_of,
        cone_of,
        torus_of,
    )
}

/// [`cshell_tessellation_with_outcomes`], additionally threading the rank-1
/// cylinder evidence readers (Milestone A / FORMAL-013-015).
///
/// The three cylinder closures are `look`'s composition-layer readers —
/// `step::cylinder::identify_source_cylinder`,
/// `step::lattice::cylinder_curve_schema_of` and
/// `step::lattice::cylinder_curve_family_of` — reduced to `Option`/tag-only
/// return types here so this crate does not depend on `look`'s error types.
/// Kept as a second entry point, rather than changing
/// [`cshell_tessellation_with_outcomes`]'s signature, so every existing
/// caller (and the STL/non-STEP paths, which have no cylinder evidence to
/// offer) compiles unchanged.
#[allow(clippy::too_many_arguments)]
pub(super) fn cshell_tessellation_with_outcomes_and_cylinder<'a, C, S>(
    shell: &CompressedShell<Point3, C, S>,
    tol: f64,
    sp: impl SP<S>,
    lattice_of: impl Fn(&S) -> CertifiedLattice + Parallelizable,
    schema_of: impl Fn(&S) -> formal::SupportSurfaceSchema + Parallelizable,
    curve_schema_of: impl Fn(&C) -> formal::CurveSchema + Parallelizable,
    cylinder_of: impl Fn(&S) -> std::result::Result<formal::CertifiedEmbeddedCylinder, &'static str>
        + Parallelizable,
    cylinder_curve_schema_of: impl Fn(&C) -> formal::CurveSchema + Parallelizable,
    cylinder_curve_family_of: impl Fn(&C) -> Option<formal::SourceCurveFamily> + Parallelizable,
) -> MeshedShellOutcome
where
    C: PolylineableCurve + 'a,
    S: PreMeshableSurface + 'a,
{
    cshell_tessellation_inner(
        shell,
        tol,
        sp,
        lattice_of,
        schema_of,
        curve_schema_of,
        cylinder_of,
        cylinder_curve_schema_of,
        cylinder_curve_family_of,
        // No conical evidence was offered, so none is claimed and the conical
        // route is unreachable. This entry point's output is unchanged by the
        // route's existence, by construction.
        |_: &S| -> std::result::Result<formal::CertifiedEmbeddedCone, &'static str> {
            Err("cone_evidence_not_provided")
        },
        |_: &S| -> std::result::Result<formal::CertifiedEmbeddedTorus, &'static str> {
            Err("torus_evidence_not_provided")
        },
    )
}

/// The one tessellation body every entry point above funnels into.
#[allow(clippy::too_many_arguments)]
fn cshell_tessellation_inner<'a, C, S>(
    shell: &CompressedShell<Point3, C, S>,
    tol: f64,
    sp: impl SP<S>,
    lattice_of: impl Fn(&S) -> CertifiedLattice + Parallelizable,
    schema_of: impl Fn(&S) -> formal::SupportSurfaceSchema + Parallelizable,
    curve_schema_of: impl Fn(&C) -> formal::CurveSchema + Parallelizable,
    cylinder_of: impl Fn(&S) -> std::result::Result<formal::CertifiedEmbeddedCylinder, &'static str>
        + Parallelizable,
    cylinder_curve_schema_of: impl Fn(&C) -> formal::CurveSchema + Parallelizable,
    cylinder_curve_family_of: impl Fn(&C) -> Option<formal::SourceCurveFamily> + Parallelizable,
    cone_of: impl Fn(&S) -> std::result::Result<formal::CertifiedEmbeddedCone, &'static str>
        + Parallelizable,
    torus_of: impl Fn(&S) -> std::result::Result<formal::CertifiedEmbeddedTorus, &'static str>
        + Parallelizable,
) -> MeshedShellOutcome
where
    C: PolylineableCurve + 'a,
    S: PreMeshableSurface + 'a,
{
    let vertices = shell.vertices.clone();
    let edge_probe = std::env::var_os("TRUCK_PROBE_EDGE").is_some();
    let evidence_probe = std::env::var_os("TRUCK_PROBE_EVIDENCE").is_some();
    let ambient_probe = std::env::var_os("TRUCK_PROBE_AMBIENT").is_some();
    // The planar vertical slice. `TRUCK_PROBE_SLICE` runs it in shadow and
    // reports; the recovery gate additionally lets a validated formal mesh
    // replace a face the *legacy path lost*, and nothing else.
    //
    // Every recovery gate below is **default-on with explicit opt-out** since
    // `WAVE-2C` — see `diagnosis::recovery_route_enabled`. Each route is
    // refinement-only (it is entered only where `failure.is_some()`), so
    // enabling one cannot change a face that already meshed; the worst it can
    // do is decline to recover. `TRUCK_FORMAL_RECOVERY=0` restores the pure
    // legacy tessellation, and any single `_ROUTE=0` restores that route's.
    let slice_probe = std::env::var_os("TRUCK_PROBE_SLICE").is_some();
    let recovery_gate = diagnosis::formal_recovery_enabled();
    // The planar-holes expansion carries its own gate under the master, so
    // "rank-0 one-bound recovery" and "rank-0 one-bound + holes recovery" are
    // separately measurable runs rather than one conflated population. Closing
    // the master gate closes this one with it.
    let holes_recovery_gate =
        recovery_gate && diagnosis::recovery_route_enabled("TRUCK_FORMAL_RECOVERY_HOLES");
    // The rank-1 cylinder route, on the identical two-tier pattern as the
    // holes route above: a stable route tag (`_CYLINDER`) under the same
    // master gate, plus its own shadow probe.
    let cylinder_probe = std::env::var_os("TRUCK_PROBE_CYLINDER").is_some();
    let cylinder_recovery_gate =
        recovery_gate && diagnosis::recovery_route_enabled("TRUCK_FORMAL_RECOVERY_CYLINDER");
    let run_cylinder_slice = cylinder_probe || cylinder_recovery_gate;
    let run_slice = slice_probe || recovery_gate;
    // The rank-1 cylinder *band* route: the two-bound annulus. Same two-tier
    // pattern again, and it carries no shadow probe of its own — the attempt
    // is reported through `MeshedShellOutcome::band_attempts`, which is typed
    // and needs no parsing, rather than through another stderr channel.
    let band_recovery_gate = diagnosis::cylinder_band_recovery_enabled();
    // The rank-2 torus annulus route. Two modes under one gate:
    // `TRUCK_PROBE_TORUS` runs the certification in shadow and records the
    // typed outcome without replacing the legacy mesh — the observer that
    // reproduces the census. `TRUCK_FORMAL_RECOVERY_TORUS` additionally lets
    // a validated torus annulus mesh replace a face the legacy path lost.
    // Nested under the master gate since `WAVE-2C`: it used to stand outside
    // it so a torus run could be measured without the planar route's
    // recoveries mixed in, and under default-on that is instead done by
    // setting `TRUCK_FORMAL_RECOVERY_TORUS=0` and diffing.
    let torus_probe = std::env::var_os("TRUCK_PROBE_TORUS").is_some();
    let torus_recovery_gate =
        recovery_gate && diagnosis::recovery_route_enabled("TRUCK_FORMAL_RECOVERY_TORUS");
    // Whether a recovery announces itself on stderr.
    //
    // While the routes were opt-in, every recovery printed a `RECOVERED` line
    // (and the planar route a `RECOVERED_VERTEX` line per vertex) because the
    // only reason to have opened the gate was to read them. Default-on makes
    // that an unconditional side effect of rendering: 525 lines on
    // `00009190`, on a tool whose stderr an agent is expected to parse. The
    // recovery is still fully reported — `MeshedShellOutcome`'s typed
    // `band_attempts`, `cone_band_attempts` and `torus_band_attempts` carry it
    // structurally, which is what the census reads — so the log is now opt-in
    // behind its own probe rather than being the only channel.
    let recovery_log = std::env::var_os("TRUCK_PROBE_RECOVERY").is_some();
    // The deck-consistent two-loop join. Unlike the routes above it does not
    // build a mesh of its own: it rebuilds the *same* boundary with the second
    // loop traversed in the direction that satisfies `Σδ = 0`, and re-runs the
    // ordinary tessellator on it. So it inherits every check the legacy path
    // makes, and adds no geometry the legacy path would not have accepted.
    let deck_join_gate = diagnosis::deck_join_recovery_enabled();
    let run_torus = torus_probe || torus_recovery_gate;
    // A per-run shell ordinal, so a `FaceKey` is unique across shells:
    // `declared_face_index` is an index *within* a shell and collides between
    // them. Assigned once per shell here, before the parallel face loop, so
    // every face of one shell shares it.
    let shell_ordinal = SHELL_ORDINAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tessellate_edge = |edge: &CompressedEdge<C>| {
        let curve = &edge.curve;
        let range = curve.range_tuple();
        if edge_probe {
            // How much of its own period a curve is asked to cover. An edge
            // whose start and end vertices coincide gives the importer no
            // independent parameter for each end -- they are the same point
            // modulo the period -- so a generic endpoint solver can resolve
            // them into copies two periods apart. The ratio, not the absolute
            // range, is what says so: a shifted circle may legitimately run
            // over [-pi, 3pi], which is still two periods.
            let span = range.1 - range.0;
            let ratio = curve.period().map(|period| span / period);
            // The radius the converted curve actually has. The source file's
            // circles are a short, known inventory, so a converted radius that
            // is not in it means the conversion built the wrong geometry for
            // the right entity; one that is in it exonerates the curve and
            // points at the face-to-surface pairing instead.
            let fitted = {
                let (t0, t1) = range;
                let (a, b, c) = (
                    curve.subs(t0),
                    curve.subs(t0 + (t1 - t0) / 3.0),
                    curve.subs(t0 + 2.0 * (t1 - t0) / 3.0),
                );
                let (ab, ac) = (b - a, c - a);
                let normal = ab.cross(ac);
                let n2 = normal.magnitude2();
                match n2 > f64::EPSILON {
                    true => {
                        let centre = a
                            + (ac.magnitude2() * ab.cross(normal)
                                - ab.magnitude2() * ac.cross(normal))
                                / (2.0 * n2);
                        Some(centre.distance(a))
                    }
                    false => None,
                }
            };
            eprintln!(
                "EDGE range=({:.6},{:.6}) span={span:.6} period={:?} span/period={:?} \
                 same_vertex={} fitted_radius={:?}",
                range.0,
                range.1,
                curve.period(),
                ratio,
                edge.vertices.0 == edge.vertices.1,
                fitted.map(|r| (r * 1.0e5).round() / 1.0e5),
            );
        }
        let mut range = curve.range_tuple();
        if edge.vertices.0 == edge.vertices.1 && (range.1 - range.0).abs() < 1e-4 {
            if let Some(period) = curve.period() {
                if period > 1e-4 {
                    range = (range.0, range.0 + period);
                }
            }
        }
        let mut poly = PolylineCurve::from_curve(curve, range, tol);
        if poly.len() <= 2 && range.1 - range.0 > 1e-4 {
            let mut pts = Vec::new();
            const STEPS: usize = 16;
            for i in 0..=STEPS {
                let t = range.0 + (i as f64 / STEPS as f64) * (range.1 - range.0);
                pts.push(curve.subs(t));
            }
            poly = PolylineCurve::from(pts);
        }
        CompressedEdge {
            vertices: edge.vertices,
            curve: poly,
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    let edges: Vec<_> = shell.edges.par_iter().map(tessellate_edge).collect();
    #[cfg(target_arch = "wasm32")]
    let edges: Vec<_> = shell.edges.iter().map(tessellate_edge).collect();
    // Which surface in this shell does a face's own boundary actually lie on?
    //
    // A residual says the boundary and the surface it was handed are
    // incompatible; it does not say whether the pairing is wrong or one of the
    // entities is built wrong. Testing the boundary against *every* surface in
    // the shell separates them. If another surface fits to near zero, this is
    // an association defect and the correct partner is named. If none fits, one
    // of the two entities is constructed incorrectly.
    if std::env::var_os("TRUCK_PROBE_ASSOC").is_some() {
        const GRID: usize = 60;
        let nearest = |surface: &S, point: Point3| {
            let (urange, vrange) = surface.try_range_tuple();
            let axis = |range: Option<(f64, f64)>, period: Option<f64>| match (range, period) {
                (Some(r), _) => r,
                (None, Some(p)) => (-p, p),
                (None, None) => (-1.0, 1.0),
            };
            let (ulo, uhi) = axis(urange, surface.u_period());
            let (vlo, vhi) = axis(vrange, surface.v_period());
            let mut best = f64::INFINITY;
            for i in 0..=GRID {
                let u = ulo + (uhi - ulo) * i as f64 / GRID as f64;
                for j in 0..=GRID {
                    let v = vlo + (vhi - vlo) * j as f64 / GRID as f64;
                    best = best.min(surface.subs(u, v).distance(point));
                }
            }
            best
        };
        for (index, face) in shell.faces.iter().enumerate() {
            // A handful of boundary samples is enough to decide a pairing.
            let samples: Vec<Point3> = face
                .boundaries
                .iter()
                .flatten()
                .filter_map(|e| edges.get(e.index))
                .flat_map(|e| e.curve.iter().copied())
                .step_by(3)
                .take(8)
                .collect();
            if samples.is_empty() {
                continue;
            }
            let worst = |surface: &S| {
                samples
                    .iter()
                    .fold(0.0_f64, |acc, p| acc.max(nearest(surface, *p)))
            };
            let own = worst(&face.surface);
            if own <= tol * 3.0 {
                continue;
            }
            let mut best = (usize::MAX, f64::INFINITY);
            for (other, candidate) in shell.faces.iter().enumerate() {
                if other == index {
                    continue;
                }
                let d = worst(&candidate.surface);
                if d < best.1 {
                    best = (other, d);
                }
            }
            eprintln!(
                "ASSOC face={index} own={own:.4e} best_other=face{} at {:.4e} tol={tol:.4e}",
                best.0, best.1
            );
        }
    }
    let tessellate_face = |(declared_face_index, face): (usize, &CompressedFace<S>)| {
        let source_face_id = face.provenance.best_id().map(SourceEntityId::get);
        let periodic_rank = u8::from(face.surface.u_period().is_some())
            + u8::from(face.surface.v_period().is_some());
        PROBE_FACE_CONTEXT.with(|context| {
            context.set((source_face_id, declared_face_index, periodic_rank));
        });
        let periodic_axes = diagnosis::PeriodicAxes {
            u: face.surface.u_period().is_some(),
            v: face.surface.v_period().is_some(),
        };
        let bound_count = face.boundaries.len();
        let diag = diagnosis::diag_enabled();
        if diag {
            diagnosis::clear_sink();
        }

        let boundaries = face.boundaries.clone();
        let surface = &face.surface;
        let lattice = lattice_of(surface);
        // The structural schema, read before the lattice erases which producer
        // said `NonPeriodic`. Nothing in the legacy chain below reads it.
        let schema = schema_of(surface);
        // Step 0: build the rewrite's input seam beside the legacy path and
        // report it. Nothing below reads it, so geometry is unchanged by
        // construction — the point is to count what the seam carries before
        // the pipeline that depends on it exists.
        if evidence_probe {
            let input = source_face_input_from_compressed(
                declared_face_index,
                source_face_id,
                face,
                &edges,
            );
            emit_evidence_probe(&input, source_face_id, declared_face_index, &lattice);
        }
        // Step 1: resolve the same lattice through the formal ambient-period
        // model and report what it concludes. Nothing below reads the result --
        // it is dropped at the end of this block -- so geometry is unchanged by
        // construction. The point is to measure where the legacy
        // representation collapses uncertainty, before anything depends on it.
        if ambient_probe {
            emit_ambient_probe(
                source_face_id,
                declared_face_index,
                shell_ordinal,
                &lattice,
                &schema,
            );
        }
        let create_edge = |edge_idx: &CompressedEdgeIndex| match edge_idx.orientation {
            true => Some(edges.get(edge_idx.index)?.curve.clone()),
            false => Some(edges.get(edge_idx.index)?.curve.inverse()),
        };
        let create_boundary = |wire: &Vec<CompressedEdgeIndex>| {
            let wire_iter = wire.iter().filter_map(create_edge);
            PolyBoundaryPiece::try_new(surface, wire_iter, &sp, tol, &lattice)
        };
        let preboundary: std::result::Result<Vec<_>, _> =
            boundaries.iter().map(create_boundary).collect();
        // G8: the same computation as before, with the failure kept rather than
        // flattened into an empty mesh.
        //
        // `surface` is left exactly as the legacy path produced it — `None`
        // when no boundary could be built, `Some(empty)` when tessellation
        // itself failed — so the meshed shell is unchanged and this commit adds
        // information without moving any face between populations. The reason
        // travels beside it instead of being destroyed.
        // The pieces are retained only for a face that can reach the two-loop
        // join at all — a periodic chart presenting exactly two bounds — so the
        // clone is paid on the band population rather than on every face.
        let deck_join_candidate = deck_join_gate
            && (lattice.declared_u_period().is_some() || lattice.declared_v_period().is_some())
            && preboundary.as_ref().is_ok_and(|pieces| pieces.len() == 2);
        let (polygon, failure) = match preboundary {
            Err(reason) => (None, Some(TessellationFailure::from(reason))),
            Ok(preboundary) => {
                let retained = deck_join_candidate.then(|| preboundary.clone());
                let boundary = PolyBoundary::new(preboundary, &surface, tol, &lattice);
                match trimming_tessellation_result(&surface, &boundary, tol, &lattice) {
                    Ok(mesh) => (Some(mesh), None),
                    // Refinement-only, structurally: the corrected join is
                    // reached only from the arm where the legacy path produced
                    // no mesh, so it can replace a failure and nothing else.
                    Err(failure) => {
                        // The DIAG-001 record, and with it the loss bucket the
                        // band routes admit on, must keep describing the legacy
                        // boundary — not a mixture of it and this second
                        // attempt.
                        let _suspension = diagnosis::SinkSuspension::new();
                        let recovered = retained.and_then(|pieces| {
                            let (boundary, outcome) = PolyBoundary::new_with_join(
                                pieces,
                                &surface,
                                tol,
                                &lattice,
                                TwoLoopJoinPolicy::DeckConsistent,
                            );
                            // Rebuilding is only worth a tessellation pass when
                            // the equation actually selected the other
                            // traversal. Every other outcome reproduces the
                            // boundary that was just tried.
                            let applied = TwoLoopJoinOutcome::ForwardResolves { applied: true };
                            (outcome == applied)
                                .then(|| {
                                    trimming_tessellation_result(&surface, &boundary, tol, &lattice)
                                })
                                .and_then(std::result::Result::ok)
                        });
                        match recovered {
                            Some(mesh) => {
                                if recovery_log {
                                    eprintln!(
                                        "RECOVERED\tsource_face_id={}\t\
                                         declared_face_index={declared_face_index}\t\
                                         triangles={}\tpath=deck_join",
                                        source_face_id
                                            .map(|id| id.to_string())
                                            .unwrap_or_else(|| "none".into()),
                                        mesh.tri_faces().len(),
                                    );
                                }
                                (Some(mesh), None)
                            }
                            // The corrected join was not attempted, or was
                            // itself refused. The legacy failure is preserved
                            // exactly.
                            None => (Some(PolygonMesh::default()), Some(failure)),
                        }
                    }
                }
            }
        };
        // The legacy verdict, classified here and not later: the cylinder-band
        // fallback admits one loss bucket, the bucket is derived from conflict
        // witnesses the sink holds, and `build_face_diagnosis` below consumes
        // them. Reading it at the seam where the legacy path finished also
        // makes it unambiguously a statement about *that* result, and not
        // about whatever a formal route replaced it with.
        let legacy_bucket = match (band_recovery_gate, &failure) {
            (true, Some(failure)) => Some(diagnosis::derived_bucket(failure.reason)),
            _ => None,
        };
        // The planar vertical slice, run beside the legacy result. Its input
        // is Step 0's source evidence, Step 1's certified lattice and the
        // structural schemas; it reads nothing the legacy path produced, so a
        // face's formal verdict is independent of whether the legacy path
        // succeeded.
        let (polygon, failure) = if !run_slice {
            (polygon, failure)
        } else {
            let outcome = run_slice_for_face(
                declared_face_index,
                source_face_id,
                shell_ordinal,
                face,
                &shell.edges,
                &shell.vertices,
                &lattice,
                &schema,
                &curve_schema_of,
                tol,
            );
            let slice_record = outcome.as_ref().map(|outcome| outcome.planar.clone());
            let holes_record = outcome.as_ref().map(|outcome| outcome.holes.clone());
            if slice_probe {
                emit_slice_probe(
                    source_face_id,
                    declared_face_index,
                    shell_ordinal,
                    &slice_record,
                );
                emit_holes_probe(
                    source_face_id,
                    declared_face_index,
                    shell_ordinal,
                    &holes_record,
                );
            }
            // The recovery gate. Every conjunct is explicit: a validated formal
            // mesh replaces a face the legacy path *lost*, and never a face it
            // meshed.
            //
            // The hole-free slice is consulted first, so opening the holes gate
            // cannot move a face that the original rank-0 path already
            // recovered — those recoveries stay bit-identical. The two
            // populations are disjoint anyway (one slice delegates wherever the
            // other applies), and the ordering makes that independent of the
            // delegation logic rather than reliant on it.
            let legacy_failed = failure.is_some();
            let resolved_mesh = match legacy_failed {
                false => None,
                true => {
                    let planar =
                        slice_record
                            .as_ref()
                            .filter(|_| recovery_gate)
                            .and_then(|record| {
                                match record.stage == formal::SliceStage::FinalValidity
                                    && record.category == formal::SliceCategory::Resolved
                                {
                                    true => {
                                        record.mesh.as_ref().map(|mesh| ("rank0_one_bound", mesh))
                                    }
                                    false => None,
                                }
                            });
                    let holes = || {
                        holes_record
                            .as_ref()
                            .filter(|_| holes_recovery_gate)
                            .and_then(|record| {
                                match !record.delegated
                                    && record.stage == formal::SliceStage::FinalValidity
                                    && record.category == formal::SliceCategory::Resolved
                                {
                                    true => record.mesh.as_ref().map(|mesh| ("rank0_holes", mesh)),
                                    false => None,
                                }
                            })
                    };
                    planar.or_else(holes)
                }
            };
            match resolved_mesh {
                Some((path, formal_mesh)) => {
                    if recovery_log {
                        eprintln!(
                            "RECOVERED\tsource_face_id={}\t\
                             declared_face_index={declared_face_index}\ttriangles={}\tpath={path}",
                            source_face_id
                                .map(|id| id.to_string())
                                .unwrap_or_else(|| "none".into()),
                            formal_mesh.triangles.len(),
                        );
                        // The recovered geometry, so a corpus face can become a
                        // regression fixture without shipping the 400 MB model
                        // it came from.
                        for position in &formal_mesh.positions {
                            eprintln!(
                                "RECOVERED_VERTEX\tsource_face_id={}\tx={:?}\ty={:?}\tz={:?}",
                                source_face_id
                                    .map(|id| id.to_string())
                                    .unwrap_or_else(|| "none".into()),
                                position.x,
                                position.y,
                                position.z,
                            );
                        }
                    }
                    (Some(planar_mesh_to_polygon(formal_mesh)), None)
                }
                None => (polygon, failure),
            }
        };
        // The rank-1 cylinder slice, on the identical additive discipline as
        // the planar slice above: it only ever runs after the planar block
        // has already had its chance, and it only ever replaces a face the
        // legacy path *still* has no mesh for — never a face the planar
        // rank-0 path just recovered, and never a successful legacy mesh.
        let (polygon, failure) = if !run_cylinder_slice {
            (polygon, failure)
        } else {
            let record = run_cylinder_slice_for_face(
                declared_face_index,
                source_face_id,
                face,
                &shell.edges,
                &shell.vertices,
                &cylinder_of,
                &cylinder_curve_schema_of,
                &cylinder_curve_family_of,
                tol,
            );
            if cylinder_probe {
                emit_cylinder_probe(source_face_id, declared_face_index, shell_ordinal, &record);
            }
            let legacy_failed = failure.is_some();
            match (
                legacy_failed,
                cylinder_recovery_gate,
                record.category == formal::SliceCategory::Resolved,
                &record.mesh,
            ) {
                (true, true, true, Some(mesh)) => {
                    if recovery_log {
                        eprintln!(
                            "RECOVERED\tsource_face_id={}\t\
                             declared_face_index={declared_face_index}\ttriangles={}\tpath=cylinder",
                            source_face_id
                                .map(|id| id.to_string())
                                .unwrap_or_else(|| "none".into()),
                            record.triangles,
                        );
                    }
                    (Some(mesh.clone()), None)
                }
                _ => (polygon, failure),
            }
        };
        // The formal cylinder-band fallback. The exact production rule:
        //
        //   legacy success                     -> the legacy mesh, unchanged
        //   legacy SyntheticSyntheticCrossing
        //     + certified cylinder support
        //     + exactly two authoritative bounds
        //                                      -> attempt `run_cylinder_band`
        //       validated mesh                 -> the formal mesh
        //       typed exit                     -> the legacy failure, preserved
        //   any other legacy failure           -> the legacy failure, preserved
        //
        // `failure.is_some()` is the "legacy success" arm: a face that has a
        // mesh at this point — because the legacy path meshed it, or because a
        // formal route above already recovered it — is never attempted. The
        // bucket, the cylinder certificate and the bound count are the other
        // three conjuncts, each checked explicitly and none of them repaired.
        let (polygon, failure, band_attempt) = match (
            band_recovery_gate,
            failure.is_some(),
            legacy_bucket == Some(diagnosis::LossBucket::SyntheticSyntheticCrossing),
        ) {
            (true, true, true) => {
                match run_cylinder_band_for_face(
                    declared_face_index,
                    source_face_id,
                    face,
                    &shell.edges,
                    &shell.vertices,
                    &cylinder_of,
                    &cylinder_curve_schema_of,
                    &cylinder_curve_family_of,
                    tol,
                ) {
                    None => (polygon, failure, None),
                    Some(Ok((mesh, conformance))) => {
                        let triangles = mesh.tri_faces().len();
                        (
                            Some(mesh),
                            None,
                            Some(CylinderBandAttempt::Recovered {
                                triangles,
                                conformance,
                            }),
                        )
                    }
                    Some(Err(exit)) => (polygon, failure, Some(CylinderBandAttempt::Refused(exit))),
                }
            }
            _ => (polygon, failure, None),
        };
        // The conical essential-band route, on the identical production rule
        // and under the identical gate. It runs only after the cylinder band
        // has had its chance and only on a face that *still* has no mesh, so
        // the two cells cannot both claim one face: `cylinder_of` and `cone_of`
        // are mutually exclusive on any one surface anyway — a revolved line is
        // either parallel to its axis or tilted from it, and each identifier
        // refuses the other's case by name — but the ordering makes that a
        // property of the pipeline rather than only of the adapters.
        let (polygon, failure, cone_band_attempt) = match (
            band_recovery_gate,
            failure.is_some(),
            legacy_bucket == Some(diagnosis::LossBucket::SyntheticSyntheticCrossing),
        ) {
            (true, true, true) => {
                match run_conical_band_for_face(
                    declared_face_index,
                    source_face_id,
                    face,
                    &shell.edges,
                    &shell.vertices,
                    &cone_of,
                    &cylinder_curve_schema_of,
                    &cylinder_curve_family_of,
                    tol,
                ) {
                    None => (polygon, failure, None),
                    Some(Ok((mesh, nappe, standing))) => {
                        let triangles = mesh.tri_faces().len();
                        (
                            Some(mesh),
                            None,
                            Some(ConeBandAttempt::Recovered {
                                triangles,
                                nappe,
                                standing,
                            }),
                        )
                    }
                    Some(Err(exit)) => (polygon, failure, Some(ConeBandAttempt::Refused(exit))),
                }
            }
            _ => (polygon, failure, None),
        };
        // The torus annulus route. Two modes:
        //
        //   shadow (TRUCK_PROBE_TORUS):  always runs on a torus face, records
        //       the typed outcome in `torus_band_attempts`, and does NOT
        //       replace the legacy mesh. This is the observer that reproduces
        //       the corrected census.
        //
        //   recovery (TRUCK_FORMAL_RECOVERY_TORUS): runs only on a face that
        //       still has no mesh (legacy failed, and no earlier formal route
        //       recovered it), and replaces the failure with the validated
        //       torus annulus mesh. Production recovery is gated separately
        //       from the observer so the census can be confirmed before any
        //       mesh is changed.
        //
        // The torus route runs after the cone route and only on a face that
        // is a toroidal surface — `torus_of` refuses every non-torus surface
        // by name, so `cylinder_of`, `cone_of`, and `torus_of` are mutually
        // exclusive on any one surface.
        let (polygon, failure, torus_band_attempt) = if run_torus {
            match run_torus_annulus_for_face(
                declared_face_index,
                source_face_id,
                face,
                &shell.edges,
                &shell.vertices,
                &torus_of,
                &cylinder_curve_family_of,
                tol,
            ) {
                None => (polygon, failure, None),
                Some(Ok((mesh, conformance))) => {
                    let triangles = mesh.tri_faces().len();
                    if torus_recovery_gate && failure.is_some() {
                        if recovery_log {
                            eprintln!(
                                "RECOVERED\tsource_face_id={}\t\
                                 declared_face_index={declared_face_index}\t\
                                 triangles={}\tpath=torus",
                                source_face_id
                                    .map(|id| id.to_string())
                                    .unwrap_or_else(|| "none".into()),
                                triangles,
                            );
                        }
                        (
                            Some(mesh),
                            None,
                            Some(TorusAnnulusAttempt::Recovered {
                                triangles,
                                conformance,
                            }),
                        )
                    } else {
                        // Shadow mode: record the outcome but preserve the
                        // legacy mesh and failure unchanged.
                        (
                            polygon,
                            failure,
                            Some(TorusAnnulusAttempt::Recovered {
                                triangles,
                                conformance,
                            }),
                        )
                    }
                }
                Some(Err(exit)) => (polygon, failure, Some(TorusAnnulusAttempt::Refused(exit))),
            }
        } else {
            (polygon, failure, None)
        };
        let result = CompressedFace {
            boundaries,
            orientation: face.orientation,
            surface: polygon,
            // Tessellation is the stage most likely to produce nothing, so it
            // is the stage that most needs to say which face produced nothing.
            // `polygon` is `None` on failure, and the identity is then the only
            // thing left that can name what was lost.
            provenance: face.provenance,
        };
        PROBE_FACE_CONTEXT.with(|context| context.set((None, usize::MAX, 0)));
        let face_diagnosis = if diag {
            if let Some(ref failure) = failure {
                let all_periods_certified = (!periodic_axes.u
                    || matches!(lattice.u, AxisPeriodStatus::Exact { .. }))
                    && (!periodic_axes.v || matches!(lattice.v, AxisPeriodStatus::Exact { .. }));
                let lift_status = diagnosis::compute_lift_status(
                    periodic_axes,
                    failure.reason,
                    all_periods_certified,
                );
                let deck_status = diagnosis::compute_deck_status(periodic_rank);
                Some(diagnosis::build_face_diagnosis(
                    source_face_id,
                    failure.reason,
                    periodic_rank,
                    periodic_axes,
                    bound_count,
                    lift_status,
                    deck_status,
                ))
            } else {
                None
            }
        } else {
            None
        };
        (
            result,
            failure,
            face_diagnosis,
            band_attempt,
            cone_band_attempt,
            torus_band_attempt,
        )
    };
    #[cfg(not(target_arch = "wasm32"))]
    let results: Vec<_> = shell
        .faces
        .par_iter()
        .enumerate()
        .map(tessellate_face)
        .collect();
    #[cfg(target_arch = "wasm32")]
    let results: Vec<_> = shell
        .faces
        .iter()
        .enumerate()
        .map(tessellate_face)
        .collect();
    let mut faces = Vec::with_capacity(results.len());
    let mut face_failures = Vec::with_capacity(results.len());
    let mut face_diagnoses = Vec::with_capacity(results.len());
    let mut band_attempts = Vec::with_capacity(results.len());
    let mut cone_band_attempts = Vec::with_capacity(results.len());
    let mut torus_band_attempts = Vec::with_capacity(results.len());
    for (f, ff, fd, ba, cba, tba) in results {
        faces.push(f);
        face_failures.push(ff);
        face_diagnoses.push(fd);
        band_attempts.push(ba);
        cone_band_attempts.push(cba);
        torus_band_attempts.push(tba);
    }
    MeshedShellOutcome {
        shell: MeshedCShell {
            vertices,
            edges,
            faces,
        },
        face_failures,
        face_diagnoses,
        band_attempts,
        cone_band_attempts,
        torus_band_attempts,
    }
}

/// Builds the source-evidence view of one compressed face.
///
/// Step 0 of the formal rewrite: this runs beside the legacy boundary
/// construction and feeds nothing but the probe below. Its purpose is to
/// establish, by measurement on the corpus rather than by reading the
/// converter, what the rewrite's input seam actually carries.
///
/// Three facts survive here that the legacy path discards, and they are the
/// three the deck solve will need:
///
/// - **edge-use structure.** `create_boundary` flattens a bound's edge uses
///   into one point vector, after which no arc has endpoints.
/// - **source vertex identity.** `CompressedEdge::vertices` is read for the
///   first time on this path; `create_edge` takes only `.curve`.
/// - **composed `s_b · s_o`.** `create_edge` applies it as `curve.inverse()`
///   and keeps nothing.
///
/// **Contracts:** retains what `TOP-005` requires; `TOP-001` identity is
/// carried but not re-checked here, since the converter's arena already
/// discharges it.
fn source_face_input_from_compressed<S, C>(
    declared_face_index: usize,
    source_face_id: Option<u64>,
    face: &CompressedFace<S>,
    edges: &[CompressedEdge<C>],
) -> std::result::Result<SourceFaceInput, SourceEvidenceError> {
    let mut bounds = Vec::with_capacity(face.boundaries.len());
    for (bound_index, wire) in face.boundaries.iter().enumerate() {
        let bound = BoundId(bound_index);
        if wire.is_empty() {
            // Not a malformed bound, and not grounds for discarding the face: a
            // collapsed `VERTEX_LOOP` contributes no trim segment by design,
            // the compressed form cannot tell that from a bound that lost its
            // edges, and the face's *other* bounds are unaffected either way.
            bounds.push(SourceBoundInput::DegenerateEvidenceUnavailable { id: bound });
            continue;
        }
        let mut edge_uses = Vec::with_capacity(wire.len());
        for (use_index, edge_idx) in wire.iter().enumerate() {
            let id = EdgeUseId::new(bound, use_index);
            let edge =
                edges
                    .get(edge_idx.index)
                    .ok_or(SourceEvidenceError::EdgeIndexOutOfRange {
                        edge_use: id,
                        index: edge_idx.index,
                    })?;
            // The edge's vertices are stated in the edge's own direction, so
            // the composed sense selects which is the *use's* start. This is
            // the one place the retained orientation is read as a fact rather
            // than performed as an inversion — and both orders are kept, so a
            // consumer cannot apply the same fact twice undetected.
            let source_vertices = (
                SourceVertexKey::ShellVertex(edge.vertices.0),
                SourceVertexKey::ShellVertex(edge.vertices.1),
            );
            let use_vertices = match edge_idx.orientation {
                true => source_vertices,
                false => (source_vertices.1, source_vertices.0),
            };
            edge_uses.push(SourceEdgeUseInput {
                id,
                source_edge_index: edge_idx.index,
                source_vertices,
                use_vertices,
                orientation: SourceEdgeOrientationEvidence {
                    bound_times_oriented_edge: OrientationEvidence::Retained {
                        forward: edge_idx.orientation,
                        origin: OrientationOrigin::BoundTimesOrientedEdge,
                    },
                    // Folded into the converted curve at import. Sound
                    // mechanism, no surviving Boolean.
                    edge_curve_same_sense: OrientationEvidence::HistoryErased {
                        mechanism:
                            ErasedOrientationMechanism::EdgeCurveSenseFoldedIntoConvertedCurve,
                    },
                    selected_curve_direction: OrientationEvidence::HistoryErased {
                        mechanism:
                            ErasedOrientationMechanism::SelectedCurveDirectionFoldedIntoConvertedCurve,
                    },
                },
            });
        }
        bounds.push(SourceBoundInput::EdgeUses {
            id: bound,
            edge_uses,
        });
    }
    Ok(SourceFaceInput {
        source_face_id,
        declared_face_index,
        bounds,
        orientation: SourceFaceOrientationEvidence {
            face_use_orientation: OrientationEvidence::Retained {
                forward: face.orientation,
                origin: OrientationOrigin::CompressedFaceOrientation,
            },
            // Folded in by `surface.invert()`, which is recorded as breaking
            // curve-on-surface incidence rather than only reversing the
            // parameterization. This asserts the erasure, not that any
            // particular face had `same_sense == false`.
            face_surface_same_sense: OrientationEvidence::HistoryErased {
                mechanism: ErasedOrientationMechanism::FaceSurfaceSenseFoldedViaSurfaceInvert,
            },
        },
    })
}

/// Per-run shell ordinal for the Step-1 `FaceKey`. Diagnostic only.
static SHELL_ORDINAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The formal envelope the ambient probe evaluates against.
///
/// **This is not a production policy.** No project document specifies values
/// for `s_max`, `n_max`, `e_max`, `w_max`, `x_max`, `v_max` or `g_max` --
/// `FORMAL_SYSTEM.md` Definition 6 says only that they are policy and that the
/// closure proof needs them finite -- so these are diagnostic constants chosen
/// to be non-binding. The one value the documents *do* fix, `r_max <= 2`, is
/// set to that maximum.
///
/// The rank clause is the only one Step 1 evaluates. Every other bound is
/// unreachable at this stage and is stated because the envelope has no
/// `Default` and must be given in full.
fn diagnostic_envelope() -> formal::FormalEnvelope {
    formal::FormalEnvelope::new(
        formal::PolicyInstanceId::new(0),
        2,
        usize::MAX,
        usize::MAX,
        usize::MAX,
        u64::MAX,
        usize::MAX,
        usize::MAX,
        usize::MAX,
    )
    .expect("r_max = 2 is Definition 6's own maximum")
}

/// Step-1 ambient-period probe.
///
/// One `eprintln!` per successfully converted compressed face, for the same
/// reason as the `EVIDENCE` probe: face tessellation is parallel and a record
/// split across calls interleaves. Consumers must parse order-independently and
/// key on `source_face_id` / `declared_face_index`.
///
/// The record reports the legacy `declared_rank` and `certified_rank` beside
/// the formal resolution so the two can be compared face by face. They are
/// **not expected to agree**: `certified_rank == 0` covers both "proved
/// aperiodic" and "periodicity declared and never certified", and separating
/// those is what this step exists for.
fn emit_ambient_probe(
    source_face_id: Option<u64>,
    declared_face_index: usize,
    shell_ordinal: u64,
    lattice: &CertifiedLattice,
    schema: &formal::SupportSurfaceSchema,
) {
    let declared_rank = usize::from(lattice.declared_u_period().is_some())
        + usize::from(lattice.declared_v_period().is_some());
    let legacy_certified_rank = lattice.certified_rank();
    let id = match source_face_id {
        Some(id) => id.to_string(),
        None => "none".to_string(),
    };

    let record = match formal::ambient_evidence_from_schema(
        schema,
        lattice,
        formal::LatticeOrigin::UnattributedLegacyLattice,
    ) {
        Err(error) => format!(
            "u_state=none\tv_state=none\tformal_resolution=adapter_error\tformal_rank=none\t\
             unresolved_reason=none\tinconsistency_reason=none\tunsupported_clause=none\t\
             diagnostic_hint_count=0\tauthoritative_generator_count=0\tadapter_error={}",
            error.tag(),
        ),
        Ok(evidence) => {
            let u_state = evidence.u.tag();
            let v_state = evidence.v.tag();
            let hints = evidence.diagnostic_hints().len();
            let generators = evidence.authoritative_generator_count();
            let face = formal::FaceKey {
                document: formal::DocumentScope::SingleDocumentRun,
                shell: formal::ShellKey::new(shell_ordinal),
                source_face_id: source_face_id.map(formal::SourceEntityKey::new),
                declared_face_index,
            };
            // An operational failure is not a semantic judgment, so it is
            // reported as itself rather than folded into a resolution.
            let (resolution, rank, unresolved, inconsistency, unsupported) =
                match formal::resolve_ambient_periods(evidence, &diagnostic_envelope(), face) {
                    Err(failure) => (failure.tag(), "none".to_string(), "none", "none", "none"),
                    Ok(outcome) => {
                        let tag = outcome.tag();
                        match &outcome {
                            formal::StageOutcome::Resolved(resolved) => {
                                (tag, resolved.rank().to_string(), "none", "none", "none")
                            }
                            formal::StageOutcome::Unresolved(report) => (
                                tag,
                                "none".to_string(),
                                report.reason().tag(),
                                "none",
                                "none",
                            ),
                            formal::StageOutcome::Inconsistent(report) => (
                                tag,
                                "none".to_string(),
                                "none",
                                report.reason().tag(),
                                "none",
                            ),
                            formal::StageOutcome::Unsupported(report) => (
                                tag,
                                "none".to_string(),
                                "none",
                                "none",
                                report.cause().tag(),
                            ),
                            formal::StageOutcome::Ambiguous(report) => (
                                tag,
                                "none".to_string(),
                                "none",
                                report.reason().tag(),
                                "none",
                            ),
                        }
                    }
                };
            format!(
                "u_state={u_state}\tv_state={v_state}\tformal_resolution={resolution}\t\
                 formal_rank={rank}\tunresolved_reason={unresolved}\t\
                 inconsistency_reason={inconsistency}\tunsupported_clause={unsupported}\t\
                 diagnostic_hint_count={hints}\tauthoritative_generator_count={generators}\t\
                 adapter_error=none"
            )
        }
    };
    let support_schema = schema.tag();
    eprintln!(
        "AMBIENT\tsource_face_id={id}\tdeclared_face_index={declared_face_index}\t\
         shell_ordinal={shell_ordinal}\tdeclared_rank={declared_rank}\t\
         legacy_certified_rank={legacy_certified_rank}\tsupport_schema={support_schema}\t\
         {record}"
    );
}

// ---------------------------------------------------------------------------
// The planar vertical slice, run beside the legacy tessellator
// ---------------------------------------------------------------------------

/// Run the formal planar slice for one face, when it is a candidate.
///
/// `None` means the face never entered: it is not a structurally identified
/// plane, or Step 1 did not resolve it to a certified rank-0 lattice. Those two
/// populations are already reported by the ambient probe, so they are counted
/// there rather than duplicated here as slice exits.
///
/// Nothing in here reads the legacy result, so a face's formal verdict is
/// independent of whether the legacy path succeeded on it.
#[allow(clippy::too_many_arguments)]
fn run_slice_for_face<S, C>(
    declared_face_index: usize,
    source_face_id: Option<u64>,
    shell_ordinal: u64,
    face: &CompressedFace<S>,
    edges: &[CompressedEdge<C>],
    vertices: &[Point3],
    lattice: &CertifiedLattice,
    schema: &formal::SupportSurfaceSchema,
    curve_schema_of: &impl Fn(&C) -> formal::CurveSchema,
    tol: f64,
) -> Option<FormalSliceOutcome> {
    // Step 1. The same call the ambient probe makes, on the same evidence.
    let plane = schema.plane()?;
    let evidence = formal::ambient_evidence_from_schema(
        schema,
        lattice,
        formal::LatticeOrigin::UnattributedLegacyLattice,
    )
    .ok()?;
    let key = formal::FaceKey {
        document: formal::DocumentScope::SingleDocumentRun,
        shell: formal::ShellKey::new(shell_ordinal),
        source_face_id: source_face_id.map(formal::SourceEntityKey::new),
        declared_face_index,
    };
    let formal::StageOutcome::Resolved(certified) =
        formal::resolve_ambient_periods(evidence, &diagnostic_envelope(), key).ok()?
    else {
        return None;
    };

    // Step 0. The same seam the evidence probe reports.
    let input =
        source_face_input_from_compressed(declared_face_index, source_face_id, face, edges).ok()?;

    let mut curve_of = |edge_index: usize| match edges.get(edge_index) {
        Some(edge) => curve_schema_of(&edge.curve),
        None => formal::CurveSchema::not_structurally_identified(
            formal::CurveSchemaFailure::NoStructuralReader {
                representation: "edge_index_out_of_range",
            },
        ),
    };
    let vertex_position = |vertex| match vertex {
        SourceVertexKey::ShellVertex(index) => vertices.get(index).copied(),
        _ => None,
    };

    // Both paths run on the same evidence. The hole-free slice is unchanged and
    // still reports `multiple_bounds_or_holes` for a multi-bound face, so its
    // funnel stays comparable with the frozen baseline; the holes slice
    // delegates on a single-bound face rather than answering for a population
    // the other module owns.
    let planar = formal::run_planar_slice(
        &input,
        plane,
        &certified,
        face.provenance.outer_bound,
        &mut curve_of,
        &vertex_position,
        tol,
    );
    let holes = formal::run_planar_holes_slice(
        &input,
        plane,
        &certified,
        face.provenance.outer_bound,
        &mut curve_of,
        &vertex_position,
        tol,
    );
    Some(FormalSliceOutcome { planar, holes })
}

/// Both rank-0 formal paths' verdicts on one face.
///
/// They are kept apart rather than merged into one record: each names the
/// population it is answerable for, and a face that delegates carries no
/// verdict from the module that declined it.
struct FormalSliceOutcome {
    /// The hole-free slice's record. Always present.
    planar: formal::SliceRecord,
    /// The planar-holes slice's record. `delegated` when the face has no inner
    /// bounds.
    holes: formal::HoleSliceRecord,
}

/// Turn a validated planar mesh into the polygon mesh the shell holds.
///
/// The normal is the support plane's chart normal, constant over the face
/// because the face is planar. It carries no material-side meaning: Step 0
/// measured the normalized physical sign as unavailable on all 110,770 edge
/// uses, and nothing here invents one.
fn planar_mesh_to_polygon(mesh: &formal::PlanarMesh) -> PolygonMesh {
    let positions = mesh.positions.clone();
    let normals = vec![mesh.chart_normal; positions.len()];
    let tri_faces: Vec<[StandardVertex; 3]> = mesh
        .triangles
        .iter()
        .map(|indices| {
            array![i => StandardVertex {
                pos: indices[i],
                uv: None,
                nor: Some(indices[i]),
            }; 3]
        })
        .collect();
    PolygonMesh::debug_new(
        StandardAttributes {
            positions,
            uv_coords: Vec::new(),
            normals,
        },
        Faces::from_tri_and_quad_faces(tri_faces, Vec::new()),
    )
}

// ---------------------------------------------------------------------------
// The rank-1 cylinder vertical slice, run beside the legacy tessellator
// (Milestone A / FORMAL-013-015)
// ---------------------------------------------------------------------------

/// Diagnostic materialization budget for [`formal::build_working_cover`].
///
/// Not a proved bound: a real rank-1 face's angular sweep is bounded by a
/// handful of source arcs, so a working cover needing thousands of deck
/// copies is already a sign of a pathological input rather than a face this
/// budget should quietly widen for. Chosen generously enough that no
/// legitimate single-face cylinder disk should ever hit it; a face that does
/// exits `OperationalFailure` rather than materializing unboundedly.
const CYLINDER_DECK_BUDGET: formal::DeckBudget = formal::DeckBudget {
    deck_width_cap: 4096,
};

/// One face's rank-1 cylinder-slice verdict, for the funnel and the recovery
/// gate.
#[derive(Debug, Clone)]
struct CylinderSliceRecord {
    /// The furthest stage reached: `"surface"`, `"evidence"`, `"traversal"`,
    /// `"witness"`, `"holonomy"`, `"cover"`, `"arrangement"`, `"mesh"`, or
    /// `"final_validity"` on success.
    stage: &'static str,
    /// The taxonomy category of the outcome. `Resolved` only when a valid
    /// physical mesh was certified.
    category: formal::SliceCategory,
    /// A stable tag naming the specific exit, or `"resolved"`.
    tag: &'static str,
    /// How many of the face's edge uses classified as an axial line.
    line_edge_uses: usize,
    /// How many classified as a circumferential arc.
    arc_edge_uses: usize,
    /// How many classified as neither (unsupported representation, or a
    /// non-circular affine image).
    unsupported_edge_uses: usize,
    /// The certified physical mesh, when fully resolved.
    mesh: Option<PolygonMesh>,
    /// Triangle count, when resolved (`0` otherwise), for the probe line.
    triangles: usize,
}

impl CylinderSliceRecord {
    fn exit(
        stage: &'static str,
        category: formal::SliceCategory,
        tag: &'static str,
        line_edge_uses: usize,
        arc_edge_uses: usize,
        unsupported_edge_uses: usize,
    ) -> Self {
        Self {
            stage,
            category,
            tag,
            line_edge_uses,
            arc_edge_uses,
            unsupported_edge_uses,
            mesh: None,
            triangles: 0,
        }
    }
}

/// Map a certified cylinder mesh's developed triangulation and physical lift
/// onto the [`PolygonMesh`] the shell holds.
///
/// Unlike [`planar_mesh_to_polygon`]'s single chart normal, a cylinder is
/// curved: each vertex gets its own outward radial normal, `normalize(x -
/// origin - axial(x) * axis)`, computed directly from the certified schema
/// rather than approximated from the mesh.
fn cylinder_mesh_to_polygon(
    mesh: &formal::CertifiedCylinderMesh,
    schema: &formal::CylinderSchema,
) -> PolygonMesh {
    cylinder_polygon_from_lifted(&mesh.physical_vertices, &mesh.developed.triangles, schema)
}

/// A `PolygonMesh` from vertices already lifted onto a certified cylinder.
///
/// Shared by the disk and band routes because the step is the same one: the
/// lift is done, the triangles index it, and the outward normal at a point of
/// a cylinder is its own radial direction. Nothing here re-derives geometry.
fn cylinder_polygon_from_lifted(
    positions: &[Point3],
    triangles: &[[usize; 3]],
    schema: &formal::CylinderSchema,
) -> PolygonMesh {
    let positions = positions.to_vec();
    let normals: Vec<Vector3> = positions
        .iter()
        .map(|p| {
            let r = *p - schema.origin();
            let radial = r - r.dot(schema.axis()) * schema.axis();
            let magnitude = radial.magnitude();
            match magnitude > 0.0 {
                true => radial / magnitude,
                false => schema.axis(),
            }
        })
        .collect();
    let tri_faces: Vec<[StandardVertex; 3]> = triangles
        .iter()
        .map(|indices| {
            array![i => StandardVertex {
                pos: indices[i],
                uv: None,
                nor: Some(indices[i]),
            }; 3]
        })
        .collect();
    PolygonMesh::debug_new(
        StandardAttributes {
            positions,
            uv_coords: Vec::new(),
            normals,
        },
        Faces::from_tri_and_quad_faces(tri_faces, Vec::new()),
    )
}

/// Run the formal rank-1 cylinder slice for one face, when it is a
/// candidate: the authoritative surface adapter, real source traversal, real
/// source curve adapters, then FORMAL-007 through FORMAL-012 unchanged.
///
/// Nothing in here reads the legacy result, so a face's formal verdict is
/// independent of whether the legacy path succeeded on it — the same
/// discipline [`run_slice_for_face`] follows for the planar rank-0 route.
#[allow(clippy::too_many_arguments)]
fn run_cylinder_slice_for_face<S, C>(
    declared_face_index: usize,
    source_face_id: Option<u64>,
    face: &CompressedFace<S>,
    edges: &[CompressedEdge<C>],
    vertices: &[Point3],
    cylinder_of: &impl Fn(&S) -> std::result::Result<formal::CertifiedEmbeddedCylinder, &'static str>,
    cylinder_curve_schema_of: &impl Fn(&C) -> formal::CurveSchema,
    cylinder_curve_family_of: &impl Fn(&C) -> Option<formal::SourceCurveFamily>,
    tol: f64,
) -> CylinderSliceRecord {
    let cylinder = match cylinder_of(&face.surface) {
        Ok(cylinder) => cylinder,
        // `tag` distinguishes "not a `CylindricalSurface` representation at
        // all" from "a `CylindricalSurface` representation that
        // `identify_cylinder` itself refused" (a cone smuggled in by a
        // degenerate transform, a zero radius, an unverified angular
        // period) — see `look::step::cylinder::CylinderSurfaceAdapterFailure`,
        // whose `.tag()` this closure forwards without `truck-meshalgo`
        // depending on that type.
        Err(tag) => {
            return CylinderSliceRecord::exit(
                "surface",
                formal::SliceCategory::Unsupported,
                tag,
                0,
                0,
                0,
            )
        }
    };

    // The curve-family funnel is counted independently of the traversal
    // gate below, so a face that fails traversal for an unrelated reason
    // (a second declared outer bound, a broken cyclic join) still reports
    // what its edges structurally *were*.
    let (mut line_edge_uses, mut arc_edge_uses, mut unsupported_edge_uses) =
        (0usize, 0usize, 0usize);
    for wire in &face.boundaries {
        for edge_idx in wire {
            match edges
                .get(edge_idx.index)
                .and_then(|edge| cylinder_curve_family_of(&edge.curve))
            {
                Some(formal::SourceCurveFamily::Line) => line_edge_uses += 1,
                // A complete circle counts in the same arc column: the funnel
                // reports which structural family an edge use presented, and
                // a closed circle is a circular arc that covers its whole
                // period rather than a separate family.
                Some(formal::SourceCurveFamily::CircularArc { .. })
                | Some(formal::SourceCurveFamily::CompleteCircle { .. }) => arc_edge_uses += 1,
                None => unsupported_edge_uses += 1,
            }
        }
    }

    let Ok(input) =
        source_face_input_from_compressed(declared_face_index, source_face_id, face, edges)
    else {
        return CylinderSliceRecord::exit(
            "evidence",
            formal::SliceCategory::Unresolved,
            "source_evidence_error",
            line_edge_uses,
            arc_edge_uses,
            unsupported_edge_uses,
        );
    };

    let mut curve_of = |edge_index: usize| match edges.get(edge_index) {
        Some(edge) => cylinder_curve_schema_of(&edge.curve),
        None => formal::CurveSchema::not_structurally_identified(
            formal::CurveSchemaFailure::NoStructuralReader {
                representation: "edge_index_out_of_range",
            },
        ),
    };

    let traversal_record = match formal::build_cylinder_face(
        source_face_id,
        cylinder,
        &input,
        face.provenance.outer_bound,
        &mut curve_of,
    ) {
        Ok(record) => record,
        Err(exit) => {
            return CylinderSliceRecord::exit(
                "traversal",
                exit.category(),
                exit.tag(),
                line_edge_uses,
                arc_edge_uses,
                unsupported_edge_uses,
            )
        }
    };

    let schema = traversal_record.cylinder.schema().clone();
    let vertex_position = |vertex| match vertex {
        SourceVertexKey::ShellVertex(index) => vertices.get(index).copied(),
        _ => None,
    };
    // The single production classification route: derives the witness class
    // and (for an arc) declared sweep from the edge's own source
    // representation, never from a caller assertion. By construction this
    // agrees with the gate `curve_of` above already applied — both route
    // through `cylinder_curve_family_of`/`cylinder_curve_schema_of`'s
    // identical `decode_transformed_circle` check — so the `Line` fallback
    // below is defensively unreachable, not a silent misclassification: a
    // genuinely wrong family here still fails the witness's own on-cylinder
    // and constant-coordinate checks rather than certifying incorrectly.
    let family_of = |edge_use: EdgeUseId| {
        traversal_record
            .traversal
            .occurrences
            .iter()
            .find(|occurrence| occurrence.edge_use == edge_use)
            .and_then(|occurrence| edges.get(occurrence.source_edge_index))
            .and_then(|edge| cylinder_curve_family_of(&edge.curve))
            .unwrap_or(formal::SourceCurveFamily::Line)
    };

    let developed = match formal::develop_traversal_from_source(
        &traversal_record.traversal,
        &schema,
        &vertex_position,
        &family_of,
    ) {
        Ok(developed) => developed,
        Err(exit) => {
            return CylinderSliceRecord::exit(
                "witness",
                exit.category(),
                exit.tag(),
                line_edge_uses,
                arc_edge_uses,
                unsupported_edge_uses,
            )
        }
    };

    let lift = match formal::propagate_and_classify_holonomy(&developed, schema.deck_generator()) {
        Ok(lift) => lift,
        Err(exit) => {
            return CylinderSliceRecord::exit(
                "holonomy",
                exit.category(),
                exit.tag(),
                line_edge_uses,
                arc_edge_uses,
                unsupported_edge_uses,
            )
        }
    };

    let cover = match formal::build_working_cover(
        &developed.witnesses,
        &lift.placements,
        schema.deck_generator(),
        CYLINDER_DECK_BUDGET,
    ) {
        Ok(cover) => cover,
        Err(exit) => {
            return CylinderSliceRecord::exit(
                "cover",
                exit.category(),
                exit.tag(),
                line_edge_uses,
                arc_edge_uses,
                unsupported_edge_uses,
            )
        }
    };

    let disk = match formal::certify_cylinder_disk(
        &developed.edge_uses,
        &developed.witnesses,
        &lift.placements,
        schema.deck_generator(),
        face.provenance.outer_bound,
        &cover.materialized_copies,
    ) {
        Ok(disk) => disk,
        Err(exit) => {
            return CylinderSliceRecord::exit(
                "arrangement",
                exit.category(),
                exit.tag(),
                line_edge_uses,
                arc_edge_uses,
                unsupported_edge_uses,
            )
        }
    };

    let occurrences = formal::placed_occurrences(
        &developed.edge_uses,
        &developed.witnesses,
        &lift.placements,
        &schema.deck_generator(),
    );
    let mesh = match formal::certify_cylinder_mesh(&disk, &occurrences, &schema, tol) {
        Ok(mesh) => mesh,
        Err(exit) => {
            return CylinderSliceRecord::exit(
                "mesh",
                exit.category(),
                exit.tag(),
                line_edge_uses,
                arc_edge_uses,
                unsupported_edge_uses,
            )
        }
    };

    let triangles = mesh.developed.triangles.len();
    let polygon = cylinder_mesh_to_polygon(&mesh, &schema);
    CylinderSliceRecord {
        stage: "final_validity",
        category: formal::SliceCategory::Resolved,
        tag: "resolved",
        line_edge_uses,
        arc_edge_uses,
        unsupported_edge_uses,
        mesh: Some(polygon),
        triangles,
    }
}

/// Run the formal cylinder-band path for one face, when it is eligible.
///
/// `None` is "not eligible, nothing was attempted": no certified cylinder
/// support, no source evidence, or a bound count other than two authoritative
/// bounds. `Some` means `run_cylinder_band` was actually called and the value
/// is its verdict.
///
/// This is an adapter and nothing more. Every input it hands the band path is
/// one production already produces:
///
/// - the **cylinder** comes from `cylinder_of`, the same authoritative surface
///   adapter the rank-1 disk route uses, so "certified cylinder support" is
///   that certificate and not a surface-kind string match;
/// - the **face** comes from [`source_face_input_from_compressed`], the same
///   Step-0 evidence seam the disk route reads;
/// - the **curve schema** and **curve family** come from the same two
///   structural readers the disk route passes;
/// - the **vertex positions** are the shell's own vertex table, keyed by
///   [`SourceVertexKey::ShellVertex`];
/// - the **outer bound** is the face's declared standing, unchanged.
///
/// The one thing built here is `family_of`, and it is built from *identity*:
/// [`EdgeUseId`] selects the source edge use, the use names its
/// `source_edge_index`, and that indexes the shell's edge table. No tessellated
/// coordinate is consulted to decide what a curve is. (The disk route resolves
/// the same map through its traversal record; there is no traversal record here
/// because the band path runs one traversal per bound *inside*
/// `certify_cylinder_band`, so the map is resolved through the evidence the
/// traversal is itself built from.)
#[allow(clippy::too_many_arguments)]
fn run_cylinder_band_for_face<S, C>(
    declared_face_index: usize,
    source_face_id: Option<u64>,
    face: &CompressedFace<S>,
    edges: &[CompressedEdge<C>],
    vertices: &[Point3],
    cylinder_of: &impl Fn(&S) -> std::result::Result<formal::CertifiedEmbeddedCylinder, &'static str>,
    cylinder_curve_schema_of: &impl Fn(&C) -> formal::CurveSchema,
    cylinder_curve_family_of: &impl Fn(&C) -> Option<formal::SourceCurveFamily>,
    tol: f64,
) -> Option<
    std::result::Result<
        (PolygonMesh, formal::cylinder_band::SourceConformance),
        formal::cylinder_band::BandExit,
    >,
> {
    let Ok(cylinder) = cylinder_of(&face.surface) else {
        return None;
    };
    let Ok(input) =
        source_face_input_from_compressed(declared_face_index, source_face_id, face, edges)
    else {
        return None;
    };
    // Exactly two bounds, and both of them authoritative. A face with a
    // degenerate-evidence bound is not a two-bound face with one bound missing;
    // it is a face this route has no evidence for, and it is left alone rather
    // than attempted and refused.
    if input.bounds.len() != 2 || input.regular_bound_count() != 2 {
        return None;
    }

    let schema = cylinder.schema().clone();
    let mut curve_of = |edge_index: usize| match edges.get(edge_index) {
        Some(edge) => cylinder_curve_schema_of(&edge.curve),
        None => formal::CurveSchema::not_structurally_identified(
            formal::CurveSchemaFailure::NoStructuralReader {
                representation: "edge_index_out_of_range",
            },
        ),
    };
    let vertex_position = |vertex| match vertex {
        SourceVertexKey::ShellVertex(index) => vertices.get(index).copied(),
        _ => None,
    };
    // The same defensive `Line` fallback the disk route carries, for the same
    // reason: an edge use whose family cannot be read is not silently
    // misclassified, because a wrong family still fails the witness's own
    // on-cylinder and constant-coordinate checks rather than certifying.
    let family_of = |edge_use: EdgeUseId| {
        input
            .edge_uses()
            .find(|use_| use_.id == edge_use)
            .and_then(|use_| edges.get(use_.source_edge_index))
            .and_then(|edge| cylinder_curve_family_of(&edge.curve))
            .unwrap_or(formal::SourceCurveFamily::Line)
    };

    Some(
        formal::cylinder_band::run_cylinder_band(
            source_face_id,
            cylinder,
            &input,
            face.provenance.outer_bound,
            &mut curve_of,
            &vertex_position,
            &family_of,
            tol,
        )
        .map(|(_, mesh)| {
            (
                cylinder_polygon_from_lifted(
                    &mesh.physical_vertices,
                    &mesh.developed.triangles,
                    &schema,
                ),
                mesh.conformance,
            )
        }),
    )
}

/// Build the recovered conical band's mesh, with a per-vertex normal read from
/// the certified cone rather than from the triangles.
///
/// The cone's outward unit normal at a point is perpendicular to the generator
/// through it and to the parallel through it, which in the certified frame is
/// the radial direction tilted back by the half-angle: `(axis_component,
/// radial_component)` proportional to `(-slope · sign(s), 1)`, normalized. It
/// is derived from the certificate — the apex, the axis and the half-angle —
/// and not averaged from adjacent facets, so a coarse band and a fine one carry
/// the same normal field.
///
/// At the apex the normal is undefined, and every vertex of a certified band is
/// strictly off it; the fallback exists only so the function is total.
fn cone_polygon_from_lifted(
    positions: &[Point3],
    triangles: &[[usize; 3]],
    schema: &formal::ConeSchema,
) -> PolygonMesh {
    let positions = positions.to_vec();
    let slope = schema.slope().get();
    let scale = 1.0 / (1.0 + slope * slope).sqrt();
    let normals: Vec<Vector3> = positions
        .iter()
        .map(|p| {
            let r = *p - schema.apex();
            let s = r.dot(schema.axis());
            let radial = r - s * schema.axis();
            let magnitude = radial.magnitude();
            match magnitude > 0.0 {
                true => scale * (radial / magnitude - slope * s.signum() * schema.axis()),
                false => schema.axis(),
            }
        })
        .collect();
    let tri_faces: Vec<[StandardVertex; 3]> = triangles
        .iter()
        .map(|indices| {
            array![i => StandardVertex {
                pos: indices[i],
                uv: None,
                nor: Some(indices[i]),
            }; 3]
        })
        .collect();
    PolygonMesh::debug_new(
        StandardAttributes {
            positions,
            uv_coords: Vec::new(),
            normals,
        },
        Faces::from_tri_and_quad_faces(tri_faces, Vec::new()),
    )
}

/// Run the formal conical essential-band path for one face, when it is
/// eligible.
///
/// `None` is "not eligible, nothing was attempted": no certified cone support,
/// no source evidence, or a bound count other than two authoritative bounds.
/// `Some` means `run_conical_essential_band` was actually called and the value
/// is its verdict.
///
/// This is an adapter and nothing more, and every input it hands over is one
/// production already produces — the same list [`run_cylinder_band_for_face`]
/// documents, with `cone_of` in place of `cylinder_of`. The two curve readers
/// are literally the same closures: they classify a source curve into a
/// [`formal::SourceCurveFamily`] and know nothing about the ambient surface, so
/// a complete source `CIRCLE` is read identically whichever cell will consume
/// it. What differs is entirely in what the cell then requires of that circle,
/// and that lives in [`formal::cone_band`].
#[allow(clippy::too_many_arguments)]
fn run_conical_band_for_face<S, C>(
    declared_face_index: usize,
    source_face_id: Option<u64>,
    face: &CompressedFace<S>,
    edges: &[CompressedEdge<C>],
    vertices: &[Point3],
    cone_of: &impl Fn(&S) -> std::result::Result<formal::CertifiedEmbeddedCone, &'static str>,
    cylinder_curve_schema_of: &impl Fn(&C) -> formal::CurveSchema,
    cylinder_curve_family_of: &impl Fn(&C) -> Option<formal::SourceCurveFamily>,
    tol: f64,
) -> Option<
    std::result::Result<
        (
            PolygonMesh,
            formal::cone::Nappe,
            formal::cone_band::ConicalSourceStanding,
        ),
        formal::cone_band::ConicalBandExit,
    >,
> {
    let Ok(cone) = cone_of(&face.surface) else {
        return None;
    };
    let Ok(input) =
        source_face_input_from_compressed(declared_face_index, source_face_id, face, edges)
    else {
        return None;
    };
    // Exactly two bounds, and both of them authoritative. A face with a
    // degenerate-evidence bound is not a two-bound face with one bound missing;
    // it is a face this route has no evidence for, and it is left alone rather
    // than attempted and refused.
    if input.bounds.len() != 2 || input.regular_bound_count() != 2 {
        return None;
    }

    let schema = cone.schema().clone();
    let mut curve_of = |edge_index: usize| match edges.get(edge_index) {
        Some(edge) => cylinder_curve_schema_of(&edge.curve),
        None => formal::CurveSchema::not_structurally_identified(
            formal::CurveSchemaFailure::NoStructuralReader {
                representation: "edge_index_out_of_range",
            },
        ),
    };
    let vertex_position = |vertex| match vertex {
        SourceVertexKey::ShellVertex(index) => vertices.get(index).copied(),
        _ => None,
    };
    // Resolved from *identity*: `EdgeUseId` selects the source edge use, the
    // use names its `source_edge_index`, and that indexes the shell's edge
    // table. No tessellated coordinate is consulted to decide what a curve is.
    //
    // Unlike the cylinder route this has no `Line` fallback, and that is a
    // deliberate difference rather than an oversight. This cell admits exactly
    // one curve family, so an edge use whose family cannot be read has nothing
    // it could safely be defaulted to; `None` reaches
    // `BoundNotACompleteSourceCircle` and says the boundary was not readable as
    // a complete circle, which is the true statement.
    let family_of = |edge_use: EdgeUseId| {
        input
            .edge_uses()
            .find(|use_| use_.id == edge_use)
            .and_then(|use_| edges.get(use_.source_edge_index))
            .and_then(|edge| cylinder_curve_family_of(&edge.curve))
    };

    Some(
        formal::cone_band::run_conical_essential_band(
            source_face_id,
            cone,
            &input,
            face.provenance.outer_bound,
            &mut curve_of,
            &vertex_position,
            &family_of,
            tol,
        )
        .map(|(_, mesh)| {
            (
                cone_polygon_from_lifted(
                    &mesh.physical_vertices,
                    &mesh.developed.triangles,
                    &schema,
                ),
                mesh.nappe,
                mesh.standing,
            )
        }),
    )
}

/// Build a `PolygonMesh` from a realized torus annulus, with per-vertex normals
/// recomputed from the certified torus surface (not averaged from adjacent
/// facets).
fn torus_polygon_from_realized(
    realized: &formal::torus_realize::RealizedTorusAnnulus,
    deck: &formal::torus::CertifiedRankTwoDeck,
) -> PolygonMesh {
    let schema = deck.schema();
    let center = schema.center();
    let axis = schema.axis();
    let large = schema.large_radius().get();
    let small = schema.small_radius().get();
    let positions = realized.vertices.clone();
    let normals: Vec<Vector3> = positions
        .iter()
        .map(|p| {
            let rel = *p - center;
            let h = rel.dot(axis);
            let radial = rel - h * axis;
            let magnitude = radial.magnitude();
            match magnitude > 0.0 {
                true => {
                    let radial_dir = radial / magnitude;
                    let cos_v = (magnitude - large) / small;
                    let sin_v = h / small;
                    let n = cos_v * radial_dir + sin_v * axis;
                    let n_mag = n.magnitude();
                    if n_mag > 0.0 {
                        n / n_mag
                    } else {
                        axis
                    }
                }
                false => axis,
            }
        })
        .collect();
    let tri_faces: Vec<[StandardVertex; 3]> = realized
        .triangles
        .iter()
        .map(|indices| {
            array![i => StandardVertex {
                pos: indices[i],
                uv: None,
                nor: Some(indices[i]),
            }; 3]
        })
        .collect();
    PolygonMesh::debug_new(
        StandardAttributes {
            positions,
            uv_coords: Vec::new(),
            normals,
        },
        Faces::from_tri_and_quad_faces(tri_faces, Vec::new()),
    )
}

/// Run the torus annulus route for one face, when it is eligible.
///
/// `None` is "not eligible, nothing was attempted": no certified torus support.
/// `Some` means the certification pipeline was actually run and the value is
/// its verdict — `Ok` for a certified and realized annulus, `Err` for a typed
/// refusal.
///
/// This adapter mirrors [`run_cylinder_band_for_face`] and
/// [`run_conical_band_for_face`]: it identifies the surface, extracts the
/// boundary loop placements from the source edges, certifies each circle on the
/// torus via the whole-interval Fourier test, checks homology and material
/// authority, and realizes the annulus mesh. The typed exit maps each formal
/// stage's refusal to one category of the corrected torus census.
#[allow(clippy::too_many_arguments)]
fn run_torus_annulus_for_face<S, C>(
    _declared_face_index: usize,
    _source_face_id: Option<u64>,
    face: &CompressedFace<S>,
    edges: &[CompressedEdge<C>],
    _vertices: &[Point3],
    torus_of: &impl Fn(&S) -> std::result::Result<formal::CertifiedEmbeddedTorus, &'static str>,
    cylinder_curve_family_of: &impl Fn(&C) -> Option<formal::SourceCurveFamily>,
    tol: f64,
) -> Option<
    std::result::Result<
        (PolygonMesh, formal::torus_cell::ConformanceTag),
        formal::TorusAnnulusExit,
    >,
>
where
    C: PolylineableCurve,
    S: PreMeshableSurface,
{
    use formal::torus_cell::{
        BoundaryLoopPlacement, ConformanceTag, SourceBoundaryComposition, TorusCellFailure,
        TwoOuterBoundMalformation,
    };
    use formal::{CircleFamily, CircleOnTorusStatus};
    use truck_topology::compress::OuterBoundStanding;

    // 1. Identify the torus. `None` (not eligible) when the surface is not a
    //    certified toroidal surface.
    let embedded = torus_of(&face.surface).ok()?;
    let deck = embedded.deck();
    let schema = deck.schema();
    let torus_center = schema.center();
    let torus_axis = schema.axis();
    let large = schema.large_radius().get();
    let small = schema.small_radius().get();

    // 2. Extract boundary loop placements. Each wire must reduce to exactly one
    //    complete source circle; any other edge or a multi-circle wire marks
    //    `extra` and the face is not a two-complete-circle annulus.
    let mut circle_placements: Vec<(formal::CompleteCirclePlacement, bool)> = Vec::new();
    let mut extra = false;
    for wire in &face.boundaries {
        let mut wire_circles: Vec<(formal::CompleteCirclePlacement, bool)> = Vec::new();
        for edge_ref in wire {
            let Some(edge) = edges.get(edge_ref.index) else {
                extra = true;
                continue;
            };
            match cylinder_curve_family_of(&edge.curve) {
                Some(formal::SourceCurveFamily::CompleteCircle { placement }) => {
                    wire_circles.push((placement, edge_ref.orientation));
                }
                _ => {
                    extra = true;
                }
            }
        }
        if wire_circles.len() == 1 {
            circle_placements.push(wire_circles[0]);
        } else {
            extra = true;
        }
    }
    if circle_placements.len() != 2 {
        return Some(Err(formal::TorusAnnulusExit::NotEligible));
    }

    // 3. Source boundary composition: the double-outer-bound malformation is
    //    detected from the face's provenance, not inferred from bound count.
    let outer_bound_malformation = match face.provenance.outer_bound {
        OuterBoundStanding::Declared { declared_count, .. } if declared_count >= 2 => {
            Some(TwoOuterBoundMalformation)
        }
        _ => None,
    };
    let composition = SourceBoundaryComposition {
        component_count: face.boundaries.len(),
        extra_source_edge: extra,
        outer_bound_malformation,
    };

    // 4. Build BoundaryLoopPlacement with a placeholder sign for the on-torus
    //    certification. The sign is not read by `certify_circle_on_torus`; it
    //    is only needed by `certify_torus_annular_cell` for the material
    //    authority check, and is filled in after the winding is certified.
    let (pa, orient_a) = circle_placements[0];
    let (pb, orient_b) = circle_placements[1];
    let placement_a = BoundaryLoopPlacement {
        center: pa.center,
        normal: pa.sweep_axis,
        radius: pa.radius,
        effective_orientation_sign: 0,
    };
    let placement_b = BoundaryLoopPlacement {
        center: pb.center,
        normal: pb.sweep_axis,
        radius: pb.radius,
        effective_orientation_sign: 0,
    };

    // 5. Certify each circle on the torus via the whole-interval Fourier test.
    //    This is scale-invariant, unlike the cell's own `certify_loop` check.
    let status_a = formal::certify_circle_on_torus(deck, &placement_a);
    let status_b = formal::certify_circle_on_torus(deck, &placement_b);

    let witness_a = match &status_a {
        CircleOnTorusStatus::CertifiedOnTorus { witness } => *witness,
        CircleOnTorusStatus::ProvedNotOnTorus { .. } => {
            return Some(Err(formal::TorusAnnulusExit::CircleNotOnTorus));
        }
        CircleOnTorusStatus::OnTorusUnresolved { .. } => {
            return Some(Err(formal::TorusAnnulusExit::CertificationFailure));
        }
        CircleOnTorusStatus::OperationalFailure => {
            return Some(Err(formal::TorusAnnulusExit::OperationalFailure));
        }
    };
    let witness_b = match &status_b {
        CircleOnTorusStatus::CertifiedOnTorus { witness } => *witness,
        CircleOnTorusStatus::ProvedNotOnTorus { .. } => {
            return Some(Err(formal::TorusAnnulusExit::CircleNotOnTorus));
        }
        CircleOnTorusStatus::OnTorusUnresolved { .. } => {
            return Some(Err(formal::TorusAnnulusExit::CertificationFailure));
        }
        CircleOnTorusStatus::OperationalFailure => {
            return Some(Err(formal::TorusAnnulusExit::OperationalFailure));
        }
    };

    // 6. Primitivity: only parallel `(±1, 0)` and meridian `(0, ±1)` windings
    //    are admitted — the two-complete-circle parallel/meridian annulus
    //    theorem. Diagonal `(±1, ±1)` and other primitive windings are refused.
    let family_a = witness_a.family;
    let family_b = witness_b.family;
    if !matches!(family_a, CircleFamily::Parallel | CircleFamily::Meridian)
        || !matches!(family_b, CircleFamily::Parallel | CircleFamily::Meridian)
    {
        return Some(Err(formal::TorusAnnulusExit::IncompatibleLoopWindings));
    }

    // 7. Homology: the two loops must have the same unsigned winding.
    let wa = witness_a.winding;
    let wb = witness_b.winding;
    if wa[0].unsigned_abs() != wb[0].unsigned_abs() || wa[1].unsigned_abs() != wb[1].unsigned_abs()
    {
        return Some(Err(formal::TorusAnnulusExit::IncompatibleLoopWindings));
    }

    // 8. Effective orientation sign. The winding from `lift_circle_winding`
    //    includes the `Processor`'s curve orientation (folded into
    //    `sweep_axis`), but the material-authority check must use the sign
    //    WITHOUT the curve orientation — the curve orientation is a property
    //    of the curve's parameterization, not of the loop's traversal in the
    //    face's boundary. The edge use orientation (already folded in
    //    `CompressedEdgeIndex::orientation`) is the correct place for the
    //    traversal direction.
    //
    //    `sign_c` (without curve orientation) = `winding_sign / curve_orientation`
    //    = `winding_sign * curve_orientation` (since orientation is ±1).
    let curve_orient_a = if pa.curve_orientation { 1 } else { -1 };
    let curve_orient_b = if pb.curve_orientation { 1 } else { -1 };
    let sign_a = (if family_a == CircleFamily::Parallel {
        wa[0]
    } else {
        wa[1]
    }) * curve_orient_a;
    let sign_b = (if family_b == CircleFamily::Parallel {
        wb[0]
    } else {
        wb[1]
    }) * curve_orient_b;
    let eff_a = (sign_a * if orient_a { 1 } else { -1 }) as i8;
    let eff_b = (sign_b * if orient_b { 1 } else { -1 }) as i8;

    let placement_a = BoundaryLoopPlacement {
        effective_orientation_sign: eff_a,
        ..placement_a
    };
    let placement_b = BoundaryLoopPlacement {
        effective_orientation_sign: eff_b,
        ..placement_b
    };

    // 9. Certify the annular cell using the pre-certified witnesses, which
    //    skips the cell's scale-relative on-torus check (already discharged
    //    by the Fourier test). The disjointness and material authority checks
    //    are identical to `certify_torus_annular_cell`.
    let cell = match formal::certify_torus_annular_cell_with_witnesses(
        deck,
        placement_a,
        placement_b,
        witness_a,
        witness_b,
        &composition,
    ) {
        Ok(cell) => cell,
        Err(TorusCellFailure::InconsistentBoundaryHomology) => {
            return Some(Err(formal::TorusAnnulusExit::InconsistentBoundaryHomology));
        }
        Err(TorusCellFailure::IntersectingBoundaries) => {
            return Some(Err(formal::TorusAnnulusExit::IntersectingBoundaries));
        }
        Err(TorusCellFailure::NonprimitiveWinding) | Err(TorusCellFailure::InhomologousLoops) => {
            return Some(Err(formal::TorusAnnulusExit::IncompatibleLoopWindings));
        }
        Err(TorusCellFailure::SourceContradiction) => {
            return Some(Err(formal::TorusAnnulusExit::CircleNotOnTorus));
        }
        Err(TorusCellFailure::WrongSourceBoundaryComponentCount)
        | Err(TorusCellFailure::ExtraSourceEdgePresent) => {
            return Some(Err(formal::TorusAnnulusExit::NotEligible));
        }
        Err(TorusCellFailure::UnresolvedMaterialAuthority) => {
            return Some(Err(formal::TorusAnnulusExit::CertificationFailure));
        }
    };

    // 10. Realize the annulus mesh. Grid resolution is derived from the
    //     tolerance and the torus radii, so chord deviation stays below `tol`.
    let nu = ((std::f64::consts::TAU * (large + small) / tol).sqrt() as usize)
        .max(8)
        .min(256);
    let nv = ((std::f64::consts::TAU * small / tol).sqrt() as usize)
        .max(4)
        .min(128);
    let realized =
        match formal::realize_torus_annulus(&cell, embedded.entity(), embedded.transform(), nu, nv)
        {
            Ok(r) => r,
            Err(_) => return Some(Err(formal::TorusAnnulusExit::RealizationFailure)),
        };

    // 11. Convert to PolygonMesh with per-vertex normals from the torus surface.
    let mesh = torus_polygon_from_realized(&realized, deck);
    let conformance = cell.material_authority().conformance();
    Some(Ok((mesh, conformance)))
}

/// `CYLINDER\t...` diagnostic probe, one line per candidate face. Emitted
/// from inside a parallel tessellation, so consumers must parse
/// order-independently and key on `source_face_id` / `declared_face_index`,
/// exactly as the `SLICE`/`HOLES` probes already require.
fn emit_cylinder_probe(
    source_face_id: Option<u64>,
    declared_face_index: usize,
    shell_ordinal: u64,
    record: &CylinderSliceRecord,
) {
    let id = source_face_id
        .map(|id| id.to_string())
        .unwrap_or_else(|| "none".to_string());
    eprintln!(
        "CYLINDER\tsource_face_id={id}\tdeclared_face_index={declared_face_index}\t\
         shell_ordinal={shell_ordinal}\tstage={}\tcategory={:?}\texit={}\t\
         line_edge_uses={}\tarc_edge_uses={}\tunsupported_edge_uses={}\t\
         triangles={}",
        record.stage,
        record.category,
        record.tag,
        record.line_edge_uses,
        record.arc_edge_uses,
        record.unsupported_edge_uses,
        record.triangles,
    );
}

/// One tab-separated record per candidate face, for the funnel and the
/// obstruction histogram.
///
/// Emitted from inside a parallel tessellation, so consumers must parse
/// order-independently and key on `source_face_id` / `declared_face_index`.
/// One tab-separated record per multi-bound candidate face, for the
/// planar-holes funnel.
///
/// A face that delegated — no inner bounds — is not reported: it belongs to the
/// `SLICE` funnel, and emitting it here would double-count it.
fn emit_holes_probe(
    source_face_id: Option<u64>,
    declared_face_index: usize,
    shell_ordinal: u64,
    record: &Option<formal::HoleSliceRecord>,
) {
    let Some(record) = record else { return };
    if record.delegated {
        return;
    }
    let id = source_face_id
        .map(|id| id.to_string())
        .unwrap_or_else(|| "none".to_string());
    let join = |counts: &[usize]| match counts.is_empty() {
        true => "none".to_string(),
        false => counts
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(","),
    };
    let curves = match record.curve_representations.is_empty() {
        true => "none".to_string(),
        false => record.curve_representations.join(","),
    };
    let exit = record.exit.map_or("none", formal::SliceExit::tag);
    let (triangles, cycles, euler) = match record.validity {
        Some(validity) => (
            validity.triangles.to_string(),
            validity.boundary_cycles.to_string(),
            validity.euler_characteristic.to_string(),
        ),
        None => ("none".into(), "none".into(), "none".into()),
    };
    eprintln!(
        "HOLES\tsource_face_id={id}\tdeclared_face_index={declared_face_index}\t\
         shell_ordinal={shell_ordinal}\tstage={}\tcategory={}\texit={exit}\t\
         bounds={}\tinner_bounds={}\tedge_uses_per_bound={}\t\
         polygon_vertices_per_bound={}\tcurves={curves}\ttriangles={triangles}\t\
         boundary_cycles={cycles}\teuler={euler}\tobstruction_bound={}",
        record.stage.tag(),
        record.category.tag(),
        record.bound_count,
        record.inner_bound_count,
        join(&record.edge_uses_per_bound),
        join(&record.polygon_vertices_per_bound),
        record
            .obstruction_bound
            .map_or("none", formal::BoundRole::tag),
    );
}

fn emit_slice_probe(
    source_face_id: Option<u64>,
    declared_face_index: usize,
    shell_ordinal: u64,
    record: &Option<formal::SliceRecord>,
) {
    let id = source_face_id
        .map(|id| id.to_string())
        .unwrap_or_else(|| "none".to_string());
    let Some(record) = record else {
        eprintln!(
            "SLICE\tsource_face_id={id}\tdeclared_face_index={declared_face_index}\t\
             shell_ordinal={shell_ordinal}\tcandidate=0\tstage=not_attempted\t\
             category=unresolved\texit=not_a_planar_rank0_candidate\tbounds=0\t\
             edge_uses=0\touter_bound=none\tcurves=none\tcertificate_route=none\t\
             polygon_vertices=none\ttriangles=none"
        );
        return;
    };
    let curves = match record.curve_representations.is_empty() {
        true => "none".to_string(),
        false => record.curve_representations.join(","),
    };
    let exit = record.exit.map_or("none", formal::SliceExit::tag);
    let route = record
        .certificate_route
        .map_or("none", formal::CertificateRoute::tag);
    let polygon_vertices = record
        .polygon_vertices
        .map_or("none".to_string(), |count| count.to_string());
    let triangles = record.validity.map_or("none".to_string(), |validity| {
        validity.triangles.to_string()
    });
    eprintln!(
        "SLICE\tsource_face_id={id}\tdeclared_face_index={declared_face_index}\t\
         shell_ordinal={shell_ordinal}\tcandidate=1\tstage={}\tcategory={}\texit={exit}\t\
         bounds={}\tedge_uses={}\touter_bound={}\tcurves={curves}\t\
         certificate_route={route}\tpolygon_vertices={polygon_vertices}\t\
         triangles={triangles}",
        record.stage.tag(),
        record.category.tag(),
        record.bound_count,
        record.edge_use_count,
        record.outer_bound.tag(),
    );
}

// Step-0 evidence-audit probe. Remove after `SourceFaceInput` becomes the
// production input, or promote these fields into the permanent census.
//
// One `eprintln!` per face, because face tessellation is parallel and a record
// split across calls interleaves. Consumers must parse order-independently and
// key on `source_face_id` / `declared_face_index`; two runs' probe output are
// not comparable byte-for-byte.
fn emit_evidence_probe(
    input: &std::result::Result<SourceFaceInput, SourceEvidenceError>,
    source_face_id: Option<u64>,
    declared_face_index: usize,
    lattice: &CertifiedLattice,
) {
    let declared_rank = usize::from(lattice.declared_u_period().is_some())
        + usize::from(lattice.declared_v_period().is_some());
    let certified_rank = lattice.certified_rank();
    let id = match source_face_id {
        Some(id) => id.to_string(),
        None => "none".to_string(),
    };
    let record = match input {
        Ok(input) => {
            // The two ratios are the quantitative result. Every other field is
            // a categorical fact about the seam, and there is deliberately no
            // aggregate orientation status: the factors are reported one apiece
            // because collapsing them is what this seam exists to avoid.
            // Per-factor *counts*, never a representative value from the first
            // edge use: that is correct only while every use is constructed
            // identically, and it would silently pick a winner the moment one
            // adapter retains a factor another erases.
            let edge_curve = input.edge_curve_sense_counts();
            let selected_curve = input.selected_curve_direction_counts();
            format!(
                "bounds_total={}\tbounds_regular={}\tbounds_degenerate={}\tedge_uses={}\t\
                 endpoint_ids_complete={}\tendpoints_consistent={}\t\
                 continuous_regular_bounds={}/{}\tcomputable_normalized_signs={}/{}\t\
                 face_use_orientation={}\tface_surface_history={}\t\
                 edge_curve_retained={}\tedge_curve_history_erased={}\tedge_curve_missing={}\t\
                 selected_curve_retained={}\tselected_curve_history_erased={}\t\
                 selected_curve_missing={}\tadapter_error=none",
                input.bounds.len(),
                input.regular_bound_count(),
                input.degenerate_bound_count(),
                input.edge_use_count(),
                u8::from(input.endpoint_ids_complete()),
                u8::from(input.endpoints_consistent()),
                input.continuous_regular_bound_count(),
                input.regular_bound_count(),
                input.computable_normalized_sign_count(),
                input.edge_use_count(),
                input.orientation.face_use_orientation.tag(),
                input.orientation.face_surface_same_sense.tag(),
                edge_curve.retained,
                edge_curve.history_erased,
                edge_curve.missing,
                selected_curve.retained,
                selected_curve.history_erased,
                selected_curve.missing,
            )
        }
        Err(error) => format!(
            "bounds_total=0\tbounds_regular=0\tbounds_degenerate=0\tedge_uses=0\t\
             endpoint_ids_complete=0\tendpoints_consistent=0\tcontinuous_regular_bounds=0/0\t\
             computable_normalized_signs=0/0\tface_use_orientation=missing\t\
             face_surface_history=missing\tedge_curve_retained=0\t\
             edge_curve_history_erased=0\tedge_curve_missing=0\t\
             selected_curve_retained=0\tselected_curve_history_erased=0\t\
             selected_curve_missing=0\tadapter_error={}",
            error.tag(),
        ),
    };
    eprintln!(
        "EVIDENCE\tsource_face_id={id}\tdeclared_face_index={declared_face_index}\t{record}\t\
         declared_rank={declared_rank}\tcertified_rank={certified_rank}"
    );
}

fn shell_create_polygon<S: PreMeshableSurface>(
    surface: &S,
    wires: Vec<Wire<Point3, PolylineCurve>>,
    orientation: bool,
    tol: f64,
    sp: impl SP<S>,
    lattice: &CertifiedLattice,
) -> Face<Point3, PolylineCurve, Option<PolygonMesh>> {
    let preboundary = wires
        .iter()
        .map(|wire: &Wire<_, _>| {
            let wire_iter = wire.iter().map(Edge::oriented_curve);
            PolyBoundaryPiece::try_new(surface, wire_iter, &sp, tol, lattice)
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .ok();
    let polygon: Option<PolygonMesh> = preboundary.map(|preboundary| {
        let boundary = PolyBoundary::new(preboundary, &surface, tol, lattice);
        trimming_tessellation(surface, &boundary, tol, lattice)
    });
    let mut new_face = Face::debug_new(wires, polygon);
    if !orientation {
        new_face.invert();
    }
    new_face
}

#[derive(Clone, Copy, Debug, derive_more::Deref, derive_more::DerefMut)]
pub(in crate::tessellation) struct SurfacePoint {
    pub(in crate::tessellation) point: Point3,
    #[deref]
    #[deref_mut]
    pub(in crate::tessellation) uv: Point2,
}

impl From<(Point2, Point3)> for SurfacePoint {
    fn from((uv, point): (Point2, Point3)) -> Self {
        Self { point, uv }
    }
}

/// Reconciles a UV step entering or leaving a detected collapsed direction.
///
/// A small derivative only proposes a chart substitution. The candidate UV
/// must also evaluate back to the associated 3D point within model tolerance.
/// An outgoing bridge retains the previous singular 3D point; only its UV
/// representative changes to connect the two chart branches.
fn reconcile_singular_transition<S: ParametricSurface3D>(
    surface: &S,
    previous_uv: Point2,
    previous_point: Point3,
    current_uv: &mut Point2,
    current_point: Point3,
    tolerance: f64,
    output: &mut Vec<SurfacePoint>,
) {
    let represents =
        |uv: Point2, point: Point3| surface.subs(uv.x, uv.y).distance(point) <= tolerance;
    if !previous_uv.x.near(&current_uv.x) && surface.uder(current_uv.x, current_uv.y).so_small() {
        let candidate = Point2::new(previous_uv.x, current_uv.y);
        if represents(candidate, current_point) {
            current_uv.x = previous_uv.x;
        }
    }
    if !previous_uv.y.near(&current_uv.y) && surface.vder(current_uv.x, current_uv.y).so_small() {
        let candidate = Point2::new(current_uv.x, previous_uv.y);
        if represents(candidate, current_point) {
            current_uv.y = previous_uv.y;
        }
    }
    if !previous_uv.x.near(&current_uv.x) && surface.uder(previous_uv.x, previous_uv.y).so_small() {
        let candidate = Point2::new(current_uv.x, previous_uv.y);
        if represents(candidate, previous_point) {
            output.push((candidate, previous_point).into());
        }
    }
    if !previous_uv.y.near(&current_uv.y) && surface.vder(previous_uv.x, previous_uv.y).so_small() {
        let candidate = Point2::new(previous_uv.x, current_uv.y);
        if represents(candidate, previous_point) {
            output.push((candidate, previous_point).into());
        }
    }
}

#[derive(Debug, Default, Clone)]
struct PolyBoundaryPiece(Vec<SurfacePoint>);

impl PolyBoundaryPiece {
    fn try_new<S: PreMeshableSurface>(
        surface: &S,
        wire: impl Iterator<Item = PolylineCurve>,
        sp: impl SP<S>,
        tol: f64,
        lattice: &CertifiedLattice,
    ) -> std::result::Result<Self, TessellationFailureReason> {
        // Audit A-ambient: periodicity now arrives as a descriptor whose type
        // distinguishes exact from accessor-only evidence. `declared_period`
        // is what this path read before, so the boundary is introduced with no
        // semantic change; moving a site to `generator` is separate and must
        // be measured on its own.
        let (up, vp) = (lattice.declared_u_period(), lattice.declared_v_period());
        let (urange, vrange) = surface.try_range_tuple();
        // How many polylines this bound is assembled from, and how long each
        // is. A bound winding twice is either fed two once-winding pieces --
        // assembly -- or one piece that the lift doubles. This separates them.
        let mut piece_lengths: Vec<usize> = Vec::new();
        let mut bdry3d: Vec<Point3> = wire
            .inspect(|poly_edge| piece_lengths.push(poly_edge.len()))
            .flat_map(|poly_edge| {
                if poly_edge.len() == 2 {
                    let p0 = poly_edge[0];
                    let p1 = poly_edge[1];
                    let mut pts = Vec::new();
                    const N: usize = 8;
                    for i in 0..N {
                        let frac = i as f64 / N as f64;
                        pts.push(p0 + (p1 - p0) * frac);
                    }
                    pts
                } else {
                    let n = poly_edge.len().saturating_sub(1);
                    poly_edge.into_iter().take(n).collect()
                }
            })
            .collect();
        // A wire that contributed no points cannot bound a face. This
        // constructor is already fallible, so say so rather than closing the
        // boundary by indexing a vector that is empty. Real exports do produce
        // such wires, and panicking here aborts the whole model.
        if bdry3d.is_empty() {
            return Err(TessellationFailureReason::BoundaryWireEmpty);
        }
        bdry3d.push(bdry3d[0]);
        let lift_probe = std::env::var_os("TRUCK_PROBE_LIFT").is_some();
        let mut previous: Option<(f64, f64)> = None;
        let mut previous_pt: Option<Point3> = None;
        let mut vec: Vec<SurfacePoint> = Vec::with_capacity(bdry3d.len());
        // Samples still to lift, most recent last. A step whose periodic
        // representative is ambiguous pushes its own chord midpoint and then
        // revisits itself, so density is spent only where the lift is unsafe
        // rather than across every edge in the model.
        // The flag marks a point this refinement invented rather than one the
        // edge supplied.
        let mut pending: Vec<(Point3, bool)> = Vec::new();
        for point in &bdry3d {
            pending.clear();
            pending.push((*point, false));
            let mut refinements = 0usize;
            while let Some((pt, synthetic)) = pending.pop() {
                let projected = sp(surface, pt, previous);
                // A midpoint is only a device for disambiguating the step, and
                // a chord midpoint of a coarse arc does not lie on the surface,
                // so its projection can legitimately fail. Dropping it costs
                // only the refinement; failing the face over it costs the face,
                // which is how this turned 276 surfaceless faces into 391.
                let (mut u, mut v) = match (projected, synthetic) {
                    (Some(uv), _) => uv,
                    (None, true) => continue,
                    (None, false) => {
                        return Err(TessellationFailureReason::BoundaryProjectionFailed)
                    }
                };
                // A nearest point is not an incidence.
                //
                // `search_nearest_parameter` answers whether or not the query
                // lies on the surface, so a boundary belonging to a different
                // face still yields a plausible parameter, and the uv path
                // built from it is smooth enough to triangulate into a large
                // wrong region. Every symptom chased downstream of this — a
                // doubled periodic winding, bounds landing in different period
                // copies, a domain spanning the whole chart — was a reading of
                // that path as though it meant something.
                //
                // The contract is that a face's boundary lies on its own
                // surface. Check it where the answer is produced, and refuse
                // the boundary rather than pass a fiction downstream. Measured
                // on shell 160039: 0.027 against a 0.003 tolerance, nine times
                // over, with no nearer solution anywhere in the domain.
                if !synthetic {
                    let residual = surface.subs(u, v).distance(pt);
                    if residual > tol * compatibility_factor() {
                        if std::env::var_os("TRUCK_PROBE_COMPAT").is_some() {
                            // The residual as a multiple of tolerance is the
                            // number the sweep needs: it says where this
                            // rejection would land under any other factor, so
                            // one run yields the whole distribution rather
                            // than one point on it.
                            eprintln!(
                                "COMPAT boundary point off surface: residual={residual:.4e} \
                                 permitted={:.4e} ratio={:.4}",
                                tol * compatibility_factor(),
                                residual / tol,
                            );
                        }
                        return Err(TessellationFailureReason::BoundaryPointOffSurface);
                    }
                }
                let raw = (u, v);
                if let (Some(up), Some((u0, _))) = (up, previous) {
                    u = get_mindiff(u, u0, up);
                }
                if let (Some(vp), Some((_, v0))) = (vp, previous) {
                    v = get_mindiff(v, v0, vp);
                }
                if lift_probe {
                    // Each sample's raw projection, the periodic representative
                    // chosen for it, and the step that choice implies. Aliasing
                    // shows up as a step near or beyond half a period, or as a
                    // step that closes a loop which should have stayed open.
                    let (du, dv) = match previous {
                        Some((u0, v0)) => (u - u0, v - v0),
                        None => (0.0, 0.0),
                    };
                    let frac = |d: f64, p: Option<f64>| p.map_or(0.0, |p| d / p);
                    eprintln!(
                        "LIFT raw=({:.6},{:.6}) chosen=({u:.6},{v:.6}) \
                         step=({du:+.6},{dv:+.6}) step/period=({:+.4},{:+.4})",
                        raw.0,
                        raw.1,
                        frac(du, up),
                        frac(dv, vp),
                    );
                }
                // Halve the step rather than guess which copy was meant. The
                // projection of the chord midpoint recovers a point the curve
                // actually passes through, so each half advances by less and
                // the nearest copy becomes unambiguous.
                if let (Some((u0, v0)), Some(previous_point)) = (previous, previous_pt) {
                    let ambiguous = |now: f64, before: f64, period: Option<f64>| {
                        period.is_some_and(|period| {
                            f64::abs(now - before) >= AMBIGUOUS_STEP_FRACTION * period
                        })
                    };
                    if ambiguous(u, u0, up) || ambiguous(v, v0, vp) {
                        if refinements < MAX_LIFT_REFINEMENTS {
                            refinements += 1;
                            pending.push((pt, synthetic));
                            pending.push((previous_point.midpoint(pt), true));
                            continue;
                        }
                        // G2. Bisection is exhausted and the step is still
                        // ambiguous, so no evidence distinguishes the two
                        // candidate period copies. Previously control fell
                        // through here and the ambiguous value was pushed with
                        // nothing recording that it was a guess — the face then
                        // proceeded as though the lift were certified. FS
                        // Def. 14 requires a continuous lift; an unresolved
                        // branch is not one.
                        return Err(TessellationFailureReason::AmbiguousLift);
                    }
                }
                vec.push((Point2::new(u, v), pt).into());
                previous = Some((u, v));
                previous_pt = Some(pt);
            }
        }
        if (bdry3d.len() <= 2 || bdry3d[0].distance(bdry3d[bdry3d.len() - 1]) < 1e-4)
            && piece_lengths.iter().all(|&l| l <= 2)
        {
            if let Some(up) = lattice.declared_u_period() {
                let p0 = bdry3d[0];
                if let Some((u0, v0)) = sp(surface, p0, None) {
                    let mut dense = Vec::new();
                    const STEPS: usize = 16;
                    for i in 0..=STEPS {
                        let frac = i as f64 / STEPS as f64;
                        let u = u0 + frac * up;
                        let pt = surface.subs(u, v0);
                        dense.push((Point2::new(u, v0), pt).into());
                    }
                    vec = dense;
                }
            } else if let Some(vp) = lattice.declared_v_period() {
                let p0 = bdry3d[0];
                if let Some((u0, v0)) = sp(surface, p0, None) {
                    let mut dense = Vec::new();
                    const STEPS: usize = 16;
                    for i in 0..=STEPS {
                        let frac = i as f64 / STEPS as f64;
                        let v = v0 + frac * vp;
                        let pt = surface.subs(u0, v);
                        dense.push((Point2::new(u0, v), pt).into());
                    }
                    vec = dense;
                }
            }
        }
        let grav = vec.iter().fold(Point2::origin(), |g, p| g + p.uv.to_vec()) / vec.len() as f64;
        let mut quot_u = 0.0;
        let mut quot_v = 0.0;
        if let (Some(up), Some((u0, _))) = (up, urange) {
            quot_u = f64::floor((grav.x - u0) / up);
            vec.iter_mut().for_each(|p| p.x -= quot_u * up);
        }
        if let (Some(vp), Some((v0, _))) = (vp, vrange) {
            quot_v = f64::floor((grav.y - v0) / vp);
            vec.iter_mut().for_each(|p| p.y -= quot_v * vp);
        }
        if lift_probe {
            // Which period copy this bound was placed in, and where it ended
            // up. The shift is chosen from this bound's own centroid alone, and
            // `try_new` runs once per wire, so two bounds of the same face are
            // normalized independently and can be placed in different copies.
            // Comparing these lines across the bounds of one face is the test.
            let (mut u_lo, mut u_hi) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut v_lo, mut v_hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for p in &vec {
                u_lo = u_lo.min(p.uv.x);
                u_hi = u_hi.max(p.uv.x);
                v_lo = v_lo.min(p.uv.y);
                v_hi = v_hi.max(p.uv.y);
            }
            let winding = |lo: f64, hi: f64, period: Option<f64>| {
                period.map_or(0.0, |period| (hi - lo) / period)
            };
            // Span conflates two different defects, so report the pair that
            // separates them. `k` is the net winding — how many periods the
            // boundary ends away from where it started — and `V` the total
            // variation, how far it travelled altogether. Circling once gives
            // |k| = 1 with V ~ 1. |k| = 1 with V ~ 2 means it went out and came
            // back, a branch chosen wrongly part way. |k| = 2 with V ~ 2 means
            // it genuinely went round twice, which is a duplicated wire or a
            // seam concatenated in both orientations.
            let (mut travel_u, mut travel_v) = (0.0, 0.0);
            for pair in vec.windows(2) {
                travel_u += f64::abs(pair[1].uv.x - pair[0].uv.x);
                travel_v += f64::abs(pair[1].uv.y - pair[0].uv.y);
            }
            let net = |period: Option<f64>, first: f64, last: f64| {
                period.map_or(0.0, |period| f64::round((last - first) / period))
            };
            let (first, last) = (vec[0].uv, vec[vec.len() - 1].uv);
            // Is the reported period a real period, and is the lift a valid
            // inverse at all? `e_p` and `e_2p` say whether `S` actually repeats
            // after one or two periods; `e_hp` catches a period that is not
            // fundamental. `e_inv` is the reconstruction residual, the distance
            // from each lifted parameter back to the 3D point it came from --
            // small residual with a doubled winding means the chart genuinely
            // takes two parameter circuits per geometric circuit.
            let anchor = vec[0].uv;
            let base = surface.subs(anchor.x, anchor.y);
            let shifted = |dv: f64| surface.subs(anchor.x, anchor.y + dv).distance(base);
            let (e_p, e_2p, e_hp) = match vp {
                Some(period) => (
                    shifted(period),
                    shifted(2.0 * period),
                    shifted(0.5 * period),
                ),
                None => (f64::NAN, f64::NAN, f64::NAN),
            };
            let e_inv = vec.iter().fold(0.0_f64, |worst, s| {
                worst.max(surface.subs(s.uv.x, s.uv.y).distance(s.point))
            });
            // Independent of `sp` entirely: brute-force the true distance from
            // the first boundary point to the surface. This separates the two
            // remaining explanations for a large residual. If the minimum is
            // also far, the point genuinely does not lie on this surface and
            // the edge has been paired with the wrong face. If the minimum is
            // near zero, a valid inverse existed and the projection search
            // failed to find it.
            let target = vec[0].point;
            let anchor_uv = vec[0].uv;
            let axis =
                |range: Option<(f64, f64)>, period: Option<f64>, centre: f64| match (range, period)
                {
                    (Some(r), _) => r,
                    (None, Some(p)) => (centre - p, centre + p),
                    (None, None) => (centre - 1.0, centre + 1.0),
                };
            let (ulo, uhi) = axis(urange, up, anchor_uv.x);
            let (vlo, vhi) = axis(vrange, vp, anchor_uv.y);
            const GRID: usize = 400;
            let mut d_min = f64::INFINITY;
            for i in 0..=GRID {
                let u = ulo + (uhi - ulo) * i as f64 / GRID as f64;
                for j in 0..=GRID {
                    let v = vlo + (vhi - vlo) * j as f64 / GRID as f64;
                    d_min = d_min.min(surface.subs(u, v).distance(target));
                }
            }
            // The structure of the residual, not just its size, says which
            // upstream error produced it. A constant world-space vector is a
            // missing translation. A constant magnitude aligned with the
            // surface normal is a radius or offset error. Neither pattern
            // holding means the entities are unrelated rather than
            // misplaced.
            let residuals: Vec<_> = vec
                .iter()
                .map(|s| s.point - surface.subs(s.uv.x, s.uv.y))
                .collect();
            let mean =
                residuals.iter().fold(Vector3::zero(), |acc, r| acc + r) / residuals.len() as f64;
            let spread = residuals
                .iter()
                .fold(0.0_f64, |worst, r| worst.max((r - mean).magnitude()));
            let (mut mag_lo, mut mag_hi) = (f64::INFINITY, 0.0_f64);
            let mut normal_alignment = 0.0;
            for (r, s) in residuals.iter().zip(&vec) {
                let magnitude = r.magnitude();
                mag_lo = mag_lo.min(magnitude);
                mag_hi = mag_hi.max(magnitude);
                let normal = surface
                    .uder(s.uv.x, s.uv.y)
                    .cross(surface.vder(s.uv.x, s.uv.y));
                if magnitude > 0.0 && normal.magnitude() > 0.0 {
                    normal_alignment += f64::abs(r.dot(normal.normalize()) / magnitude);
                }
            }
            normal_alignment /= residuals.len() as f64;
            eprintln!(
                "PERIOD e_p={e_p:.3e} e_2p={e_2p:.3e} e_hp={e_hp:.3e} e_inv={e_inv:.3e} \
                 d_min={d_min:.3e}"
            );
            eprintln!(
                "RESID |mean|={:.4e} spread={spread:.4e} |r|=[{mag_lo:.4e},{mag_hi:.4e}] \
                 normal_align={normal_alignment:.3}",
                mean.magnitude(),
            );
            // The same incidence certificate the source file satisfies exactly,
            // recomputed on the *converted* geometry. Both frames are recovered
            // from three sampled points, so this needs no knowledge of how the
            // conversion stores them and introduces no grid error. The source
            // gives e_angle = 0, e_axis <= 1e-14, e_radius = 0; whichever of
            // the three is non-zero here names the field the conversion
            // damaged.
            let circle_through = |a: Point3, b: Point3, c: Point3| {
                let (ab, ac) = (b - a, c - a);
                let normal = ab.cross(ac);
                let n2 = normal.magnitude2();
                if n2 < f64::EPSILON {
                    return None;
                }
                let centre = a
                    + (ac.magnitude2() * ab.cross(normal) - ab.magnitude2() * ac.cross(normal))
                        / (2.0 * n2);
                Some((centre, normal.normalize(), centre.distance(a)))
            };
            // Boundary curve: three points spread along it.
            let n = vec.len();
            let boundary_fit = if n >= 3 {
                circle_through(vec[0].point, vec[n / 3].point, vec[2 * n / 3].point)
            } else {
                None
            };
            // Surface cross-section: three points around one periodic axis.
            let anchor = vec[0].uv;
            let cross_fit = match (up, vp) {
                (Some(p), _) => circle_through(
                    surface.subs(anchor.x, anchor.y),
                    surface.subs(anchor.x + p / 3.0, anchor.y),
                    surface.subs(anchor.x + 2.0 * p / 3.0, anchor.y),
                ),
                (_, Some(p)) => circle_through(
                    surface.subs(anchor.x, anchor.y),
                    surface.subs(anchor.x, anchor.y + p / 3.0),
                    surface.subs(anchor.x, anchor.y + 2.0 * p / 3.0),
                ),
                _ => None,
            };
            if let (Some((cc, cn, cr)), Some((sc, sn, sr))) = (boundary_fit, cross_fit) {
                let e_angle = f64::acos(f64::min(1.0, f64::abs(cn.dot(sn)))).to_degrees();
                let offset = cc - sc;
                let e_axis = (offset - sn * offset.dot(sn)).magnitude();
                eprintln!(
                    "INCID e_angle={e_angle:.4}deg e_axis={e_axis:.4e} \
                     e_radius={:.4e} r_curve={cr:.5} r_surf={sr:.5}",
                    f64::abs(cr - sr),
                );
            }
            eprintln!(
                "BOUND pieces={piece_lengths:?} pts={} k=({:+.0},{:+.0}) V=({:.2},{:.2}) \
                 quot=({quot_u:+.0},{quot_v:+.0}) \
                 u=[{u_lo:.4},{u_hi:.4}] v=[{v_lo:.4},{v_hi:.4}] \
                 span/period=({:.3},{:.3})",
                vec.len(),
                net(up, first.x, last.x),
                net(vp, first.y, last.y),
                up.map_or(0.0, |p| travel_u / p),
                vp.map_or(0.0, |p| travel_v / p),
                winding(u_lo, u_hi, up),
                winding(v_lo, v_hi, vp),
            );
        }
        let last = *vec.last().unwrap();
        if !vec[0].near(&last) {
            let Point2 { x: u0, y: v0 } = last.uv;
            if surface.uder(u0, v0).so_small() || surface.vder(u0, v0).so_small() {
                vec.push(vec[0]);
            }
        }
        Ok(Self(vec))
    }
}

fn get_mindiff(u: f64, u0: f64, up: f64) -> f64 {
    // The nearest periodic copy outright, rather than the nearest among five.
    // The old search covered only two periods either side, so a boundary that
    // wrapped further was silently pulled back; rounding has no such bound and
    // is cheaper.
    u + f64::round((u0 - u) / up) * up
}

/// How far a boundary point may sit from its own surface, as a multiple of the
/// chord tolerance, before the pairing is refused.
///
/// A face's boundary is required to lie on that face's surface; this is the
/// slack allowed for the chord approximation and for imperfect exports, not a
/// licence to trim a surface with a curve belonging to something else.
///
/// **Off by default, deliberately.** The violation this detects is real —
/// swept on `00009190`, the rejected points sit at a median of 191x tolerance
/// and a maximum of 617x, and loosening the factor twentyfold removes only 62
/// of 315 rejections, so this is a population and not a threshold. But
/// rejecting them repairs nothing visible: the blob shells are byte-identical
/// with the gate on and off, while the gate costs 292 faces and 21,131
/// triangles, a tenth of the model. Deleting real geometry to fix nothing is
/// the wrong default. Set `TRUCK_COMPAT_FACTOR=5` to measure the population;
/// turn it on for real only once something downstream can use the refusal.
const COMPATIBILITY_FACTOR: f64 = f64::INFINITY;

/// The factor in force, overridable by `TRUCK_COMPAT_FACTOR`.
///
/// Sweeping the factor is what established that the gate names a real
/// population rather than a threshold, and a rebuild per sample would have made
/// that a five-build afternoon. Read once — this sits in the per-boundary-point
/// loop, and an env lookup there would be a measurable cost charged to every
/// model.
fn compatibility_factor() -> f64 {
    static FACTOR: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *FACTOR.get_or_init(|| {
        std::env::var("TRUCK_COMPAT_FACTOR")
            .ok()
            .and_then(|raw| raw.trim().parse::<f64>().ok())
            .filter(|factor| *factor > 0.0 && !factor.is_nan())
            .unwrap_or(COMPATIBILITY_FACTOR)
    })
}

/// How far a step may advance, as a fraction of the period, before the periodic
/// representative it implies is treated as ambiguous.
///
/// [`get_mindiff`] takes the copy nearest the previous parameter, which is the
/// right answer only while the true step is under half a period. At exactly
/// half, the two candidates are equidistant and the tie is broken arbitrarily —
/// measured advancing `-0.5` of a period where the curve went `+0.5`, which
/// folds a full turn onto itself and makes a period-wrapping boundary look like
/// a closed loop. The margin below `0.5` keeps numerical noise clear of the tie.
const AMBIGUOUS_STEP_FRACTION: f64 = 0.45;

/// How many times a single step may be halved before refinement gives up.
const MAX_LIFT_REFINEMENTS: usize = 8;

/// How many independent ray directions [`PolyBoundary::include`] may try before
/// reporting that containment is undecidable at a point.
///
/// Cost is confined to the abort path — `find_map` stops at the first ray that
/// decides. Measured on ABC `00009190`: 18% of the aborts left by a single cast
/// are resolved by alternate directions, and every one of those resolved to
/// *outside*, changing no output. Whatever still aborts after eight directions
/// is classified by [`PolyBoundary::on_boundary`], a direct predicate, rather
/// than by inferring a location from the rays having failed.
const INCLUSION_RAY_ATTEMPTS: u32 = 8;

/// Where a point lies relative to a trimmed domain.
///
/// `Boundary` and `Indeterminate` are deliberately distinct. The first is a
/// positive result from a direct point-on-segment test; the second means no
/// method established a location. Collapsing them would reintroduce, one level
/// up, the conflation this type exists to remove.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PointLocation {
    /// Strictly inside the material domain.
    Inside,
    /// Strictly outside it.
    Outside,
    /// On a boundary segment within `TOLERANCE`, by direct test.
    Boundary,
    /// Not established by any available method.
    Indeterminate,
}

/// Result of boundary loop closure evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryClosure {
    /// Endpoints are within Euclidean UV tolerance.
    EuclideanClosed,
    /// Endpoints close modulo parameter period.
    PeriodicClosed {
        /// Integer winding displacement along [u, v].
        displacement: [i64; 2],
    },
    /// Boundary loop is un-closed (open).
    Open,
}

/// Evaluates periodic displacement winding modulo period if residual <= tolerance.
fn periodic_displacement(start: f64, end: f64, period: f64, tolerance: f64) -> Option<i64> {
    if period <= 1e-6 {
        return None;
    }
    let winding = ((end - start) / period).round() as i64;
    let residual = (end - start) - winding as f64 * period;
    if residual.abs() <= tolerance {
        Some(winding)
    } else {
        None
    }
}

/// Where one boundary segment came from.
///
/// **G6, phase 2A.** `PolyBoundary::new` stitches synthesised closure and seam
/// segments into the *same* point vectors as source-derived trim, after which
/// nothing distinguishes them — so `insert_to` tagged every segment
/// `PhysicalBoundary`, fabricated geometry included. The fix is not to
/// reclassify afterwards but to record the origin where the segment is created,
/// which is the only place it is known.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum SegmentOrigin {
    /// Carries source boundary evidence: a lifted source edge use.
    Source,
    /// Synthesised to close an open piece against the working extent. No source
    /// entity describes it (`DOM-ARTIFICIAL-CLOSURE-001`).
    SyntheticClosure,
    /// Synthesised to bridge a collapsed periodic pair — a seam across a
    /// degenerate direction rather than a trim boundary.
    Seam,
}

impl SegmentOrigin {
    /// The constraint role this origin justifies.
    ///
    /// Deliberately **behaviour-preserving**: the synthetic roles still toggle
    /// material parity, exactly as they did while masquerading as
    /// `PhysicalBoundary`. This makes the populations nameable and countable;
    /// deciding what a synthesised segment *should* do to material state is a
    /// separate change that must be measured on its own.
    fn role(self) -> ConstraintRole {
        match self {
            Self::Source => ConstraintRole::PhysicalBoundary,
            Self::SyntheticClosure | Self::Seam => ConstraintRole::UnresolvedSyntheticClosure,
        }
    }
}

/// A closed boundary loop in parameter space, carrying each segment's origin.
///
/// `origins[i]` describes the segment from `points[i]` to `points[i + 1]`,
/// cyclically, so the two vectors have equal length by construction.
#[derive(Debug, Default, Clone)]
struct BoundaryLoop {
    points: Vec<SurfacePoint>,
    origins: Vec<SegmentOrigin>,
}

impl BoundaryLoop {
    /// Build from parts that are known to chain end-to-start, closing back on
    /// the first part's start. Every join is a shared endpoint, so no segment
    /// is invented; this is the stitching case, where each run was constructed
    /// to begin where the previous one ended.
    fn chained(parts: impl IntoIterator<Item = (Vec<SurfacePoint>, SegmentOrigin)>) -> Self {
        let mut path = BoundaryPath::default();
        for (part, origin) in parts {
            path.append(part, origin, PartJoin::SharedEndpoint);
        }
        path.close(PartJoin::SharedEndpoint)
    }

    /// Cut the cyclic loop open at its wrap segment, yielding a path whose
    /// origins are retained.
    ///
    /// The wrap's own origin is dropped because that segment ceases to exist;
    /// every other segment keeps the label it was created with. This is what
    /// lets a loop be re-joined to something else without its provenance being
    /// rebuilt from scratch — taking `.points` and relabelling would, for
    /// instance, silently turn a periodic walk's deck seam back into `Source`.
    fn into_path_cutting_wrap(self) -> BoundaryPath {
        let Self {
            points,
            mut origins,
        } = self;
        origins.pop();
        BoundaryPath { points, origins }
    }

    /// Checked constructor. The equal-length relation is the type's whole
    /// invariant, so it is enforced rather than documented.
    fn new(points: Vec<SurfacePoint>, origins: Vec<SegmentOrigin>) -> Self {
        assert_eq!(
            points.len(),
            origins.len(),
            "every boundary segment must carry exactly one origin",
        );
        Self { points, origins }
    }

    /// A loop whose duplicate endpoint has already been removed, so every
    /// cyclic segment — including the wrap from the last point back to the
    /// first — is source-derived.
    fn euclidean_source_loop(points: Vec<SurfacePoint>) -> Self {
        let origins = vec![SegmentOrigin::Source; points.len()];
        Self::new(points, origins)
    }

    /// A lifted walk that closes only *modulo the lattice*: its last point is
    /// `first + Lδ`, a distinct parameter point, and is retained.
    ///
    /// The wrap segment is therefore **not** another source trim segment — it
    /// is the deck closure, and labelling it `Source` would feed the material
    /// solve a boundary no source entity describes. Properly this should not be
    /// a geometric segment at all but a deck identification; until the quotient
    /// stage exists to hold that relation, it is marked `Seam`, which keeps the
    /// current toggling behaviour while naming what it is.
    fn periodic_source_walk(points: Vec<SurfacePoint>) -> Self {
        let mut origins = vec![SegmentOrigin::Source; points.len()];
        if let Some(wrap) = origins.last_mut() {
            *wrap = SegmentOrigin::Seam;
        }
        Self::new(points, origins)
    }
}

/// How one boundary part meets the next.
///
/// **Stated by the caller, never inferred.** An earlier version decided this by
/// testing `tail.uv.distance(next[0].uv) < TOLERANCE`, which is wrong twice
/// over. A UV epsilon cannot distinguish a retained shared endpoint from a deck
/// identification, a singular attachment, or an unresolved relation — they are
/// different facts that can present with the same coordinates — and its
/// tolerance has no fixed physical meaning across parameterisations. The
/// stitching site already knows which case it is building; the type now makes
/// it say so.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PartJoin {
    /// The next part begins at the point the previous one ended on. The
    /// duplicate is dropped and **no segment is created**.
    SharedEndpoint,
    /// The parts do not meet. A segment is created between them, carrying the
    /// given origin because neither part supplied it.
    Bridge(SegmentOrigin),
}

/// An *open* chain of boundary segments, before it is closed into a loop.
///
/// `origins[i]` labels the segment `points[i] -> points[i + 1]`, so there is
/// exactly one fewer origin than point. Keeping the open case in its own type
/// is what makes the closing segment an explicit decision rather than an
/// accident of indexing.
#[derive(Debug, Default, Clone)]
struct BoundaryPath {
    points: Vec<SurfacePoint>,
    origins: Vec<SegmentOrigin>,
}

impl BoundaryPath {
    fn start(points: Vec<SurfacePoint>, origin: SegmentOrigin) -> Self {
        let origins = vec![origin; points.len().saturating_sub(1)];
        Self { points, origins }
    }

    /// Append a part, saying explicitly how it meets what is already here.
    ///
    /// A shared endpoint drops the duplicate point and creates no segment. A
    /// bridge keeps **both** endpoints and inserts one labelled segment between
    /// them — which is the case the previous implementation got wrong: it
    /// dropped every part's final point unconditionally, so a bridge silently
    /// replaced `a1 -> a2 -> b0` with the shortcut `a1 -> b0`, deleting a real
    /// source segment precisely when the distinction mattered most.
    fn append(&mut self, mut part: Vec<SurfacePoint>, origin: SegmentOrigin, join: PartJoin) {
        if part.is_empty() {
            return;
        }
        if self.points.is_empty() {
            *self = Self::start(part, origin);
            return;
        }
        match join {
            PartJoin::SharedEndpoint => {
                part.remove(0);
            }
            PartJoin::Bridge(bridge) => self.origins.push(bridge),
        }
        self.origins
            .extend(std::iter::repeat_n(origin, part.len().saturating_sub(1)));
        if !part.is_empty() {
            // The segment from the current tail into the first retained point
            // of `part` belongs to `part` when they shared an endpoint, and was
            // already labelled as the bridge otherwise.
            if matches!(join, PartJoin::SharedEndpoint) {
                self.origins.push(origin);
            }
            self.points.extend(part);
        }
    }

    /// Append another path, preserving its per-segment origins.
    fn append_path(&mut self, other: BoundaryPath, join: PartJoin) {
        if other.points.is_empty() {
            return;
        }
        if self.points.is_empty() {
            *self = other;
            return;
        }
        let BoundaryPath {
            mut points,
            origins,
        } = other;
        match join {
            PartJoin::SharedEndpoint => {
                points.remove(0);
            }
            PartJoin::Bridge(bridge) => self.origins.push(bridge),
        }
        self.origins.extend(origins);
        self.points.extend(points);
    }

    /// Reverse traversal.
    ///
    /// Sound on an open path precisely *because* it is open: with `origins[i]`
    /// labelling `points[i] -> points[i + 1]`, reversing both vectors maps
    /// segment `i` to old segment `n - 2 - i`, the same segment travelled
    /// backwards. The cyclic case is **not** this — reversing a loop's two
    /// vectors directly is off by one, because the wrap segment does not move
    /// with the rest. Cutting a loop into a path first removes the need to
    /// reason about where the cut went.
    fn reverse(&mut self) {
        self.points.reverse();
        self.origins.reverse();
    }

    /// Close the path into a cyclic loop, saying what the closing segment is.
    ///
    /// `SharedEndpoint` means the path already returns to its start, so the
    /// duplicate final point is dropped and the existing last segment becomes
    /// the wrap. `Bridge` keeps every point and adds one labelled wrap segment.
    fn close(mut self, join: PartJoin) -> BoundaryLoop {
        match join {
            PartJoin::SharedEndpoint => {
                self.points.pop();
            }
            PartJoin::Bridge(bridge) => self.origins.push(bridge),
        }
        BoundaryLoop::new(self.points, self.origins)
    }
}

impl BoundaryLoop {
    fn len(&self) -> usize {
        self.points.len()
    }
}

#[derive(Debug, Default, Clone)]
struct PolyBoundary(Vec<BoundaryLoop>);

fn normalize_range(curve: &mut Vec<SurfacePoint>, compidx: usize, (u0, u1): (f64, f64)) {
    let p = curve[0];
    let q = curve[curve.len() - 1];
    let tmp = f64::min(p[compidx], q[compidx]) + TOLERANCE;
    let del = f64::floor((tmp - u0) / (u1 - u0)) * (u1 - u0);
    curve.iter_mut().for_each(|p| p[compidx] -= del);
    let Some(i) = curve
        .iter()
        .position(|p| (curve[0][compidx] - u1) * (p[compidx] - u1) < 0.0)
    else {
        return;
    };
    let mut curve1 = curve.split_off(i + 1);
    curve1.pop();
    curve1.insert(0, curve[i]);
    match curve[0][compidx] < curve[curve.len() - 1][compidx] {
        true => curve1.iter_mut(),
        false => curve.iter_mut(),
    }
    .for_each(|p| p[compidx] -= u1 - u0);
    curve1.append(curve);
    *curve = curve1;
}

/// Twice the signed area of a closed `uv` loop, by the shoelace formula.
///
/// Diagnostic only. Its *sign* must not be used to decide what a face
/// contains: it negates under an orientation-reversing reparameterization,
/// which no observer of the solid can detect, so any predicate built on it
/// classifies the same face differently depending on how its surface happens
/// to be parameterized. Relative sign between loops is invariant; absolute
/// sign is not.
#[allow(dead_code)]
/// Record one boundary piece's deck evidence into the DIAG-001 sink.
///
/// The winding sign uses the *same* degeneracy threshold the two-closed-loop
/// branch tests against, so a piece recorded as sign `0` is exactly a piece
/// that branch would admit. Reporting a different threshold here would make
/// the record describe a decision nothing takes.
fn record_piece_deck(
    piece_index: usize,
    points: &[SurfacePoint],
    closure: ObservedClosure,
    ku: i64,
    kv: i64,
) {
    let area = signed_area(points);
    let uv = |p: &SurfacePoint| (p.uv.x, p.uv.y);
    let (start_uv, end_uv) = match (points.first(), points.last()) {
        (Some(first), Some(last)) => (uv(first), uv(last)),
        _ => ((f64::NAN, f64::NAN), (f64::NAN, f64::NAN)),
    };
    diagnosis::record_boundary_piece(diagnosis::BoundaryPieceDeck {
        piece_index,
        closure,
        ku,
        kv,
        winding_sign: match area {
            a if a.abs() < DEGENERATE_LOOP_AREA => 0,
            a if a > 0.0 => 1,
            _ => -1,
        },
        signed_area: area,
        representative: start_uv,
        start_uv,
        end_uv,
        point_count: points.len(),
    });
}

/// Record what the two-closed-loop join did to the deck sum.
///
/// `Σδᵢ = Δ_walk`, and `Δ_walk = 0` for a contractible regular boundary. The
/// branch traverses loop 1 **reversed**, unconditionally, so the sum it
/// realises is `δ₀ − δ₁`. `forward_would_close` is the discriminator: it is
/// true exactly when the reversal is what broke the equation and traversing
/// forward would satisfy it, which is the case package 1 is about.
fn record_two_loop_join(
    loop0_displacement: [i64; 2],
    loop1_displacement: [i64; 2],
    mean_translate: [i64; 2],
    loop1_reversed: bool,
    loop0_path: &BoundaryPath,
    loop1_path: &BoundaryPath,
) {
    let uv = |p: &SurfacePoint| (p.uv.x, p.uv.y);
    let nan = (f64::NAN, f64::NAN);
    let ends = |path: &BoundaryPath| match (path.points.first(), path.points.last()) {
        (Some(first), Some(last)) => (uv(first), uv(last)),
        _ => (nan, nan),
    };
    let (loop0_start, loop0_end) = ends(loop0_path);
    let (loop1_start, loop1_end) = ends(loop1_path);
    // The sum the chosen traversal realises, and the sum the other one would.
    let sign = if loop1_reversed { -1 } else { 1 };
    let deck_sum_u = loop0_displacement[0] + sign * loop1_displacement[0];
    let deck_sum_v = loop0_displacement[1] + sign * loop1_displacement[1];
    let other_u = loop0_displacement[0] - sign * loop1_displacement[0];
    let other_v = loop0_displacement[1] - sign * loop1_displacement[1];
    let deck_consistent = deck_sum_u == 0 && deck_sum_v == 0;
    diagnosis::record_two_loop_join(diagnosis::TwoLoopJoinRecord {
        loop0_displacement,
        loop1_displacement,
        loop1_reversed,
        mean_translate,
        deck_sum_u,
        deck_sum_v,
        deck_consistent,
        forward_would_close: !deck_consistent && other_u == 0 && other_v == 0,
        // The join appends loop 1's reversed path after loop 0's, then closes
        // back to loop 0's start: two bridges, and these are their endpoints.
        bridge0: [loop0_end, loop1_start],
        bridge1: [loop1_end, loop0_start],
    });
}

/// The parameter-space area below which a closed loop is treated as degenerate
/// — a band's boundary circle, which encloses no area in the chart because it
/// *is* a chart-crossing line.
///
/// Named because the two-closed-loop branch and the DIAG-001 record must test
/// the same number: a record taken against a different threshold would describe
/// a population no code path acts on.
const DEGENERATE_LOOP_AREA: f64 = 1e-4;

fn signed_area(curve: &[SurfacePoint]) -> f64 {
    curve
        .iter()
        .circular_tuple_windows()
        .fold(0.0, |sum, (p, q)| sum + (q.x + p.x) * (q.y - p.y))
}

/// The working parameter rectangle of a trimmed face, derived from the face's
/// own bounds.
///
/// **`PAR-RANGE-INHERITANCE-001`.** A surface's declared `parameter_range` is a
/// property of the primitive it was constructed from, not of any face that
/// references it. `Line::parameter_range` is `[0, 1]` unconditionally and
/// `RevolutedCurve` inherits it, so a cone built as a revolved line declares
/// `[0, 1] x [0, 2pi)` — one unit of generatrix starting at the STEP reference
/// radius, chosen by the primitive and unrelated to the face. Stitching an open
/// boundary piece against the edge of that rectangle fabricates trim geometry
/// no source entity describes (`DOM-ARTIFICIAL-CLOSURE-001`), and when the
/// piece already lies on the edge the enclosed area is zero
/// (`DOM-ZERO-AREA-001`).
///
/// Measured: extending that range by a constant instead recovers 348 NIST faces
/// and destroys 268 others, in a disjoint set of models — one part in two
/// encodings loses 148 cone faces under whichever window excludes it. Any
/// fixed-size window trades one population for another, because whether a
/// face's material interval falls inside is decided by where its exporter put
/// the reference circle. The extent has to come from the face.
///
/// **A periodic axis is different in kind**: its extent *is* determined, by the
/// period, and the seam handling below relies on the wrap interval being one
/// full period. Only non-periodic axes are re-derived.
///
/// Returns `None` for an axis the bounds do not determine. A degenerate extent
/// means the material region is not recoverable from the boundary alone, and
/// the caller must refuse rather than invent one — a collapsed single-vertex
/// bound is exactly that case, marking a point the domain must reach while
/// contributing no trim segment (`QUO-005`, `SNG-COLLAPSED-DIRECTION-001`).
fn working_range(
    pieces: &[PolyBoundaryPiece],
    surface: &impl PreMeshableSurface,
) -> (Option<(f64, f64)>, Option<(f64, f64)>) {
    let (udeclared, vdeclared) = surface.try_range_tuple();
    let axis = |idx: usize, period: Option<f64>, declared: Option<(f64, f64)>| {
        // A period determines the extent; the bounds do not get a say.
        if period.is_some() {
            return declared;
        }
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        pieces.iter().for_each(|PolyBoundaryPiece(vec)| {
            vec.iter().for_each(|p| {
                lo = f64::min(lo, p[idx]);
                hi = f64::max(hi, p[idx]);
            })
        });
        (hi - lo > TOLERANCE).then_some((lo, hi))
    };
    (
        axis(0, surface.u_period(), udeclared),
        axis(1, surface.v_period(), vdeclared),
    )
}

/// How the two-closed-loop branch traverses the second loop.
///
/// The branch has always reversed loop 1 unconditionally. For a quotient-closed
/// boundary walk `Σδᵢ = Δ_walk`, with `Δ_walk = 0` for a contractible regular
/// boundary, so the reversal is only correct when the two loops wind the *same*
/// way. The two boundary circles of a band wind opposite — as they must, for
/// the face boundary to be coherently oriented — and there the reversal makes
/// `Σδ = ±2`, which is exactly the crossing the CDT then refuses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TwoLoopJoinPolicy {
    /// Reverse loop 1 unconditionally.
    Legacy,
    /// Traverse loop 1 in whichever direction satisfies `Σδ = 0`, and fall back
    /// to [`Self::Legacy`] when no direction does or both do. The equation is
    /// decidable, so this guesses nothing: a direction is chosen only when it
    /// is the unique solution.
    DeckConsistent,
}

/// What the two-closed-loop branch concluded about its deck equation.
///
/// Reported rather than inferred, so a caller can tell "the correction changed
/// the boundary" from "there was nothing to correct" without rebuilding and
/// comparing meshes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TwoLoopJoinOutcome {
    /// The branch did not run: the face does not present two degenerate closed
    /// loops on a periodic chart.
    NotAttempted,
    /// `Σδ = 0` already holds for the reversed traversal.
    LegacyDeckConsistent,
    /// `Σδ ≠ 0` reversed, and forward traversal is the unique solution.
    /// `applied` says whether the policy let it be taken.
    ForwardResolves {
        /// Whether the forward traversal was used.
        applied: bool,
    },
    /// Neither traversal satisfies the equation. Refused: the legacy join is
    /// retained and the face keeps whatever typed failure it had.
    Inconsistent,
    /// Both traversals satisfy it, so the deck equation does not select one.
    /// Refused for the same reason.
    Unresolved,
}

impl PolyBoundary {
    fn new(
        pieces: Vec<PolyBoundaryPiece>,
        surface: &impl PreMeshableSurface,
        tol: f64,
        lattice: &CertifiedLattice,
    ) -> Self {
        Self::new_with_join(pieces, surface, tol, lattice, TwoLoopJoinPolicy::Legacy).0
    }

    fn new_with_join(
        pieces: Vec<PolyBoundaryPiece>,
        surface: &impl PreMeshableSurface,
        tol: f64,
        lattice: &CertifiedLattice,
        join_policy: TwoLoopJoinPolicy,
    ) -> (Self, TwoLoopJoinOutcome) {
        let mut join_outcome = TwoLoopJoinOutcome::NotAttempted;
        let probe = std::env::var_os("TRUCK_PROBE_BOUNDARY").is_some();
        let had_source_pieces = !pieces.is_empty();
        // EXPERIMENT (TRUCK_FACE_DOMAIN): take the working rectangle from the
        // face's own bounds rather than from the supporting primitive's
        // declared range. Default off until swept.
        let range = match std::env::var_os("TRUCK_FACE_DOMAIN").is_some() {
            true => working_range(&pieces, surface),
            false => surface.try_range_tuple(),
        };
        let (mut closed, mut open) = (Vec::new(), Vec::new());
        // The lattice displacement of each closed loop, parallel to `closed`.
        // The `BoundaryLoop` the classification produces does not retain it,
        // and the two-closed-loop branch below needs it to say what its join
        // does to the deck sum — recovering it afterwards from normalised
        // points would re-derive an integer the classifier already decided.
        let mut closed_displacements: Vec<[i64; 2]> = Vec::new();
        let u_period = lattice.declared_u_period();
        let v_period = lattice.declared_v_period();
        // DIAG-001 deck evidence. Recorded where each piece is classified, so
        // the displacement written down is the one the pipeline acted on rather
        // than one recovered later from already-normalised points.
        let diag = diagnosis::diag_enabled();
        pieces
            .into_iter()
            .enumerate()
            .for_each(|(piece_index, PolyBoundaryPiece(mut vec))| {
            let p0 = vec[0].uv;
            let p1 = vec[vec.len() - 1].uv;

            let closure = if p0.distance(p1) < 1.0e-3 {
                BoundaryClosure::EuclideanClosed
            } else {
                let ku = u_period
                    .and_then(|up| periodic_displacement(p0.x, p1.x, up, 1e-3))
                    .unwrap_or(0);
                let kv = v_period
                    .and_then(|vp| periodic_displacement(p0.y, p1.y, vp, 1e-3))
                    .unwrap_or(0);
                if (ku != 0 || kv != 0) && vec[0].point.distance(vec[vec.len() - 1].point) < 1e-3 {
                    BoundaryClosure::PeriodicClosed {
                        displacement: [ku, kv],
                    }
                } else {
                    BoundaryClosure::Open
                }
            };

            if probe {
                let perimeter: f64 = vec
                    .windows(2)
                    .map(|w| w[0].uv.distance(w[1].uv))
                    .sum::<f64>();
                eprintln!(
                    "PROBE piece pts={} gap={:.6e} perimeter={perimeter:.6e} \
                     closure={:?}",
                    vec.len(),
                    p0.distance(p1),
                    closure,
                );
            }

            match closure {
                BoundaryClosure::EuclideanClosed => {
                    vec.pop();
                    if diag {
                        record_piece_deck(piece_index, &vec, ObservedClosure::EuclideanClosed, 0, 0);
                    }
                    closed_displacements.push([0, 0]);
                    closed.push(BoundaryLoop::euclidean_source_loop(vec));
                }
                BoundaryClosure::PeriodicClosed {
                    displacement: [ku, kv],
                } => {
                    if let Some(up) = u_period {
                        if ku != 0 {
                            for p in &mut vec {
                                p.uv.x -= (ku as f64) * up;
                            }
                            vec.last_mut().unwrap().uv.x = vec[0].uv.x + (ku as f64) * up;
                        }
                    }
                    if let Some(vp) = v_period {
                        if kv != 0 {
                            for p in &mut vec {
                                p.uv.y -= (kv as f64) * vp;
                            }
                            vec.last_mut().unwrap().uv.y = vec[0].uv.y + (kv as f64) * vp;
                        }
                    }
                    if diag {
                        record_piece_deck(
                            piece_index,
                            &vec,
                            ObservedClosure::PeriodicClosed,
                            ku,
                            kv,
                        );
                    }
                    closed_displacements.push([ku, kv]);
                    closed.push(BoundaryLoop::periodic_source_walk(vec));
                }
                BoundaryClosure::Open => {
                    if diag {
                        record_piece_deck(piece_index, &vec, ObservedClosure::Open, 0, 0);
                    }
                    open.push(vec)
                }
            }
        });
        if closed.len() == 2
            && (lattice.declared_u_period().is_some() || lattice.declared_v_period().is_some())
        {
            let area0 = signed_area(&closed[0].points);
            let area1 = signed_area(&closed[1].points);
            if area0.abs() < DEGENERATE_LOOP_AREA && area1.abs() < DEGENERATE_LOOP_AREA {
                let loop0_displacement = closed_displacements[0];
                let loop1_displacement = closed_displacements[1];
                let mut mean_translate = [0i64, 0i64];
                let loop0 = closed.remove(0);
                let mut loop1 = closed.remove(0);
                let u0_mean: f64 =
                    loop0.points.iter().map(|p| p.uv.x).sum::<f64>() / loop0.points.len() as f64;
                let u1_mean: f64 =
                    loop1.points.iter().map(|p| p.uv.x).sum::<f64>() / loop1.points.len() as f64;
                if let Some(up) = lattice.declared_u_period() {
                    let ku = ((u0_mean - u1_mean) / up).round();
                    mean_translate[0] = ku as i64;
                    if ku != 0.0 {
                        for p in &mut loop1.points {
                            p.uv.x += ku * up;
                        }
                    }
                }
                let v0_mean: f64 =
                    loop0.points.iter().map(|p| p.uv.y).sum::<f64>() / loop0.len() as f64;
                let v1_mean: f64 =
                    loop1.points.iter().map(|p| p.uv.y).sum::<f64>() / loop1.len() as f64;
                if let Some(vp) = lattice.declared_v_period() {
                    let kv = ((v0_mean - v1_mean) / vp).round();
                    mean_translate[1] = kv as i64;
                    if kv != 0.0 {
                        for p in &mut loop1.points {
                            p.uv.y += kv * vp;
                        }
                    }
                }
                // Both halves are source-derived, but joining a loop to a
                // *reversed* loop introduces two segments that neither
                // supplied: the jump from `loop0`'s end to `loop1`'s reversed
                // start, and the closing wrap back to `loop0`'s start. Building
                // this by parts labels those bridges instead of letting them
                // inherit `Source`.
                // Solve the deck equation before choosing a traversal. Reversing
                // loop 1 realises `δ₀ − δ₁`; traversing it forward realises
                // `δ₀ + δ₁`. `Δ_walk = 0`, so each direction is admissible
                // exactly when its sum vanishes, and the direction is *chosen*
                // only when precisely one does.
                let reversed_closes = loop0_displacement[0] == loop1_displacement[0]
                    && loop0_displacement[1] == loop1_displacement[1];
                let forward_closes = loop0_displacement[0] == -loop1_displacement[0]
                    && loop0_displacement[1] == -loop1_displacement[1];
                let take_forward = match (reversed_closes, forward_closes) {
                    (false, true) => {
                        let applied = join_policy == TwoLoopJoinPolicy::DeckConsistent;
                        join_outcome = TwoLoopJoinOutcome::ForwardResolves { applied };
                        applied
                    }
                    (true, false) => {
                        join_outcome = TwoLoopJoinOutcome::LegacyDeckConsistent;
                        false
                    }
                    // Both zero — the loops are Euclidean-closed on a periodic
                    // chart, so the equation says nothing about direction — or
                    // neither, which is a boundary the deck model does not
                    // describe. Refuse in both cases and keep the legacy
                    // traversal, so a face can only be recovered on a decided
                    // equation and never on a coin toss.
                    (true, true) => {
                        join_outcome = TwoLoopJoinOutcome::Unresolved;
                        false
                    }
                    (false, false) => {
                        join_outcome = TwoLoopJoinOutcome::Inconsistent;
                        false
                    }
                };
                let mut loop1_path = loop1.into_path_cutting_wrap();
                if !take_forward {
                    loop1_path.reverse();
                }
                let mut path = loop0.into_path_cutting_wrap();
                if diag {
                    record_two_loop_join(
                        loop0_displacement,
                        loop1_displacement,
                        mean_translate,
                        !take_forward,
                        &path,
                        &loop1_path,
                    );
                }
                // The two loops are disconnected: joining them creates a
                // segment neither supplied, and so does closing back to the
                // start. Both are declared, so no source point is dropped to
                // manufacture a shortcut.
                path.append_path(loop1_path, PartJoin::Bridge(SegmentOrigin::Seam));
                closed.push(path.close(PartJoin::Bridge(SegmentOrigin::Seam)));
            }
        } else if let Some(pair) = CollapsedPeriodicBoundaryPair::try_classify(
            surface,
            &closed.iter().map(|l| l.points.clone()).collect::<Vec<_>>(),
            &open,
            range,
            lattice,
        ) {
            let mut loop0 = closed.remove(0).points;
            if loop0.len() > 1 && loop0[0].uv.distance(loop0.last().unwrap().uv) < 1e-3 {
                loop0.pop();
            }
            let is_v = lattice.declared_v_period().is_some_and(|p| p > 1e-6);
            let period = if is_v {
                lattice.declared_v_period().unwrap()
            } else {
                lattice.declared_u_period().unwrap()
            };

            let mut loop0_full = loop0.clone();
            let mut last_p = *loop0.last().unwrap();
            if is_v {
                last_p.uv.y = loop0[0].uv.y + period;
            } else {
                last_p.uv.x = loop0[0].uv.x + period;
            }
            loop0_full.push(last_p);

            let loop1_full: Vec<SurfacePoint> = loop0_full
                .iter()
                .map(|p| {
                    let uv = if is_v {
                        Point2::new(pair.apex_u, p.uv.y)
                    } else {
                        Point2::new(p.uv.x, pair.apex_u)
                    };
                    (uv, surface.subs(uv.x, uv.y)).into()
                })
                .collect();

            let end_loop0 = *loop0_full.last().unwrap();
            let loop1_rev: Vec<SurfacePoint> = loop1_full.into_iter().rev().collect();
            let start_loop1 = loop1_rev[0];
            let end_loop1 = *loop1_rev.last().unwrap();
            let start_loop0 = loop0_full[0];

            let seam_down = polyline_on_surface(surface, end_loop0, start_loop1, tol);
            let seam_up = polyline_on_surface(surface, end_loop1, start_loop0, tol);

            // Only `loop0_full` carries source evidence, and even it ends with
            // an appended period-wrap point rather than a source sample. The
            // apex branch `loop1_rev` is *evaluated from the surface* at
            // `pair.apex_u` — synthesised geometry that no source edge
            // describes — and the two joining runs are seams across the
            // collapsed direction.
            closed.push(BoundaryLoop::chained([
                (loop0_full, SegmentOrigin::Source),
                (seam_down, SegmentOrigin::Seam),
                (loop1_rev, SegmentOrigin::Seam),
                (seam_up, SegmentOrigin::Seam),
            ]));
        }
        let (n_closed_in, n_open_in) = (closed.len(), open.len());
        // `connect_edges` used to live here. It dropped each part's last point
        // unconditionally, which is correct only when parts chain — the
        // assumption `BoundaryPath::append` now makes the caller state, so the
        // helper has no remaining callers.
        match open.len() {
            1 => {
                let mut curve = open.pop().unwrap();
                let p = curve[0];
                let q = curve[curve.len() - 1];
                if let (Some((u0, u1)), Some((v0, v1))) = range {
                    if p.x < q.x - TOLERANCE {
                        normalize_range(&mut curve, 0, (u0, u1));
                        let p = curve[0];
                        let q = curve[curve.len() - 1];
                        let x = (Point2::new(u0, v1), surface.subs(u0, v1)).into();
                        let y = (Point2::new(u1, v1), surface.subs(u1, v1)).into();
                        let vec0 = polyline_on_surface(surface, q, y, tol);
                        let vec1 = polyline_on_surface(surface, y, x, tol);
                        let vec2 = polyline_on_surface(surface, x, p, tol);
                        closed.push(BoundaryLoop::chained([
                            (vec0, SegmentOrigin::SyntheticClosure),
                            (vec1, SegmentOrigin::SyntheticClosure),
                            (vec2, SegmentOrigin::SyntheticClosure),
                            (curve, SegmentOrigin::Source),
                        ]));
                    } else if q.x < p.x - TOLERANCE {
                        normalize_range(&mut curve, 0, (u0, u1));
                        let p = curve[0];
                        let q = curve[curve.len() - 1];
                        let x = (Point2::new(u1, v0), surface.subs(u1, v0)).into();
                        let y = (Point2::new(u0, v0), surface.subs(u0, v0)).into();
                        let vec0 = polyline_on_surface(surface, q, y, tol);
                        let vec1 = polyline_on_surface(surface, y, x, tol);
                        let vec2 = polyline_on_surface(surface, x, p, tol);
                        closed.push(BoundaryLoop::chained([
                            (vec0, SegmentOrigin::SyntheticClosure),
                            (vec1, SegmentOrigin::SyntheticClosure),
                            (vec2, SegmentOrigin::SyntheticClosure),
                            (curve, SegmentOrigin::Source),
                        ]));
                    } else if p.y < q.y - TOLERANCE {
                        normalize_range(&mut curve, 1, (v0, v1));
                        let p = curve[0];
                        let q = curve[curve.len() - 1];
                        let x = (Point2::new(u0, v0), surface.subs(u0, v0)).into();
                        let y = (Point2::new(u0, v1), surface.subs(u0, v1)).into();
                        let vec0 = polyline_on_surface(surface, q, y, tol);
                        let vec1 = polyline_on_surface(surface, y, x, tol);
                        let vec2 = polyline_on_surface(surface, x, p, tol);
                        closed.push(BoundaryLoop::chained([
                            (vec0, SegmentOrigin::SyntheticClosure),
                            (vec1, SegmentOrigin::SyntheticClosure),
                            (vec2, SegmentOrigin::SyntheticClosure),
                            (curve, SegmentOrigin::Source),
                        ]));
                    } else if q.y < p.y - TOLERANCE {
                        normalize_range(&mut curve, 1, (v0, v1));
                        let p = curve[0];
                        let q = curve[curve.len() - 1];
                        let x = (Point2::new(u1, v1), surface.subs(u1, v1)).into();
                        let y = (Point2::new(u1, v0), surface.subs(u1, v0)).into();
                        let vec0 = polyline_on_surface(surface, q, y, tol);
                        let vec1 = polyline_on_surface(surface, y, x, tol);
                        let vec2 = polyline_on_surface(surface, x, p, tol);
                        closed.push(BoundaryLoop::chained([
                            (vec0, SegmentOrigin::SyntheticClosure),
                            (vec1, SegmentOrigin::SyntheticClosure),
                            (vec2, SegmentOrigin::SyntheticClosure),
                            (curve, SegmentOrigin::Source),
                        ]));
                    }
                }
            }
            2 => {
                let mut curve1 = open.pop().unwrap();
                let mut curve0 = open.pop().unwrap();
                fn end_pts<T: Copy>(vec: &[T]) -> (T, T) {
                    (vec[0], vec[vec.len() - 1])
                }
                let ((p0, p1), (q0, q1)) = (end_pts(&curve0), end_pts(&curve1));
                if !p0.x.near(&p1.x) && !q0.x.near(&q1.x) {
                    if let (Some(urange), _) = range {
                        normalize_range(&mut curve0, 0, urange);
                        normalize_range(&mut curve1, 0, urange);
                    }
                } else if !p0.y.near(&p1.y) && !q0.y.near(&q1.y) {
                    if let (_, Some(vrange)) = range {
                        normalize_range(&mut curve0, 1, vrange);
                        normalize_range(&mut curve1, 1, vrange);
                    }
                }
                let ((p0, p1), (q0, q1)) = (end_pts(&curve0), end_pts(&curve1));
                let vec0 = polyline_on_surface(surface, p1, q0, tol);
                let vec1 = polyline_on_surface(surface, q1, p0, tol);
                closed.push(BoundaryLoop::chained([
                    (curve0, SegmentOrigin::Source),
                    (vec0, SegmentOrigin::SyntheticClosure),
                    (curve1, SegmentOrigin::Source),
                    (vec1, SegmentOrigin::SyntheticClosure),
                ]));
            }
            _ => {}
        }
        if probe {
            let areas: Vec<String> = closed
                .iter()
                .map(|c| format!("{:+.4e}", signed_area(&c.points)))
                .collect();
            let range = surface.try_range_tuple();
            let has_rect = matches!(range, (Some(_), Some(_)));
            eprintln!(
                "PROBE in_closed={n_closed_in} in_open={n_open_in} loops={} \
                 areas=[{}] uperiod={:?} vperiod={:?} range={} rect={}",
                closed.len(),
                areas.join(","),
                surface.u_period(),
                surface.v_period(),
                has_rect,
                closed.is_empty() && has_rect,
            );
        }
        // Only a face with no enclosing loop takes its domain from the surface.
        if closed.is_empty() && !had_source_pieces {
            if let (Some((u0, u1)), Some((v0, v1))) = range {
                let p = [
                    (Point2::new(u0, v0), surface.subs(u0, v0)).into(),
                    (Point2::new(u1, v0), surface.subs(u1, v0)).into(),
                    (Point2::new(u1, v1), surface.subs(u1, v1)).into(),
                    (Point2::new(u0, v1), surface.subs(u0, v1)).into(),
                ];
                let vec0 = polyline_on_surface(surface, p[0], p[1], tol);
                let vec1 = polyline_on_surface(surface, p[1], p[2], tol);
                let vec2 = polyline_on_surface(surface, p[2], p[3], tol);
                let vec3 = polyline_on_surface(surface, p[3], p[0], tol);
                closed.push(BoundaryLoop::chained([
                    (vec0, SegmentOrigin::SyntheticClosure),
                    (vec1, SegmentOrigin::SyntheticClosure),
                    (vec2, SegmentOrigin::SyntheticClosure),
                    (vec3, SegmentOrigin::SyntheticClosure),
                ]));
            }
        }
        (Self(closed), join_outcome)
    }

    /// Where `c` lies relative to the domain bounded by `self`.
    ///
    /// **G7a.** Previously this returned a `bool`, and a ray cast that aborted
    /// was reported as `false` — *outside* — which is an answer the computation
    /// did not have.
    ///
    /// The two failure modes are separated here rather than inferred from each
    /// other. `Boundary` is decided by a direct point-on-segment predicate, so
    /// it is a positive result about `c`. `Inside` and `Outside` come from ray
    /// casting. `Indeterminate` means every tried ray aborted *and* the direct
    /// predicate did not fire — the location is simply not established, and the
    /// type says so instead of naming a side.
    ///
    /// An earlier revision claimed the residue after eight rays *was* boundary
    /// membership. That was an inference from a negative result: an aborted
    /// cast in floating point can equally be near-boundary numerical
    /// degeneracy or an unlucky family of seeds, and "no ray decided" licenses
    /// neither conclusion. Measuring it directly happens to confirm the guess —
    /// on ABC `00009190`, 117,145 samples test as `Boundary` and **zero** come
    /// back `Indeterminate`, with triangle and failure counts unchanged — but
    /// it is now a positive result that would report `Indeterminate` the moment
    /// that stopped being true, rather than a conclusion drawn from silence.
    fn locate(&self, c: Point2) -> PointLocation {
        // A positive test first, so boundary membership is established rather
        // than inferred from rays failing to decide.
        if self.on_boundary(c) {
            return PointLocation::Boundary;
        }
        // Deterministic, so a face tessellates identically across runs.
        match (0..INCLUSION_RAY_ATTEMPTS).find_map(|attempt| self.include_along_ray(c, attempt)) {
            Some(true) => PointLocation::Inside,
            Some(false) => PointLocation::Outside,
            None => PointLocation::Indeterminate,
        }
    }

    /// Whether `c` lies on a boundary segment, within `TOLERANCE`.
    ///
    /// Direct and ray-independent: the nearest point of each segment is
    /// computed and compared to `c`. This is what entitles [`Self::locate`] to
    /// report `Boundary` as a fact rather than as "the rays gave up".
    fn on_boundary(&self, c: Point2) -> bool {
        self.0
            .iter()
            .flat_map(|loop_| loop_.points.iter().circular_tuple_windows())
            .any(|(p0, p1)| {
                let (a, b) = (**p0, **p1);
                let ab = b - a;
                let len2 = ab.magnitude2();
                let t = match len2 <= f64::EPSILON {
                    true => 0.0,
                    false => ((c - a).dot(ab) / len2).clamp(0.0, 1.0),
                };
                (a + ab * t).distance2(c) <= TOLERANCE * TOLERANCE
            })
    }

    /// One ray cast. `None` when this ray is degenerate against the boundary.
    fn include_along_ray(&self, c: Point2, attempt: u32) -> Option<bool> {
        // Offsetting the seed per attempt keeps successive rays unrelated
        // rather than merely rotated by a fixed step, which a boundary with
        // regularly spaced vertices could otherwise defeat repeatedly.
        let seed = HashGen::hash1(c) + f64::from(attempt) * std::f64::consts::FRAC_1_PI;
        let t = 2.0 * std::f64::consts::PI * seed.fract();
        let r = Vector2::new(f64::cos(t), f64::sin(t));
        self.0
            .iter()
            .flat_map(|loop_| loop_.points.iter().circular_tuple_windows())
            .try_fold(0_i32, move |crossings, (p0, p1)| {
                let a = **p0 - c;
                let b = **p1 - c;
                let s0 = r.x * a.y - r.y * a.x; // v times a
                let s1 = r.x * b.y - r.y * b.x; // v times b
                let s2 = a.x * b.y - a.y * b.x; // a times b
                let x = s2 / (s1 - s0);
                if x.so_small() && s0 * s1 < 0.0 {
                    None
                } else if x > 0.0 && ((s0 <= 0.0 && s1 > 0.0) || (s0 >= 0.0 && s1 < 0.0)) {
                    Some(crossings + 1)
                } else {
                    Some(crossings)
                }
            })
            // `None` propagates: a degenerate ray decides nothing, and the
            // caller retries with another direction rather than reading the
            // abort as "outside".
            .map(|crossings| crossings % 2 == 1)
    }

    /// Inserts points and adds constraint
    fn insert_to(
        &self,
        triangulation: &mut Cdt,
        boundary_map: &mut HashMap<FixedVertexHandle, Point3>,
        roles: &mut ConstraintRoles,
    ) -> std::result::Result<(), TessellationFailureReason> {
        // The first refusal, kept typed. The loop continues after recording it
        // so the probe counters below still see the whole face.
        let mut failure: Option<TessellationFailureReason> = None;
        let probe = std::env::var_os("TRUCK_PROBE_FAIL").is_some();
        let probe_face_id = if probe {
            std::env::var("TRUCK_PROBE_FACE_ID")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
        } else {
            None
        };
        if let Some(target) = probe_face_id {
            let (source_face_id, declared_face_index, periodic_rank) =
                PROBE_FACE_CONTEXT.with(std::cell::Cell::get);
            if source_face_id == Some(target) {
                for (piece_index, piece) in self.0.iter().enumerate() {
                    for (point_index, point) in piece.points.iter().enumerate() {
                        eprintln!(
                            "WL\tsource_face_id={target}\tdeclared_face_index={declared_face_index}\t\
                             periodic_rank={periodic_rank}\tpost_stitch_piece={piece_index}\t\
                             point_index={point_index}\tuv={:.17e},{:.17e}",
                            point.uv.x, point.uv.y,
                        );
                    }
                }
            }
        }
        let mut probe_point_fail = 0usize;
        let mut probe_degenerate = 0usize;
        let mut probe_already_direct = 0usize;
        let mut probe_refused_with_conflicts = 0usize;
        let mut probe_refused_without_conflicts = 0usize;
        let mut probe_conflicting_edges = 0usize;
        let mut probe_add_returned_false = 0usize;
        // These are post-stitch loop/segment proxies, not source-edge
        // provenance. Keep all direct contributors because duplicates exist.
        let mut installed_origins =
            probe.then(HashMap::<FixedUndirectedEdgeHandle, Vec<(usize, usize)>>::default);
        let mut first_conflict = None;
        // DIAG-001: diagnostic capture. Gated on TRUCK_FACE_DIAG_JSONL. When
        // disabled, none of this code has any effect — the edge map is never
        // populated and the sink is never written. This instrumentation must
        // not alter insertion order or insertion behaviour.
        let diag = diagnosis::diag_enabled();
        let mut diag_edge_map: HashMap<FixedUndirectedEdgeHandle, u64> = HashMap::default();
        for (piece_index, piece) in self.0.iter().enumerate() {
            let poly2tri: Vec<Option<FixedVertexHandle>> = piece
                .points
                .iter()
                .map(|pt| {
                    let sp = SPoint2::new(spade_round(pt.uv.x), spade_round(pt.uv.y));
                    if let Some(idx) = triangulation
                        .vertices()
                        .find(|v| sp.distance_2(*v.as_ref()) < 1e-12)
                        .map(|v| v.fix())
                    {
                        boundary_map.insert(idx, pt.point);
                        Some(idx)
                    } else {
                        match triangulation.insert(sp) {
                            Ok(idx) => {
                                boundary_map.insert(idx, pt.point);
                                Some(idx)
                            }
                            Err(_) => None,
                        }
                    }
                })
                .collect();

            if poly2tri.iter().any(|v| v.is_none()) {
                probe_point_fail += 1;
                if diag {
                    diagnosis::set_vertex_insertion_failed();
                }
                failure.get_or_insert(TessellationFailureReason::ConstraintInsertionIncomplete);
                continue;
            }
            let len = poly2tri.len();
            if len < 3 {
                continue;
            }
            for k in 0..len {
                let i = k;
                let j = (k + 1) % len;
                let vi = poly2tri[i].unwrap();
                let vj = poly2tri[j].unwrap();
                if vi == vj {
                    probe_degenerate += 1;
                    continue;
                }
                // ARR-003: has *this face* already constrained this exact edge?
                //
                // A well-formed loop traverses each edge once. If the direct
                // edge is already a constraint that this face's own role table
                // claims, the boundary is traversing it a second time — a
                // duplicate or collinear-overlapping segment, which the
                // envelope does not admit.
                //
                // The previous code rejected this case, but only as a side
                // effect of treating "already a constraint" as a failure, which
                // also refused segments that were legitimately already fully
                // represented — 5 faces on `00009190`. Separating the two keeps
                // the refusal and drops the false positive, and gives the
                // overlap its own typed reason instead of reporting it as an
                // insertion failure.
                // G6: the role this segment is entitled to, decided by where
                // the segment came from rather than by which vector it ended up
                // in. `PolyBoundary::new` stitches synthesised closure and seam
                // segments into the same pieces as source trim, so before the
                // origin was recorded at creation every one of them arrived
                // here indistinguishable from a real boundary.
                let segment_origin = piece.origins.get(i).unwrap_or(SegmentOrigin::Source);
                let segment_role = segment_origin.role();
                let diag_seg_id = if diag {
                    diagnosis::record_segment(segment_origin, Some(piece_index), k as u32)
                } else {
                    0
                };
                let overlapping = triangulation
                    .get_edge_from_neighbors(vi, vj)
                    .filter(|e| e.is_constraint_edge())
                    .map(|e| e.as_undirected().fix())
                    .is_some_and(|handle| roles.role_of(handle).is_some());
                if overlapping {
                    failure.get_or_insert(TessellationFailureReason::ConstraintOverlapUnsupported);
                    continue;
                }
                // G5a: ask once, and label what was actually realized.
                //
                // The previous sequence was `get_edge_from_neighbors` +
                // `can_add_constraint` + `add_constraint` +
                // `get_edge_from_neighbors`: three traversals to decide, then a
                // fourth to rediscover the outcome. That last lookup is where
                // the role table lost its entries, because Spade may realize
                // one requested segment as a *chain* when an existing vertex
                // lies on it — its own documentation says
                // `exists_constraint(from, to)` is then not true — and the
                // direct edge `(vi, vj)` simply does not exist to be found.
                //
                // `try_add_constraint` returns the realized chain instead, so
                // every edge of it can be labelled. Its contract is exactly the
                // one this site needs:
                //
                //   - empty      => refused; the segment properly crosses an
                //                   existing constraint, and the triangulation
                //                   is left unchanged (it is atomic on
                //                   conflict, so refusal cannot half-apply);
                //   - non-empty  => realized, including any edge that was
                //                   already present.
                //
                // "Already fully represented" therefore stops being a separate
                // case that had to be inferred from a Boolean.
                let chain = triangulation.try_add_constraint(vi, vj);
                if !chain.is_empty() {
                    for directed in &chain {
                        let handle = triangulation.directed_edge(*directed).as_undirected().fix();
                        // Audit A1: every segment reaching here comes from a
                        // `PolyBoundary` piece, so it is treated as physical
                        // boundary. That is deliberately *not* the whole
                        // truth — `PolyBoundary::new` also stitches synthetic
                        // closure segments into these same pieces, and those
                        // are `UnresolvedSyntheticClosure` in reality (audit
                        // A6). Distinguishing them needs per-piece
                        // provenance, which would be a second semantic change
                        // in the same experiment. Tagged as it behaves today;
                        // A6 splits the population.
                        roles.record(handle, segment_role);
                        *roles.origin_census.entry(segment_origin).or_insert(0) += 1;
                        if let Some(installed_origins) = installed_origins.as_mut() {
                            installed_origins
                                .entry(handle)
                                .or_default()
                                .push((piece_index, k));
                        }
                        if diag {
                            diag_edge_map.entry(handle).or_insert(diag_seg_id);
                            diagnosis::record_realized_edge(segment_role, diag_seg_id);
                        }
                    }
                } else {
                    if probe {
                        let conflicts: Vec<_> = triangulation
                            .get_conflicting_edges_between_vertices(vi, vj)
                            .map(|edge| {
                                (
                                    edge.as_undirected().fix(),
                                    edge.from().position(),
                                    edge.to().position(),
                                )
                            })
                            .collect();
                        probe_conflicting_edges += conflicts.len();
                        if conflicts.is_empty() {
                            probe_refused_without_conflicts += 1;
                        } else {
                            probe_refused_with_conflicts += 1;
                            if first_conflict.is_none() {
                                let mapped_conflicts = conflicts
                                    .iter()
                                    .filter(|(handle, _, _)| {
                                        installed_origins
                                            .as_ref()
                                            .is_some_and(|origins| origins.contains_key(handle))
                                    })
                                    .count();
                                // Prefer a resolvable direct origin, but make
                                // the selection bias explicit in the record.
                                let selected_conflict_index = conflicts
                                    .iter()
                                    .position(|(handle, _, _)| {
                                        installed_origins
                                            .as_ref()
                                            .is_some_and(|origins| origins.contains_key(handle))
                                    })
                                    .unwrap_or(0);
                                let selected = &conflicts[selected_conflict_index];
                                let existing_origins = installed_origins
                                    .as_ref()
                                    .and_then(|origins| origins.get(&selected.0))
                                    .cloned()
                                    .unwrap_or_default();
                                first_conflict = Some((
                                    piece_index,
                                    k,
                                    piece.points[i].uv,
                                    piece.points[j].uv,
                                    selected_conflict_index,
                                    conflicts.len(),
                                    mapped_conflicts,
                                    existing_origins,
                                    selected.1,
                                    selected.2,
                                ));
                            }
                        }
                    }
                    if diag {
                        let diag_conflicts: Vec<_> = triangulation
                            .get_conflicting_edges_between_vertices(vi, vj)
                            .map(|edge| edge.as_undirected().fix())
                            .collect();
                        for handle in &diag_conflicts {
                            if let Some(&blocking_id) = diag_edge_map.get(handle) {
                                diagnosis::record_conflict(
                                    diag_seg_id,
                                    blocking_id,
                                    diagnosis::PresentedSegmentRelation::ProperInteriorCrossing,
                                );
                            }
                        }
                    }
                    failure.get_or_insert(TessellationFailureReason::ConstraintInsertionIncomplete);
                }
            }
        }
        if probe {
            if let Some((
                proposed_piece,
                proposed_segment,
                proposed_a,
                proposed_b,
                selected_conflict_index,
                selected_refusal_conflicts,
                mapped_conflicts,
                existing_origins,
                existing_a,
                existing_b,
            )) = first_conflict
            {
                let (source_face_id, declared_face_index, periodic_rank) =
                    PROBE_FACE_CONTEXT.with(std::cell::Cell::get);
                let existing_first_origin = existing_origins.first().copied();
                let selection = if mapped_conflicts == 0 {
                    "first_reported"
                } else {
                    "first_mappable"
                };
                let origin_resolution = if existing_first_origin.is_some() {
                    "direct"
                } else {
                    "missing"
                };
                eprintln!(
                    "CW\tsource_face_id={source_face_id:?}\tdeclared_face_index={declared_face_index}\t\
                     periodic_rank={periodic_rank}\tselection={selection}\t\
                     selected_conflict_index={selected_conflict_index}\t\
                     selected_refusal_conflicts={selected_refusal_conflicts}\t\
                     mapped_conflicts={mapped_conflicts}\t\
                     proposed_post_stitch_piece={proposed_piece}\t\
                     proposed_post_stitch_segment={proposed_segment}\t\
                     existing_origin_resolution={origin_resolution}\t\
                     existing_direct_contributors={}\t\
                     existing_first_post_stitch_origin={existing_first_origin:?}\t\
                     proposed_uv={:.17e},{:.17e};{:.17e},{:.17e}\t\
                     existing_uv={:.17e},{:.17e};{:.17e},{:.17e}\t\
                     face_conflicting_edges={probe_conflicting_edges}",
                    existing_origins.len(), proposed_a.x, proposed_a.y, proposed_b.x,
                    proposed_b.y, existing_a.x, existing_a.y, existing_b.x, existing_b.y,
                );
            }
        }
        if probe && (failure.is_some() || probe_degenerate != 0 || probe_add_returned_false != 0) {
            eprintln!(
                "PF,{},{probe_point_fail},{probe_degenerate},\
                 {probe_already_direct},{probe_refused_with_conflicts},\
                 {probe_refused_without_conflicts},{probe_conflicting_edges},\
                 {probe_add_returned_false}",
                u8::from(failure.is_none()),
            );
        }
        match failure {
            Some(reason) => Err(reason),
            None => Ok(()),
        }
    }
}

fn spade_round(x: f64) -> f64 {
    match f64::abs(x) < MIN_ALLOWED_VALUE {
        true => 0.0,
        false => x,
    }
}

/// What a Spade constraint edge *means*, so that material semantics can be
/// derived from it.
///
/// **Audit A1.** The dual-parity flood in [`triangulation_into_polymesh_outcome`]
/// asks `edge.is_constraint_edge()` and toggles material parity on every `true`.
/// That is one bit where the formal system requires a role: `insert_surface`
/// adds constraints across the *interior sampling grid*, wholly inside the
/// material region, and those toggle parity exactly as a trim segment does.
///
/// FORMAL_SYSTEM.md §IX distinguishes `Physical`, `ArtificialCut`,
/// `NativeBoundary` and `SingularLink`; Definition 20 gives them different
/// material constraints — a physical half-edge pins `μ_L = 1, μ_R = 0` while an
/// artificial cut requires `μ_L = μ_R`. This enum is that distinction, carried
/// far enough to make the A1 experiment causal.
///
/// It is deliberately *not* the general cell-constraint solver. Parity is still
/// the mechanism; the only thing that changes is which edges are entitled to
/// flip it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub enum ConstraintRole {
    /// A trim segment carrying source boundary evidence. Toggles material side.
    PhysicalBoundary,
    /// A chart cut or seam introduced to make the domain simply connected.
    /// Both sides are the same material state, so it must not toggle.
    ArtificialCut,
    /// The ambient parameter domain's own edge. Does not itself toggle;
    /// its interpretation is schema-dependent (FORMAL_SYSTEM Definition 20).
    NativeBoundary,
    /// An interior grid constraint inserted by [`insert_surface`] to control
    /// triangle shape. Carries no material meaning whatsoever.
    SurfaceSampling,
    /// A closure segment synthesised by [`PolyBoundary::new`] that no source
    /// entity justifies. Its role is genuinely unknown; it must not be silently
    /// assigned equality semantics (audit A6).
    UnresolvedSyntheticClosure,
}

/// Roles for the constraint edges of one face's triangulation.
///
/// A side table rather than a Spade edge-data parameter: `add_constraint` does
/// not hand back the edges it marked, so a role can only be attached by looking
/// the edge up afterwards. When that lookup fails — Spade may mark a collinear
/// chain rather than the direct edge — the role is genuinely unresolved, and
/// [`Self::role_of`] says so rather than guessing.
#[derive(Debug, Default)]
struct ConstraintRoles {
    roles: HashMap<FixedUndirectedEdgeHandle, ConstraintRole>,
    /// Constraint edges the flood met that no `record` call had claimed.
    /// Counted, not assumed: this is the size of the gap between what we asked
    /// Spade to constrain and what we can name (CDT-001, CDT-002).
    unresolved_at_flood: std::cell::Cell<usize>,
    /// How many edges each role claimed, for the experiment's own report.
    recorded: HashMap<ConstraintRole, usize>,
    /// How many edges each *origin* claimed. Distinct from `recorded`: several
    /// origins deliberately share one role while the material semantics of
    /// synthesised geometry stay unchanged, so without this the synthetic
    /// populations are indistinguishable in the census.
    origin_census: HashMap<SegmentOrigin, usize>,
}

impl ConstraintRoles {
    fn record(&mut self, edge: FixedUndirectedEdgeHandle, role: ConstraintRole) {
        // First claim wins. A later, weaker claim must not overwrite a physical
        // boundary: an interior grid segment that happens to land on an edge a
        // trim segment already constrained is still a trim segment.
        if !self.roles.contains_key(&edge) {
            self.roles.insert(edge, role);
            *self.recorded.entry(role).or_insert(0) += 1;
        }
    }

    fn role_of(&self, edge: FixedUndirectedEdgeHandle) -> Option<ConstraintRole> {
        self.roles.get(&edge).copied()
    }

    /// Whether a constraint edge is entitled to flip material parity.
    ///
    /// `None` means the edge carries **no resolvable role**, which is not a
    /// material category and must not be answered with one.
    ///
    /// **G5b: fail closed.** This previously returned `true` for an unresolved
    /// edge — an unjustified material assertion. Answering `false` instead
    /// would have been the same mistake facing the other way: both invent a
    /// semantics for an edge the code cannot name. Since G5a labels the whole
    /// realized chain, every constraint edge this face requested has a role, and
    /// an unresolved one is an invariant violation rather than a legitimate
    /// category — so it is reported, not guessed.
    ///
    /// Measured after G5a: zero occurrences on ABC `00009190`, so this guard
    /// lands provably non-firing.
    fn toggles_material(&self, edge: FixedUndirectedEdgeHandle) -> Option<bool> {
        match self.role_of(edge) {
            Some(ConstraintRole::SurfaceSampling) => Some(false),
            Some(ConstraintRole::ArtificialCut) => Some(false),
            Some(ConstraintRole::PhysicalBoundary) => Some(true),
            // FORMAL_SYSTEM Definition 20 says a native ambient boundary does
            // not itself toggle; its interpretation comes from incident
            // physical constraints. Neither this nor the synthetic-closure case
            // is ever constructed today, so both keep their legacy answer and
            // are decided when G6 first builds them.
            Some(ConstraintRole::NativeBoundary) => Some(true),
            Some(ConstraintRole::UnresolvedSyntheticClosure) => Some(true),
            None => {
                self.unresolved_at_flood
                    .set(self.unresolved_at_flood.get() + 1);
                None
            }
        }
    }
}

/// Why one face could not be tessellated.
///
/// Seven of these variants are declared but never constructed, and are marked
/// as such below. They are retained rather than deleted because each names a
/// stage the formal system requires and this implementation does not yet have —
/// deleting them would erase the fact that the case is unhandled, which is the
/// opposite of what a typed outcome is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub enum TessellationFailureReason {
    /// No lifted boundary could be built, for a reason the lift does not name.
    ///
    /// Retained as the residual bucket now that the lift reports its own
    /// causes. A face arriving here means a refusal path was added without a
    /// reason to go with it.
    BoundaryConstructionFailed,
    /// A bound contributed no points at all, so it cannot bound anything.
    BoundaryWireEmpty,
    /// A boundary sample had no parameter on the face's own surface.
    BoundaryProjectionFailed,
    /// A boundary sample lay further from its own surface than the
    /// compatibility policy permits (GEO-005). Only reachable when
    /// `TRUCK_COMPAT_FACTOR` is set; the gate is off by default.
    BoundaryPointOffSurface,
    /// The periodic branch of a lift step could not be resolved.
    ///
    /// `get_mindiff` picks the period copy nearest the previous sample, which
    /// is correct only while the true step is under half a period. Beyond
    /// `AMBIGUOUS_STEP_FRACTION` the two candidates are not distinguishable
    /// that way, and the step is bisected to shorten it. When bisection is
    /// exhausted the branch is genuinely unresolved: the code previously
    /// accepted the ambiguous value silently, which is how a period-wrapping
    /// boundary could fold onto itself and read as closed (FS Def. 14).
    AmbiguousLift,
    /// The parity flood assigned a cell two different labels around a cycle.
    ///
    /// This is a *proved* inconsistency, not a heuristic giving up: the
    /// boundary as realized cannot carry a coherent material assignment.
    ContradictoryDualParity,
    /// The flood completed but selected no material cell, so there is nothing
    /// to mesh.
    NoOddParityRegion,
    /// A constraint chain did not close. **Never constructed.**
    ConstraintChainNotClosed,
    /// At least one boundary segment could not be represented as a constraint.
    ///
    /// Almost always a proper crossing of an earlier segment of this same
    /// face's boundary — which is what a folded lift produces.
    ConstraintInsertionIncomplete,
    /// A certified intersection was found that the envelope does not admit.
    /// **Never constructed** — no intersection classification stage exists yet.
    ConstraintIntersectionUnsupported,
    /// A collinear overlap was found that the envelope does not admit.
    /// **Never constructed** — no overlap normalization stage exists yet.
    ConstraintOverlapUnsupported,
    /// The triangulation could not be built at all. **Never constructed.**
    CdtConstructionFailed,
    /// A vertex evaluated to a non-finite 3D position.
    NonFinitePosition,
    /// A constraint edge carried no resolvable role. **Never constructed** —
    /// today an unresolved role silently keeps its legacy toggling behaviour.
    ConstraintRoleMissing,
    /// A constraint chain degenerated to a point. **Never constructed.**
    DegenerateConstraintChain,
    /// Parity selected cells but none yielded a finite triangle.
    /// **Never constructed** — the empty case reports [`Self::NoOddParityRegion`].
    NoFiniteTrianglesAfterParity,
}

/// Why one face failed to tessellate, and where.
///
/// The locating fields are best-effort and mostly unpopulated today; the
/// `reason` is the load-bearing part.
#[derive(Clone, Debug)]
pub struct TessellationFailure {
    /// What went wrong.
    pub reason: TessellationFailureReason,
    /// Which of the face's bounds, when known.
    pub source_bound: Option<usize>,
    /// Which edge use within that bound, when known.
    pub source_edge_use: Option<usize>,
    /// Constraint identifiers implicated, when known.
    pub constraint_ids: Vec<usize>,
    /// Where in parameter space, when known.
    pub uv_location: Option<Point2>,
}

impl From<TessellationFailureReason> for TessellationFailure {
    fn from(reason: TessellationFailureReason) -> Self {
        Self {
            reason,
            source_bound: None,
            source_edge_use: None,
            constraint_ids: Vec::new(),
            uv_location: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VertexGeneration {
    SurfaceEvaluation,
    SourceEdgeSample,
    ConstraintIntersection,
    SingularRealization,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct VertexRoles(pub u16);

impl VertexRoles {
    pub const PHYSICAL_BOUNDARY: u16 = 1 << 0;
    pub const ARTIFICIAL_SEAM: u16 = 1 << 1;
    pub const SINGULAR_COLLAPSE: u16 = 1 << 2;
    pub const CONSTRAINT_INTERSECT: u16 = 1 << 3;

    pub fn contains(&self, role: u16) -> bool {
        (self.0 & role) == role
    }
    pub fn insert(&mut self, role: u16) {
        self.0 |= role;
    }
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }
}

#[derive(Clone, Debug)]
pub struct SeamPair {
    pub first_chain: Vec<usize>,
    pub second_chain: Vec<usize>,
    pub correspondence: Vec<(usize, usize)>,
    pub orientation_reversed: bool,
    pub deck_displacement: [i64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SingularKind {
    Apex,
    Pole,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SingularLinkKind {
    InteriorCycle,
    BoundaryInterval,
}

#[derive(Clone, Debug)]
pub struct SingularGroup {
    pub vertices: Vec<usize>,
    pub canonical_point: Point3,
    pub kind: SingularKind,
    pub link_kind: SingularLinkKind,
}

#[derive(Clone, Debug)]
pub struct VertexMetadata {
    pub uv: Point2,
    pub generation: VertexGeneration,
    pub roles: VertexRoles,
    pub source_edge_use: Option<usize>,
    pub seam_pair: Option<usize>,
    pub singular_group: Option<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct TessellationDiagnostics {
    pub vertex_metadata: Vec<VertexMetadata>,
    pub seam_pairs: Vec<SeamPair>,
    pub singular_groups: Vec<SingularGroup>,
}

#[derive(Clone, Debug)]
pub struct FaceTessellation {
    pub mesh: PolygonMesh,
    pub diagnostics: TessellationDiagnostics,
}

#[derive(Clone, Debug)]
pub enum TessellationOutcome {
    Mesh(FaceTessellation),
    Failed(TessellationFailure),
}

/// Tessellates one surface trimmed by polyline, returning `TessellationOutcome`.
fn trimming_tessellation_with_diagnostics<S>(
    surface: &S,
    polyboundary: &PolyBoundary,
    tol: f64,
    lattice: &CertifiedLattice,
) -> TessellationOutcome
where
    S: PreMeshableSurface,
{
    let mut triangulation = Cdt::new();
    let mut boundary_map = HashMap::<FixedVertexHandle, Point3>::default();
    let mut roles = ConstraintRoles::default();
    if let Err(reason) = polyboundary.insert_to(&mut triangulation, &mut boundary_map, &mut roles) {
        return TessellationOutcome::Failed(reason.into());
    }
    let (samples_on_boundary, sampling_location_unresolved) =
        insert_surface(&mut triangulation, surface, polyboundary, tol, &mut roles);
    let outcome = triangulation_into_polymesh_outcome(
        &triangulation,
        surface,
        polyboundary,
        &boundary_map,
        &roles,
        lattice,
    );
    if std::env::var_os("TRUCK_PROBE_ROLES").is_some() {
        // The population sizes the A1 comparison rests on. `unresolved` is the
        // honest gap: constraint edges the flood met that no `record` call had
        // claimed, which keep their legacy toggling behaviour. A large number
        // here would mean the experiment is less causal than it looks.
        let count = |role| roles.recorded.get(&role).copied().unwrap_or(0);
        let by_origin = |o| roles.origin_census.get(&o).copied().unwrap_or(0);
        eprintln!(
            "ROLES\tphysical={}\tsampling={}\tartificial={}\tnative={}\t\
             unresolved_synth={}\tunresolved_at_flood={}\t\
             samples_on_boundary={}\tsampling_location_unresolved={}\t\
             origin_source={}\torigin_synthetic={}\torigin_seam={}",
            count(ConstraintRole::PhysicalBoundary),
            count(ConstraintRole::SurfaceSampling),
            count(ConstraintRole::ArtificialCut),
            count(ConstraintRole::NativeBoundary),
            count(ConstraintRole::UnresolvedSyntheticClosure),
            roles.unresolved_at_flood.get(),
            samples_on_boundary,
            sampling_location_unresolved,
            by_origin(SegmentOrigin::Source),
            by_origin(SegmentOrigin::SyntheticClosure),
            by_origin(SegmentOrigin::Seam),
        );
    }
    outcome
}

/// Tessellates one surface trimmed by polyline, returning `TessellationOutcome`.
#[allow(dead_code)]
fn trimming_tessellation_with_outcome<S>(
    surface: &S,
    polyboundary: &PolyBoundary,
    tol: f64,
    lattice: &CertifiedLattice,
) -> TessellationOutcome
where
    S: PreMeshableSurface,
{
    trimming_tessellation_with_diagnostics(surface, polyboundary, tol, lattice)
}

/// Tessellates one surface trimmed by polyline, preserving why it failed.
///
/// **G8.** The mesh-only form below discards a fully-formed
/// [`TessellationFailure`] — including `ContradictoryDualParity`, which is a
/// *proved* inconsistency — and returns an empty mesh that the caller cannot
/// distinguish from a face that legitimately meshed to nothing. Detection was
/// never the missing part; the value was constructed and then destroyed one
/// line later. This form is the same computation with the result kept.
fn trimming_tessellation_result<S>(
    surface: &S,
    polyboundary: &PolyBoundary,
    tol: f64,
    lattice: &CertifiedLattice,
    // `Result` here is `truck_topology::Result<T>`, which fixes the error type, so
    // the standard two-parameter form must be named explicitly.
) -> std::result::Result<PolygonMesh, TessellationFailure>
where
    S: PreMeshableSurface,
{
    match trimming_tessellation_with_diagnostics(surface, polyboundary, tol, lattice) {
        TessellationOutcome::Mesh(ft) => {
            let mut mesh = ft.mesh;
            mesh.make_face_compatible_to_normal();
            Ok(mesh)
        }
        TessellationOutcome::Failed(f) => {
            if std::env::var_os("TRUCK_PROBE_FAIL").is_some() {
                eprintln!("PROBE_FAIL reason={:?}", f.reason);
            }
            Err(f)
        }
    }
}

/// Tessellates one surface trimmed by polyline, discarding why it failed.
///
/// Legacy shape, retained for the entry points that cannot carry an outcome.
/// Prefer [`trimming_tessellation_result`].
fn trimming_tessellation<S>(
    surface: &S,
    polyboundary: &PolyBoundary,
    tol: f64,
    lattice: &CertifiedLattice,
) -> PolygonMesh
where
    S: PreMeshableSurface,
{
    trimming_tessellation_result(surface, polyboundary, tol, lattice).unwrap_or_default()
}

/// Inserts parameter divisions into triangulation.
///
/// Returns how many grid samples lay on the boundary, and how many had no
/// established location at all (G7a).
fn insert_surface(
    triangulation: &mut Cdt,
    surface: impl PreMeshableSurface,
    polyline: &PolyBoundary,
    tol: f64,
    roles: &mut ConstraintRoles,
) -> (usize, usize) {
    // Grid samples on a boundary segment, by direct test.
    let mut on_boundary = 0usize;
    // Grid samples no method located. Not "outside", and not known to be on the
    // boundary either — simply unestablished.
    let mut location_unresolved = 0usize;
    // Audit A1: every constraint added below is an interior sampling edge. It
    // exists to control triangle shape and lies wholly inside the material
    // region — `polyline.include` gated the insertion of both its endpoints.
    // It carries no material meaning and must not toggle parity.
    // G5a, and the more consequential half of it.
    //
    // A trim segment that loses its role is still treated as a trim segment,
    // because the unresolved default toggles. A *sampling grid* segment that
    // loses its role is treated as a trim segment too — and that is exactly the
    // defect audit A1 removed, reappearing through the chain-splitting hole
    // rather than through the one-bit test A1 fixed. Labelling the whole
    // realized chain closes it.
    let mut constrain = |triangulation: &mut Cdt, a: FixedVertexHandle, b: FixedVertexHandle| {
        for directed in triangulation.try_add_constraint(a, b) {
            let handle = triangulation.directed_edge(directed).as_undirected().fix();
            roles.record(handle, ConstraintRole::SurfaceSampling);
        }
    };
    let bdb: BoundingBox<Point2> = polyline
        .0
        .iter()
        .flat_map(|loop_| loop_.points.iter())
        .map(std::ops::Deref::deref)
        .collect();
    let range = ((bdb.min()[0], bdb.max()[0]), (bdb.min()[1], bdb.max()[1]));
    let (udiv, vdiv) = surface.parameter_division(range, tol);
    let insert_res: Vec<Vec<Option<_>>> = udiv
        .into_iter()
        .map(|u| {
            vdiv.iter()
                // G7a. This call site asks "may I place an interior sampling
                // vertex here", not "is this point material". Only `Inside`
                // earns a vertex; the other three decline.
                //
                // Declining asserts nothing about material state — that is
                // decided later by constraint roles and the dual labelling,
                // never by this predicate — so skipping is the correct
                // conservative answer to the question actually being asked, and
                // refusing the whole face over a shape-control sample would
                // discard geometry for no semantic gain.
                //
                // What was wrong before was not the skip but the silence: an
                // aborted ray was folded into `false`, so a point the algorithm
                // could not classify was indistinguishable from one it had
                // classified as outside. Both residual populations are now
                // counted, and separately, because "on the boundary" and "not
                // established" are different facts.
                .map(|v| match polyline.locate(Point2::new(u, *v)) {
                    PointLocation::Inside => triangulation.insert(SPoint2::new(u, *v)).ok(),
                    PointLocation::Outside => None,
                    PointLocation::Boundary => {
                        on_boundary += 1;
                        None
                    }
                    PointLocation::Indeterminate => {
                        location_unresolved += 1;
                        None
                    }
                })
                .collect()
        })
        .collect();
    insert_res.windows(2).for_each(|vec| {
        vec[0].windows(2).zip(&vec[1]).for_each(|(a, z)| {
            if let Some(x) = a[0] {
                if let Some(y) = a[1] {
                    constrain(triangulation, x, y);
                }
                if let Some(z) = z {
                    constrain(triangulation, x, *z);
                }
            }
        });
        let idx = vec[0].len() - 1;
        if let (Some(x), Some(y)) = (vec[0][idx], vec[1][idx]) {
            constrain(triangulation, x, y);
        }
    });
    (on_boundary, location_unresolved)
}

/// Converts triangulation into `TessellationOutcome`.
fn triangulation_into_polymesh_outcome<S: ParametricSurface3D>(
    triangulation: &Cdt,
    surface: &S,
    _polyline: &PolyBoundary,
    boundary_map: &HashMap<FixedVertexHandle, Point3>,
    roles: &ConstraintRoles,
    lattice: &CertifiedLattice,
) -> TessellationOutcome {
    use std::collections::{HashMap as StdHashMap, VecDeque};

    // 1. Parity-labeled CDT dual traversal across domain-boundary constraint edges
    let mut face_parity = StdHashMap::<usize, u32>::new();
    let mut queue = VecDeque::new();

    let outer = triangulation.outer_face();
    face_parity.insert(outer.index(), 0);
    queue.push_back((outer.fix(), 0));

    let mut contradictory_parity = false;
    while let Some((ffh, current_parity)) = queue.pop_front() {
        let face = triangulation.face(ffh);
        let edges = if let Some(inner) = face.as_inner() {
            inner.adjacent_edges()
        } else {
            let e0 = face.adjacent_edge().unwrap();
            let e1 = e0.next();
            let e2 = e1.next();
            [e0, e1, e2]
        };
        for e in edges {
            // Audit A1. This was `e.is_constraint_edge()` — one bit, so an
            // interior sampling constraint flipped material parity exactly as a
            // trim segment did. A constraint edge is now only a material
            // transition if its role says so.
            let is_domain_boundary = if e.is_constraint_edge() {
                // G5b: an edge with no resolvable role stops the face rather
                // than being assigned a material meaning it does not have.
                match roles.toggles_material(e.as_undirected().fix()) {
                    Some(toggles) => toggles,
                    None => {
                        return TessellationOutcome::Failed(
                            TessellationFailureReason::ConstraintRoleMissing.into(),
                        )
                    }
                }
            } else {
                false
            };
            let next_parity = if is_domain_boundary {
                current_parity ^ 1
            } else {
                current_parity
            };
            let adj_face = e.rev().face();
            let adj_idx = adj_face.index();
            if let Some(&existing_parity) = face_parity.get(&adj_idx) {
                if existing_parity != next_parity {
                    contradictory_parity = true;
                }
            } else {
                face_parity.insert(adj_idx, next_parity);
                queue.push_back((adj_face.fix(), next_parity));
            }
        }
    }

    if contradictory_parity {
        return TessellationOutcome::Failed(
            TessellationFailureReason::ContradictoryDualParity.into(),
        );
    }

    // 2. Vertex positions, parameter coordinates, and roles
    let mut positions = Vec::<Point3>::new();
    let mut uv_coords = Vec::<Vector2>::new();
    let mut normals = Vec::<Vector3>::new();
    let mut vertex_metadata = Vec::<VertexMetadata>::new();

    let u_period = lattice.declared_u_period();
    let v_period = lattice.declared_v_period();

    let vmap: HashMap<_, _> = triangulation
        .vertices()
        .enumerate()
        .map(|(i, v)| {
            let p = *v.as_ref();
            let idx = v.fix();
            let point = match boundary_map.get(&idx) {
                Some(point) => *point,
                None => surface.subs(p.x, p.y),
            };
            if !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite() {
                return (idx, usize::MAX);
            }
            positions.push(point);
            let uv = Vector2::new(p.x, p.y);
            uv_coords.push(uv);

            // Determine vertex roles (bitflags)
            let mut roles = VertexRoles::default();
            if boundary_map.contains_key(&idx) {
                let is_seam = (u_period.is_some()
                    && (p.x <= 1e-4 || (p.x - u_period.unwrap()).abs() <= 1e-4))
                    || (v_period.is_some()
                        && (p.y <= 1e-4 || (p.y - v_period.unwrap()).abs() <= 1e-4));
                if is_seam {
                    roles.insert(VertexRoles::ARTIFICIAL_SEAM);
                } else {
                    roles.insert(VertexRoles::PHYSICAL_BOUNDARY);
                }
            } else {
                // INTERIOR is derived downstream as the absence of any
                // boundary/seam/singular role; it is not stored as an
                // overlapping flag (PR 4A.1 design, sidecar diagnostics).
            }

            let n = surface.normal(p.x, p.y);
            let n_valid = if n.x.is_finite()
                && n.y.is_finite()
                && n.z.is_finite()
                && n.magnitude2() > 1e-12
            {
                n.normalize()
            } else {
                roles.insert(VertexRoles::SINGULAR_COLLAPSE);
                Vector3::zero()
            };
            normals.push(n_valid);

            vertex_metadata.push(VertexMetadata {
                uv: Point2::new(p.x, p.y),
                generation: if boundary_map.contains_key(&idx) {
                    VertexGeneration::SourceEdgeSample
                } else {
                    VertexGeneration::SurfaceEvaluation
                },
                roles,
                source_edge_use: None,
                seam_pair: None,
                singular_group: None,
            });

            (idx, i)
        })
        .collect();

    if vmap.values().any(|&i| i == usize::MAX) {
        return TessellationOutcome::Failed(TessellationFailureReason::NonFinitePosition.into());
    }

    // 3. Material triangles selection (odd parity = 1)
    let tri_faces_raw: Vec<[usize; 3]> = triangulation
        .inner_faces()
        .filter(|face| face_parity.get(&face.index()) == Some(&1))
        .map(|tri| tri.vertices())
        .filter_map(|tri| {
            let idcs = [
                vmap[&tri[0].fix()],
                vmap[&tri[1].fix()],
                vmap[&tri[2].fix()],
            ];
            if idcs[0] == idcs[1] || idcs[1] == idcs[2] || idcs[0] == idcs[2] {
                return None;
            }
            let p0 = positions[idcs[0]];
            let p1 = positions[idcs[1]];
            let p2 = positions[idcs[2]];
            let cross = (p1 - p0).cross(p2 - p0);
            let area = 0.5 * cross.magnitude();
            if area <= 1e-12 || !area.is_finite() {
                return None;
            }
            Some(idcs)
        })
        .collect();

    if tri_faces_raw.is_empty() {
        return TessellationOutcome::Failed(TessellationFailureReason::NoOddParityRegion.into());
    }

    // 4. Singular Vertex Normal Repair
    let mut vertex_incident_normals = vec![Vector3::zero(); positions.len()];
    for &[i0, i1, i2] in &tri_faces_raw {
        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];
        let cross = (p1 - p0).cross(p2 - p0);
        let mag = cross.magnitude();
        if mag > 1e-12 {
            let fnorm = cross / mag;
            vertex_incident_normals[i0] += fnorm * mag;
            vertex_incident_normals[i1] += fnorm * mag;
            vertex_incident_normals[i2] += fnorm * mag;
        }
    }

    for (i, norm) in normals.iter_mut().enumerate() {
        if norm.so_small() || !norm.x.is_finite() {
            let inc = vertex_incident_normals[i];
            if inc.magnitude2() > 1e-12 {
                *norm = inc.normalize();
            } else {
                *norm = Vector3::unit_z();
            }
        }
    }

    let tri_faces: Vec<[StandardVertex; 3]> = tri_faces_raw
        .into_iter()
        .map(|idcs| array![i => [idcs[i], idcs[i], idcs[i]].into(); 3])
        .collect();

    // PR 4A.1 invariant: sidecar metadata must stay aligned with mesh positions,
    // one record per vertex, through every transformation below. Enforced in the
    // test build; the census (release) relies on it holding by construction here.
    debug_assert_eq!(vertex_metadata.len(), positions.len());

    let mesh = PolygonMesh::debug_new(
        StandardAttributes {
            positions,
            uv_coords,
            normals,
        },
        Faces::from_tri_and_quad_faces(tri_faces, Vec::new()),
    );

    TessellationOutcome::Mesh(FaceTessellation {
        mesh,
        diagnostics: TessellationDiagnostics {
            vertex_metadata,
            seam_pairs: Vec::new(),
            singular_groups: Vec::new(),
        },
    })
}

#[allow(dead_code)]
fn triangulation_into_polymesh<S: ParametricSurface3D>(
    triangulation: &Cdt,
    surface: &S,
    polyline: &PolyBoundary,
    boundary_map: &HashMap<FixedVertexHandle, Point3>,
    roles: &ConstraintRoles,
    lattice: &CertifiedLattice,
) -> PolygonMesh {
    match triangulation_into_polymesh_outcome(
        triangulation,
        surface,
        polyline,
        boundary_map,
        roles,
        lattice,
    ) {
        TessellationOutcome::Mesh(ft) => ft.mesh,
        TessellationOutcome::Failed(_) => PolygonMesh::default(),
    }
}

fn polyline_on_surface(
    surface: impl PreMeshableSurface,
    p: SurfacePoint,
    q: SurfacePoint,
    tol: f64,
) -> Vec<SurfacePoint> {
    use truck_geometry::prelude::*;
    let tol = tol.max(TOLERANCE);
    let line = Line(p.uv, q.uv);
    let pcurve = PCurve::new(line, &surface);
    let (vec, _) = pcurve.parameter_division(pcurve.range_tuple(), tol);
    vec.into_iter()
        .map(|t| {
            let uv = line.subs(t);
            (uv, surface.subs(uv.x, uv.y)).into()
        })
        .collect()
}

/// Explicit classification for a conical/revoluted face with one circular boundary and a collapsed apex.
#[allow(dead_code, unused)]
#[derive(Debug, Clone, Copy)]
pub struct CollapsedPeriodicBoundaryPair {
    pub base_u: f64,
    pub apex_u: f64,
}

impl CollapsedPeriodicBoundaryPair {
    /// Classify whether a face's boundary consists of a single regular periodic loop and a collapsed apex.
    pub fn try_classify<S: PreMeshableSurface>(
        surface: &S,
        closed: &[Vec<SurfacePoint>],
        open: &[Vec<SurfacePoint>],
        _range: (Option<(f64, f64)>, Option<(f64, f64)>),
        lattice: &CertifiedLattice,
    ) -> Option<Self> {
        // 1. Exactly one regular closed periodic boundary and no open generator/sector curves
        if closed.len() != 1 || !open.is_empty() {
            return None;
        }

        let (period, is_v) = match (lattice.declared_v_period(), lattice.declared_u_period()) {
            (Some(vp), _) if vp > 1e-6 => (vp, true),
            (_, Some(up)) if up > 1e-6 => (up, false),
            _ => return None,
        };
        let loop0 = &closed[0];

        // 2. Regular loop must span the periodic parameter (winding ±1, span ~ period)
        let (p_min, p_max) =
            loop0
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), p| {
                    let val = if is_v { p.uv.y } else { p.uv.x };
                    (mn.min(val), mx.max(val))
                });
        if (p_max - p_min) < 0.75 * period {
            return None;
        }

        let base_r: f64 = loop0
            .iter()
            .map(|p| if is_v { p.uv.x } else { p.uv.y })
            .sum::<f64>()
            / loop0.len() as f64;

        // 3. Analytically compute exact apex parameter where angular orbit collapses in 3D.
        use cgmath::InnerSpace;
        let w = |r: f64| -> Vector3 {
            let (p0, p_half) = if is_v {
                (surface.subs(r, 0.0), surface.subs(r, 0.5 * period))
            } else {
                (surface.subs(0.0, r), surface.subs(0.5 * period, r))
            };
            p0 - p_half
        };

        let w0 = w(0.0);
        let w1 = w(1.0);
        let dw = w1 - w0;
        let dw2 = dw.magnitude2();

        if dw2 < 1e-12 {
            return None;
        }

        let apex_r = -w0.dot(dw) / dw2;

        // 4. Certificate guard: verify 3D point collapse at apex
        if w(apex_r).magnitude() > 1e-3 {
            return None;
        }

        // 5. Axial separation between base loop and apex must be nonzero
        if (base_r - apex_r).abs() < 1e-6 {
            return None;
        }

        Some(Self {
            base_u: base_r,
            apex_u: apex_r,
        })
    }
}

#[allow(dead_code, unused)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoundaryStratum {
    #[allow(dead_code, unused)]
    RegularCurve,
    #[allow(dead_code, unused)]
    CollapsedToPoint,
    #[allow(dead_code, unused)]
    PeriodicIdentification,
}

#[allow(dead_code, unused)]
#[derive(Clone, Debug)]
pub struct WireAssembledFace {
    pub loops: Vec<Vec<SurfacePoint>>,
    pub strata: Vec<BoundaryStratum>,
}

#[allow(dead_code, unused)]
#[derive(Clone, Debug)]
pub struct QuotientLiftCertificate {
    #[allow(dead_code, unused)]
    pub period_shifts: Vec<(i32, i32)>,
    #[allow(dead_code, unused)]
    pub max_junction_residual: f64,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct QuotientResolvedFace {
    pub resolved_loops: Vec<Vec<SurfacePoint>>,
    pub certificate: QuotientLiftCertificate,
}

#[allow(dead_code)]
pub fn solve_quotient_lift<S: PreMeshableSurface>(
    surface: &S,
    loops: &[Vec<SurfacePoint>],
) -> Option<QuotientResolvedFace> {
    let u_period = surface.u_period();
    let v_period = surface.v_period();

    let mut resolved_loops = Vec::with_capacity(loops.len());
    let mut period_shifts = Vec::with_capacity(loops.len());
    let mut max_junction_residual = 0.0f64;

    for loop_pts in loops {
        if loop_pts.len() < 2 {
            resolved_loops.push(loop_pts.clone());
            period_shifts.push((0, 0));
            continue;
        }

        let p0 = loop_pts[0].uv;
        let p1 = loop_pts[loop_pts.len() - 1].uv;
        let du = p1.x - p0.x;
        let dv = p1.y - p0.y;

        let ku = match u_period {
            Some(pu) if pu > 1e-6 => (du / pu).round() as i32,
            _ => 0,
        };
        let kv = match v_period {
            Some(pv) if pv > 1e-6 => (dv / pv).round() as i32,
            _ => 0,
        };

        period_shifts.push((ku, kv));

        let u_shift = match u_period {
            Some(pu) => (ku as f64) * pu,
            None => 0.0,
        };
        let v_shift = match v_period {
            Some(pv) => (kv as f64) * pv,
            None => 0.0,
        };

        let mut lifted_loop = loop_pts.clone();
        let n = lifted_loop.len();
        for (idx, pt) in lifted_loop.iter_mut().enumerate() {
            let frac = idx as f64 / (n - 1).max(1) as f64;
            pt.uv.x -= frac * u_shift;
            pt.uv.y -= frac * v_shift;
        }

        let residual = lifted_loop[0].uv.distance(lifted_loop[n - 1].uv);
        max_junction_residual = max_junction_residual.max(residual);
        resolved_loops.push(lifted_loop);
    }

    Some(QuotientResolvedFace {
        resolved_loops,
        certificate: QuotientLiftCertificate {
            period_shifts,
            max_junction_residual,
        },
    })
}

#[test]
fn test_global_quotient_lift_solver() {
    use truck_modeling::{BSplineSurface, KnotVec};
    let knot_vec = KnotVec::from(vec![0.0, 0.0, 1.0, 1.0]);
    let ctrl_pts = vec![
        vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
        vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
    ];
    let surface = BSplineSurface::new((knot_vec.clone(), knot_vec), ctrl_pts);
    let loop_pts = vec![
        (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
        (Point2::new(1.0, 1.0), Point3::new(1.0, 1.0, 0.0)).into(),
    ];
    let resolved = solve_quotient_lift(&surface, &[loop_pts]);
    assert!(resolved.is_some());
    let res = resolved.unwrap();
    assert_eq!(res.certificate.period_shifts, vec![(0, 0)]);
}

/*
#[test]
#[ignore]
#[cfg(not(target_arch = "wasm32"))]
fn par_bench() {
    use std::time::Instant;
    use truck_modeling::*;
    const JSON: &str = include_str!("../../resources/shape/bottle.json");
    let solid: Solid = serde_json::from_str(JSON).unwrap();
    let shell = solid.into_boundaries().pop().unwrap();

    let instant = Instant::now();
    (0..100).for_each(|_| {
        let _shell = shell_tessellation(&shell, 0.01, by_search_parameter);
    });
    println!("{}ms", instant.elapsed().as_millis());

    let instant = Instant::now();
    (0..100).for_each(|_| {
        let _shell = shell_tessellation_single_thread(&shell, 0.01, by_search_parameter);
    });
    println!("{}ms", instant.elapsed().as_millis());
}
*/

#[cfg(test)]
mod cone_topology_tests {
    use super::*;
    use std::f64::consts::PI;
    use truck_modeling::{Line, Point2, Point3, RevolutedCurve, Vector3};

    fn make_test_cone(r_base: f64, r_apex: f64, h: f64) -> RevolutedCurve<Line<Point3>> {
        let p0 = Point3::new(r_apex, 0.0, 0.0);
        let p1 = Point3::new(r_base, 0.0, h);
        RevolutedCurve::by_revolution(Line(p0, p1), Point3::origin(), Vector3::unit_z())
    }

    #[test]
    fn test_cone_lateral_face_forward_classified() {
        let cone = make_test_cone(10.0, 0.0, 10.0);
        let loop0: Vec<SurfacePoint> = (0..=10)
            .map(|i| {
                let v = (i as f64 / 10.0) * 2.0 * PI;
                let uv = Point2::new(1.0, v);
                (uv, cone.subs(uv.x, uv.y)).into()
            })
            .collect();
        let range = (Some((0.0, 1.0)), None);
        let res = CollapsedPeriodicBoundaryPair::try_classify(
            &cone,
            &[loop0],
            &[],
            range,
            &unevidenced_lattice(&cone),
        );
        assert!(res.is_some());
        let pair = res.unwrap();
        assert!((pair.apex_u - 0.0).abs() < 1e-6);
        assert!((pair.base_u - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_truncated_cone_two_loops_rejected() {
        let cone = make_test_cone(10.0, 5.0, 10.0);
        let loop0: Vec<SurfacePoint> = (0..=10)
            .map(|i| {
                let v = (i as f64 / 10.0) * 2.0 * PI;
                let uv = Point2::new(1.0, v);
                (uv, cone.subs(uv.x, uv.y)).into()
            })
            .collect();
        let loop1: Vec<SurfacePoint> = (0..=10)
            .map(|i| {
                let v = (i as f64 / 10.0) * 2.0 * PI;
                let uv = Point2::new(0.0, v);
                (uv, cone.subs(uv.x, uv.y)).into()
            })
            .collect();
        let range = (Some((0.0, 1.0)), None);
        let res = CollapsedPeriodicBoundaryPair::try_classify(
            &cone,
            &[loop0, loop1],
            &[],
            range,
            &unevidenced_lattice(&cone),
        );
        assert!(res.is_none());
    }

    #[test]
    fn test_cone_sector_with_open_generator_edges_rejected() {
        let cone = make_test_cone(10.0, 0.0, 10.0);
        let loop0: Vec<SurfacePoint> = (0..=10)
            .map(|i| {
                let v = (i as f64 / 10.0) * PI;
                let uv = Point2::new(1.0, v);
                (uv, cone.subs(uv.x, uv.y)).into()
            })
            .collect();
        let open_edge: Vec<SurfacePoint> = vec![
            (Point2::new(0.0, 0.0), cone.subs(0.0, 0.0)).into(),
            (Point2::new(1.0, 0.0), cone.subs(1.0, 0.0)).into(),
        ];
        let range = (Some((0.0, 1.0)), None);
        let res = CollapsedPeriodicBoundaryPair::try_classify(
            &cone,
            &[loop0],
            &[open_edge],
            range,
            &unevidenced_lattice(&cone),
        );
        assert!(res.is_none());
    }

    #[test]
    fn test_zero_area_non_cone_cylinder_rejected() {
        let cylinder = RevolutedCurve::by_revolution(
            Line(Point3::new(10.0, 0.0, 0.0), Point3::new(10.0, 0.0, 10.0)),
            Point3::origin(),
            Vector3::unit_z(),
        );
        let loop0: Vec<SurfacePoint> = (0..=10)
            .map(|i| {
                let v = (i as f64 / 10.0) * 2.0 * PI;
                let uv = Point2::new(1.0, v);
                (uv, cylinder.subs(uv.x, uv.y)).into()
            })
            .collect();
        let range = (Some((0.0, 1.0)), None);
        let res = CollapsedPeriodicBoundaryPair::try_classify(
            &cylinder,
            &[loop0],
            &[],
            range,
            &unevidenced_lattice(&cylinder),
        );
        assert!(res.is_none());
    }

    /// A cylindrical band presented the way STEP presents one: two boundary
    /// circles winding **opposite** ways, as they must for the face boundary to
    /// be coherently oriented.
    fn opposite_winding_band_pieces() -> (RevolutedCurve<Line<Point3>>, Vec<PolyBoundaryPiece>) {
        let cylinder = RevolutedCurve::by_revolution(
            Line(Point3::new(10.0, 0.0, 0.0), Point3::new(10.0, 0.0, 10.0)),
            Point3::origin(),
            Vector3::unit_z(),
        );
        let circle = |u: f64, sign: f64| -> PolyBoundaryPiece {
            PolyBoundaryPiece(
                (0..=32)
                    .map(|i| {
                        let v = sign * (i as f64 / 32.0) * 2.0 * PI;
                        let uv = Point2::new(u, v);
                        (uv, cylinder.subs(uv.x, uv.y)).into()
                    })
                    .collect(),
            )
        };
        (cylinder, vec![circle(0.2, 1.0), circle(0.8, -1.0)])
    }

    /// The deck equation, on the geometry it was written for: reversing loop 1
    /// gives `Σδ = ±2`, so the legacy join is refused and forward traversal is
    /// named as the unique solution.
    #[test]
    fn opposite_winding_band_selects_forward_traversal() {
        let (cylinder, pieces) = opposite_winding_band_pieces();
        let lattice = unevidenced_lattice(&cylinder);
        let (_, legacy) = PolyBoundary::new_with_join(
            pieces.clone(),
            &cylinder,
            0.01,
            &lattice,
            TwoLoopJoinPolicy::Legacy,
        );
        assert_eq!(
            legacy,
            TwoLoopJoinOutcome::ForwardResolves { applied: false },
            "the legacy policy diagnoses the case but does not act on it",
        );
        let (_, corrected) = PolyBoundary::new_with_join(
            pieces,
            &cylinder,
            0.01,
            &lattice,
            TwoLoopJoinPolicy::DeckConsistent,
        );
        assert_eq!(
            corrected,
            TwoLoopJoinOutcome::ForwardResolves { applied: true },
            "the deck-consistent policy takes the solution",
        );
    }

    /// The corrected join is what makes the band tessellate: the legacy
    /// traversal's two bridges cross, and the CDT refuses the second one.
    #[test]
    fn opposite_winding_band_tessellates_only_when_deck_consistent() {
        let (cylinder, pieces) = opposite_winding_band_pieces();
        let lattice = unevidenced_lattice(&cylinder);
        let legacy = PolyBoundary::new(pieces.clone(), &cylinder, 0.01, &lattice);
        assert_eq!(
            trimming_tessellation_result(&cylinder, &legacy, 0.01, &lattice)
                .err()
                .map(|failure| failure.reason),
            Some(TessellationFailureReason::ConstraintInsertionIncomplete),
            "the crossing bridges are the failure this package is about",
        );
        let (corrected, _) = PolyBoundary::new_with_join(
            pieces,
            &cylinder,
            0.01,
            &lattice,
            TwoLoopJoinPolicy::DeckConsistent,
        );
        let mesh = trimming_tessellation_result(&cylinder, &corrected, 0.01, &lattice)
            .expect("the deck-consistent boundary tessellates");
        assert!(!mesh.tri_faces().is_empty(), "and produces triangles");
    }

    /// Two loops winding the *same* way need the reversal, and must not be
    /// disturbed: the equation selects the legacy traversal there.
    #[test]
    fn same_winding_band_keeps_the_legacy_traversal() {
        let cylinder = RevolutedCurve::by_revolution(
            Line(Point3::new(10.0, 0.0, 0.0), Point3::new(10.0, 0.0, 10.0)),
            Point3::origin(),
            Vector3::unit_z(),
        );
        let circle = |u: f64| -> PolyBoundaryPiece {
            PolyBoundaryPiece(
                (0..=32)
                    .map(|i| {
                        let v = (i as f64 / 32.0) * 2.0 * PI;
                        let uv = Point2::new(u, v);
                        (uv, cylinder.subs(uv.x, uv.y)).into()
                    })
                    .collect(),
            )
        };
        let (_, outcome) = PolyBoundary::new_with_join(
            vec![circle(0.2), circle(0.8)],
            &cylinder,
            0.01,
            &unevidenced_lattice(&cylinder),
            TwoLoopJoinPolicy::DeckConsistent,
        );
        assert_eq!(outcome, TwoLoopJoinOutcome::LegacyDeckConsistent);
    }

    #[test]
    fn test_traversal_semantics_periodic_circle() {
        use crate::tessellation::domain::projection::TraversalSemantics;
        use truck_geometry::prelude::*;
        let circle = UnitCircle::<Point3>::new();
        let cone = make_test_cone(10.0, 0.0, 10.0);
        let semantics = TraversalSemantics::resolve(&circle, &cone, 1e-4);
        assert!(matches!(semantics, TraversalSemantics::FullPeriod { .. }));
    }

    #[test]
    fn test_traversal_semantics_degenerate_point() {
        use crate::tessellation::domain::projection::TraversalSemantics;
        use truck_geometry::prelude::*;
        let line = Line(Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0));
        let cone = make_test_cone(10.0, 0.0, 10.0);
        let semantics = TraversalSemantics::resolve(&line, &cone, 1e-4);
        assert_eq!(semantics, TraversalSemantics::DegeneratePoint);
    }

    #[test]
    fn test_shared_boundary_projection_processor_wrapped_cone() {
        use crate::tessellation::domain::projection::{project_boundary_curve, TraversalSemantics};
        use truck_geometry::prelude::*;
        let cone = make_test_cone(10.0, 0.0, 10.0);
        let p_uv = cone.subs(0.5, 0.0);
        let tr = Matrix4::from_translation(Vector3::new(10.0, 20.0, 30.0));
        let proc_cone = Processor::with_transform(cone, tr);
        let circle = Processor::with_transform(TrimmedCurve::new(Line(p_uv, p_uv), (0.0, 1.0)), tr);
        let sem = TraversalSemantics::resolve(&circle, &proc_cone, 1e-4);
        let path = project_boundary_curve(&circle, &proc_cone, sem, 1.0).unwrap();
        assert!(!path.samples.is_empty());
    }

    #[test]
    fn test_periodic_closure_cone_tessellation() {
        let cone = make_test_cone(10.0, 0.0, 10.0);
        let tol = 0.1;
        let n = 32usize;
        let circle_pts: Vec<SurfacePoint> = (0..=n)
            .map(|i| {
                let v = (i as f64 / n as f64) * 2.0 * PI;
                let uv = Point2::new(1.0, v);
                (uv, cone.subs(uv.x, uv.y)).into()
            })
            .collect();
        let piece = PolyBoundaryPiece(circle_pts);
        let boundary = PolyBoundary::new(vec![piece], &cone, tol, &unevidenced_lattice(&cone));
        let mesh = trimming_tessellation(&cone, &boundary, tol, &unevidenced_lattice(&cone));
        assert!(
            !mesh.faces().is_empty(),
            "Cone C-APEX-DISK face must tessellate to non-empty mesh"
        );
    }

    #[test]
    fn test_parity_single_disk() {
        use truck_geometry::prelude::*;
        let plane = Plane::new(
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        let tol = 0.01;
        let loop0: Vec<SurfacePoint> = vec![
            (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
            (Point2::new(10.0, 0.0), Point3::new(10.0, 0.0, 0.0)).into(),
            (Point2::new(10.0, 10.0), Point3::new(10.0, 10.0, 0.0)).into(),
            (Point2::new(0.0, 10.0), Point3::new(0.0, 10.0, 0.0)).into(),
            (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
        ];
        let boundary = PolyBoundary::new(
            vec![PolyBoundaryPiece(loop0)],
            &plane,
            tol,
            &unevidenced_lattice(&plane),
        );
        let mesh = trimming_tessellation(&plane, &boundary, tol, &unevidenced_lattice(&plane));
        assert!(!mesh.faces().is_empty());
    }

    #[test]
    fn test_parity_disk_with_hole() {
        use truck_geometry::prelude::*;
        let plane = Plane::new(
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        let tol = 0.01;
        let outer: Vec<SurfacePoint> = vec![
            (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
            (Point2::new(10.0, 0.0), Point3::new(10.0, 0.0, 0.0)).into(),
            (Point2::new(10.0, 10.0), Point3::new(10.0, 10.0, 0.0)).into(),
            (Point2::new(0.0, 10.0), Point3::new(0.0, 10.0, 0.0)).into(),
            (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
        ];
        let hole: Vec<SurfacePoint> = vec![
            (Point2::new(3.0, 3.0), Point3::new(3.0, 3.0, 0.0)).into(),
            (Point2::new(7.0, 3.0), Point3::new(7.0, 3.0, 0.0)).into(),
            (Point2::new(7.0, 7.0), Point3::new(7.0, 7.0, 0.0)).into(),
            (Point2::new(3.0, 7.0), Point3::new(3.0, 7.0, 0.0)).into(),
            (Point2::new(3.0, 3.0), Point3::new(3.0, 3.0, 0.0)).into(),
        ];
        let boundary = PolyBoundary::new(
            vec![PolyBoundaryPiece(outer), PolyBoundaryPiece(hole)],
            &plane,
            tol,
            &unevidenced_lattice(&plane),
        );
        let mesh = trimming_tessellation(&plane, &boundary, tol, &unevidenced_lattice(&plane));
        assert!(!mesh.faces().is_empty());
        // Verify hole centroid (5.0, 5.0) has no triangle containing it
        for tri in mesh.faces().tri_faces() {
            let p0 = mesh.uv_coords()[tri[0].pos];
            let p1 = mesh.uv_coords()[tri[1].pos];
            let p2 = mesh.uv_coords()[tri[2].pos];
            let center = (p0 + p1 + p2) / 3.0;
            assert!(
                !(center.x > 3.1 && center.x < 6.9 && center.y > 3.1 && center.y < 6.9),
                "Hole interior must not contain triangles"
            );
        }
    }

    #[test]
    fn test_parity_concave_outer_loop() {
        use truck_geometry::prelude::*;
        let plane = Plane::new(
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        let tol = 0.01;
        // L-shaped concave outer loop
        let loop0: Vec<SurfacePoint> = vec![
            (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
            (Point2::new(10.0, 0.0), Point3::new(10.0, 0.0, 0.0)).into(),
            (Point2::new(10.0, 5.0), Point3::new(10.0, 5.0, 0.0)).into(),
            (Point2::new(5.0, 5.0), Point3::new(5.0, 5.0, 0.0)).into(),
            (Point2::new(5.0, 10.0), Point3::new(5.0, 10.0, 0.0)).into(),
            (Point2::new(0.0, 10.0), Point3::new(0.0, 10.0, 0.0)).into(),
            (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
        ];
        let boundary = PolyBoundary::new(
            vec![PolyBoundaryPiece(loop0)],
            &plane,
            tol,
            &unevidenced_lattice(&plane),
        );
        let mesh = trimming_tessellation(&plane, &boundary, tol, &unevidenced_lattice(&plane));
        assert!(!mesh.faces().is_empty());
    }

    #[test]
    fn test_parity_intersecting_constraints_rejected() {
        use truck_geometry::prelude::*;
        let plane = Plane::new(
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        let tol = 0.01;
        // Self-overlapping loop traversing same segment (0,0)->(10,0) twice in forward direction
        let loop0: Vec<SurfacePoint> = vec![
            (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
            (Point2::new(10.0, 0.0), Point3::new(10.0, 0.0, 0.0)).into(),
            (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
            (Point2::new(10.0, 0.0), Point3::new(10.0, 0.0, 0.0)).into(),
            (Point2::new(10.0, 10.0), Point3::new(10.0, 10.0, 0.0)).into(),
            (Point2::new(0.0, 10.0), Point3::new(0.0, 10.0, 0.0)).into(),
            (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
        ];
        let boundary = PolyBoundary::new(
            vec![PolyBoundaryPiece(loop0)],
            &plane,
            tol,
            &unevidenced_lattice(&plane),
        );
        let mesh = trimming_tessellation(&plane, &boundary, tol, &unevidenced_lattice(&plane));
        assert!(
            mesh.faces().is_empty(),
            "Self-overlapping degenerate loop must fail or produce empty mesh"
        );
    }
}

#[cfg(test)]
mod singular_transition_tests {
    use super::*;
    use truck_modeling::{Point2, Point3, Vector3};

    /// `v` collapses at `u = 1`: the whole `u = 1` row maps to the single
    /// point `(0, 1, 0)`, so every `(1, v)` is a legitimate UV representative
    /// of that one singular 3D point. This is the abstract shape of the
    /// representative face #54588 collapse.
    #[derive(Clone, Copy)]
    struct CollapsedPatch;

    impl ParametricSurface for CollapsedPatch {
        type Point = Point3;
        type Vector = Vector3;
        fn subs(&self, u: f64, v: f64) -> Point3 {
            Point3::new((1.0 - u) * v, u, 0.0)
        }
        fn uder(&self, _: f64, v: f64) -> Vector3 {
            Vector3::new(-v, 1.0, 0.0)
        }
        fn vder(&self, u: f64, _: f64) -> Vector3 {
            Vector3::new(1.0 - u, 0.0, 0.0)
        }
        fn uuder(&self, _: f64, _: f64) -> Vector3 {
            Vector3::zero()
        }
        fn uvder(&self, _: f64, _: f64) -> Vector3 {
            Vector3::new(-1.0, 0.0, 0.0)
        }
        fn vvder(&self, _: f64, _: f64) -> Vector3 {
            Vector3::zero()
        }
        fn der_mn(&self, m: usize, n: usize, u: f64, v: f64) -> Vector3 {
            match (m, n) {
                (0, 0) => self.subs(u, v).to_vec(),
                (1, 0) => self.uder(u, v),
                (0, 1) => self.vder(u, v),
                (1, 1) => self.uvder(u, v),
                _ => Vector3::zero(),
            }
        }
    }

    impl ParametricSurface3D for CollapsedPatch {}

    impl ParameterDivision2D for CollapsedPatch {
        fn parameter_division(
            &self,
            ((u0, u1), (v0, v1)): ((f64, f64), (f64, f64)),
            _: f64,
        ) -> (Vec<f64>, Vec<f64>) {
            (vec![u0, u1], vec![v0, v1])
        }
    }

    /// Mirror of `CollapsedPatch` with the axes swapped, so `u` collapses at
    /// `v = 1`. Used to exercise the u-direction enter/leave branches.
    #[derive(Clone, Copy)]
    struct CollapsedPatchU;

    impl ParametricSurface for CollapsedPatchU {
        type Point = Point3;
        type Vector = Vector3;
        fn subs(&self, u: f64, v: f64) -> Point3 {
            Point3::new((1.0 - v) * u, v, 0.0)
        }
        fn uder(&self, _: f64, v: f64) -> Vector3 {
            Vector3::new(1.0 - v, 0.0, 0.0)
        }
        fn vder(&self, u: f64, _: f64) -> Vector3 {
            Vector3::new(-u, 1.0, 0.0)
        }
        fn uuder(&self, _: f64, _: f64) -> Vector3 {
            Vector3::zero()
        }
        fn uvder(&self, _: f64, _: f64) -> Vector3 {
            Vector3::new(-1.0, 0.0, 0.0)
        }
        fn vvder(&self, _: f64, _: f64) -> Vector3 {
            Vector3::zero()
        }
        fn der_mn(&self, m: usize, n: usize, u: f64, v: f64) -> Vector3 {
            match (m, n) {
                (0, 0) => self.subs(u, v).to_vec(),
                (1, 0) => self.uder(u, v),
                (0, 1) => self.vder(u, v),
                (1, 1) => self.uvder(u, v),
                _ => Vector3::zero(),
            }
        }
    }

    impl ParametricSurface3D for CollapsedPatchU {}

    impl ParameterDivision2D for CollapsedPatchU {
        fn parameter_division(
            &self,
            ((u0, u1), (v0, v1)): ((f64, f64), (f64, f64)),
            _: f64,
        ) -> (Vec<f64>, Vec<f64>) {
            (vec![u0, u1], vec![v0, v1])
        }
    }

    /// Items 5-7: at least one triangle, no zero-area triangles, and a single
    /// consistent orientation sign. CollapsedPatch is planar (z = 0), so the
    /// orientation reduces to the sign of the z component of the cross product.
    fn assert_no_zero_area_or_orientation_flip(mesh: &PolygonMesh) {
        assert!(
            !mesh.faces().is_empty(),
            "tessellation produced no triangles"
        );
        let mut orientation = 0.0_f64;
        for tri in mesh.faces().tri_faces() {
            let [p0, p1, p2] = tri.map(|v| mesh.positions()[v.pos]);
            let cross = (p1 - p0).cross(p2 - p0);
            assert!(cross.magnitude2() > 1.0e-18, "zero-area triangle");
            if orientation == 0.0 {
                orientation = cross.z.signum();
            }
            assert_eq!(
                cross.z.signum(),
                orientation,
                "inconsistent triangle orientation"
            );
        }
    }

    /// Items 1-8: entering the collapsed row preserves the incoming UV branch,
    /// leaving it appends a bridge paired with the singular 3D point, every
    /// touched point re-evaluates within tolerance, the reconstructed boundary
    /// does not backtrack across the singular row, and the surface tessellates
    /// with consistent triangle orientation. The reverse-traversal shape is
    /// checked at the end.
    #[test]
    fn singular_transition_preserves_incidence_and_tessellates() {
        let surface = CollapsedPatch;
        let tolerance = 1.0e-9;
        let uv = |u, v| Point2::new(u, v);
        let sp = |p: Point2| (p, surface.subs(p.x, p.y)).into();
        // Approach the singular row along v = 0, leave it along v = 1.
        let mut points: Vec<SurfacePoint> =
            [uv(0.0, 1.0), uv(0.0, 0.5), uv(0.0, 0.0), uv(0.5, 0.0)]
                .into_iter()
                .map(sp)
                .collect();

        // Enter the collapse. The raw lift proposes (1, 1.1006...) but the
        // singular row u = 1 evaluates it to (0, 1, 0).
        let previous_uv = uv(0.5, 0.0);
        let previous_point = surface.subs(previous_uv.x, previous_uv.y);
        let mut entering_uv = uv(1.0, 1.100_620_084_301_750_6);
        let singular_point = surface.subs(entering_uv.x, entering_uv.y);
        assert!(singular_point.near(&Point3::new(0.0, 1.0, 0.0)));
        reconcile_singular_transition(
            &surface,
            previous_uv,
            previous_point,
            &mut entering_uv,
            singular_point,
            tolerance,
            &mut points,
        );
        // (1) The incoming v = 0 branch is preserved.
        assert!(entering_uv.near(&uv(1.0, 0.0)));
        points.push((entering_uv, singular_point).into());

        // Leave the collapse toward v = 1.
        let mut leaving_uv = uv(0.5, 1.0);
        let leaving_point = surface.subs(leaving_uv.x, leaving_uv.y);
        reconcile_singular_transition(
            &surface,
            entering_uv,
            singular_point,
            &mut leaving_uv,
            leaving_point,
            tolerance,
            &mut points,
        );
        // (2) A bridge was appended, paired with the singular 3D point.
        let bridge = *points.last().unwrap();
        assert!(bridge.uv.near(&uv(1.0, 1.0)));
        assert!(bridge.point.near(&singular_point));
        points.push((leaving_uv, leaving_point).into());

        // (3) Every modified or inserted point re-evaluates within tolerance.
        assert!(points
            .iter()
            .all(|p| surface.subs(p.uv.x, p.uv.y).distance(p.point) <= tolerance));

        // (4) No improper backtracking crossing. The original defect produced
        // the spike (0.5,0) -> (1,1.1006) -> (1,1); the repaired boundary
        // enters and leaves the singular row u = 1 exactly once each.
        let u_visits: Vec<f64> = points.iter().map(|p| p.uv.x).collect();
        let near_row = |x: f64| (x - 1.0).abs() < 1.0e-7;
        let crossings = u_visits
            .windows(2)
            .filter(|w| near_row(w[0]) ^ near_row(w[1]))
            .count();
        assert_eq!(
            crossings, 2,
            "boundary must enter and leave u = 1 exactly once each"
        );

        // Close the loop and tessellate.
        points.push(sp(uv(0.0, 1.0)));
        let boundary = PolyBoundary::new(
            vec![PolyBoundaryPiece(points)],
            &surface,
            tolerance,
            &unevidenced_lattice(&surface),
        );
        let mesh = trimming_tessellation(
            &surface,
            &boundary,
            tolerance,
            &unevidenced_lattice(&surface),
        );
        // (5),(6),(7) at least one triangle, no zero-area, consistent orientation.
        assert_no_zero_area_or_orientation_flip(&mesh);

        // (8) Reverse traversal: walk the same collapse the other way. The
        // singular 3D point must still be preserved on both enter and leave,
        // with a bridge inserted on the leave.
        let mut rev: Vec<SurfacePoint> = [uv(0.5, 1.0)].into_iter().map(sp).collect();
        let prev_uv = uv(0.5, 1.0);
        let prev_pt = surface.subs(prev_uv.x, prev_uv.y);
        let mut enter_uv = uv(1.0, 0.9);
        let singular_pt = surface.subs(enter_uv.x, enter_uv.y);
        assert!(singular_pt.near(&Point3::new(0.0, 1.0, 0.0)));
        reconcile_singular_transition(
            &surface,
            prev_uv,
            prev_pt,
            &mut enter_uv,
            singular_pt,
            tolerance,
            &mut rev,
        );
        assert!(
            enter_uv.near(&uv(1.0, 1.0)),
            "reverse enter preserves the singular row"
        );
        rev.push((enter_uv, singular_pt).into());
        let mut leave_uv = uv(0.5, 0.0);
        let leave_pt = surface.subs(leave_uv.x, leave_uv.y);
        reconcile_singular_transition(
            &surface,
            enter_uv,
            singular_pt,
            &mut leave_uv,
            leave_pt,
            tolerance,
            &mut rev,
        );
        let rev_bridge = *rev.last().unwrap();
        assert!(
            rev_bridge.point.near(&singular_pt),
            "reverse leave bridges with the singular 3D point"
        );
        assert!(
            rev.iter()
                .all(|p| surface.subs(p.uv.x, p.uv.y).distance(p.point) <= tolerance),
            "reverse traversal: every point re-evaluates within tolerance"
        );
    }

    /// Item 9: the u-direction enter/leave branches on a surface whose `u`
    /// collapses at `v = 1`.
    #[test]
    fn singular_transition_both_parameter_directions() {
        let surface = CollapsedPatchU;
        let tolerance = 1.0e-9;
        let uv = |u, v| Point2::new(u, v);
        let sp = |p: Point2| (p, surface.subs(p.x, p.y)).into();
        let mut points: Vec<SurfacePoint> =
            [uv(1.0, 0.0), uv(0.5, 0.0), uv(0.0, 0.0), uv(0.0, 0.5)]
                .into_iter()
                .map(sp)
                .collect();
        let previous_uv = uv(0.0, 0.5);
        let previous_point = surface.subs(previous_uv.x, previous_uv.y);
        let mut entering_uv = uv(1.100_620_084_301_750_6, 1.0);
        let singular_point = surface.subs(entering_uv.x, entering_uv.y);
        assert!(singular_point.near(&Point3::new(0.0, 1.0, 0.0)));
        reconcile_singular_transition(
            &surface,
            previous_uv,
            previous_point,
            &mut entering_uv,
            singular_point,
            tolerance,
            &mut points,
        );
        assert!(
            entering_uv.near(&uv(0.0, 1.0)),
            "u-collapse enter preserves the incoming u branch"
        );
        points.push((entering_uv, singular_point).into());

        let mut leaving_uv = uv(1.0, 0.5);
        let leaving_point = surface.subs(leaving_uv.x, leaving_uv.y);
        reconcile_singular_transition(
            &surface,
            entering_uv,
            singular_point,
            &mut leaving_uv,
            leaving_point,
            tolerance,
            &mut points,
        );
        let bridge = *points.last().unwrap();
        assert!(bridge.uv.near(&uv(1.0, 1.0)));
        assert!(bridge.point.near(&singular_point));
        points.push((leaving_uv, leaving_point).into());
        assert!(
            points
                .iter()
                .all(|p| surface.subs(p.uv.x, p.uv.y).distance(p.point) <= tolerance),
            "u-direction: every point re-evaluates within tolerance"
        );
    }

    /// Item 10: a derivative that is small but whose candidate UV does NOT
    /// reconstruct the associated 3D point must not be substituted.
    #[test]
    fn singular_transition_negative_case_rejects_non_reconstructing_candidate() {
        let surface = CollapsedPatch;
        let tolerance = 1.0e-9;
        // At u = 1 - delta the v-derivative has magnitude delta < TOLERANCE, so
        // `so_small()` proposes a substitution, but the point still genuinely
        // depends on v, so the residual check must reject it.
        let delta = 5.0e-7;
        let previous_uv = Point2::new(0.5, 0.0);
        let previous_point = surface.subs(previous_uv.x, previous_uv.y);
        let raw_uv = Point2::new(1.0 - delta, 0.6);
        let raw_point = surface.subs(raw_uv.x, raw_uv.y);
        // Sanity: the v-derivative really is small here, and the candidate
        // really would move the point beyond tolerance.
        assert!(surface.vder(raw_uv.x, raw_uv.y).so_small());
        let candidate = Point2::new(raw_uv.x, previous_uv.y);
        assert!(
            surface.subs(candidate.x, candidate.y).distance(raw_point) > tolerance,
            "test setup: candidate must fail the residual check"
        );

        let mut current_uv = raw_uv;
        let mut out: Vec<SurfacePoint> = Vec::new();
        reconcile_singular_transition(
            &surface,
            previous_uv,
            previous_point,
            &mut current_uv,
            raw_point,
            tolerance,
            &mut out,
        );
        // The candidate did not reconstruct the point: nothing changed.
        assert!(
            current_uv.near(&raw_uv),
            "helper must not substitute a non-reconstructing candidate"
        );
        assert!(
            out.is_empty(),
            "no bridge may be appended when the residual check fails"
        );
    }
}

#[cfg(test)]
mod segment_origin_tests {
    use super::*;

    fn pt(x: f64, y: f64) -> SurfacePoint {
        (Point2::new(x, y), Point3::new(x, y, 0.0)).into()
    }

    /// A shared endpoint drops the duplicate and creates no segment.
    #[test]
    fn shared_endpoint_creates_no_segment() {
        let mut path = BoundaryPath::start(vec![pt(0.0, 0.0), pt(1.0, 0.0)], SegmentOrigin::Source);
        path.append(
            vec![pt(1.0, 0.0), pt(1.0, 1.0)],
            SegmentOrigin::SyntheticClosure,
            PartJoin::SharedEndpoint,
        );
        assert_eq!(path.points.len(), 3, "the duplicate join point is dropped");
        assert_eq!(
            path.origins,
            vec![SegmentOrigin::Source, SegmentOrigin::SyntheticClosure],
        );
    }

    /// A bridge keeps **both** endpoints and inserts exactly one labelled
    /// segment between them.
    ///
    /// This is the case an earlier implementation got wrong: it dropped every
    /// part's last point unconditionally, so `a1 -> a2 -> b0` silently became
    /// the shortcut `a1 -> b0`, deleting a source segment. The point count is
    /// the assertion that matters — metadata retention must not change the
    /// polygon.
    #[test]
    fn a_bridge_preserves_both_endpoints() {
        let mut path = BoundaryPath::start(
            vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(2.0, 0.0)],
            SegmentOrigin::Source,
        );
        path.append(
            vec![pt(5.0, 5.0), pt(6.0, 5.0), pt(7.0, 5.0)],
            SegmentOrigin::Source,
            PartJoin::Bridge(SegmentOrigin::Seam),
        );
        assert_eq!(path.points.len(), 6, "no source point may be dropped");
        assert_eq!(
            path.points[2].uv,
            pt(2.0, 0.0).uv,
            "the first part keeps its tail"
        );
        assert_eq!(
            path.points[3].uv,
            pt(5.0, 5.0).uv,
            "the second keeps its head"
        );
        assert_eq!(
            path.origins,
            vec![
                SegmentOrigin::Source, // 0 -> 1
                SegmentOrigin::Source, // 1 -> 2
                SegmentOrigin::Seam,   // 2 -> 3, the bridge
                SegmentOrigin::Source, // 3 -> 4
                SegmentOrigin::Source, // 4 -> 5
            ],
        );
    }

    /// Closing on a shared endpoint drops the repeated point; the existing last
    /// segment becomes the cyclic wrap.
    #[test]
    fn closing_on_a_shared_endpoint_reuses_the_last_segment() {
        let mut path = BoundaryPath::start(vec![pt(0.0, 0.0), pt(1.0, 0.0)], SegmentOrigin::Source);
        path.append(
            vec![pt(1.0, 0.0), pt(0.0, 0.0)],
            SegmentOrigin::SyntheticClosure,
            PartJoin::SharedEndpoint,
        );
        let loop_ = path.close(PartJoin::SharedEndpoint);
        assert_eq!(loop_.points.len(), 2);
        assert_eq!(
            loop_.points.len(),
            loop_.origins.len(),
            "one origin per segment"
        );
        assert_eq!(
            loop_.origins,
            vec![SegmentOrigin::Source, SegmentOrigin::SyntheticClosure],
        );
    }

    /// Closing across a gap adds one labelled wrap segment and keeps every point.
    #[test]
    fn closing_across_a_gap_adds_a_labelled_wrap() {
        let path = BoundaryPath::start(
            vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(1.0, 1.0)],
            SegmentOrigin::Source,
        );
        let loop_ = path.close(PartJoin::Bridge(SegmentOrigin::SyntheticClosure));
        assert_eq!(loop_.points.len(), 3);
        assert_eq!(loop_.points.len(), loop_.origins.len());
        assert_eq!(
            *loop_.origins.last().unwrap(),
            SegmentOrigin::SyntheticClosure,
            "the wrap is not another source segment",
        );
    }

    /// A periodically closed walk retains its endpoint at `first + L·δ`, so the
    /// cyclic wrap is the deck closure and must not be labelled `Source`.
    #[test]
    fn a_periodic_walk_does_not_call_its_wrap_a_source_segment() {
        let walk =
            BoundaryLoop::periodic_source_walk(vec![pt(0.0, 0.0), pt(0.0, 1.0), pt(0.0, 2.0)]);
        assert_eq!(walk.points.len(), walk.origins.len());
        assert_eq!(
            walk.origins,
            vec![
                SegmentOrigin::Source,
                SegmentOrigin::Source,
                SegmentOrigin::Seam,
            ],
        );
    }

    /// A Euclidean loop has already had its duplicate endpoint removed, so
    /// every cyclic segment including the wrap is genuine source trim.
    #[test]
    fn a_euclidean_loop_wraps_on_a_source_segment() {
        let loop_ =
            BoundaryLoop::euclidean_source_loop(vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(1.0, 1.0)]);
        assert!(loop_.origins.iter().all(|o| *o == SegmentOrigin::Source));
        assert_eq!(loop_.points.len(), loop_.origins.len());
    }

    /// Every constructed loop must carry exactly one origin per segment.
    #[test]
    fn chained_parts_yield_one_origin_per_segment() {
        let loop_ = BoundaryLoop::chained([
            (vec![pt(0.0, 0.0), pt(1.0, 0.0)], SegmentOrigin::Source),
            (
                vec![pt(1.0, 0.0), pt(1.0, 1.0)],
                SegmentOrigin::SyntheticClosure,
            ),
            (vec![pt(1.0, 1.0), pt(0.0, 0.0)], SegmentOrigin::Seam),
        ]);
        assert_eq!(loop_.points.len(), loop_.origins.len());
        assert_eq!(loop_.points.len(), 3, "join duplicates are dropped");
    }
}
