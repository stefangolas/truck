//! DIAG-001: structured diagnostic records for failed-face geometry census.
//!
//! This module provides instrumentation that classifies what kind of geometry
//! is present in faces that fail tessellation. It observes and classifies
//! existing failures without repairing geometry, changing tessellation
//! decisions, or beginning ARR-003.
//!
//! # Governing correctness rule
//!
//! Approximation may guide computation only. Every classification derives from
//! evidence already established by the pipeline: typed failure reasons, exact
//! predicates, certified enclosures, authoritative source identity, segment
//! origins, and the exact foreign-library refusal contract.
//!
//! # When capture runs
//!
//! Capture is requested explicitly by `TRUCK_FACE_DIAG_JSONL`, and is also
//! turned on by any formal route that *consumes* a witness to decide what it
//! may attempt — the cylinder-band and conical-band routes admit exactly the
//! `SyntheticSyntheticCrossing` bucket, which is derived from these records.
//! Since `WAVE-2C` made those routes default-on, a default run collects
//! witnesses; `TRUCK_FORMAL_RECOVERY=0` turns the routes and the capture off
//! together. See [`diag_enabled`].

use serde::Serialize;

use super::triangulation::{ConstraintRole, SegmentOrigin, TessellationFailureReason};

// ---------------------------------------------------------------------------
// Origin classification
// ---------------------------------------------------------------------------

/// A stable, coarse classification of a segment's origin for aggregation.
///
/// Derived from [`SegmentOrigin`] without erasing the detailed origin from the
/// raw record. "Authoritative source trim" means geometry directly supported
/// by STEP boundary evidence. Synthetic closure, seam links, and other
/// generated geometry remain separate classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum OriginClass {
    /// Geometry directly supported by STEP boundary evidence.
    AuthoritativeSourceTrim,
    /// Synthesised to close an open piece against the working extent.
    SyntheticClosure,
    /// A seam or periodization bridge across a degenerate direction.
    SeamOrPeriodization,
    /// Reconstructed or derived geometry not directly from source trim.
    ReconstructedOrDerived,
    /// The origin is not established.
    Unknown,
}

impl From<SegmentOrigin> for OriginClass {
    fn from(origin: SegmentOrigin) -> Self {
        match origin {
            SegmentOrigin::Source => Self::AuthoritativeSourceTrim,
            SegmentOrigin::SyntheticClosure => Self::SyntheticClosure,
            SegmentOrigin::Seam => Self::SeamOrPeriodization,
        }
    }
}

impl OriginClass {
    /// Whether this origin is an authoritative source trim.
    pub fn is_authoritative_source(self) -> bool {
        matches!(self, Self::AuthoritativeSourceTrim)
    }

    /// Whether this origin is synthetic (not authoritative source trim).
    pub fn is_synthetic(self) -> bool {
        !self.is_authoritative_source()
    }
}

// ---------------------------------------------------------------------------
// Surface and chart classification
// ---------------------------------------------------------------------------

/// Which axes of a surface's parameter chart are periodic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct PeriodicAxes {
    /// Whether the `u` axis is periodic.
    pub u: bool,
    /// Whether the `v` axis is periodic.
    pub v: bool,
}

/// A coarse family for the support surface, for aggregation.
///
/// Filled by the corpus runner from the STEP surface enum; `Unknown` when
/// produced inside the tessellation pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum SurfaceFamily {
    /// A plane.
    Plane,
    /// A cylinder.
    Cylinder,
    /// A cone.
    Cone,
    /// A sphere.
    Sphere,
    /// A torus.
    Torus,
    /// An extruded surface.
    Extruded,
    /// A surface of revolution.
    Revolved,
    /// A B-spline surface.
    Bspline,
    /// A NURBS surface.
    Nurbs,
    /// An offset surface.
    Offset,
    /// The surface family is not established.
    Unknown,
}

// ---------------------------------------------------------------------------
// Observed statuses
// ---------------------------------------------------------------------------

/// What the pipeline established about the lift, observed without
/// reconstruction.
///
/// Only `Certified` or `Ambiguous` are emitted when the existing pipeline
/// establishes it. A periodic surface with no retained lift evidence is
/// `Unavailable`, not a guessed bad lift.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ObservedLiftStatus {
    /// The surface is not periodic; no lift is needed.
    NotPeriodic,
    /// The lattice certified the periods with representation-derived evidence.
    Certified,
    /// The periodic branch of a lift step could not be resolved.
    Ambiguous,
    /// A periodic surface with no retained lift evidence.
    Unavailable,
}

/// What the pipeline established about the deck, observed without
/// reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ObservedDeckStatus {
    /// The chart rank is zero; no deck is needed.
    Rank0,
    /// The deck certified a relative placement.
    CertifiedRelativePlacement,
    /// A free gauge is present.
    FreeGaugePresent,
    /// The deck found a contradiction.
    Contradiction,
    /// The deck status is not established.
    Unavailable,
}

/// What the pipeline established about boundary projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ObservedProjectionStatus {
    /// Projection succeeded; the failure occurred later in the pipeline.
    Successful,
    /// A typed projection failure occurred.
    FailedTyped,
    /// Projection was not reached.
    Unavailable,
}

// ---------------------------------------------------------------------------
// Presented segment relation
// ---------------------------------------------------------------------------

/// How a presented constraint segment relates to the blocking segment.
///
/// Initially only relations that can be established by existing exact
/// predicates or by the foreign library's documented refusal contract are
/// supported. Relations requiring a new approximate tolerance test are left
/// `Unknown`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum PresentedSegmentRelation {
    /// The segment properly crosses an existing constraint in its interior.
    ///
    /// Established by Spade's `try_add_constraint` returning an empty chain,
    /// which its contract defines as a proper interior crossing refusal.
    ProperInteriorCrossing,
    /// An endpoint of the segment lies on the interior of an existing
    /// constraint. Not yet distinguished from `ProperInteriorCrossing`.
    EndpointOnInterior,
    /// The segment is collinear with and overlaps an existing constraint.
    CollinearOverlap,
    /// The segment traverses the same edge as an existing constraint.
    DuplicateTraversal,
    /// A vertex could not be inserted into the triangulation.
    VertexInsertionFailure,
    /// The relation cannot be established by existing evidence.
    Unknown,
}

// ---------------------------------------------------------------------------
// Semantic segment reference and witnesses
// ---------------------------------------------------------------------------

/// A reference to one semantic constraint segment.
///
/// The `origin` field retains the detailed [`SegmentOrigin`]; [`OriginClass`]
/// is derived separately for aggregation and does not erase it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SemanticSegmentRef {
    /// A sequential identifier unique within one face's diagnosis.
    pub semantic_constraint_id: u64,
    /// The detailed segment origin, retained without erasure.
    pub origin: SegmentOrigin,
    /// The boundary component (loop/piece index) the segment belongs to.
    pub boundary_component: Option<usize>,
    /// The index of the segment within its boundary component.
    pub segment_index: u32,
    /// The source bound identifier, when available.
    pub source_bound: Option<usize>,
    /// The source edge use identifier, when available.
    pub source_edge_use: Option<usize>,
}

/// A compact 2D parameter-space enclosure.
///
/// Two intervals, not large geometry objects.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ParameterEnclosure2 {
    /// The `u` interval `[lo, hi]`.
    pub u: (f64, f64),
    /// The `v` interval `[lo, hi]`.
    pub v: (f64, f64),
}

/// One witnessed conflict during constraint insertion.
///
/// For every `ConstraintInsertionIncomplete` failure, at least one witness (or
/// an explicit `InsertionUnknown` bucket) is produced. Both segment origins
/// and source identifiers are retained where available.
#[derive(Clone, Debug, Serialize)]
pub struct ConstraintConflictWitness {
    /// The segment being inserted that was refused.
    pub incoming: SemanticSegmentRef,
    /// The existing segment that blocked the insertion.
    pub blocking: SemanticSegmentRef,
    /// The established relation between the segments.
    pub relation: PresentedSegmentRelation,
    /// Whether both segments belong to the same bound, when established.
    pub same_bound: Option<bool>,
    /// Whether both segments share the same source edge use, when established.
    pub same_source_edge_use: Option<bool>,
    /// A certified enclosure of the intersection point, when computed.
    pub intersection_enclosure: Option<ParameterEnclosure2>,
}

/// Metadata retained on each realized foreign constraint chain edge.
///
/// The triangulation library may realize one requested semantic constraint as
/// a chain of foreign edges. This preserves semantic metadata on every
/// realized chain edge, so every returned edge from `try_add_constraint` maps
/// to the semantic segment that requested it.
#[derive(Clone, Debug, Serialize)]
pub struct RealizedConstraintMetadata {
    /// The constraint role assigned to this edge.
    pub role: ConstraintRole,
    /// The semantic segment that requested this realized edge.
    pub semantic_segment: SemanticSegmentRef,
}

// ---------------------------------------------------------------------------
// Loss bucket
// ---------------------------------------------------------------------------

/// The deterministic loss bucket derived from the raw evidence.
///
/// These names describe presented geometry and terminal behaviour. They do
/// not claim the upstream STEP interpretation is correct.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum LossBucket {
    /// A typed boundary projection failure.
    ProjectionFailure,
    /// An ambiguous periodic lift.
    LiftAmbiguous,
    /// A constraint overlap that the envelope does not admit.
    UnsupportedOverlap,
    /// A parity contradiction around a cycle.
    ParityContradiction,
    /// No material region was selected by the parity flood.
    NoMaterialRegion,
    /// All witnessed conflicts are source/source within one bound.
    SourceSourceSameBoundCrossing,
    /// All witnessed conflicts are source/source between different bounds.
    SourceSourceInterBoundCrossing,
    /// At least one source/synthetic conflict, no synthetic/synthetic.
    SourceSyntheticCrossing,
    /// All witnessed conflicts are synthetic/synthetic.
    SyntheticSyntheticCrossing,
    /// Only vertex-insertion failures are observed.
    VertexInsertionFailure,
    /// Materially different conflict classes occur.
    MixedConstraintConflict,
    /// No usable witness exists.
    InsertionUnknown,
    /// A typed failure not in the above categories.
    OtherTypedFailure,
}

// ---------------------------------------------------------------------------
// Failed face diagnosis
// ---------------------------------------------------------------------------

/// One structured record per failed face.
///
/// The raw evidence fields are more important than the final bucket.
/// `model_id` and `surface_family` are filled by the corpus runner; other
/// fields are produced inside the tessellation pipeline.
#[derive(Clone, Debug, Serialize)]
pub struct FailedFaceDiagnosis {
    /// The model identifier (STEP file name), filled by the corpus runner.
    pub model_id: String,
    /// The document-local source face entity id, when available.
    pub source_face_id: Option<u64>,
    /// The typed terminal failure reason.
    pub terminal_reason: TessellationFailureReason,
    /// The coarse surface family, filled by the corpus runner.
    pub surface_family: SurfaceFamily,
    /// The parameter-space chart rank (0, 1, or 2).
    pub chart_rank: u8,
    /// Which axes are periodic.
    pub periodic_axes: PeriodicAxes,
    /// The number of bounds on the face.
    pub bound_count: usize,
    /// The number of authoritative-source-trim segments.
    pub source_segment_count: usize,
    /// The number of synthetic (closure + seam) segments.
    pub synthetic_segment_count: usize,
    /// The observed lift status.
    pub lift_status: ObservedLiftStatus,
    /// The observed deck status.
    pub deck_status: ObservedDeckStatus,
    /// The observed projection status.
    pub projection_status: ObservedProjectionStatus,
    /// The structured conflict witnesses, when the failure is an insertion
    /// failure.
    pub insertion_conflicts: Vec<ConstraintConflictWitness>,
    /// The deterministic loss bucket.
    pub derived_bucket: LossBucket,
}

// ---------------------------------------------------------------------------
// Derivation functions
// ---------------------------------------------------------------------------

/// Derive the loss bucket from the raw evidence.
///
/// Follows the documented priority order. Existing typed terminal failures
/// are not replaced by a weaker insertion-derived bucket.
pub fn derive_loss_bucket(
    reason: TessellationFailureReason,
    witnesses: &[ConstraintConflictWitness],
    vertex_insertion_failed: bool,
) -> LossBucket {
    use TessellationFailureReason as R;
    match reason {
        R::BoundaryProjectionFailed | R::BoundaryPointOffSurface => LossBucket::ProjectionFailure,
        R::AmbiguousLift => LossBucket::LiftAmbiguous,
        R::ConstraintOverlapUnsupported => LossBucket::UnsupportedOverlap,
        R::ContradictoryDualParity => LossBucket::ParityContradiction,
        R::NoOddParityRegion => LossBucket::NoMaterialRegion,
        R::ConstraintInsertionIncomplete => {
            derive_insertion_bucket(witnesses, vertex_insertion_failed)
        }
        _ => LossBucket::OtherTypedFailure,
    }
}

/// Derive the insertion-specific bucket from witnesses.
fn derive_insertion_bucket(
    witnesses: &[ConstraintConflictWitness],
    vertex_insertion_failed: bool,
) -> LossBucket {
    if witnesses.is_empty() {
        return if vertex_insertion_failed {
            LossBucket::VertexInsertionFailure
        } else {
            LossBucket::InsertionUnknown
        };
    }
    let origin_of = |seg: &SemanticSegmentRef| OriginClass::from(seg.origin);
    // All source/source?
    let all_source_source = witnesses.iter().all(|w| {
        origin_of(&w.incoming).is_authoritative_source()
            && origin_of(&w.blocking).is_authoritative_source()
    });
    if all_source_source {
        let all_same_bound = witnesses.iter().all(|w| w.same_bound == Some(true));
        return if all_same_bound {
            LossBucket::SourceSourceSameBoundCrossing
        } else {
            LossBucket::SourceSourceInterBoundCrossing
        };
    }
    // All synthetic/synthetic?
    let all_synthetic_synthetic = witnesses
        .iter()
        .all(|w| origin_of(&w.incoming).is_synthetic() && origin_of(&w.blocking).is_synthetic());
    if all_synthetic_synthetic {
        return LossBucket::SyntheticSyntheticCrossing;
    }
    // At least one source/synthetic and no synthetic/synthetic?
    let has_synthetic_synthetic = witnesses
        .iter()
        .any(|w| origin_of(&w.incoming).is_synthetic() && origin_of(&w.blocking).is_synthetic());
    let has_source_synthetic = witnesses.iter().any(|w| {
        (origin_of(&w.incoming).is_authoritative_source() && origin_of(&w.blocking).is_synthetic())
            || (origin_of(&w.incoming).is_synthetic()
                && origin_of(&w.blocking).is_authoritative_source())
    });
    if has_source_synthetic && !has_synthetic_synthetic {
        return LossBucket::SourceSyntheticCrossing;
    }
    LossBucket::MixedConstraintConflict
}

/// Derive the projection status from the terminal reason.
pub fn derive_projection_status(reason: TessellationFailureReason) -> ObservedProjectionStatus {
    use TessellationFailureReason as R;
    match reason {
        R::BoundaryProjectionFailed | R::BoundaryPointOffSurface => {
            ObservedProjectionStatus::FailedTyped
        }
        R::BoundaryWireEmpty | R::BoundaryConstructionFailed => {
            ObservedProjectionStatus::Unavailable
        }
        _ => ObservedProjectionStatus::Successful,
    }
}

/// Derive the lift status from the periodic axes, failure reason, and
/// whether the lattice certified all periodic axes.
pub fn compute_lift_status(
    periodic_axes: PeriodicAxes,
    reason: TessellationFailureReason,
    all_periods_certified: bool,
) -> ObservedLiftStatus {
    if !periodic_axes.u && !periodic_axes.v {
        return ObservedLiftStatus::NotPeriodic;
    }
    if reason == TessellationFailureReason::AmbiguousLift {
        return ObservedLiftStatus::Ambiguous;
    }
    if all_periods_certified {
        ObservedLiftStatus::Certified
    } else {
        ObservedLiftStatus::Unavailable
    }
}

/// Derive the deck status from the chart rank.
pub fn compute_deck_status(chart_rank: u8) -> ObservedDeckStatus {
    if chart_rank == 0 {
        ObservedDeckStatus::Rank0
    } else {
        // The deck computation happens inside the pipeline but its status is
        // not surfaced as a separate typed result today. Do not guess.
        ObservedDeckStatus::Unavailable
    }
}

// ---------------------------------------------------------------------------
// Diagnostic sink (thread-local)
// ---------------------------------------------------------------------------

/// Whether a formal recovery route is enabled, by its environment variable.
///
/// **Default-on with explicit opt-out** (`WAVE-2C`). Every formal recovery
/// route shipped off-by-default while it was being proven, so that production
/// output was unchanged by construction and each route's contribution could be
/// read as the delta between a run with its variable set and a run without.
/// That discipline did its job: the routes are certified, validated, and
/// refinement-only — each one replaces a mesh only where `failure.is_some()`,
/// so it cannot un-render a face the legacy path already meshed. Shipping them
/// switched off therefore no longer buys safety; it only withholds recovery
/// that has already been proven.
///
/// The measurement property is preserved rather than lost. A route's own
/// contribution is still one subtraction — set its variable to `0` and diff
/// against the default — so the per-route population stays readable without
/// any route having to stand outside the master gate to be measurable. That
/// was the sole reason `_BAND` and `_TORUS` were not nested under
/// `TRUCK_FORMAL_RECOVERY`, and it no longer applies, so they are nested now
/// and the master variable is a single kill switch for the whole formal
/// chain.
///
/// Only an explicit negative disables: `0`, `off`, `false`, or `no`, in any
/// case. An unset variable, or one set to anything else, leaves the route on.
pub fn recovery_route_enabled(variable: &str) -> bool {
    match std::env::var(variable) {
        Err(_) => true,
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
    }
}

/// Whether the whole formal recovery chain is enabled.
///
/// The master kill switch. With `TRUCK_FORMAL_RECOVERY=0` every formal route
/// is off and the tessellator produces exactly the legacy meshes, which is the
/// bisect and rollback path.
pub fn formal_recovery_enabled() -> bool {
    recovery_route_enabled("TRUCK_FORMAL_RECOVERY")
}

/// Whether the formal cylinder-band fallback is active.
///
/// Nested under the master gate — see [`recovery_route_enabled`] for why it no
/// longer stands outside it.
///
/// It lives in this module, rather than beside the other gates, because
/// [`diag_enabled`] has to agree with it — see there.
pub fn cylinder_band_recovery_enabled() -> bool {
    formal_recovery_enabled() && recovery_route_enabled("TRUCK_FORMAL_RECOVERY_BAND")
}

/// Whether diagnostic capture is enabled.
///
/// Enabled by `TRUCK_FACE_DIAG_JSONL`, and independently by the cylinder-band
/// fallback — not as a convenience: that route's admitted population *is* the
/// `SyntheticSyntheticCrossing` bucket, and the bucket is derived from these
/// witnesses. The sink is an input to a production decision, not only a
/// report, so it must be filled whether or not anyone asked for the JSONL.
///
/// **Since the band route became default-on (`WAVE-2C`), this is true on a
/// default run**, where it used to be false. That is a deliberate cost, and it
/// is the price of the band and cone routes' admission rule rather than a
/// diagnostic left switched on by accident: without the witnesses there is no
/// certified way to tell a seam/seam crossing from any other insertion
/// failure, and the routes would have to attempt every lost face instead of
/// the population they are proven for. `TRUCK_FORMAL_RECOVERY=0` turns it back
/// off along with the routes that need it.
pub fn diag_enabled() -> bool {
    std::env::var_os("TRUCK_FACE_DIAG_JSONL").is_some() || cylinder_band_recovery_enabled()
}

/// The thread-local diagnostic sink.
///
/// Accumulates evidence during `insert_to` and is read by the
/// `tessellate_face` closure after each face. Cleared before each face.
#[derive(Default)]
struct DiagnosisSink {
    segments: Vec<SemanticSegmentRef>,
    witnesses: Vec<ConstraintConflictWitness>,
    realized_chain: Vec<RealizedConstraintMetadata>,
    vertex_insertion_failed: bool,
    source_segment_count: usize,
    synthetic_segment_count: usize,
}

impl DiagnosisSink {
    fn clear(&mut self) {
        self.segments.clear();
        self.witnesses.clear();
        self.realized_chain.clear();
        self.vertex_insertion_failed = false;
        self.source_segment_count = 0;
        self.synthetic_segment_count = 0;
    }
}

std::thread_local! {
    static FACE_DIAGNOSIS_SINK: std::cell::RefCell<DiagnosisSink> =
        std::cell::RefCell::new(DiagnosisSink::default());
}

/// Clear the diagnostic sink before a new face.
pub(crate) fn clear_sink() {
    FACE_DIAGNOSIS_SINK.with(|s| s.borrow_mut().clear());
}

/// Record a new semantic segment and return its assigned id.
///
/// Called from `insert_to` for each boundary segment when diagnostics are
/// enabled.
pub(crate) fn record_segment(
    origin: SegmentOrigin,
    boundary_component: Option<usize>,
    segment_index: u32,
) -> u64 {
    FACE_DIAGNOSIS_SINK.with(|sink| {
        let sink = &mut *sink.borrow_mut();
        let id = sink.segments.len() as u64;
        match origin {
            SegmentOrigin::Source => sink.source_segment_count += 1,
            _ => sink.synthetic_segment_count += 1,
        }
        sink.segments.push(SemanticSegmentRef {
            semantic_constraint_id: id,
            origin,
            boundary_component,
            segment_index,
            source_bound: None,
            source_edge_use: None,
        });
        id
    })
}

/// Record a realized constraint chain edge, mapping it to the semantic
/// segment that requested it.
pub(crate) fn record_realized_edge(role: ConstraintRole, semantic_segment_id: u64) {
    FACE_DIAGNOSIS_SINK.with(|sink| {
        let sink = &mut *sink.borrow_mut();
        let semantic_segment = sink
            .segments
            .get(semantic_segment_id as usize)
            .cloned()
            .expect("semantic segment id must be valid");
        sink.realized_chain.push(RealizedConstraintMetadata {
            role,
            semantic_segment,
        });
    });
}

/// Record a conflict witness between two semantic segments.
pub(crate) fn record_conflict(
    incoming_id: u64,
    blocking_id: u64,
    relation: PresentedSegmentRelation,
) {
    FACE_DIAGNOSIS_SINK.with(|sink| {
        let sink = &mut *sink.borrow_mut();
        let incoming = sink
            .segments
            .get(incoming_id as usize)
            .cloned()
            .expect("incoming segment id must be valid");
        let blocking = sink
            .segments
            .get(blocking_id as usize)
            .cloned()
            .expect("blocking segment id must be valid");
        let same_bound = match (incoming.boundary_component, blocking.boundary_component) {
            (Some(a), Some(b)) => Some(a == b),
            _ => None,
        };
        sink.witnesses.push(ConstraintConflictWitness {
            incoming,
            blocking,
            relation,
            same_bound,
            same_source_edge_use: None,
            intersection_enclosure: None,
        });
    });
}

/// Record that a vertex insertion failed.
pub(crate) fn set_vertex_insertion_failed() {
    FACE_DIAGNOSIS_SINK.with(|sink| {
        sink.borrow_mut().vertex_insertion_failed = true;
    });
}

/// Snapshot of the realized constraint chain, for testing.
///
/// Verifies that every returned chain edge maps to the semantic segment that
/// requested it.
#[cfg(test)]
pub(crate) fn realized_chain_snapshot() -> Vec<RealizedConstraintMetadata> {
    FACE_DIAGNOSIS_SINK.with(|sink| sink.borrow().realized_chain.clone())
}

/// The loss bucket the sink's current evidence derives, without consuming it.
///
/// [`build_face_diagnosis`] takes the witnesses and clears the sink, which is
/// correct for a record built once at the end of a face. A consumer that has to
/// *classify the legacy failure while the face is still being worked on* — the
/// cylinder-band fallback — needs the same derivation earlier and must leave
/// the sink intact for the record that still follows. Same
/// [`derive_loss_bucket`], same inputs; only the borrow differs.
pub(crate) fn derived_bucket(terminal_reason: TessellationFailureReason) -> LossBucket {
    FACE_DIAGNOSIS_SINK.with(|sink| {
        let sink = sink.borrow();
        derive_loss_bucket(
            terminal_reason,
            &sink.witnesses,
            sink.vertex_insertion_failed,
        )
    })
}

/// Build the per-face diagnosis from the sink.
///
/// Reads and clears the sink. The `model_id` and `surface_family` fields are
/// filled by the corpus runner; here they default to empty and `Unknown`.
pub(crate) fn build_face_diagnosis(
    source_face_id: Option<u64>,
    terminal_reason: TessellationFailureReason,
    chart_rank: u8,
    periodic_axes: PeriodicAxes,
    bound_count: usize,
    lift_status: ObservedLiftStatus,
    deck_status: ObservedDeckStatus,
) -> FailedFaceDiagnosis {
    FACE_DIAGNOSIS_SINK.with(|sink| {
        let mut sink = sink.borrow_mut();
        let vertex_insertion_failed = sink.vertex_insertion_failed;
        let witnesses = std::mem::take(&mut sink.witnesses);
        let source_segment_count = sink.source_segment_count;
        let synthetic_segment_count = sink.synthetic_segment_count;
        sink.clear();
        let derived_bucket =
            derive_loss_bucket(terminal_reason, &witnesses, vertex_insertion_failed);
        let projection_status = derive_projection_status(terminal_reason);
        FailedFaceDiagnosis {
            model_id: String::new(),
            source_face_id,
            terminal_reason,
            surface_family: SurfaceFamily::Unknown,
            chart_rank,
            periodic_axes,
            bound_count,
            source_segment_count,
            synthetic_segment_count,
            lift_status,
            deck_status,
            projection_status,
            insertion_conflicts: witnesses,
            derived_bucket,
        }
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn seg_ref(id: u64, origin: SegmentOrigin, bound: Option<usize>) -> SemanticSegmentRef {
        SemanticSegmentRef {
            semantic_constraint_id: id,
            origin,
            boundary_component: bound,
            segment_index: 0,
            source_bound: None,
            source_edge_use: None,
        }
    }

    fn witness(
        incoming: SemanticSegmentRef,
        blocking: SemanticSegmentRef,
        same_bound: Option<bool>,
    ) -> ConstraintConflictWitness {
        ConstraintConflictWitness {
            incoming,
            blocking,
            relation: PresentedSegmentRelation::ProperInteriorCrossing,
            same_bound,
            same_source_edge_use: None,
            intersection_enclosure: None,
        }
    }

    // Test 1: source/source crossing within one bound.
    #[test]
    fn source_source_same_bound_crossing() {
        let incoming = seg_ref(0, SegmentOrigin::Source, Some(0));
        let blocking = seg_ref(1, SegmentOrigin::Source, Some(0));
        let witnesses = vec![witness(incoming, blocking, Some(true))];
        let bucket = derive_loss_bucket(
            TessellationFailureReason::ConstraintInsertionIncomplete,
            &witnesses,
            false,
        );
        assert_eq!(bucket, LossBucket::SourceSourceSameBoundCrossing);
    }

    // Test 2: source/source crossing between bounds.
    #[test]
    fn source_source_inter_bound_crossing() {
        let incoming = seg_ref(0, SegmentOrigin::Source, Some(0));
        let blocking = seg_ref(1, SegmentOrigin::Source, Some(1));
        let witnesses = vec![witness(incoming, blocking, Some(false))];
        let bucket = derive_loss_bucket(
            TessellationFailureReason::ConstraintInsertionIncomplete,
            &witnesses,
            false,
        );
        assert_eq!(bucket, LossBucket::SourceSourceInterBoundCrossing);
    }

    // Test 3: source/synthetic crossing.
    #[test]
    fn source_synthetic_crossing() {
        let incoming = seg_ref(0, SegmentOrigin::Source, Some(0));
        let blocking = seg_ref(1, SegmentOrigin::SyntheticClosure, Some(0));
        let witnesses = vec![witness(incoming, blocking, Some(true))];
        let bucket = derive_loss_bucket(
            TessellationFailureReason::ConstraintInsertionIncomplete,
            &witnesses,
            false,
        );
        assert_eq!(bucket, LossBucket::SourceSyntheticCrossing);
    }

    // Test 4: synthetic/synthetic crossing.
    #[test]
    fn synthetic_synthetic_crossing() {
        let incoming = seg_ref(0, SegmentOrigin::SyntheticClosure, Some(0));
        let blocking = seg_ref(1, SegmentOrigin::Seam, Some(0));
        let witnesses = vec![witness(incoming, blocking, Some(true))];
        let bucket = derive_loss_bucket(
            TessellationFailureReason::ConstraintInsertionIncomplete,
            &witnesses,
            false,
        );
        assert_eq!(bucket, LossBucket::SyntheticSyntheticCrossing);
    }

    // Test 6: multiple conflict classes producing MixedConstraintConflict.
    #[test]
    fn mixed_constraint_conflict() {
        let w1 = witness(
            seg_ref(0, SegmentOrigin::Source, Some(0)),
            seg_ref(1, SegmentOrigin::Source, Some(0)),
            Some(true),
        );
        let w2 = witness(
            seg_ref(2, SegmentOrigin::SyntheticClosure, Some(0)),
            seg_ref(3, SegmentOrigin::SyntheticClosure, Some(0)),
            Some(true),
        );
        let witnesses = vec![w1, w2];
        let bucket = derive_loss_bucket(
            TessellationFailureReason::ConstraintInsertionIncomplete,
            &witnesses,
            false,
        );
        assert_eq!(bucket, LossBucket::MixedConstraintConflict);
    }

    // Test 7: unavailable conflict witness producing InsertionUnknown.
    #[test]
    fn insertion_unknown_when_no_witness() {
        let bucket = derive_loss_bucket(
            TessellationFailureReason::ConstraintInsertionIncomplete,
            &[],
            false,
        );
        assert_eq!(bucket, LossBucket::InsertionUnknown);
    }

    // Test 8: vertex insertion failure.
    #[test]
    fn vertex_insertion_failure_bucket() {
        let bucket = derive_loss_bucket(
            TessellationFailureReason::ConstraintInsertionIncomplete,
            &[],
            true,
        );
        assert_eq!(bucket, LossBucket::VertexInsertionFailure);
    }

    // Test 9: periodic face with unavailable lift data remaining Unavailable.
    #[test]
    fn periodic_unavailable_lift() {
        let status = compute_lift_status(
            PeriodicAxes { u: true, v: false },
            TessellationFailureReason::ConstraintInsertionIncomplete,
            false,
        );
        assert_eq!(status, ObservedLiftStatus::Unavailable);
    }

    // Test 10: deterministic serialization and aggregation ordering.
    #[test]
    fn deterministic_serialization() {
        let diag = FailedFaceDiagnosis {
            model_id: "model.step".into(),
            source_face_id: Some(42),
            terminal_reason: TessellationFailureReason::ConstraintInsertionIncomplete,
            surface_family: SurfaceFamily::Cylinder,
            chart_rank: 1,
            periodic_axes: PeriodicAxes { u: false, v: true },
            bound_count: 1,
            source_segment_count: 4,
            synthetic_segment_count: 1,
            lift_status: ObservedLiftStatus::Unavailable,
            deck_status: ObservedDeckStatus::Unavailable,
            projection_status: ObservedProjectionStatus::Successful,
            insertion_conflicts: vec![witness(
                seg_ref(0, SegmentOrigin::Source, Some(0)),
                seg_ref(1, SegmentOrigin::Source, Some(0)),
                Some(true),
            )],
            derived_bucket: LossBucket::SourceSourceSameBoundCrossing,
        };
        let json1 = serde_json::to_string(&diag).unwrap();
        let json2 = serde_json::to_string(&diag).unwrap();
        assert_eq!(json1, json2, "serialization must be deterministic");
        // Re-deserializing the bucket field name confirms stable output.
        let v: serde_json::Value = serde_json::from_str(&json1).unwrap();
        assert_eq!(v["derived_bucket"], "SourceSourceSameBoundCrossing");
        assert_eq!(v["terminal_reason"], "ConstraintInsertionIncomplete");
    }

    // Test 11: typed non-insertion failures retaining their original categories.
    #[test]
    fn typed_failures_retain_categories() {
        assert_eq!(
            derive_loss_bucket(
                TessellationFailureReason::BoundaryProjectionFailed,
                &[],
                false
            ),
            LossBucket::ProjectionFailure,
        );
        assert_eq!(
            derive_loss_bucket(TessellationFailureReason::AmbiguousLift, &[], false),
            LossBucket::LiftAmbiguous,
        );
        assert_eq!(
            derive_loss_bucket(
                TessellationFailureReason::ConstraintOverlapUnsupported,
                &[],
                false
            ),
            LossBucket::UnsupportedOverlap,
        );
        assert_eq!(
            derive_loss_bucket(
                TessellationFailureReason::ContradictoryDualParity,
                &[],
                false
            ),
            LossBucket::ParityContradiction,
        );
        assert_eq!(
            derive_loss_bucket(TessellationFailureReason::NoOddParityRegion, &[], false),
            LossBucket::NoMaterialRegion,
        );
        assert_eq!(
            derive_loss_bucket(TessellationFailureReason::BoundaryWireEmpty, &[], false),
            LossBucket::OtherTypedFailure,
        );
    }

    // Test 12: the diagnostic sink follows the routes that consume it.
    //
    // This used to assert that an unset `TRUCK_FACE_DIAG_JSONL` left the sink
    // off, which was the whole no-observable-change argument while the formal
    // routes were opt-in. `WAVE-2C` made the cylinder-band route default-on,
    // and that route's admission rule *is* the derived witness bucket, so the
    // sink is now filled on a default run by construction. The invariant that
    // replaces it is the one that still carries weight: the sink is on exactly
    // when something needs it, and the master kill switch turns both off
    // together.
    #[test]
    fn diag_follows_the_routes_that_consume_it() {
        // These tests share a process, so the variables are set explicitly
        // rather than assumed absent.
        std::env::remove_var("TRUCK_FACE_DIAG_JSONL");
        std::env::remove_var("TRUCK_FORMAL_RECOVERY_BAND");
        std::env::remove_var("TRUCK_FORMAL_RECOVERY");
        // Default: the band route is on, so its witnesses are collected.
        assert!(cylinder_band_recovery_enabled());
        assert!(diag_enabled());

        // Master kill switch: no route needs the sink, so it is off again and
        // the legacy no-observable-change property is recovered in full.
        std::env::set_var("TRUCK_FORMAL_RECOVERY", "0");
        assert!(!cylinder_band_recovery_enabled());
        assert!(!diag_enabled());

        // ...but an explicit request for the JSONL still turns it on, with
        // every route switched off.
        std::env::set_var("TRUCK_FACE_DIAG_JSONL", "diag.jsonl");
        assert!(diag_enabled());

        std::env::remove_var("TRUCK_FACE_DIAG_JSONL");
        std::env::remove_var("TRUCK_FORMAL_RECOVERY");
    }

    /// Only an explicit negative disables a route; anything else leaves it on.
    #[test]
    fn recovery_route_opt_out_is_explicit() {
        const VAR: &str = "TRUCK_TEST_ROUTE_GATE";
        std::env::remove_var(VAR);
        assert!(recovery_route_enabled(VAR), "unset means on");
        for off in ["0", "off", "false", "no", "OFF", "False", " no "] {
            std::env::set_var(VAR, off);
            assert!(!recovery_route_enabled(VAR), "{off:?} must disable");
        }
        for on in ["1", "yes", "on", "true", ""] {
            std::env::set_var(VAR, on);
            assert!(recovery_route_enabled(VAR), "{on:?} must not disable");
        }
        std::env::remove_var(VAR);
    }

    // Test 5: realized constraint chain preserving one semantic segment identity.
    #[test]
    fn realized_chain_preserves_semantic_identity() {
        clear_sink();
        let id = record_segment(SegmentOrigin::Source, Some(0), 3);
        // A chain of three realized edges, all from the same semantic segment.
        record_realized_edge(ConstraintRole::PhysicalBoundary, id);
        record_realized_edge(ConstraintRole::PhysicalBoundary, id);
        record_realized_edge(ConstraintRole::PhysicalBoundary, id);
        let chain = realized_chain_snapshot();
        assert_eq!(chain.len(), 3, "three edges realized");
        assert!(
            chain
                .iter()
                .all(|m| m.semantic_segment.semantic_constraint_id == id),
            "every chain edge maps to the requesting semantic segment"
        );
        clear_sink();
    }

    /// A source/source same-bound + source/synthetic mix with no synth/synth
    /// is SourceSynthetic (not Mixed), per the priority order: rule 3 applies
    /// when at least one source/synthetic pair exists and no synth/synth.
    #[test]
    fn source_source_and_source_synthetic_mixed_is_source_synthetic() {
        let w1 = witness(
            seg_ref(0, SegmentOrigin::Source, Some(0)),
            seg_ref(1, SegmentOrigin::Source, Some(0)),
            Some(true),
        );
        let w2 = witness(
            seg_ref(2, SegmentOrigin::Source, Some(0)),
            seg_ref(3, SegmentOrigin::SyntheticClosure, Some(0)),
            Some(true),
        );
        let bucket = derive_loss_bucket(
            TessellationFailureReason::ConstraintInsertionIncomplete,
            &[w1, w2],
            false,
        );
        assert_eq!(bucket, LossBucket::SourceSyntheticCrossing);
    }

    /// Projection status derivation for typed failures.
    #[test]
    fn projection_status_derivation() {
        assert_eq!(
            derive_projection_status(TessellationFailureReason::BoundaryProjectionFailed),
            ObservedProjectionStatus::FailedTyped,
        );
        assert_eq!(
            derive_projection_status(TessellationFailureReason::ConstraintInsertionIncomplete),
            ObservedProjectionStatus::Successful,
        );
        assert_eq!(
            derive_projection_status(TessellationFailureReason::BoundaryWireEmpty),
            ObservedProjectionStatus::Unavailable,
        );
    }

    /// Lift status: certified when all periodic axes are exact.
    #[test]
    fn lift_status_certified() {
        let status = compute_lift_status(
            PeriodicAxes { u: false, v: true },
            TessellationFailureReason::ConstraintInsertionIncomplete,
            true,
        );
        assert_eq!(status, ObservedLiftStatus::Certified);
    }

    /// Lift status: not periodic when no axes are periodic.
    #[test]
    fn lift_status_not_periodic() {
        let status = compute_lift_status(
            PeriodicAxes { u: false, v: false },
            TessellationFailureReason::ContradictoryDualParity,
            false,
        );
        assert_eq!(status, ObservedLiftStatus::NotPeriodic);
    }

    /// Lift status: ambiguous when the failure reason is AmbiguousLift.
    #[test]
    fn lift_status_ambiguous() {
        let status = compute_lift_status(
            PeriodicAxes { u: false, v: true },
            TessellationFailureReason::AmbiguousLift,
            true,
        );
        assert_eq!(status, ObservedLiftStatus::Ambiguous);
    }
}
