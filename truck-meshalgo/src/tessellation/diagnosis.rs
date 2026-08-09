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
    /// A chart-closure run of a periodic cap: meridian seam or pole line.
    ChartClosure,
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
            SegmentOrigin::ChartClosure => Self::ChartClosure,
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

/// The two parameter-space endpoints of one presented segment.
///
/// Recorded unconditionally at every conflict site, because the endpoints are
/// in hand there and are gone by the time the terminal enum is read. Kept
/// separate from [`ParameterEnclosure2`], which is reserved for a *certified*
/// enclosure of the intersection and stays opt-in: a bounding box of two
/// endpoints cannot say which endpoint is which, and the chord-versus-analytic
/// question needs the ordered pair.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SegmentEndpoints2 {
    /// The segment's start, in parameter space.
    pub a: (f64, f64),
    /// The segment's end, in parameter space.
    pub b: (f64, f64),
}

// ---------------------------------------------------------------------------
// Projection witness (PROJ-002)
// ---------------------------------------------------------------------------

/// Which start produced the best nearest solution for a failing point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum NearestRoute {
    /// The caller's previous-UV hint, or the hintless presearch start — the two
    /// starts production's own chain already uses.
    ProductionStart,
    /// A structural (knot-span) seed.
    StructuralSeed,
    /// No search produced a usable solution.
    None,
}

/// What the deep inverse probe found for one failing boundary point.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum PointVerdict {
    /// A start production already uses reaches `residual <= tol`. Production
    /// rejected the point only because Newton's convergence test was not met.
    ProductionMiss,
    /// Only a structural seed reaches `residual <= tol`.
    SeedBasinGap,
    /// A stable solution exists and its world residual genuinely exceeds `tol`.
    NearestTooFar,
    /// A solution within `tol` exists but lies outside the declared parameter
    /// range.
    DomainOrContractIssue,
    /// No search produced a credible nearest solution.
    NoInverseFound,
    /// A probe cap or non-finite numerics prevents classification.
    Inconclusive,
}

/// Why a within-tolerance candidate lies outside the declared parameter range
/// (PROJ-003 Stage C).
///
/// The face-level `DomainOrContractIssue` verdict is not one mechanism. A
/// candidate whose world residual is within the caller tolerance but whose UV
/// is outside the declared range can be: an equivalent representative of a
/// genuinely periodic surface, a point numerically epsilon outside a closed
/// boundary, a sign that the represented parameter range does not cover the
/// surface's actual geometry, or simply a parameter the surface contract does
/// not cover. The class is derived from the candidate, the declared range, and
/// the certified lattice; it is what decides whether a principled recovery
/// exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum DomainRecoveryClass {
    /// The candidate differs from an in-range representative by an integer
    /// number of *certified* surface periods on a certified periodic axis.
    PeriodicEquivalent,
    /// The candidate sits only microscopically outside a closed boundary of
    /// the declared range, within a scale-aware epsilon of it.
    BoundaryEpsilon,
    /// The candidate's residual is within tolerance at a parameter far outside
    /// the declared range, so the represented range and the surface's actual
    /// (geometric) extent disagree in a nontrivial way.
    RepresentationRangeMismatch,
    /// The candidate lies genuinely outside the declared domain and no
    /// principled transformation brings it back in.
    TrueOutOfDomain,
    /// The class could not be established.
    Unknown,
}

/// One probed failing boundary point's structured evidence (PROJ-002/PROJ-003).
///
/// Holds the best candidate the deep probe found — its UV and re-evaluable
/// residual — alongside the verdict the probe classified, so a downstream
/// analysis can inspect *where* the candidate landed (e.g. how far outside the
/// declared range a `DomainOrContractIssue` candidate sits) rather than only
/// its label.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ProjectionPointEvidence {
    /// The per-point verdict.
    pub verdict: PointVerdict,
    /// The route that produced the best candidate.
    pub route: NearestRoute,
    /// The best candidate's parameter.
    pub best_uv: (f64, f64),
    /// The best candidate's world residual.
    pub best_residual: f64,
    /// The domain class, when the point is a domain/contract issue.
    pub domain_class: Option<DomainRecoveryClass>,
}

/// The face-level projection witness, aggregated over its probed points.
#[derive(Clone, Debug, Serialize)]
pub struct ProjectionWitness {
    /// Boundary points the walk failed to project.
    pub failed_points: usize,
    /// Boundary points the walk presented.
    pub boundary_points: usize,
    /// Failing points the deep probe examined, after the per-face cap.
    pub probed_points: usize,
    /// Whether the per-face point cap truncated the probe.
    pub point_cap_hit: bool,
    /// Whether any probed point hit the per-point seed cap.
    pub seed_cap_hit: bool,
    /// Structural seeds the surface offered. Zero means the seed route cannot
    /// help this family at all.
    pub seeds_offered: usize,
    /// The caller's chord tolerance.
    pub tolerance: f64,
    /// Best world residual over every probed point's best solution.
    pub best_residual: Option<f64>,
    /// `best_residual / tolerance`.
    pub best_residual_over_tol: Option<f64>,
    /// The worst of the per-point best residuals — the point that would still
    /// fail if the best solution were admitted everywhere.
    pub worst_residual_over_tol: Option<f64>,
    /// Which start won on the point that decided the face verdict.
    pub winning_route: NearestRoute,
    /// Per-point verdict counts, in the face's own order of severity.
    pub point_verdicts: Vec<PointVerdict>,
    /// The structured evidence of each probed point, when the deep probe ran.
    pub point_evidence: Vec<ProjectionPointEvidence>,
    /// Searches that stopped on a singular Jacobian.
    pub degenerate_hits: usize,
    /// Searches that exhausted their trial budget.
    pub nonconvergent: usize,
    /// The face verdict, derived from the point verdicts.
    pub verdict: PointVerdict,
}

/// Derive the face verdict from its points.
///
/// The face is only recoverable if **every** failing point is: one bad point
/// fails the face just as thoroughly as all of them. So the face takes its
/// worst point's verdict, ordered by how much would have to change to fix it.
pub fn derive_face_verdict(points: &[PointVerdict]) -> PointVerdict {
    let severity = |v: &PointVerdict| match v {
        PointVerdict::ProductionMiss => 0,
        PointVerdict::SeedBasinGap => 1,
        PointVerdict::DomainOrContractIssue => 2,
        PointVerdict::NearestTooFar => 3,
        PointVerdict::NoInverseFound => 4,
        PointVerdict::Inconclusive => 5,
    };
    points
        .iter()
        .copied()
        .max_by_key(|v| severity(v))
        .unwrap_or(PointVerdict::Inconclusive)
}

// ---------------------------------------------------------------------------
// Route selection
// ---------------------------------------------------------------------------

/// A formal recovery route, named where the dispatcher considers it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum RecoveryRoute {
    /// The formal cylinder-band fallback.
    CylinderBand,
    /// The conical essential-band route.
    ConeBand,
    /// The torus annulus route.
    TorusAnnulus,
    /// The winding-parity retry.
    WindingParity,
}

/// Why a route function declined a face without attempting anything.
///
/// These are the bare `None` returns the cone investigation ran into. Exit 3 in
/// particular — the bound-count test — is exactly "a one-bound apex cone never
/// enters the two-bound conical-band route, and therefore falls into the
/// generic lift", which was invisible until it was typed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum RouteIneligible {
    /// The surface identifier refused: this is not the route's surface.
    SurfaceNotCertified,
    /// The source face input could not be read from the compressed face.
    SourceInputUnavailable,
    /// The face does not present exactly two authoritative bounds.
    BoundsNotTwoAuthoritative,
}

/// What became of one route on one face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum RouteOutcome {
    /// The route's environment gate is closed.
    GateClosed,
    /// The gate is open, but the dispatcher's precondition did not hold — the
    /// face already had a mesh, or its loss bucket is not the admitted one.
    PreconditionUnmet,
    /// The route was entered and declined the face before attempting anything.
    Ineligible(RouteIneligible),
    /// The route attempted the face and returned a typed exit.
    Refused,
    /// The route produced a validated mesh that replaced the legacy failure.
    Recovered,
}

/// What the dispatcher and one route did about one face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RouteDecisionRecord {
    /// Which route.
    pub route: RecoveryRoute,
    /// Whether the route's environment gate was open.
    pub gate_open: bool,
    /// What became of it.
    pub outcome: RouteOutcome,
    /// The typed exit's stable tag, when the route refused.
    pub refusal_tag: Option<&'static str>,
}

/// The six stage counts of the CDT and material pipeline.
///
/// Every one of these numbers is already computed and then discarded. The
/// motivating case is `NoOddParityRegion`, which is raised when the final
/// triangle vector is empty and therefore **conflates two different
/// failures**: parity selected no region at all, and parity selected a region
/// whose triangles were then all removed as degenerate or zero-area. Splitting
/// the count either side of that filter separates them.
///
/// The three CDT counts are `Option` because `insert_to` can refuse before the
/// triangulation is ever read, and "the stage was not reached" is a different
/// statement from "the stage produced zero".
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct CdtStageVector {
    /// Boundary points that reached a vertex handle in the triangulation.
    pub boundary_vertices: usize,
    /// Constraint segments presented to the triangulation.
    pub constraints_presented: usize,
    /// Constraint segments the triangulation realized, as a chain or directly.
    pub constraints_inserted: usize,
    /// Triangles in the raw CDT, before any material selection.
    pub raw_cdt_triangles: Option<usize>,
    /// Triangles the odd-parity material filter selected.
    pub material_selected: Option<usize>,
    /// Triangles surviving the degenerate and zero-area validation.
    pub final_valid: Option<usize>,
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
    /// The parameter-space endpoints of the segment being inserted.
    pub incoming_segment: Option<SegmentEndpoints2>,
    /// The parameter-space endpoints of the segment that blocked it.
    pub blocking_segment: Option<SegmentEndpoints2>,
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
// Deck evidence for periodic boundary pieces
// ---------------------------------------------------------------------------

/// How one boundary piece closed, recorded without erasing its displacement.
///
/// Mirrors `triangulation::BoundaryClosure`, which is not `Serialize` and
/// carries the displacement inside one variant. Splitting the kind from the
/// displacement lets a record be aggregated on either.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ObservedClosure {
    /// The endpoints met within Euclidean UV tolerance.
    EuclideanClosed,
    /// The endpoints met only modulo the lattice.
    PeriodicClosed,
    /// The piece did not close.
    Open,
}

/// The deck evidence carried by one closed boundary piece.
///
/// This is the per-piece half of what package 1 needs to decide a two-loop
/// join: the lattice displacement the walk actually accumulated, the winding
/// sign of its parameter-space traversal, and the fundamental-domain
/// representative it was normalised onto. Recorded where the piece is
/// classified, because that is the only place all three are known at once.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct BoundaryPieceDeck {
    /// The piece's index in the face's boundary list.
    pub piece_index: usize,
    /// How the piece closed.
    pub closure: ObservedClosure,
    /// The integer lattice displacement along `u`.
    pub ku: i64,
    /// The integer lattice displacement along `v`.
    pub kv: i64,
    /// The sign of the piece's signed parameter-space area: `+1`, `-1`, or `0`
    /// when the area is below the degeneracy threshold — which is exactly the
    /// case the two-loop branch admits.
    pub winding_sign: i8,
    /// The signed parameter-space area itself, retained because the sign alone
    /// cannot distinguish "degenerate" from "nearly degenerate".
    pub signed_area: f64,
    /// The fundamental-domain representative: the piece's first point after
    /// normalisation.
    pub representative: (f64, f64),
    /// The piece's first parameter point, after normalisation.
    pub start_uv: (f64, f64),
    /// The piece's last parameter point, after normalisation.
    pub end_uv: (f64, f64),
    /// The number of parameter samples in the piece.
    pub point_count: usize,
}

/// What the two-closed-loop branch of `PolyBoundary::new` did, and whether the
/// deck equation it implies is satisfiable.
///
/// The branch cuts both loops open, **reverses one unconditionally**, and
/// bridges them with a pair of seam segments. For a quotient-closed boundary
/// walk `Σδᵢ = Δ_walk`, and `Δ_walk = 0` for a contractible regular boundary.
/// Traversing loop1 reversed contributes `−δ₁`, so the sum the branch actually
/// realises is `δ₀ − δ₁`. When the two loops carry **opposite** winding — as
/// the two boundary circles of a band must, for the face boundary to be
/// coherently oriented — that sum is `±2`, not `0`, and the two bridges become
/// crossing diagonals rather than the vertical cut edges of a rectangle.
///
/// This record does not repair anything. It states which traversal the branch
/// chose and what `Σδ` that choice produced, so the population can be counted
/// before a repair is designed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct TwoLoopJoinRecord {
    /// Loop 0's lattice displacement `[ku, kv]`.
    pub loop0_displacement: [i64; 2],
    /// Loop 1's lattice displacement `[ku, kv]`.
    pub loop1_displacement: [i64; 2],
    /// Whether loop 1 was traversed reversed. Currently always `true`: the
    /// branch reverses unconditionally.
    pub loop1_reversed: bool,
    /// The lattice translate applied to loop 1 by the mean-parameter alignment,
    /// `[ku, kv]`.
    pub mean_translate: [i64; 2],
    /// `Σδ` along `u` for the traversal the branch chose.
    pub deck_sum_u: i64,
    /// `Σδ` along `v` for the traversal the branch chose.
    pub deck_sum_v: i64,
    /// Whether `Σδ = Δ_walk = 0` holds for that choice.
    pub deck_consistent: bool,
    /// Whether the traversal *not* taken would have satisfied the equation.
    /// With the legacy reversal that is the forward traversal, and this is the
    /// discriminator the repair turns on; once the repair applies it is false,
    /// because the taken traversal is then the one that closes.
    pub forward_would_close: bool,
    /// The first bridge's endpoints, `(from, to)` in parameter space.
    pub bridge0: [(f64, f64); 2],
    /// The closing bridge's endpoints, `(from, to)` in parameter space.
    pub bridge1: [(f64, f64); 2],
}

/// How strong the evidence for one P3b cap-theorem hypothesis is.
///
/// The cap theorem (see `PeriodicCapClosure` in `triangulation.rs`) discharges
/// H1–H5; each hypothesis is admitted only on evidence of the strength this
/// enum states. A `Candidate` may *nominate* the cap route; it may not silently
/// become a certified source fact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum CapHypothesisEvidence {
    /// Established from the representation (a period or pole witness read from
    /// the primitive's own parameterisation).
    Certified,
    /// Established by a bounded constructive numerical step (e.g. integer
    /// winding from a certified period plus a residual bound).
    Constructive,
    /// Recognized by a heuristic/numerical recognizer; may nominate the route
    /// but does not establish the source-level fact.
    Candidate,
    /// Not established at all; the hypothesis failed.
    NotEstablished,
}

/// Why the periodic-cap route was activated or declined for one loop.
///
/// Recorded when `PeriodicCapClosure::try_build` runs its gate, so a census can
/// answer, per face: why was cap recovery considered, what evidence existed for
/// each hypothesis, and which gate declined it. This is the minimum epistemic
/// contract: the theorem's H1–H5 are reported with their evidence strength, not
/// folded into a single `Some/None`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CapActivationRecord {
    /// Which parameter axis carried the period the gate keyed on.
    pub periodic_axis: PeriodicAxis,
    /// H1: a genuine period. `None` when no certified generator existed.
    pub period: Option<CapHypothesisEvidence>,
    /// H2: winding `|k| = 1`. `Some` only after a certified period and a
    /// bounded residual made the integer constructive.
    pub winding: Option<CapHypothesisEvidence>,
    /// H3: the loop is the single 1D latitude-walk signature (tiny signed area,
    /// non-periodic span small). Recognizer-level unless proven otherwise.
    pub cap_signature: CapHypothesisEvidence,
    /// H4: the orbit genuinely collapses on the material side. Certified only
    /// for a representation-derived pole (sphere); otherwise candidate.
    pub collapse: CapHypothesisEvidence,
    /// H5: the selected pole lies on the source-derived material side.
    /// `None` when the gate declined before material-side selection.
    pub material_side: Option<CapHypothesisEvidence>,
    /// Whether the gate ultimately built the cap cell.
    pub activated: bool,
    /// Why the gate declined, when it did not activate.
    pub declined_reason: Option<&'static str>,
}

/// Which parameter axis carried a periodic-cap boundary's period.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum PeriodicAxis {
    /// The `u` axis.
    U,
    /// The `v` axis.
    V,
}

/// The mechanism-level subtype of a seam-involved insertion failure.
///
/// `SyntheticSyntheticCrossing` names *which segments* collided; it says
/// nothing about *why*. This says why, and only from recorded evidence: a face
/// is `OppositeWindingReversed` when the branch ran, the loops wound opposite,
/// and traversing loop 1 forward would have closed the deck equation. That is
/// the population package 1's repair is for. Everything else stays separately
/// named rather than being folded in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum SeamMechanism {
    /// The two-loop join ran, the loops wound opposite, and forward traversal
    /// closes `Σδ = 0`. The hypothesised repairable case.
    OppositeWindingReversed,
    /// The two-loop join ran and its deck equation is already satisfied, so
    /// the crossing has another cause.
    JoinDeckConsistent,
    /// The two-loop join ran and *neither* traversal closes the equation.
    JoinDeckUnsatisfiable,
    /// Seam segments are present but the two-loop join did not run — the seams
    /// came from a periodic walk's wrap, the collapsed-pair branch, or an open
    /// piece's synthetic closure.
    SeamWithoutTwoLoopJoin,
    /// No seam evidence was recorded for this face.
    NoSeamEvidence,
}

/// Derive the seam mechanism from the recorded deck evidence.
///
/// Deliberately total and evidence-only: with no join record and no seam
/// segments there is nothing to say, and saying `NoSeamEvidence` is the honest
/// answer rather than a guess about which branch ran.
pub fn derive_seam_mechanism(
    join: Option<&TwoLoopJoinRecord>,
    seam_segment_count: usize,
) -> SeamMechanism {
    match join {
        Some(join) if join.deck_consistent => SeamMechanism::JoinDeckConsistent,
        Some(join) if join.forward_would_close => SeamMechanism::OppositeWindingReversed,
        Some(_) => SeamMechanism::JoinDeckUnsatisfiable,
        None if seam_segment_count > 0 => SeamMechanism::SeamWithoutTwoLoopJoin,
        None => SeamMechanism::NoSeamEvidence,
    }
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
    /// A FACE-VALIDITY hard rejection: the face was certified intrinsically
    /// degenerate before tessellation.
    IntrinsicDegenerate,
    /// A P2 certified rejection: the face was certified singular-ambiguous at a
    /// rank-deficient periodic transition where the incident source geometry
    /// underdetermines the continuation.
    IntrinsicAmbiguous,
    /// A typed failure not in the above categories.
    OtherTypedFailure,
}

// ---------------------------------------------------------------------------
// ARR-TAIL: mechanistic signature of a residual failed face
// ---------------------------------------------------------------------------

/// The true pipeline stage a residual failed face was lost at.
///
/// ARR-TAIL wants the *mechanism*, not the terminal enum alone: two faces with
/// the same `NoOddParityRegion` reason are different work depending on whether
/// the parity flood selected nothing or selected only degenerate triangles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Default)]
pub enum ArrFailureStage {
    /// The boundary could not be constructed (projection or lift).
    BoundaryConstruction,
    /// A constraint was rejected before insertion (overlap / duplicate).
    ConstraintRejected,
    /// A constraint was refused by the triangulation during insertion.
    ConstraintIncomplete,
    /// The raw CDT produced no triangles.
    CdtEmpty,
    /// The parity flood selected no material region.
    MaterialEmpty,
    /// Parity selected triangles that validation then removed as degenerate.
    PostMaterialDegenerate,
    /// An ambiguous periodic lift.
    LiftAmbiguous,
    /// The parity flood contradicted itself.
    ParityContradiction,
    /// The face was certified intrinsically degenerate (FACE-VALIDITY) and
    /// rejected before tessellation.
    RejectedDegenerate,
    /// The face was certified singular-ambiguous (P2) and rejected before
    /// material solving: a rank-deficient periodic transition whose incident
    /// source geometry underdetermines the lift branch.
    RejectedAmbiguous,
    /// The stage could not be established from retained evidence.
    #[default]
    Unknown,
}

/// The CDT/material pipeline class of a residual failed face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Default)]
pub enum ArrMaterialStage {
    /// The failure happened before the CDT was ever read.
    NotReached,
    /// The CDT produced triangles but no material count was retained.
    CdtOnly,
    /// The material filter selected zero triangles.
    MaterialEmpty,
    /// The material filter selected triangles, all later removed as degenerate.
    MaterialDegenerated,
    /// Material survived validation, yet the face is still lost downstream.
    MaterialSurvived,
    /// Not established.
    #[default]
    Unknown,
}

/// The origin-pair class of a witnessed constraint conflict.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Default)]
pub enum ArrProvenanceClass {
    /// Both segments are authoritative source trim.
    SourceSource,
    /// At least one segment is synthesised (closure or seam).
    SourceSynthetic,
    /// Both segments are synthesised.
    SyntheticSynthetic,
    /// No pair evidence was retained.
    #[default]
    None,
}

/// Whether a seam or deck bridge is implicated in a residual failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Default)]
pub enum ArrSeamOrDeck {
    /// No seam segment and no two-loop join record.
    #[default]
    None,
    /// At least one seam segment is present.
    Seam,
    /// A two-loop join ran (periodic/deck context).
    Deck,
    /// Both seam segments and a two-loop join record are present.
    SeamAndDeck,
}

/// The curve-family pair class of a witnessed constraint conflict.
///
/// Not derivable from retained evidence today: the presented segments are
/// parameter-space polylines by the time the CDT sees them, and the source
/// curve family is not threaded into the insertion path. The variants name
/// what the class would be once that evidence exists; everything today is
/// `Unknown`, reported honestly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Default)]
pub enum CurvePairClass {
    /// The pair class could not be established from retained evidence.
    #[default]
    Unknown,
}

/// The mechanistic signature of one residual failed face (ARR-TAIL-001).
///
/// Diagnostic-only: constructed at diagnosis-build time from evidence the
/// pipeline already retained, it cannot change rendered geometry. It exists to
/// cluster the several-thousand-face residual tail by *mechanism* rather than
/// by terminal-reason histogram. Fields that would require a major
/// architecture change to fill (e.g. the source curve family of a presented
/// constraint segment) are left at their `None`/`Unknown` value and reported
/// honestly as such.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ArrSignature {
    /// The pipeline stage the face was actually lost at.
    pub failure_stage: ArrFailureStage,
    /// The coarse bound-count class: `0`, `1`, `2`, or `>= 3`.
    pub bound_count_bucket: usize,
    /// Whether a seam or two-loop-join deck context is implicated.
    pub seam_or_deck: ArrSeamOrDeck,
    /// The dominant presented-segment relation among the retained witnesses.
    pub pair_relation: Option<PresentedSegmentRelation>,
    /// The origin-pair class of the dominant witnessed conflict.
    pub pair_provenance: ArrProvenanceClass,
    /// Whether the dominant conflict is within one bound.
    pub same_bound: Option<bool>,
    /// The curve-family pair class. Unavailable for constraint segments today.
    pub curve_pair_class: CurvePairClass,
    /// The CDT/material pipeline class.
    pub material_stage: ArrMaterialStage,
    /// The projection verdict class of a projection-failed face.
    pub projection_verdict: Option<PointVerdict>,
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
    /// The number of seam segments, separated from the closure segments that
    /// [`Self::synthetic_segment_count`] merges them with.
    pub seam_segment_count: usize,
    /// The deck evidence for each boundary piece.
    pub boundary_pieces: Vec<BoundaryPieceDeck>,
    /// What the two-closed-loop join did, when it ran.
    pub two_loop_join: Option<TwoLoopJoinRecord>,
    /// The mechanism-level subtype of a seam-involved failure.
    pub seam_mechanism: SeamMechanism,
    /// The structured conflict witnesses, when the failure is an insertion
    /// failure.
    pub insertion_conflicts: Vec<ConstraintConflictWitness>,
    /// The structured witnesses for constraint *overlaps*, held apart from
    /// [`Self::insertion_conflicts`] because the loss bucket is derived from
    /// that vector and must keep meaning what it meant.
    pub overlap_conflicts: Vec<ConstraintConflictWitness>,
    /// Overlaps whose blocking edge could not be named. Non-zero means pair
    /// evidence is genuinely missing, rather than absent because there was no
    /// overlap.
    pub unattributed_overlaps: usize,
    /// The CDT and material pipeline stage counts.
    pub cdt_stages: CdtStageVector,
    /// The deep projection witness (PROJ-002), when the probe ran.
    pub projection_witness: Option<ProjectionWitness>,
    /// FACE-VALIDITY: the certificate backing a hard degenerate rejection,
    /// when the face was rejected as intrinsically non-renderable.
    pub validity_certificate: Option<crate::tessellation::validity::FaceValidityCertificate>,
    /// What each formal recovery route decided about this face.
    pub route_decisions: Vec<RouteDecisionRecord>,
    /// P3b: the periodic-cap route's activation evidence for this face.
    pub cap_activation: Option<CapActivationRecord>,
    /// The deterministic loss bucket.
    pub derived_bucket: LossBucket,
    /// The mechanistic ARR-TAIL signature, always present for a failed face.
    pub arr: ArrSignature,
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
        R::RejectedDegenerate => LossBucket::IntrinsicDegenerate,
        R::RejectedAmbiguous => LossBucket::IntrinsicAmbiguous,
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
    if reason == TessellationFailureReason::AmbiguousLift
        || reason == TessellationFailureReason::RejectedAmbiguous
    {
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

/// Whether the deck-consistent two-loop join is active.
///
/// Nested under the master gate, like every other formal route. The route is
/// refinement-only by construction: the legacy boundary is built and
/// tessellated first, and the corrected join is attempted **only** on a face
/// that still has no mesh, so it can replace nothing but a failure.
pub fn deck_join_recovery_enabled() -> bool {
    formal_recovery_enabled() && recovery_route_enabled("TRUCK_FORMAL_RECOVERY_DECK_JOIN")
}

/// Whether the structural-seed retry of the parameter inverse is active.
///
/// Refinement-only in the same structural sense as the routes above, one level
/// down: the retry is the last link of the projection chain, so it is reached
/// only for a point every existing attempt already returned `None` for.
pub fn spline_seed_recovery_enabled() -> bool {
    formal_recovery_enabled() && recovery_route_enabled("TRUCK_FORMAL_RECOVERY_SEED")
}

/// Whether PROJ-003 Stage A is active: residual-certified admission of a
/// production-start iterate that the legacy projection chain rejected.
///
/// Refinement-only in the same structural sense as the routes above: it fires
/// only where the legacy projection returned `None`, so a face that projected
/// through the legacy chain is byte-identical with this on or off. The
/// admission contract is strict — finite UV, inside the declared parameter
/// range, `|S(u,v) - P| <= tol` — and Newton's `near2` condition is
/// deliberately not required, because the whole finding is that `near2` throws
/// away geometrically adequate answers. Set
/// `TRUCK_FORMAL_RECOVERY_PROJ_STAGE_A=0` to disable (emergency withdrawal).
pub fn proj_residual_recovery_enabled() -> bool {
    formal_recovery_enabled() && recovery_route_enabled("TRUCK_FORMAL_RECOVERY_PROJ_STAGE_A")
}

/// Whether PROJ-003 Stage B is active: residual-certified admission of a
/// structural-seed nearest iterate that the legacy chain (including the
/// spline-seed `search_parameter` retry) rejected.
///
/// Refinement-only in the same structural sense as Stage A: it fires only
/// where the whole legacy projection chain returned `None` *and* Stage A did
/// not admit a production-start iterate, so a face that projects through the
/// legacy chain or is recovered by Stage A is byte-identical with this on or
/// off. The admission contract is the same one Stage A enforces — finite UV,
/// inside the declared parameter range, `|S(u, v) - P| <= tol` — with the
/// candidate taken from the bounded nearest searches launched from the
/// structural (knot-span) seeds rather than from the production starts. Set
/// `TRUCK_FORMAL_RECOVERY_PROJ_STAGE_B=0` to disable (emergency withdrawal).
pub fn proj_seed_recovery_enabled() -> bool {
    formal_recovery_enabled() && recovery_route_enabled("TRUCK_FORMAL_RECOVERY_PROJ_STAGE_B")
}

/// Whether PROJ-003 Stage C is active: domain/contract recovery of a
/// within-tolerance iterate that lies outside the declared parameter range.
///
/// Refinement-only like the stages before it: it runs only where the whole
/// legacy chain, Stage A, and Stage B all returned `None` for a point. Unlike
/// A and B it may *transform* the candidate's coordinates, but only through a
/// principled domain/periodicity semantics — an integer number of certified
/// surface periods on a certified periodic axis, or a clamped boundary for a
/// microscopically-outside candidate — and every admission is re-certified
/// with the existing caller tolerance (finite UV, in-domain after the
/// transformation, finite evaluation, `|S(u, v) - P| <= tol`). Set
/// `TRUCK_FORMAL_RECOVERY_PROJ_STAGE_C=0` to disable (emergency withdrawal).
pub fn proj_domain_recovery_enabled() -> bool {
    formal_recovery_enabled() && recovery_route_enabled("TRUCK_FORMAL_RECOVERY_PROJ_STAGE_C")
}

/// Whether the winding-number reading of material parity is active.
///
/// The parity flood floods over the *set* of realized constraint edges, so an
/// edge two boundary segments both traversed toggles once where mod 2 it
/// should toggle not at all. That is the whole of `ContradictoryDualParity`:
/// on `00009190` every one of the 126 contradicting faces has a repeated
/// traversal and none of the 23,258 clean ones does.
///
/// Refinement-only in the same structural sense as the routes above: the
/// second reading is asked for only after the set reading's flood contradicted
/// itself, i.e. only on a face that already has no mesh.
pub fn winding_parity_enabled() -> bool {
    formal_recovery_enabled() && recovery_route_enabled("TRUCK_FORMAL_RECOVERY_PARITY")
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
    /// Overlap witnesses, kept in their **own** vector rather than appended to
    /// `witnesses`.
    ///
    /// This separation is load-bearing, not tidiness. `derive_loss_bucket`
    /// reads `witnesses`, and the band routes read the derived bucket as an
    /// admission rule, so a witness added to that vector is a production input.
    /// `ConstraintOverlapUnsupported` previously recorded nothing at all; had
    /// these gone into `witnesses`, every face where an overlap follows a
    /// crossing — the terminal reason there is still
    /// `ConstraintInsertionIncomplete`, because `failure.get_or_insert` keeps
    /// the first — would derive a different bucket than it does today, and the
    /// cylinder and cone bands would admit a different population. Held apart,
    /// the derivation is bit-identical by construction rather than by review.
    overlap_witnesses: Vec<ConstraintConflictWitness>,
    /// Overlaps whose blocking edge could not be attributed to a semantic
    /// segment this face recorded. Counted rather than dropped, so "no witness"
    /// stays distinguishable from "no overlap".
    unattributed_overlaps: usize,
    /// The CDT and material stage counts. Respects suspension, because the
    /// record must keep describing the legacy attempt.
    cdt_stages: CdtStageVector,
    /// The deep projection witness, when the probe ran.
    projection_witness: Option<ProjectionWitness>,
    /// FACE-VALIDITY: the certificate backing a hard degenerate rejection, when
    /// one was produced.
    validity_certificate: Option<crate::tessellation::validity::FaceValidityCertificate>,
    realized_chain: Vec<RealizedConstraintMetadata>,
    vertex_insertion_failed: bool,
    source_segment_count: usize,
    synthetic_segment_count: usize,
    seam_segment_count: usize,
    boundary_pieces: Vec<BoundaryPieceDeck>,
    two_loop_join: Option<TwoLoopJoinRecord>,
    /// P3b: the periodic-cap route's activation evidence for this face.
    cap_activation: Option<CapActivationRecord>,
}

impl DiagnosisSink {
    fn clear(&mut self) {
        self.segments.clear();
        self.witnesses.clear();
        self.overlap_witnesses.clear();
        self.unattributed_overlaps = 0;
        self.cdt_stages = CdtStageVector::default();
        self.projection_witness = None;
        self.validity_certificate = None;
        self.realized_chain.clear();
        self.vertex_insertion_failed = false;
        self.source_segment_count = 0;
        self.synthetic_segment_count = 0;
        self.seam_segment_count = 0;
        self.boundary_pieces.clear();
        self.two_loop_join = None;
        self.cap_activation = None;
    }
}

std::thread_local! {
    static FACE_DIAGNOSIS_SINK: std::cell::RefCell<DiagnosisSink> =
        std::cell::RefCell::new(DiagnosisSink::default());
    /// Whether recording is currently suspended. See [`SinkSuspension`].
    static SINK_SUSPENDED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Route decisions, in a **separate sub-sink that suspension does not
    /// silence**.
    ///
    /// This split is the one semantic decision in the whole sweep, and getting
    /// it backwards would be a production behaviour change wearing a
    /// diagnostic's clothes. Lift, projection, conflict and CDT records must
    /// respect [`SinkSuspension`]: the loss bucket is derived from them and the
    /// band routes read that bucket as an admission rule, so a record mixing
    /// two tessellation attempts changes what those routes admit. A route
    /// *decision* is the opposite case — the entire point is to know that a
    /// route was entered, and a route is entered precisely during the window
    /// suspension covers. Silencing these would record every recovery attempt
    /// as never having happened.
    static ROUTE_DECISIONS: std::cell::RefCell<Vec<RouteDecisionRecord>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Suspends recording for as long as it is held.
///
/// A face is tessellated once per boundary the pipeline is willing to try, and
/// the deck-consistent join adds a second attempt on a face the first one lost.
/// Without this, that second attempt would append its own segments and
/// witnesses to the first one's, and the derived bucket — which the band routes
/// read as an admission rule — would describe neither attempt. The DIAG-001
/// record therefore stays a statement about the *legacy* boundary, which is
/// what every number taken from it so far means.
pub(crate) struct SinkSuspension(());

impl SinkSuspension {
    /// Suspend recording until the returned guard is dropped.
    pub(crate) fn new() -> Self {
        SINK_SUSPENDED.with(|flag| flag.set(true));
        Self(())
    }
}

impl Drop for SinkSuspension {
    fn drop(&mut self) {
        SINK_SUSPENDED.with(|flag| flag.set(false));
    }
}

/// Whether recording is suspended.
fn suspended() -> bool {
    SINK_SUSPENDED.with(|flag| flag.get())
}

/// Clear the diagnostic sink before a new face.
pub(crate) fn clear_sink() {
    FACE_DIAGNOSIS_SINK.with(|s| s.borrow_mut().clear());
    ROUTE_DECISIONS.with(|s| s.borrow_mut().clear());
}

/// Whether diagnostics are on, decided once.
///
/// [`diag_enabled`] reads the environment on every call. That is fine where it
/// is asked once per face, but the route recorders are called from inside the
/// route functions, which have no `diag` in scope to be gated by at the call
/// site. Cached for the same reason `projection_probe_enabled` is: the value
/// cannot change within a process.
fn diag_enabled_cached() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(diag_enabled)
}

/// Record what a route decided about this face.
///
/// Deliberately does **not** consult [`suspended`]; see [`ROUTE_DECISIONS`].
///
/// It *does* consult [`diag_enabled_cached`], and must: `clear_sink` is only
/// called when diagnostics are on, so recording unconditionally would grow this
/// vector for the life of the thread on a run with diagnostics off.
pub(crate) fn record_route_decision(
    route: RecoveryRoute,
    gate_open: bool,
    outcome: RouteOutcome,
    refusal_tag: Option<&'static str>,
) {
    if !diag_enabled_cached() {
        return;
    }
    ROUTE_DECISIONS.with(|sink| {
        sink.borrow_mut().push(RouteDecisionRecord {
            route,
            gate_open,
            outcome,
            refusal_tag,
        })
    });
}

/// Record that a route was entered and declined the face outright.
///
/// Called from the route functions in place of a bare `return None`, which is
/// where the cone signature was hiding: three separate early exits all reported
/// themselves as the same absence of a record.
pub(crate) fn record_route_ineligible(route: RecoveryRoute, reason: RouteIneligible) {
    record_route_decision(route, true, RouteOutcome::Ineligible(reason), None);
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
    if suspended() {
        // A caller still needs an id to hand back to `record_conflict`, which
        // ignores it for the same reason this is ignored.
        return u64::MAX;
    }
    FACE_DIAGNOSIS_SINK.with(|sink| {
        let sink = &mut *sink.borrow_mut();
        let id = sink.segments.len() as u64;
        match origin {
            SegmentOrigin::Source => sink.source_segment_count += 1,
            SegmentOrigin::Seam => {
                sink.synthetic_segment_count += 1;
                sink.seam_segment_count += 1;
            }
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
    if suspended() {
        return;
    }
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

/// Build a witness from two recorded segment ids, or `None` if either id does
/// not name a segment this face recorded.
fn build_witness(
    sink: &DiagnosisSink,
    incoming_id: u64,
    blocking_id: u64,
    relation: PresentedSegmentRelation,
    incoming_segment: Option<SegmentEndpoints2>,
    blocking_segment: Option<SegmentEndpoints2>,
) -> Option<ConstraintConflictWitness> {
    let incoming = sink.segments.get(incoming_id as usize).cloned()?;
    let blocking = sink.segments.get(blocking_id as usize).cloned()?;
    let same_bound = match (incoming.boundary_component, blocking.boundary_component) {
        (Some(a), Some(b)) => Some(a == b),
        _ => None,
    };
    Some(ConstraintConflictWitness {
        incoming,
        blocking,
        relation,
        same_bound,
        same_source_edge_use: None,
        intersection_enclosure: None,
        incoming_segment,
        blocking_segment,
    })
}

/// Record a witness for a constraint overlap.
///
/// `ConstraintOverlapUnsupported` is raised on the segment that would traverse
/// an edge this face's own role table already claims, and until now it recorded
/// nothing whatsoever — so the whole population carried zero pair evidence and
/// was indistinguishable from an unwitnessed insertion failure.
///
/// Goes to `overlap_witnesses`, never to `witnesses`; see the field's
/// documentation for why that separation is a correctness requirement rather
/// than a filing decision.
pub(crate) fn record_overlap_conflict(
    incoming_id: u64,
    blocking_id: Option<u64>,
    relation: PresentedSegmentRelation,
    incoming_segment: Option<SegmentEndpoints2>,
    blocking_segment: Option<SegmentEndpoints2>,
) {
    if suspended() {
        return;
    }
    FACE_DIAGNOSIS_SINK.with(|sink| {
        let sink = &mut *sink.borrow_mut();
        let witness = blocking_id.and_then(|blocking_id| {
            build_witness(
                sink,
                incoming_id,
                blocking_id,
                relation,
                incoming_segment,
                blocking_segment,
            )
        });
        match witness {
            Some(witness) => sink.overlap_witnesses.push(witness),
            None => sink.unattributed_overlaps += 1,
        }
    });
}

/// Record what `insert_to` presented to the triangulation and what it realized.
///
/// Arithmetic, not tracing: three counters the function already maintains in
/// locals and drops on return.
pub(crate) fn record_insertion_counts(
    boundary_vertices: usize,
    constraints_presented: usize,
    constraints_inserted: usize,
) {
    if suspended() {
        return;
    }
    FACE_DIAGNOSIS_SINK.with(|sink| {
        let stages = &mut sink.borrow_mut().cdt_stages;
        stages.boundary_vertices = boundary_vertices;
        stages.constraints_presented = constraints_presented;
        stages.constraints_inserted = constraints_inserted;
    });
}

/// Record the deep projection witness for a face.
///
/// Respects suspension, like every other evidence record: the row must keep
/// describing the legacy attempt.
pub(crate) fn record_projection_witness(witness: ProjectionWitness) {
    if suspended() {
        return;
    }
    FACE_DIAGNOSIS_SINK.with(|sink| sink.borrow_mut().projection_witness = Some(witness));
}

/// Record the three triangle counts either side of material selection.
///
/// Separating `material_selected` from `final_valid` is the point: they are
/// computed by one chained iterator today, so a face where parity chose nothing
/// and a face where parity chose only degenerate triangles both arrive at
/// `NoOddParityRegion` indistinguishable.
pub(crate) fn record_cdt_stages(
    raw_cdt_triangles: usize,
    material_selected: usize,
    final_valid: usize,
) {
    if suspended() {
        return;
    }
    FACE_DIAGNOSIS_SINK.with(|sink| {
        let stages = &mut sink.borrow_mut().cdt_stages;
        stages.raw_cdt_triangles = Some(raw_cdt_triangles);
        stages.material_selected = Some(material_selected);
        stages.final_valid = Some(final_valid);
    });
}

/// Record one boundary piece's deck evidence.
///
/// Called from `PolyBoundary::new` as each piece is classified, so the
/// displacement recorded is the one the pipeline acted on rather than one
/// recomputed later from normalised points.
pub(crate) fn record_boundary_piece(piece: BoundaryPieceDeck) {
    if suspended() {
        return;
    }
    FACE_DIAGNOSIS_SINK.with(|sink| sink.borrow_mut().boundary_pieces.push(piece));
}

/// Record what the two-closed-loop join did.
pub(crate) fn record_two_loop_join(record: TwoLoopJoinRecord) {
    if suspended() {
        return;
    }
    FACE_DIAGNOSIS_SINK.with(|sink| sink.borrow_mut().two_loop_join = Some(record));
}

/// Record the periodic-cap route's activation evidence.
///
/// Suspension does not apply: the cap is part of the legacy boundary's own
/// classification (`PolyBoundary::new_with_join`), so its evidence belongs to
/// the same record the face's legacy verdict is built from, and silencing it
/// while a second attempt runs would lose the only statement about the first.
pub(crate) fn record_cap_activation(record: CapActivationRecord) {
    FACE_DIAGNOSIS_SINK.with(|sink| sink.borrow_mut().cap_activation = Some(record));
}

/// Record the FACE-VALIDITY certificate backing a hard degenerate rejection.
///
/// The certificate is the evidence that no positive-area trim region exists at
/// tolerance; it rides in the face's diagnosis so a census can classify the
/// face as `rejected_intrinsic` rather than as a generic tessellation failure.
pub(crate) fn record_face_rejection(
    certificate: crate::tessellation::validity::FaceValidityCertificate,
) {
    if suspended() {
        return;
    }
    FACE_DIAGNOSIS_SINK.with(|sink| sink.borrow_mut().validity_certificate = Some(certificate));
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

/// Derive the ARR-TAIL mechanistic signature from retained evidence.
///
/// Purely observational: every input is already retained in the sink or in the
/// face-level parameters. The signature is what lets the census cluster the
/// residual tail by mechanism without a new diagnostic sweep.
fn derive_arr_signature(
    terminal_reason: TessellationFailureReason,
    bound_count: usize,
    seam_segment_count: usize,
    two_loop_join: Option<&TwoLoopJoinRecord>,
    witnesses: &[ConstraintConflictWitness],
    overlap_conflicts: &[ConstraintConflictWitness],
    cdt_stages: CdtStageVector,
    projection_witness: Option<&ProjectionWitness>,
) -> ArrSignature {
    use TessellationFailureReason as R;
    // The failure stage, from the terminal reason and the material counts that
    // separate empty-from-degenerate `NoOddParityRegion`.
    let material_stage = match cdt_stages.raw_cdt_triangles {
        None => ArrMaterialStage::NotReached,
        Some(_) => match cdt_stages.material_selected {
            None => ArrMaterialStage::CdtOnly,
            Some(0) => ArrMaterialStage::MaterialEmpty,
            Some(_) => match cdt_stages.final_valid {
                Some(0) => ArrMaterialStage::MaterialDegenerated,
                Some(_) => ArrMaterialStage::MaterialSurvived,
                None => ArrMaterialStage::Unknown,
            },
        },
    };
    let failure_stage = match terminal_reason {
        R::BoundaryProjectionFailed | R::BoundaryPointOffSurface | R::BoundaryWireEmpty => {
            ArrFailureStage::BoundaryConstruction
        }
        R::AmbiguousLift => ArrFailureStage::LiftAmbiguous,
        R::RejectedDegenerate => ArrFailureStage::RejectedDegenerate,
        R::RejectedAmbiguous => ArrFailureStage::RejectedAmbiguous,
        R::ConstraintOverlapUnsupported => ArrFailureStage::ConstraintRejected,
        R::ConstraintInsertionIncomplete => ArrFailureStage::ConstraintIncomplete,
        R::ContradictoryDualParity => ArrFailureStage::ParityContradiction,
        R::NoOddParityRegion => match material_stage {
            ArrMaterialStage::MaterialDegenerated => ArrFailureStage::PostMaterialDegenerate,
            ArrMaterialStage::MaterialEmpty | ArrMaterialStage::CdtOnly => {
                ArrFailureStage::MaterialEmpty
            }
            _ => ArrFailureStage::Unknown,
        },
        _ => ArrFailureStage::Unknown,
    };
    let seam_or_deck = match (seam_segment_count > 0, two_loop_join.is_some()) {
        (true, true) => ArrSeamOrDeck::SeamAndDeck,
        (true, false) => ArrSeamOrDeck::Seam,
        (false, true) => ArrSeamOrDeck::Deck,
        (false, false) => ArrSeamOrDeck::None,
    };
    // The dominant witnessed relation and its provenance. Overlaps and
    // insertion conflicts are held apart in the sink but are the same pair
    // evidence for the purposes of naming the mechanism.
    let all: Vec<&ConstraintConflictWitness> =
        witnesses.iter().chain(overlap_conflicts.iter()).collect();
    let mut relation_counts: Vec<&PresentedSegmentRelation> =
        all.iter().map(|w| &w.relation).collect();
    relation_counts.sort_by_key(|r| format!("{r:?}"));
    relation_counts.dedup();
    let pair_relation = relation_counts
        .into_iter()
        .max_by_key(|r| all.iter().filter(|w| &w.relation == *r).count())
        .copied();
    let provenance_of = |w: &ConstraintConflictWitness| {
        let incoming = OriginClass::from(w.incoming.origin);
        let blocking = OriginClass::from(w.blocking.origin);
        match (
            incoming.is_authoritative_source(),
            blocking.is_authoritative_source(),
        ) {
            (true, true) => ArrProvenanceClass::SourceSource,
            (false, false) => ArrProvenanceClass::SyntheticSynthetic,
            _ => ArrProvenanceClass::SourceSynthetic,
        }
    };
    let dominant =
        pair_relation.and_then(|relation| all.iter().find(|w| w.relation == relation).copied());
    let (pair_provenance, same_bound) = match dominant {
        Some(w) => (provenance_of(w), w.same_bound),
        None => (ArrProvenanceClass::None, None),
    };
    let projection_verdict = projection_witness.map(|w| w.verdict);
    ArrSignature {
        failure_stage,
        bound_count_bucket: match bound_count {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => 3,
        },
        seam_or_deck,
        pair_relation,
        pair_provenance,
        same_bound,
        curve_pair_class: CurvePairClass::Unknown,
        material_stage,
        projection_verdict,
    }
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
        let overlap_conflicts = std::mem::take(&mut sink.overlap_witnesses);
        let unattributed_overlaps = sink.unattributed_overlaps;
        let cdt_stages = sink.cdt_stages;
        let projection_witness = sink.projection_witness.take();
        let validity_certificate = sink.validity_certificate.take();
        let route_decisions = ROUTE_DECISIONS.with(|s| std::mem::take(&mut *s.borrow_mut()));
        let source_segment_count = sink.source_segment_count;
        let synthetic_segment_count = sink.synthetic_segment_count;
        let seam_segment_count = sink.seam_segment_count;
        let boundary_pieces = std::mem::take(&mut sink.boundary_pieces);
        let two_loop_join = sink.two_loop_join;
        let cap_activation = sink.cap_activation.take();
        let seam_mechanism = derive_seam_mechanism(two_loop_join.as_ref(), seam_segment_count);
        sink.clear();
        let derived_bucket =
            derive_loss_bucket(terminal_reason, &witnesses, vertex_insertion_failed);
        let projection_status = derive_projection_status(terminal_reason);
        let arr = derive_arr_signature(
            terminal_reason,
            bound_count,
            seam_segment_count,
            two_loop_join.as_ref(),
            &witnesses,
            &overlap_conflicts,
            cdt_stages,
            projection_witness.as_ref(),
        );
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
            seam_segment_count,
            boundary_pieces,
            two_loop_join,
            seam_mechanism,
            insertion_conflicts: witnesses,
            overlap_conflicts,
            unattributed_overlaps,
            cdt_stages,
            projection_witness,
            validity_certificate,
            route_decisions,
            cap_activation,
            derived_bucket,
            arr,
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
            incoming_segment: None,
            blocking_segment: None,
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
            seam_segment_count: 0,
            boundary_pieces: Vec::new(),
            two_loop_join: None,
            seam_mechanism: SeamMechanism::NoSeamEvidence,
            insertion_conflicts: vec![witness(
                seg_ref(0, SegmentOrigin::Source, Some(0)),
                seg_ref(1, SegmentOrigin::Source, Some(0)),
                Some(true),
            )],
            overlap_conflicts: Vec::new(),
            unattributed_overlaps: 0,
            cdt_stages: CdtStageVector::default(),
            projection_witness: None,
            validity_certificate: None,
            route_decisions: Vec::new(),
            cap_activation: None,
            derived_bucket: LossBucket::SourceSourceSameBoundCrossing,
            arr: ArrSignature::default(),
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

    fn join(loop0: [i64; 2], loop1: [i64; 2], reversed: bool) -> TwoLoopJoinRecord {
        let sign = if reversed { -1 } else { 1 };
        let deck_sum_u = loop0[0] + sign * loop1[0];
        let deck_sum_v = loop0[1] + sign * loop1[1];
        let forward_u = loop0[0] + loop1[0];
        let forward_v = loop0[1] + loop1[1];
        TwoLoopJoinRecord {
            loop0_displacement: loop0,
            loop1_displacement: loop1,
            loop1_reversed: reversed,
            mean_translate: [0, 0],
            deck_sum_u,
            deck_sum_v,
            deck_consistent: deck_sum_u == 0 && deck_sum_v == 0,
            forward_would_close: reversed && forward_u == 0 && forward_v == 0,
            bridge0: [(0.0, 0.0); 2],
            bridge1: [(0.0, 0.0); 2],
        }
    }

    /// The band case: two boundary circles wound opposite, loop 1 reversed.
    /// `Σδ = +2`, and traversing forward would have closed it.
    #[test]
    fn opposite_winding_reversed_is_the_repairable_case() {
        let record = join([1, 0], [-1, 0], true);
        assert_eq!(record.deck_sum_u, 2, "the reversal doubles the winding");
        assert!(!record.deck_consistent);
        assert!(record.forward_would_close);
        assert_eq!(
            derive_seam_mechanism(Some(&record), 2),
            SeamMechanism::OppositeWindingReversed,
        );
    }

    /// Two loops wound the same way: the unconditional reverse is already
    /// right, so a crossing here has some other cause and must not be folded
    /// into the repairable population.
    #[test]
    fn same_winding_reversed_is_deck_consistent() {
        let record = join([1, 0], [1, 0], true);
        assert!(record.deck_consistent);
        assert!(!record.forward_would_close);
        assert_eq!(
            derive_seam_mechanism(Some(&record), 2),
            SeamMechanism::JoinDeckConsistent,
        );
    }

    /// Neither traversal closes the equation — refuse, do not guess.
    #[test]
    fn unsatisfiable_deck_equation_is_named_separately() {
        let record = join([2, 0], [-1, 0], true);
        assert!(!record.deck_consistent);
        assert!(!record.forward_would_close);
        assert_eq!(
            derive_seam_mechanism(Some(&record), 2),
            SeamMechanism::JoinDeckUnsatisfiable,
        );
    }

    /// Seams with no join record came from another branch entirely.
    #[test]
    fn seams_without_a_join_are_not_attributed_to_it() {
        assert_eq!(
            derive_seam_mechanism(None, 3),
            SeamMechanism::SeamWithoutTwoLoopJoin,
        );
        assert_eq!(
            derive_seam_mechanism(None, 0),
            SeamMechanism::NoSeamEvidence
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
