//! Generic contact and event contracts: the surface ARR-003 consumes.
//!
//! ARR-002's [`super::intersection::PairIntersectionResult`] is pairwise —
//! exactly two curve incidences per result, isolated points only, overlap a
//! bare tag. GEN-001 lifts these into a family-independent model:
//!
//! - an isolated event carries an **arbitrary number** of incident branches
//!   ([`IsolatedEvent2`]), so a genuine triple event is one event, not three
//!   pairwise pseudo-events;
//! - contact is either an isolated event or a positive-dimensional common arc
//!   ([`ContactComponent2`]);
//! - identity is construction- and provenance-based ([`EventIdentity`]), never
//!   coordinate proximity.
//!
//! GEN-001A provides the types and a faithful adapter ([`lift_pair_result`])
//! from the existing pairwise result. The adapter turns each ARR-002 isolated
//! intersection into one two-branch isolated event under a stable identity; it
//! does **not** merge pairwise results into shared multi-branch events. That
//! merge is ARR-003's responsibility, and it uses the stable identities
//! introduced here.
//!
//! # A7 — the chord-side representative is hidden, not yet repaired
//!
//! ARR-002's chord-side membership test reads rounded evaluated arc endpoints
//! (`arc.start.point()` / `arc.end.point()`); see `GEN-001.md` assumption A7.
//! It is sound today only under the ≤π monotone-piece precondition. The generic
//! contract repairs this *structurally* in GEN-001A: every incident branch
//! carries a certified [`super::span::BranchGerm`], and ARR-003 consumes germs —
//! never representative coordinates — for ordering and topology. The analytic
//! orient test itself is replaced by a germ-based test in GEN-001C.

use super::curve2d::CurveOccurrenceProvenance;
use super::intersection::{
    CertifiedIntersection2, ContactKind, IntersectionIdentity, PairIntersectionResult,
    PairUnsupported, PairUnresolved, ParameterEnclosure, ParameterLocation,
};
use super::quotient::DeckLabel;
use super::span::{BranchGerm, CurveSpan2, SpanId};
use super::super::source_evidence::{EdgeUseId, SourceVertexKey};
use truck_geometry::prelude::Point2;

/// Why a generic contact computation could not be certified.
///
/// Frozen in GEN-001A so the later phases (B: Bézier isolation, C: germs, E:
/// common arcs) populate the variants without changing the ARR-003-facing data
/// model. A tolerance may produce one of these; it may never create topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenericUnresolved {
    /// Clustered roots the isolation cascade cannot separate at this precision.
    ClusteredRoots,
    /// An inseparable multiple (high-multiplicity) root.
    InseparableMultipleRoots,
    /// The isolation Jacobian is singular at the candidate root.
    SingularJacobian,
    /// A singular branch whose local topology is not yet classified.
    UnsupportedSingularBranch,
    /// A positive-dimensional common component beyond the minimal supported cases.
    GeneralCommonComponent,
    /// A Bézier root lying exactly on a parameter-domain boundary, where no
    /// endpoint certificate has been implemented yet (GEN-001C endpoint policy).
    UnresolvedBoundaryRoot,
    /// A certified root whose tangent-numerator enclosure could not certify a
    /// nonzero first derivative: a stationary (derivative-zero) candidate.
    /// Uncertainty here is never promoted to an ordinary tangent.
    UnresolvedStationaryBranch,
    /// A certified root whose transverse determinant could not be certified
    /// nonzero: a possible tangency or singularity, not a regular crossing.
    UnresolvedTangencyOrSingularity,
    /// The pair-local root ordinals could not be backed by a certified total
    /// order: two certified roots have overlapping source-parameter enclosures
    /// on both participants, so no interval-separation order exists. The roots
    /// are never ordered by traversal or discovery order.
    UnresolvedIdentityOrdering,
    /// A derived (subdivided) piece's root could not be matched to exactly one
    /// root of the parent (unsplit) pair isolation. A sub-span cannot acquire a
    /// geometric intersection its parent lacks, so an unmatched or ambiguous
    /// match is `Unresolved`, never a new root identity.
    UnresolvedIdentityReferBack,
    /// A reused ARR-002 pairwise cause.
    Pair(PairUnresolved),
}

impl GenericUnresolved {
    /// A short stable tag, for diagnostics.
    pub fn tag(self) -> &'static str {
        match self {
            Self::ClusteredRoots => "unresolved_clustered_roots",
            Self::InseparableMultipleRoots => "unresolved_inseparable_multiple_roots",
            Self::SingularJacobian => "unresolved_singular_jacobian",
            Self::UnsupportedSingularBranch => "unresolved_unsupported_singular_branch",
            Self::GeneralCommonComponent => "unresolved_general_common_component",
            Self::UnresolvedBoundaryRoot => "unresolved_boundary_root",
            Self::UnresolvedStationaryBranch => "unresolved_stationary_branch",
            Self::UnresolvedTangencyOrSingularity => "unresolved_tangency_or_singularity",
            Self::UnresolvedIdentityOrdering => "unresolved_identity_ordering",
            Self::UnresolvedIdentityReferBack => "unresolved_identity_refer_back",
            Self::Pair(_) => "unresolved_pair",
        }
    }
}

/// How two regular branches cross at an isolated event.
///
/// For a two-branch event this is the ARR-002 `ContactKind` lifted intact. For
/// a merged high-multiplicity event ARR-003 derives the classification from the
/// [`BranchGerm`] configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossingClassification {
    /// The vertical order of the two branches swaps across the event.
    Transverse,
    /// The vertical order is preserved (a tangential contact).
    Tangent,
    /// A stationary/singular contact not reducible to transverse/tangent.
    Stationary,
}

impl CrossingClassification {
    /// Lift an ARR-002 pairwise contact kind.
    pub fn from_pair(contact: ContactKind) -> Self {
        match contact {
            ContactKind::Transverse => Self::Transverse,
            ContactKind::Tangent => Self::Tangent,
        }
    }
}

/// Which end of a common arc an endpoint event sits at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommonArcEnd {
    /// The start end of the common interval.
    Start,
    /// The end end of the common interval.
    End,
}

/// One participant's occurrence identity within an isolated-root key.
///
/// Reversal-stable: [`CurveOccurrenceProvenance::reversed`] swaps only the
/// traversal start/end vertices and preserves the edge-use and source-edge ids,
/// so reparameterizing the same occurrence keeps this identity while reversing
/// only the branch-incidence orientation. A different twin edge use is a
/// distinct B-rep incidence with a distinct span id and is never conflated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IsolatedRootParticipant {
    /// The stable span/occurrence identity.
    pub span_id: SpanId,
}

/// The canonical identity of one isolated root of a curve pair.
///
/// Construction- and provenance-based, never coordinate proximity: the two
/// participant occurrences (sorted) plus a deterministic pair-local root
/// ordinal. See [`EventIdentity::IsolatedRoot`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IsolatedRootKey {
    /// The two participant occurrences, sorted canonically.
    pub participants: [IsolatedRootParticipant; 2],
    /// The deterministic pair-local root ordinal.
    pub ordinal: u32,
}

impl IsolatedRootKey {
    /// The canonical key for the given pair of span occurrences and ordinal.
    ///
    /// The participants are sorted, so an operand swap re-sorts to the same
    /// pair. Reversal does not change the span ids ([`CurveOccurrenceProvenance::reversed`]
    /// preserves them), so reparameterization reversal keeps the same pair.
    pub fn new(first_span_id: SpanId, second_span_id: SpanId, ordinal: u32) -> Self {
        let mut participants = [
            IsolatedRootParticipant { span_id: first_span_id },
            IsolatedRootParticipant { span_id: second_span_id },
        ];
        participants.sort();
        IsolatedRootKey { participants, ordinal }
    }
}

/// The stable identity of one arrangement event, for deduplication and merging.
///
/// Based on construction and provenance, never on coordinate proximity. Two
/// pairwise intersections that share a source vertex produce the same
/// [`EventIdentity::SharedSourceVertex`], so ARR-003 can merge them into one
/// multi-branch event. A representative `Point2<f64>` may be stored elsewhere
/// for rendering; it must not define equality or ordering identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EventIdentity {
    /// A physical source vertex shared by incident branches.
    SharedSourceVertex(SourceVertexKey),
    /// An artificial monotone-critical split shared by incident branches.
    SharedMonotoneCritical {
        /// The edge use the split belongs to.
        edge_use_id: EdgeUseId,
        /// The split's critical index.
        critical_index: i64,
    },
    /// One isolated certified root of a curve pair.
    ///
    /// **Canonical (GEN-001C).** The key is construction data, never
    /// coordinates: the two participant occurrences sorted canonically plus a
    /// deterministic pair-local root ordinal. The ordinal is assigned by one
    /// canonical isolation run, ordering the pair's certified roots by interval
    /// separation of their orientation-normalized authoritative source
    /// parameter boxes. It is stable under operand swap, reversal of either
    /// span, reversal of both spans, and deterministic repetition. Certified
    /// parameter boxes are evidence carried on the incident branches, never part
    /// of this identity.
    IsolatedRoot(IsolatedRootKey),
    /// A merged high-multiplicity event (≥3 branches), keyed by a stable
    /// construction id assigned by ARR-003 when it merges.
    MergedHighMultiplicity {
        /// A stable id for the merged event.
        construction_id: u64,
    },
    /// An endpoint of a common arc.
    CommonArcEndpoint {
        /// The common arc's stable id.
        arc_id: u64,
        /// Which end.
        end: CommonArcEnd,
    },
    /// An event identified with another by a certified deck translation.
    DeckTranslated {
        /// A stable id for the identified event.
        construction_id: u64,
        /// The deck displacement identifying the copy.
        deck: DeckLabel,
    },
    /// A collapsed quotient stratum (pole/apex). Representable in GEN-001A;
    /// populated only once the evidence seam certifies strata.
    CollapsedStratum {
        /// The collapsed stratum's stable id.
        stratum_id: u64,
    },
}

impl EventIdentity {
    /// Map an ARR-002 pairwise intersection identity to the generic identity.
    ///
    /// Two pairwise intersections sharing a source vertex map to the same
    /// [`EventIdentity::SharedSourceVertex`]; that is the handle ARR-003 merges
    /// on. An isolated curve-curve intersection maps to the canonical
    /// [`EventIdentity::IsolatedRoot`] keyed by the two span occurrences and
    /// the pair-local ordinal assigned by the caller (see [`lift_pair_result`]:
    /// the pair's roots are ordered by interval separation of their certified
    /// source-parameter boxes, never by discovery order or a per-curve index).
    pub fn from_pair(
        identity: &IntersectionIdentity,
        lhs_span_id: SpanId,
        rhs_span_id: SpanId,
        ordinal: u32,
    ) -> Self {
        match identity {
            IntersectionIdentity::SourceVertex(vertex) => Self::SharedSourceVertex(*vertex),
            IntersectionIdentity::ArtificialMonotoneSplit {
                edge_use_id,
                critical_index,
            } => Self::SharedMonotoneCritical {
                edge_use_id: *edge_use_id,
                critical_index: *critical_index,
            },
            IntersectionIdentity::CurveIntersection { .. } => {
                Self::IsolatedRoot(IsolatedRootKey::new(lhs_span_id, rhs_span_id, ordinal))
            }
        }
    }
}

/// One curve's involvement in one contact event.
///
/// The arrangement-facing atom: a span, the certified parameter enclosure at
/// which it meets the event, its certified local germ, its deck displacement,
/// and its provenance. The representative point is an evaluation hint, never
/// identity (A7: ARR-003 consumes the germ and the parameter enclosure, not the
/// representative coordinate).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BranchIncidence {
    /// The span this branch is a piece of.
    pub span_id: SpanId,
    /// The source occurrence provenance.
    pub provenance: CurveOccurrenceProvenance,
    /// Certified enclosure of the meeting parameter on this branch's
    /// authoritative domain.
    pub parameter: ParameterEnclosure,
    /// Where on the piece the meeting lies.
    pub location: ParameterLocation,
    /// The certified local branch behavior at the meeting parameter.
    pub germ: BranchGerm,
    /// The deck displacement of this branch's chart copy (rank-0 in GEN-001A;
    /// GEN-001D carries rank-1/rank-2 labels).
    pub deck: DeckLabel,
    /// A representative point (evaluation hint), never identity.
    pub representative: Point2,
}

/// An isolated contact event with an arbitrary number of incident branches.
#[derive(Debug, Clone, PartialEq)]
pub struct IsolatedEvent2 {
    /// The stable identity for deduplication and merging.
    pub identity: EventIdentity,
    /// How the branches cross. For a two-branch GEN-001A lift this is the
    /// ARR-002 contact kind; for a merged event ARR-003 derives it from the
    /// [`BranchGerm`] configuration.
    pub crossing: CrossingClassification,
    /// The incident branches. Two for an ARR-002 pairwise lift; ≥3 after
    /// ARR-003 merges by identity.
    pub branches: Vec<BranchIncidence>,
    /// A representative point (evaluation hint), never identity.
    pub representative: Point2,
}

/// The certified basis on which common support is claimed for a common arc.
///
/// GEN-001E populates the minimal cases; full common-factor extraction stays
/// deferred, but the variant set does not preclude it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommonSupportBasis {
    /// Identical source provenance: the same occurrence, traversed together.
    IdenticalSourceProvenance,
    /// Identical analytic support with certified overlapping parameter intervals.
    IdenticalAnalyticSupport,
    /// Identical homogeneous Bézier representation (identity or certified
    /// affine parameter reversal).
    IdenticalHomogeneousRepresentation,
    /// A general common component not covered by the minimal cases (deferred).
    Deferred,
}

/// The relative orientation of one occurrence along a common arc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrientationAlongSupport {
    /// Traversed in the same direction as the common support.
    Codirected,
    /// Traversed opposite the common support.
    Opposed,
}

/// One occurrence's participation in a common arc.
#[derive(Debug, Clone, PartialEq)]
pub struct ArcParticipant {
    /// The span this participant is a piece of.
    pub span_id: SpanId,
    /// The source occurrence provenance.
    pub provenance: CurveOccurrenceProvenance,
    /// The parameter interval of the common sub-arc on this occurrence
    /// `(start, end)`, as certified enclosures.
    pub parameter_interval: (ParameterEnclosure, ParameterEnclosure),
    /// Relative orientation along the common support.
    pub orientation: OrientationAlongSupport,
    /// This occurrence's multiplicity contribution.
    pub multiplicity: u8,
    /// The deck displacement of this participant's chart copy.
    pub deck: DeckLabel,
}

/// A positive-dimensional common-arc contact component.
///
/// One geometric support fragment shared by every participant, with
/// per-occurrence parameter correspondence, relative orientation and
/// multiplicity. GEN-001E populates the minimal certified cases
/// ([`CommonSupportBasis`]); the type does not preclude full common-factor
/// extraction, which stays deferred.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonArc2 {
    /// The stable identity (typically the common-arc endpoint identities).
    pub identity: EventIdentity,
    /// The participating occurrences.
    pub participants: Vec<ArcParticipant>,
    /// The certified basis on which common support is claimed.
    pub support_basis: CommonSupportBasis,
}

/// One contact component: an isolated event or a common arc.
#[derive(Debug, Clone, PartialEq)]
pub enum ContactComponent2 {
    /// An isolated (0-dimensional) event.
    IsolatedEvent(IsolatedEvent2),
    /// A positive-dimensional common arc.
    CommonArc(CommonArc2),
}

/// The generic pair-contact result.
///
/// Mirrors ARR-002's [`PairIntersectionResult`] in generic terms. `Unsupported`
/// is reused verbatim so no formal diagnostic is lost; `Unresolved` is widened
/// to [`GenericUnresolved`] so the Bézier isolation phases can report their own
/// typed non-results without changing the data model.
#[derive(Debug, Clone, PartialEq)]
pub enum PairContactResult {
    /// No contact.
    Disjoint,
    /// One or more contact components.
    Components(Vec<ContactComponent2>),
    /// The contact is valid geometry outside the admitted envelope.
    Unsupported(PairUnsupported),
    /// The contact cannot be certified under the declared numerical policy.
    Unresolved(GenericUnresolved),
}

impl PairContactResult {
    /// A short stable tag, for diagnostics.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Disjoint => "contact_disjoint",
            Self::Components(_) => "contact_components",
            Self::Unsupported(_) => "contact_unsupported",
            Self::Unresolved(_) => "contact_unresolved",
        }
    }
}

/// Build the two branch incidences of one ARR-002 pairwise intersection.
///
/// Every analytic branch is certified [`BranchGerm::Regular`]: a line/circle has
/// a nonzero first derivative everywhere on a nondegenerate piece. The deck
/// label is rank-0 zero in GEN-001A (GEN-001D carries the real label).
fn branches_from_pair(
    lhs_span: &CurveSpan2,
    rhs_span: &CurveSpan2,
    intersection: &CertifiedIntersection2,
) -> [BranchIncidence; 2] {
    let lhs = BranchIncidence {
        span_id: lhs_span.span_id(),
        provenance: *lhs_span.provenance(),
        parameter: intersection.lhs_parameter,
        location: intersection.lhs_location,
        germ: BranchGerm::Regular,
        deck: DeckLabel::ZERO,
        representative: intersection.point,
    };
    let rhs = BranchIncidence {
        span_id: rhs_span.span_id(),
        provenance: *rhs_span.provenance(),
        parameter: intersection.rhs_parameter,
        location: intersection.rhs_location,
        germ: BranchGerm::Regular,
        deck: DeckLabel::ZERO,
        representative: intersection.point,
    };
    [lhs, rhs]
}

/// Order a pair's intersections canonically for ordinal assignment.
///
/// The pair-local ordinal must be stable under operand swap and must not depend
/// on discovery order. Order by interval separation of the certified
/// source-parameter enclosures (lexicographic on the interval endpoints, which
/// agrees with separation order for disjoint certified roots). The rounded
/// midpoint is never used as an ordering key.
fn sort_intersections_canonically(intersections: &mut [CertifiedIntersection2]) {
    intersections.sort_by(|a, b| {
        let ka = (
            a.lhs_parameter.lo,
            a.lhs_parameter.hi,
            a.rhs_parameter.lo,
            a.rhs_parameter.hi,
        );
        let kb = (
            b.lhs_parameter.lo,
            b.lhs_parameter.hi,
            b.rhs_parameter.lo,
            b.rhs_parameter.hi,
        );
        ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Lift an ARR-002 pairwise result into the generic contact contracts.
///
/// Faithful: each isolated intersection becomes one [`IsolatedEvent2`] with two
/// [`BranchIncidence`] under the [`EventIdentity`] mapped from its
/// [`IntersectionIdentity`]. The pair's intersections are ordered canonically
/// before ordinals are assigned, so the identity of each isolated root is
/// stable under operand swap and does not depend on discovery order. `Disjoint`,
/// `Unsupported` and `Unresolved` pass through (the last wrapped in
/// [`GenericUnresolved::Pair`]). This does not merge pairwise results into
/// shared multi-branch events — that is ARR-003's job, using the stable
/// identities introduced here.
///
/// The germ — not the representative point — is what ARR-003 reads, so the A7
/// representative-endpoint issue is hidden beneath the generic contract here and
/// repaired in GEN-001C.
pub fn lift_pair_result(
    lhs_span: &CurveSpan2,
    rhs_span: &CurveSpan2,
    result: &PairIntersectionResult,
) -> PairContactResult {
    match result {
        PairIntersectionResult::Disjoint => PairContactResult::Disjoint,
        PairIntersectionResult::Unsupported(cause) => PairContactResult::Unsupported(*cause),
        PairIntersectionResult::Unresolved(cause) => {
            PairContactResult::Unresolved(GenericUnresolved::Pair(*cause))
        }
        PairIntersectionResult::Intersections(intersections) => {
            let mut intersections = intersections.clone();
            sort_intersections_canonically(&mut intersections);
            let lhs_span_id = lhs_span.span_id();
            let rhs_span_id = rhs_span.span_id();
            let components: Vec<ContactComponent2> = intersections
                .iter()
                .enumerate()
                .map(|(ordinal, intersection)| {
                    let [lhs_branch, rhs_branch] =
                        branches_from_pair(lhs_span, rhs_span, intersection);
                    ContactComponent2::IsolatedEvent(IsolatedEvent2 {
                        identity: EventIdentity::from_pair(
                            &intersection.identity,
                            lhs_span_id,
                            rhs_span_id,
                            ordinal as u32,
                        ),
                        crossing: CrossingClassification::from_pair(intersection.contact),
                        branches: vec![lhs_branch, rhs_branch],
                        representative: intersection.point,
                    })
                })
                .collect();
            if components.is_empty() {
                PairContactResult::Disjoint
            } else {
                PairContactResult::Components(components)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::curve2d::{DevelopedCurve2D, LineSegment2, SourceEdgeId, SourceEntityId, SourceFaceId};
    use super::super::intersection::{intersect_x_monotone, IntersectionPolicy};
    use super::super::xmonotone::{make_x_monotone, NumericalPolicy, XMonotonePiece2};
    use super::super::super::source_evidence::{BoundId, EdgeUseId, SourceVertexKey};
    use truck_geometry::prelude::Point2;

    fn provenance_with(
        edge_index: usize,
        start: SourceVertexKey,
        end: SourceVertexKey,
    ) -> CurveOccurrenceProvenance {
        CurveOccurrenceProvenance {
            source_face_id: Some(SourceFaceId(1)),
            bound_id: BoundId(0),
            edge_use_id: EdgeUseId::new(BoundId(0), edge_index),
            source_edge_id: SourceEdgeId(edge_index),
            start_vertex_id: start,
            end_vertex_id: end,
            source_curve_entity_id: Some(SourceEntityId(100 + edge_index as u64)),
        }
    }

    /// A line occurrence and its single x-monotone piece, with a span over the
    /// same occurrence.
    fn line_span_and_piece(
        start: Point2,
        end: Point2,
        edge_index: usize,
        sv: SourceVertexKey,
        ev: SourceVertexKey,
    ) -> (CurveSpan2, XMonotonePiece2) {
        let segment = LineSegment2 {
            start,
            end,
            provenance: provenance_with(edge_index, sv, ev),
        };
        let occ = DevelopedCurve2D::Line(segment);
        let piece = make_x_monotone(&occ, &NumericalPolicy::standard())
            .unwrap()
            .remove(0);
        (CurveSpan2::from_line(segment), piece)
    }

    fn intersect(
        lhs: &(CurveSpan2, XMonotonePiece2),
        rhs: &(CurveSpan2, XMonotonePiece2),
    ) -> PairContactResult {
        let result = intersect_x_monotone(&lhs.1, &rhs.1, &IntersectionPolicy::standard());
        lift_pair_result(&lhs.0, &rhs.0, &result)
    }

    #[test]
    fn disjoint_lifts_to_disjoint() {
        let a = line_span_and_piece(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            0,
            SourceVertexKey::ShellVertex(1),
            SourceVertexKey::ShellVertex(2),
        );
        let b = line_span_and_piece(
            Point2::new(0.0, 5.0),
            Point2::new(1.0, 5.0),
            1,
            SourceVertexKey::ShellVertex(3),
            SourceVertexKey::ShellVertex(4),
        );
        assert_eq!(intersect(&a, &b), PairContactResult::Disjoint);
    }

    #[test]
    fn transverse_interior_crossing_lifts_to_two_branch_event() {
        let a = line_span_and_piece(
            Point2::new(0.0, 0.5),
            Point2::new(1.0, 0.5),
            0,
            SourceVertexKey::ShellVertex(1),
            SourceVertexKey::ShellVertex(2),
        );
        let b = line_span_and_piece(
            Point2::new(0.5, 0.0),
            Point2::new(0.5, 1.0),
            1,
            SourceVertexKey::ShellVertex(3),
            SourceVertexKey::ShellVertex(4),
        );
        let lifted = intersect(&a, &b);
        let PairContactResult::Components(comps) = lifted else {
            panic!("expected components, got {lifted:?}");
        };
        assert_eq!(comps.len(), 1, "one transverse crossing");
        let ContactComponent2::IsolatedEvent(event) = &comps[0] else {
            panic!("expected an isolated event");
        };
        assert_eq!(event.crossing, CrossingClassification::Transverse);
        assert_eq!(event.branches.len(), 2, "two incident branches");
        assert!(
            event.branches.iter().all(|br| br.germ == BranchGerm::Regular),
            "analytic branches are regular"
        );
        assert!(
            event.branches.iter().all(|br| br.deck.is_zero()),
            "rank-0 deck label"
        );
        // Interior crossing: neither branch location is a source endpoint.
        assert!(event
            .branches
            .iter()
            .all(|br| br.location == ParameterLocation::PieceInterior));
        // Identity is a canonical isolated root (no shared source vertex here),
        // never a point.
        assert!(matches!(
            event.identity,
            EventIdentity::IsolatedRoot(_)
        ));
    }

    #[test]
    fn shared_vertex_lifts_to_shared_source_vertex_identity() {
        let shared = SourceVertexKey::ShellVertex(9);
        let a = line_span_and_piece(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            0,
            SourceVertexKey::ShellVertex(1),
            shared,
        );
        // A straight continuation through the shared vertex: collinear tangent
        // contact admitted as a source-declared join.
        let b = line_span_and_piece(
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            1,
            shared,
            SourceVertexKey::ShellVertex(2),
        );
        let lifted = intersect(&a, &b);
        let PairContactResult::Components(comps) = lifted else {
            panic!("expected components, got {lifted:?}");
        };
        let ContactComponent2::IsolatedEvent(event) = &comps[0] else {
            panic!("expected an isolated event");
        };
        assert_eq!(event.identity, EventIdentity::SharedSourceVertex(shared));
        assert_eq!(event.crossing, CrossingClassification::Tangent);
    }

    #[test]
    fn overlap_passes_through_as_unsupported() {
        let a = line_span_and_piece(
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            0,
            SourceVertexKey::ShellVertex(1),
            SourceVertexKey::ShellVertex(2),
        );
        let b = line_span_and_piece(
            Point2::new(1.0, 0.0),
            Point2::new(3.0, 0.0),
            1,
            SourceVertexKey::ShellVertex(3),
            SourceVertexKey::ShellVertex(4),
        );
        assert_eq!(
            intersect(&a, &b),
            PairContactResult::Unsupported(PairUnsupported::Overlap)
        );
    }

    #[test]
    fn identity_invariant_under_curve_order_swap_single_root() {
        // SINGLE-ROOT pair only. The canonical participants and ordinal 0 are
        // the same either way. A two-root operand-swap test -- proving each
        // geometric root keeps its identity -- is exercised in GEN-001C on the
        // generic Bézier path, where identities are keyed by canonical pair
        // ordinals rather than a per-curve index.
        let a = line_span_and_piece(
            Point2::new(0.0, 0.5),
            Point2::new(1.0, 0.5),
            0,
            SourceVertexKey::ShellVertex(1),
            SourceVertexKey::ShellVertex(2),
        );
        let b = line_span_and_piece(
            Point2::new(0.5, 0.0),
            Point2::new(0.5, 1.0),
            1,
            SourceVertexKey::ShellVertex(3),
            SourceVertexKey::ShellVertex(4),
        );
        let PairContactResult::Components(ab) = intersect(&a, &b) else {
            unreachable!()
        };
        let PairContactResult::Components(ba) = intersect(&b, &a) else {
            unreachable!()
        };
        let id_ab = match &ab[0] {
            ContactComponent2::IsolatedEvent(e) => e.identity.clone(),
            _ => unreachable!(),
        };
        let id_ba = match &ba[0] {
            ContactComponent2::IsolatedEvent(e) => e.identity.clone(),
            _ => unreachable!(),
        };
        // Single crossing: index 0 either way, so the identities are equal.
        // (Multi-root invariance is the GEN-001C test.)
        assert_eq!(id_ab, id_ba, "single-root event identity is invariant under swap");
    }
}
