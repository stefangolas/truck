//! The two-bound cylindrical band: an annulus bounded by two complete,
//! essential, oppositely oriented cylinder parallels.
//!
//! # What the legacy path does, and why this is not that
//!
//! A face trimmed by two independently closed periodic loops has no planar
//! boundary. The legacy tessellator manufactures one anyway, by joining the
//! two loops with artificial segments so that a single planar polygon exists
//! to hand to a triangulator. Those segments are not boundaries of anything,
//! they are placed by proximity, and on `00009190` they cross each other —
//! which is exactly the `SyntheticSyntheticCrossing` population this module
//! targets.
//!
//! The band is not a planar region and no planar polygon represents it. What
//! *does* represent it is a cut-open fundamental domain: cut the annulus once
//! transversally, develop the result into the cylinder's universal cover as a
//! single rectangle-like patch, and carry an explicit **edge identification**
//! saying that the patch's two artificial sides are one and the same cut. The
//! identification is discharged after triangulation by merging vertex
//! *identities* — never by welding coordinates — so no artificial edge
//! survives into the mesh.
//!
//! # The admitted case, in full
//!
//! - the support is a certified embedded cylinder ([`super::cylinder`]);
//! - the source face has exactly two authoritative bounds;
//! - each bound develops into one complete **simple** cylinder parallel:
//!   every occurrence is a circumferential arc, the developed chain is
//!   strictly monotone in the angular coordinate, and the terminal holonomy
//!   is primitive (`±1`);
//! - the two induced homologies are opposite;
//! - the two complete physical circle carriers are certified **distinct**,
//!   through a strict separation of their certified axial enclosures;
//! - therefore the carriers are strictly ordered along the axis and disjoint,
//!   and the compact strip between them is the material region.
//!
//! Two complete parallels on one cylinder are either the same circle or
//! separated along the axis; there is no third possibility. So a pair whose
//! enclosures overlap is *not* an annulus, and is reported as
//! [`BandExit::SameCarrier`] (when source vertex identity certifies the two
//! bounds run over the same vertices) or [`BandExit::CarrierIdentityUndecided`]
//! (when nothing in the source separates them). Neither reaches the realizer.
//!
//! Cones, tori, holes, islands, multiply wound boundaries, intersecting or
//! tangent boundaries and singular supports are refused by name, not
//! attempted.
//!
//! # Why the material region is forced, not chosen
//!
//! Two disjoint parallels cut the cylinder into three pieces: the compact
//! strip between them, and two non-compact ends. Only the strip is bounded by
//! *both* circles, so a face whose entire boundary is those two circles is
//! the strip. Nothing here reverses a boundary to make an orientation work:
//! the opposite-homology requirement is checked and refused, never repaired.
//!
//! # Why the patch is not ear-clipped
//!
//! [`super::planar_slice::triangulate`] is reused everywhere the formal
//! subtree needs a *disk*, and this module reuses its certificate
//! ([`certified_polygonal_region`]) and its checker ([`final_validity`])
//! unchanged. It cannot supply the patch's triangles, for a structural
//! reason rather than a stylistic one: ear clipping a convex patch fans it
//! from one vertex, and that apex is one of the two developed lifts of a cut
//! vertex, so the fan always contains a triangle carrying *both* lifts. That
//! triangle is degenerate the moment the identification is discharged. The
//! band's own triangulator ([`triangulate_band_patch`]) walks the two
//! boundary chains instead, which by construction never puts both lifts of
//! one cut vertex in one triangle — and its output is then handed to the
//! *same* [`final_validity`] battery, so the strengthening is in the producer
//! only, never in the checking.

use std::collections::{BTreeMap, BTreeSet};

use super::super::source_evidence::{
    BoundId, EdgeUseId, SourceBoundInput, SourceFaceInput, SourceVertexKey,
};
use super::cylinder::{CertifiedEmbeddedCylinder, CylinderSchema};
use super::curve_witness::{SourceCurveFamily, WitnessClass};
use super::cylinder_lift::{
    develop_traversal_from_source, propagate_placements, CylinderLiftExit,
};
use super::cylinder_mesh::lift_to_cylinder;
use super::numeric::NonNegativeFinite;
use super::planar_slice::{
    bounded_material_region, certified_polygonal_region, final_validity, jordan_arrangement_of,
    traverse_bound, CertificateRoute, CertifiedPlanarCurveOccurrence, CertifiedPolygonalRegion,
    FinalValidityReport, Rank0Displacement, Rank0DevelopedBoundary, SliceCategory, SliceExit,
    TriangulatedRegion,
};
use super::support::CurveSchema;
use truck_geometry::prelude::{Point2, Point3};
use truck_topology::compress::OuterBoundStanding;

/// The bound identity the two artificial cut sides are recorded under.
///
/// The cut sides are *not* source boundaries and must never be mistaken for
/// one, so they carry an identity no source bound can hold: source bounds are
/// indexed from zero by their position in the face, so `usize::MAX` is
/// unreachable for them by construction. Everything downstream that keys on
/// [`EdgeUseId`] therefore sees the cut for what it is.
const CUT_BOUND: BoundId = BoundId(usize::MAX);

/// The edge-use identity of the patch's right (leading) cut side.
const RIGHT_CUT: EdgeUseId = EdgeUseId::new(CUT_BOUND, 0);

/// The edge-use identity of the patch's left (trailing) cut side.
const LEFT_CUT: EdgeUseId = EdgeUseId::new(CUT_BOUND, 1);

/// The largest number of segments one developed arc is subdivided into.
///
/// The subdivision exists so the *lifted* chord stays within tolerance of the
/// physical circle (see [`arc_segment_count`]); the cap bounds the cost of
/// [`jordan_arrangement_of`]'s exhaustive `O(n^2)` pairwise certification on a
/// pathological tolerance, and a face that would need more is refused by the
/// polygonal certificate rather than approximated more coarsely in silence.
const MAXIMUM_ARC_SEGMENTS: usize = 512;

/// The fewest segments each boundary chain is cut into.
///
/// Three, for a topological reason and not a quality one. A chain with a
/// single segment puts its two endpoints — which are the two developed lifts
/// of one cut vertex — in one triangle of *any* triangulation, and that
/// triangle collapses when the identification is discharged. Two is already
/// enough to prevent it; three leaves a margin. Any physical tolerance asks
/// for far more than this, so the floor only guards a degenerate one.
const MINIMUM_CHAIN_SEGMENTS: usize = 3;

// ---------------------------------------------------------------------------
// Exits
// ---------------------------------------------------------------------------

/// Every way a face can fail to become a certified cylinder band, or fail to
/// be realized as one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BandExit {
    /// The face does not have exactly two authoritative bounds.
    NotTwoBounds {
        /// How many it declared.
        bounds: usize,
    },
    /// One bound's authoritative traversal did not close.
    BoundTraversal {
        /// Which bound.
        bound: BoundId,
        /// That stage's own exit, unchanged.
        exit: SliceExit,
    },
    /// One bound could not be developed into the universal cover, or its deck
    /// placement did not resolve.
    BoundDevelopment {
        /// Which bound.
        bound: BoundId,
        /// That stage's own exit, unchanged.
        exit: CylinderLiftExit,
    },
    /// One bound carries an occurrence that is not a circumferential arc, so
    /// it is not a cylinder parallel at all.
    BoundNotAParallel {
        /// Which bound.
        bound: BoundId,
    },
    /// One bound's developed chain reverses direction, so it is not a
    /// *simple* parallel.
    BoundNotMonotone {
        /// Which bound.
        bound: BoundId,
    },
    /// One bound's certified homology is not primitive: `0` is contractible
    /// (the [`super::cylinder_arrangement`] population, not this one) and
    /// `|h| > 1` is multiply wound, which this slice does not model.
    BoundNotPrimitive {
        /// Which bound.
        bound: BoundId,
        /// The certified holonomy.
        homology: i64,
    },
    /// The two complete circles are certified to be the same circle: the
    /// enclosures overlap and the two bounds run over the same source
    /// vertices. A degenerate face, never an annulus.
    SameCarrier,
    /// The two enclosures overlap and no source fact separates or identifies
    /// the two circles, so neither carrier relation is established.
    CarrierIdentityUndecided,
    /// The two induced boundary homologies have the same sign. Refused, not
    /// repaired by reversing one of them.
    OrientationIncompatible {
        /// The first bound's homology.
        first: i64,
        /// The second bound's homology.
        second: i64,
    },
    /// No authoritative source endpoint offered a uniquely certified periodic
    /// coordinate to cut at.
    CutCoordinateUnavailable,
    /// The cut-open patch failed one of the reused planar obligations —
    /// Jordan simplicity, material selection, the polygonal certificate, or
    /// the final validity battery. Carries that stage's exit unchanged.
    Patch(SliceExit),
    /// The identification collapsed a triangle: two of its corners are the
    /// same regluded vertex.
    RegluedDegenerateTriangle,
    /// An artificial cut edge survived the identification as a physical mesh
    /// boundary.
    RegluedCutSurvives,
    /// The regluded complex is not connected.
    RegluedNotConnected,
    /// The regluded complex does not have exactly two boundary components.
    RegluedBoundaryComponents {
        /// How many it has.
        components: usize,
    },
    /// The regluded complex's Euler characteristic is not zero.
    RegluedEulerCharacteristic {
        /// What it is.
        characteristic: i64,
    },
    /// Two triangles traverse a shared edge in the same direction, so the
    /// regluded complex is not consistently oriented.
    RegluedOrientationInconsistent,
    /// A lifted vertex was not finite.
    LiftNotFinite,
}

impl BandExit {
    /// Which semantic category this exit belongs to.
    ///
    /// The line is authority, as everywhere else in the subtree. A face that
    /// simply is not a band — wrong bound count, a non-parallel bound, a
    /// non-primitive or same-signed homology, a coincident carrier — is
    /// `Unsupported`: a proved fact about the face, outside the admitted
    /// subset. Missing evidence (an undecided carrier, no cut coordinate) is
    /// `Unresolved`. A reglue predicate failing *after* every input
    /// obligation was discharged is a defect in this module, not a verdict
    /// about the face, so it is `OperationalFailure`.
    pub fn category(self) -> SliceCategory {
        match self {
            Self::NotTwoBounds { .. }
            | Self::BoundNotAParallel { .. }
            | Self::BoundNotMonotone { .. }
            | Self::BoundNotPrimitive { .. }
            | Self::SameCarrier
            | Self::OrientationIncompatible { .. } => SliceCategory::Unsupported,

            Self::CarrierIdentityUndecided | Self::CutCoordinateUnavailable => {
                SliceCategory::Unresolved
            }

            Self::BoundTraversal { exit, .. } | Self::Patch(exit) => exit.category(),
            Self::BoundDevelopment { exit, .. } => exit.category(),

            Self::RegluedDegenerateTriangle
            | Self::RegluedCutSurvives
            | Self::RegluedNotConnected
            | Self::RegluedBoundaryComponents { .. }
            | Self::RegluedEulerCharacteristic { .. }
            | Self::RegluedOrientationInconsistent
            | Self::LiftNotFinite => SliceCategory::OperationalFailure,
        }
    }

    /// A short stable tag, for probe and DIAG-001 records.
    pub fn tag(self) -> &'static str {
        match self {
            Self::NotTwoBounds { .. } => "band_not_two_bounds",
            Self::BoundTraversal { exit, .. } => exit.tag(),
            Self::BoundDevelopment { exit, .. } => exit.tag(),
            Self::BoundNotAParallel { .. } => "band_bound_not_a_parallel",
            Self::BoundNotMonotone { .. } => "band_bound_not_monotone",
            Self::BoundNotPrimitive { .. } => "band_bound_not_primitive",
            Self::SameCarrier => "band_same_carrier",
            Self::CarrierIdentityUndecided => "band_carrier_identity_undecided",
            Self::OrientationIncompatible { .. } => "band_orientation_incompatible",
            Self::CutCoordinateUnavailable => "band_cut_coordinate_unavailable",
            Self::Patch(exit) => exit.tag(),
            Self::RegluedDegenerateTriangle => "band_reglue_degenerate_triangle",
            Self::RegluedCutSurvives => "band_reglue_cut_survives",
            Self::RegluedNotConnected => "band_reglue_not_connected",
            Self::RegluedBoundaryComponents { .. } => "band_reglue_boundary_components",
            Self::RegluedEulerCharacteristic { .. } => "band_reglue_euler_characteristic",
            Self::RegluedOrientationInconsistent => "band_reglue_orientation_inconsistent",
            Self::LiftNotFinite => "band_lift_not_finite",
        }
    }

    /// The stage this exit left from, for the funnel.
    pub fn stage(self) -> &'static str {
        match self {
            Self::NotTwoBounds { .. } => "bounds",
            Self::BoundTraversal { .. } => "traversal",
            Self::BoundDevelopment { .. } => "development",
            Self::BoundNotAParallel { .. }
            | Self::BoundNotMonotone { .. }
            | Self::BoundNotPrimitive { .. } => "parallel",
            Self::SameCarrier | Self::CarrierIdentityUndecided => "carrier",
            Self::OrientationIncompatible { .. } => "band",
            Self::CutCoordinateUnavailable => "plan",
            Self::Patch(_) => "patch",
            Self::RegluedDegenerateTriangle
            | Self::RegluedCutSurvives
            | Self::RegluedNotConnected
            | Self::RegluedBoundaryComponents { .. }
            | Self::RegluedEulerCharacteristic { .. }
            | Self::RegluedOrientationInconsistent
            | Self::LiftNotFinite => "reglue",
        }
    }
}

// ---------------------------------------------------------------------------
// Boundary components
// ---------------------------------------------------------------------------

/// One authoritative bound, developed into one complete simple cylinder
/// parallel with a certified primitive homology.
///
/// Everything the source declared survives: the [`BoundId`], the ordered
/// [`EdgeUseId`] sequence, and the source vertex identities the traversal
/// joined on. Nothing here is keyed on a coordinate.
#[derive(Debug, Clone)]
pub struct CompleteParallel {
    /// Which authoritative bound this is.
    pub bound: BoundId,
    /// The ordered edge-use identities, in source cyclic order.
    pub edge_uses: Vec<EdgeUseId>,
    /// The source vertices the traversal starts each occurrence at, in the
    /// same order.
    pub start_vertices: Vec<SourceVertexKey>,
    /// The developed start point of each occurrence, already placed on its
    /// certified deck copy: `(axial, angular)`, angular unwrapped.
    pub starts: Vec<Point2>,
    /// The developed end point of the *last* occurrence, placed. This is the
    /// second developed lift of `starts[0]`'s source vertex, one full deck
    /// period away — the boundary is closed because `end = start + h g`, not
    /// because a segment joins them.
    pub terminal: Point2,
    /// The certified terminal holonomy: `+1` or `-1`.
    pub homology: i64,
}

impl CompleteParallel {
    /// The developed angular coordinate this component's traversal begins at.
    pub fn start_angular(&self) -> f64 {
        self.starts[0].y
    }

    /// The source vertex the cut may be taken at: the first occurrence's
    /// start vertex.
    pub fn cut_vertex(&self) -> SourceVertexKey {
        self.start_vertices[0]
    }

    /// Every source vertex this component visits.
    pub fn source_vertices(&self) -> BTreeSet<SourceVertexKey> {
        self.start_vertices.iter().copied().collect()
    }
}

/// Develop one authoritative bound and certify it a complete simple parallel.
///
/// Reuses [`traverse_bound`] (Step 2's authoritative, source-identity-only
/// traversal), [`develop_traversal_from_source`] (Step 4's production witness
/// route, which reads each occurrence's curve family and declared sweep from
/// its own source representation) and [`propagate_placements`] (Step 5's deck
/// walk) without modification. The three obligations added here are exactly
/// the ones "complete simple parallel" means beyond "closed developed chain":
/// every occurrence circumferential, the chain strictly monotone, and the
/// holonomy primitive.
pub fn develop_complete_parallel(
    bound: &SourceBoundInput,
    schema: &CylinderSchema,
    curves: &mut impl FnMut(usize) -> CurveSchema,
    vertex_position: &impl Fn(SourceVertexKey) -> Option<Point3>,
    family_of: &impl Fn(EdgeUseId) -> SourceCurveFamily,
) -> Result<CompleteParallel, BandExit> {
    let id = bound.id();
    let traversal = traverse_bound(bound, curves)
        .map_err(|exit| BandExit::BoundTraversal { bound: id, exit })?;
    let developed =
        develop_traversal_from_source(&traversal, schema, vertex_position, family_of)
            .map_err(|exit| BandExit::BoundDevelopment { bound: id, exit })?;

    // A parallel is a circle at constant axial coordinate. An axial line in
    // the bound proves the component is not one, before any placement is
    // consulted.
    if developed
        .witnesses
        .iter()
        .any(|witness| witness.class != WitnessClass::CircumferentialArc)
    {
        return Err(BandExit::BoundNotAParallel { bound: id });
    }

    let generator = schema.deck_generator();
    let walk = propagate_placements(&developed, generator)
        .map_err(|exit| BandExit::BoundDevelopment { bound: id, exit })?;

    // Primitive: exactly one turn around the cylinder. `0` is contractible
    // and `|h| > 1` is multiply wound; both are refused by name.
    if walk.holonomy != 1 && walk.holonomy != -1 {
        return Err(BandExit::BoundNotPrimitive {
            bound: id,
            homology: walk.holonomy,
        });
    }

    // Place every occurrence on its certified deck copy. This is the only
    // arithmetic applied to a witness, and it is the deck displacement the
    // walk certified — never a nearest-copy choice.
    let period = generator.signed_period().get();
    let starts: Vec<Point2> = developed
        .witnesses
        .iter()
        .zip(&walk.placements)
        .map(|(witness, &placement)| {
            Point2::new(witness.start.x, witness.start.y + placement as f64 * period)
        })
        .collect();
    let last = developed.witnesses.len() - 1;
    let terminal = Point2::new(
        developed.witnesses[last].end.x,
        developed.witnesses[last].end.y + walk.placements[last] as f64 * period,
    );

    // Simple: the developed chain advances in one angular direction only. A
    // chain that reverses covers some angle twice and is not a simple circle,
    // even when its holonomy is primitive.
    let direction = walk.holonomy as f64 * period.signum();
    let mut previous = starts[0].y;
    for point in starts[1..].iter().chain(std::iter::once(&terminal)) {
        if (point.y - previous) * direction <= 0.0 {
            return Err(BandExit::BoundNotMonotone { bound: id });
        }
        previous = point.y;
    }

    Ok(CompleteParallel {
        bound: id,
        edge_uses: traversal
            .occurrences
            .iter()
            .map(|occurrence| occurrence.edge_use)
            .collect(),
        start_vertices: traversal
            .occurrences
            .iter()
            .map(|occurrence| occurrence.start_vertex)
            .collect(),
        starts,
        terminal,
        homology: walk.holonomy,
    })
}

// ---------------------------------------------------------------------------
// Physical carriers
// ---------------------------------------------------------------------------

/// The complete physical circle a boundary component is carried by.
///
/// A circle on a certified embedded cylinder is fixed by four facts, and this
/// type carries all four: the support cylinder (shared by both carriers of a
/// band, and recorded by the caller), the circle's radius, the axis-normal
/// plane it lies in — named by the **observed extent** of every developed
/// endpoint the complete component visits, together with the certified bound
/// each of those was evaluated to, rather than by a single rounded level —
/// and the complete extent, which is `winding = ±1` because the component was
/// certified a complete simple parallel before a carrier was built for it.
///
/// The observed extent is `[observed_low, observed_high]` over *every*
/// endpoint of the complete circle, and `enclosure` is the radius-scaled
/// bound [`super::curve_witness`] already certifies each witness's
/// on-cylinder and constant-axial-coordinate claims at. The circle's true
/// axial level therefore lies in `[axial_low(), axial_high()]`. No mean, no
/// centroid, no single sampled point, no polyline point count and no epsilon
/// chosen here enters any of it.
#[derive(Debug, Clone, PartialEq)]
pub struct CylinderCircleCarrier {
    /// Which bound this carrier came from.
    pub bound: BoundId,
    /// The lowest axial coordinate observed anywhere on the complete circle.
    pub observed_low: f64,
    /// The highest axial coordinate observed anywhere on the complete circle.
    pub observed_high: f64,
    /// The certified per-coordinate evaluation bound.
    pub enclosure: f64,
    /// The circle's radius: the cylinder's own, since the circle is a
    /// complete parallel of it.
    pub radius: f64,
    /// The complete extent, in turns: `+1` or `-1`.
    pub winding: i64,
}

impl CylinderCircleCarrier {
    /// The low end of the certified enclosure of the circle's axial level.
    pub fn axial_low(&self) -> f64 {
        self.observed_low - self.enclosure
    }

    /// The high end of the certified enclosure of the circle's axial level.
    pub fn axial_high(&self) -> f64 {
        self.observed_high + self.enclosure
    }

    /// The circle's centre, given the cylinder it is a parallel of: the axis
    /// point at the enclosure's midpoint.
    pub fn centre(&self, schema: &CylinderSchema) -> Point3 {
        schema.origin() + 0.5 * (self.axial_low() + self.axial_high()) * schema.axis()
    }
}

/// The radius-scaled bound each developed axial coordinate is certified to.
///
/// This is [`super::curve_witness`]'s own `RELATIVE_TOLERANCE * max(radius,
/// 1)` — the bound at which every witness feeding this carrier already had
/// its "on the cylinder" and "at constant axial coordinate" claims accepted.
/// Reusing it, rather than choosing a fresh constant here, is what makes the
/// enclosure a restatement of an obligation already discharged instead of a
/// new tolerance.
fn axial_enclosure_bound(schema: &CylinderSchema) -> f64 {
    1.0e-9 * schema.radius().get().max(1.0)
}

/// Build the complete physical circle carrier for a certified parallel.
pub fn carrier_of(parallel: &CompleteParallel, schema: &CylinderSchema) -> CylinderCircleCarrier {
    let bound = axial_enclosure_bound(schema);
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for point in parallel.starts.iter().chain(std::iter::once(&parallel.terminal)) {
        low = low.min(point.x);
        high = high.max(point.x);
    }
    CylinderCircleCarrier {
        bound: parallel.bound,
        observed_low: low,
        observed_high: high,
        enclosure: bound,
        radius: schema.radius().get(),
        winding: parallel.homology,
    }
}

/// How two complete physical circles on one cylinder relate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CarrierRelation {
    /// Certified distinct: the two axial enclosures are strictly disjoint, so
    /// the circles cannot be the same circle and are strictly ordered along
    /// the axis. Names which carrier is lower.
    DistinctCarrier {
        /// `true` when the first carrier is the lower one.
        first_is_lower: bool,
        /// The certified gap between the two enclosures.
        separation: f64,
    },
    /// Certified coincident: the two complete circles have the same support
    /// cylinder, the same radius, the same complete extent, and one certified
    /// enclosure covers every axial coordinate observed on *either* of them —
    /// so they are the same circle.
    SameCarrier,
    /// Neither relation is established: the enclosures meet, so the circles
    /// are not separated, but the observed extents are together too wide for
    /// one enclosure to cover, so they are not certified equal either.
    Undecided,
}

/// Classify two complete physical circles on one cylinder, from their
/// complete geometry alone.
///
/// Two complete parallels of one cylinder are the same circle exactly when
/// they have the same radius, the same complete extent and the same axial
/// level. All three are decided here on the *complete* certified geometry —
/// the observed axial extent of every endpoint of each circle, against the
/// bound those coordinates were certified to — never on a mean parameter
/// value, a centroid, one sampled point, one projected vertex, a polyline
/// point count, a source-representation coincidence, or a threshold chosen
/// here.
///
/// The classifier is deliberately three-way, because floating evaluation of a
/// level admits exactly three states and not two:
///
/// - **Distinct.** The two enclosures are strictly disjoint, so no pair of
///   true levels inside them can be equal. The circles are proved different
///   and strictly ordered along the axis.
/// - **Same.** The union of the two *observed* extents is no wider than one
///   certified enclosure, so every axial coordinate either circle presents
///   is consistent with one single level — at the very precision at which
///   every constituent witness was already accepted onto this cylinder.
///   Together with equal radius and equal complete extent, that is complete
///   circle equality.
/// - **Undecided.** Everything else. Not rounded into either verdict.
///
/// Only the first admits a band, so all three outcomes are safe.
///
/// Source-representation facts are deliberately *not* consulted here. Two
/// bounds running over the identical source vertex cycle is strong evidence
/// of a duplicated representation, and
/// [`carriers_share_source_cycle`] reports it, but it is a statement about
/// how the file was written rather than about where the two circles are, so
/// it corroborates this verdict and never substitutes for it.
pub fn classify_carriers(
    first: &CylinderCircleCarrier,
    second: &CylinderCircleCarrier,
) -> CarrierRelation {
    if first.axial_high() < second.axial_low() {
        return CarrierRelation::DistinctCarrier {
            first_is_lower: true,
            separation: second.axial_low() - first.axial_high(),
        };
    }
    if second.axial_high() < first.axial_low() {
        return CarrierRelation::DistinctCarrier {
            first_is_lower: false,
            separation: first.axial_low() - second.axial_high(),
        };
    }

    // Same support cylinder, same complete extent, same circle plane. The
    // radius comparison is exact because both carriers read it from the one
    // certified schema the face is trimmed from; it is stated rather than
    // assumed so the certificate names every fact a circle is fixed by.
    let same_circle_family =
        first.radius == second.radius && first.winding.abs() == second.winding.abs();
    let union_low = first.observed_low.min(second.observed_low);
    let union_high = first.observed_high.max(second.observed_high);
    let covered = union_high - union_low <= first.enclosure.max(second.enclosure);
    match same_circle_family && covered {
        true => CarrierRelation::SameCarrier,
        false => CarrierRelation::Undecided,
    }
}

/// Whether two boundary components run over the identical set of source
/// vertices.
///
/// Corroboration only, never a carrier test: a duplicated representation of
/// one circle presents this way, but so could two genuinely distinct circles
/// if a file reused vertex entities, and a single circle written out twice
/// with fresh vertices would not present this way at all. The physical
/// verdict is [`classify_carriers`]'s.
pub fn carriers_share_source_cycle(first: &CompleteParallel, second: &CompleteParallel) -> bool {
    let (first, second) = (first.source_vertices(), second.source_vertices());
    !first.is_empty()
        && first.iter().all(|vertex| vertex.is_identified())
        && first == second
}

// ---------------------------------------------------------------------------
// Band certification
// ---------------------------------------------------------------------------

/// A certified cylinder band: the compact strip between two distinct,
/// essential, oppositely oriented complete parallels of one cylinder.
#[derive(Debug, Clone)]
pub struct CertifiedCylinderBand {
    /// The document entity this face came from, when the importer retained it.
    pub source_face_id: Option<u64>,
    /// The certified embedded cylinder the band is trimmed from.
    pub cylinder: CertifiedEmbeddedCylinder,
    /// The boundary component whose carrier is lower along the axis.
    pub lower_boundary: CompleteParallel,
    /// The boundary component whose carrier is upper along the axis.
    pub upper_boundary: CompleteParallel,
    /// The lower carrier.
    pub lower_carrier: CylinderCircleCarrier,
    /// The upper carrier.
    pub upper_carrier: CylinderCircleCarrier,
    /// The certified gap between the two carriers' enclosures.
    pub separation: f64,
    /// The cylinder's signed deck period.
    pub period: f64,
}

impl CertifiedCylinderBand {
    /// The two boundary components in the source's own bound order, which is
    /// the order the cut-open patch traverses them in.
    fn in_source_order(&self) -> (&CompleteParallel, &CompleteParallel) {
        match self.lower_boundary.bound.0 <= self.upper_boundary.bound.0 {
            true => (&self.lower_boundary, &self.upper_boundary),
            false => (&self.upper_boundary, &self.lower_boundary),
        }
    }
}

/// Certify a two-bound cylinder face as a band.
///
/// Every conjunct of the admitted case is checked here, in an order chosen so
/// the *specific* obstruction is named: bound count, then each component's own
/// completeness, then the carriers, then the orientation. Nothing is repaired
/// to make a later conjunct hold.
pub fn certify_cylinder_band(
    source_face_id: Option<u64>,
    cylinder: CertifiedEmbeddedCylinder,
    input: &SourceFaceInput,
    curves: &mut impl FnMut(usize) -> CurveSchema,
    vertex_position: &impl Fn(SourceVertexKey) -> Option<Point3>,
    family_of: &impl Fn(EdgeUseId) -> SourceCurveFamily,
) -> Result<CertifiedCylinderBand, BandExit> {
    if input.bounds.len() != 2 {
        return Err(BandExit::NotTwoBounds {
            bounds: input.bounds.len(),
        });
    }
    let schema = cylinder.schema().clone();

    let first = develop_complete_parallel(
        &input.bounds[0],
        &schema,
        curves,
        vertex_position,
        family_of,
    )?;
    let second = develop_complete_parallel(
        &input.bounds[1],
        &schema,
        curves,
        vertex_position,
        family_of,
    )?;

    // Opposite induced orientations. Checked before the carriers so that a
    // same-signed pair is reported as the orientation fact it is, and never
    // repaired by reversing one component.
    if first.homology + second.homology != 0 {
        return Err(BandExit::OrientationIncompatible {
            first: first.homology,
            second: second.homology,
        });
    }

    let first_carrier = carrier_of(&first, &schema);
    let second_carrier = carrier_of(&second, &schema);
    let (first_is_lower, separation) = match classify_carriers(&first_carrier, &second_carrier) {
        CarrierRelation::DistinctCarrier {
            first_is_lower,
            separation,
        } => (first_is_lower, separation),
        CarrierRelation::SameCarrier => return Err(BandExit::SameCarrier),
        CarrierRelation::Undecided => return Err(BandExit::CarrierIdentityUndecided),
    };

    let (lower_boundary, upper_boundary, lower_carrier, upper_carrier) = match first_is_lower {
        true => (first, second, first_carrier, second_carrier),
        false => (second, first, second_carrier, first_carrier),
    };

    Ok(CertifiedCylinderBand {
        source_face_id,
        period: schema.deck_generator().signed_period().get(),
        cylinder,
        lower_boundary,
        upper_boundary,
        lower_carrier,
        upper_carrier,
        separation,
    })
}

// ---------------------------------------------------------------------------
// Cut-open construction
// ---------------------------------------------------------------------------

/// The explicit identification of the patch's two artificial sides.
///
/// The two sides are the two developed lifts of one cut arc, so the pairing
/// is `(left, v) ~ (right, v)` vertex by vertex. Each pair names two cycle
/// positions that are the two lifts of *one source vertex*, one deck period
/// apart — which is why discharging the identification is a statement about
/// identity and not about proximity.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeIdentification {
    /// The identified vertex pairs, as `(left cycle index, right cycle
    /// index)`.
    pub pairs: Vec<(usize, usize)>,
    /// The cycle position of the right cut segment.
    pub right_segment: usize,
    /// The cycle position of the left cut segment.
    pub left_segment: usize,
}

/// The plan for cutting one certified band open into a single patch.
#[derive(Debug, Clone)]
pub struct CutOpenDomainPlan {
    /// The periodic coordinate the band is cut at.
    pub cut_angular: f64,
    /// The authoritative source vertex that coordinate was read from.
    pub cut_vertex: SourceVertexKey,
    /// The deck shift applied to the first (source-order) component, in
    /// periods.
    pub first_shift: i64,
    /// The deck shift applied to the second component, in periods.
    pub second_shift: i64,
}

/// Choose the band's cut, deterministically and from source data alone.
///
/// The cut coordinate is the periodic coordinate of the lexicographically
/// earliest authoritative source endpoint that has a uniquely certified one:
/// bounds are ordered by their source position, occurrences by their position
/// within the bound, so the earliest such endpoint is the first occurrence of
/// the earlier bound. No centroid, no midpoint, and no geometric optimisation
/// enters the choice — two runs over the same file cut in the same place.
///
/// Each component is then placed in the single deck copy that carries it into
/// the period window *starting at the cut*: the copy whose lowest developed
/// angular coordinate lies in `[cut, cut + period)`. Both components then
/// cover the same turn of the cylinder, which is what "cut here" means — the
/// component that supplied the cut lands on `[cut, cut + period]` exactly,
/// whichever way round it winds.
///
/// This is a placement, not a re-derivation: it shifts a whole certified
/// chain by an integer number of periods and changes no coordinate's
/// relationship to any other. Choosing the copy by the chain's *start*
/// instead would put a component that winds the other way a full period below
/// the cut, which is a legal developed image of the same band but a useless
/// one — the two chains would then share no angular extent at all.
pub fn plan_cut_open(band: &CertifiedCylinderBand) -> Result<CutOpenDomainPlan, BandExit> {
    let (first, second) = band.in_source_order();
    let cut_vertex = first.cut_vertex();
    if !cut_vertex.is_identified() {
        return Err(BandExit::CutCoordinateUnavailable);
    }
    let cut_angular = first.start_angular();
    if !cut_angular.is_finite() {
        return Err(BandExit::CutCoordinateUnavailable);
    }
    let magnitude = band.period.abs();
    let lowest = |parallel: &CompleteParallel| -> f64 {
        parallel
            .starts
            .iter()
            .chain(std::iter::once(&parallel.terminal))
            .fold(f64::INFINITY, |low, point| low.min(point.y))
    };
    let shift_of = |low: f64| -> i64 { -(((low - cut_angular) / magnitude).floor() as i64) };
    Ok(CutOpenDomainPlan {
        cut_angular,
        cut_vertex,
        first_shift: shift_of(lowest(first)),
        second_shift: shift_of(lowest(second)),
    })
}

/// The cut-open planar patch: one rectangle-like domain in the developed
/// cylinder chart, certified simple, with its material region selected and
/// its artificial sides recorded as an identification rather than as
/// boundary.
#[derive(Debug, Clone)]
pub struct PlanarPatch {
    /// The certified polygonal region, through the reused planar
    /// certificates.
    pub region: CertifiedPolygonalRegion,
    /// The identification of the two artificial cut sides.
    pub identification: EdgeIdentification,
    /// How many vertices of the cycle belong to the lower-in-cycle-order
    /// chain, counting both lifts of its cut vertex.
    pub first_chain_vertices: usize,
    /// The sign of the angular direction both chains advance in.
    pub direction: f64,
}

/// How many segments one developed arc is cut into so that its *lifted* chord
/// stays within `tolerance` of the physical circle.
///
/// A chord subtending `delta` on a circle of radius `r` departs from it by at
/// most the sagitta `r (1 - cos(delta / 2))`, so `k` segments over a sweep
/// `s` need `r (1 - cos(s / 2k)) <= tolerance`, i.e. `k >= s / (2 acos(1 -
/// tolerance / r))`. That is a bound on the realized geometry, derived from
/// the certified radius and the caller's own tolerance; it is not a shape
/// heuristic and it does not touch the developed chart, where the arc is
/// exactly the straight segment the witness already represents.
fn arc_segment_count(sweep: f64, radius: f64, tolerance: f64, minimum: usize) -> usize {
    let minimum = minimum.max(1);
    let sweep = sweep.abs();
    if !sweep.is_finite() || !radius.is_finite() || radius <= 0.0 {
        return MAXIMUM_ARC_SEGMENTS;
    }
    if !(tolerance > 0.0) {
        return MAXIMUM_ARC_SEGMENTS;
    }
    let cosine = (1.0 - tolerance / radius).clamp(-1.0, 1.0);
    let half = cosine.acos();
    let needed = match half > 0.0 {
        true => (sweep / (2.0 * half)).ceil(),
        false => MAXIMUM_ARC_SEGMENTS as f64,
    };
    if !needed.is_finite() || needed >= MAXIMUM_ARC_SEGMENTS as f64 {
        return MAXIMUM_ARC_SEGMENTS.max(minimum);
    }
    (needed as usize).clamp(minimum, MAXIMUM_ARC_SEGMENTS)
}

/// Subdivide one placed developed arc into `count` collinear points.
///
/// The developed image of a circumferential arc *is* a straight segment, so
/// every interpolated point lies exactly on it: the subdivision adds
/// resolution to the lift without adding any approximation to the developed
/// chart. That is why the patch's polygonal certificate stays exact, and why
/// [`certified_polygonal_region`]'s exactness guard is satisfied here for the
/// same reason it is for a line-bounded planar face, not by an exemption.
fn subdivide(start: Point2, end: Point2, count: usize) -> Vec<Point2> {
    let mut points = Vec::with_capacity(count + 1);
    points.push(start);
    for step in 1..count {
        let t = step as f64 / count as f64;
        points.push(Point2::new(
            start.x + t * (end.x - start.x),
            start.y + t * (end.y - start.y),
        ));
    }
    points.push(end);
    points
}

/// Build one chain's placed, subdivided occurrences.
///
/// Each occurrence runs from its own placed developed start to the *next*
/// occurrence's placed developed start — the terminal for the last one — so
/// the chain is closed by the source's own vertex identities and by the deck
/// placement, never by reconciling two independently rounded copies of one
/// endpoint.
fn chain_occurrences(
    parallel: &CompleteParallel,
    shift: f64,
    radius: f64,
    tolerance: f64,
) -> Vec<CertifiedPlanarCurveOccurrence> {
    let count = parallel.starts.len();
    let minimum_per_occurrence = MINIMUM_CHAIN_SEGMENTS.div_ceil(count);
    let shifted = |p: Point2| Point2::new(p.x, p.y + shift);
    (0..count)
        .map(|index| {
            let start = shifted(parallel.starts[index]);
            let end = match index + 1 == count {
                true => shifted(parallel.terminal),
                false => shifted(parallel.starts[index + 1]),
            };
            let segments = arc_segment_count(
                end.y - start.y,
                radius,
                tolerance,
                minimum_per_occurrence,
            );
            CertifiedPlanarCurveOccurrence {
                edge_use: parallel.edge_uses[index],
                start_vertex: parallel.start_vertices[index],
                end_vertex: match index + 1 == count {
                    true => parallel.start_vertices[0],
                    false => parallel.start_vertices[index + 1],
                },
                points: subdivide(start, end, segments),
                route: CertificateRoute::AnalyticCylinderDevelopment,
                endpoint_reconciliation: NonNegativeFinite::new(0.0)
                    .expect("zero is a valid nonnegative bound"),
            }
        })
        .collect()
}

/// One artificial cut side, as an occurrence.
///
/// It carries [`CUT_BOUND`], which no source bound can hold, so nothing
/// downstream can mistake it for a physical boundary. It exists only to close
/// the patch's cycle and is removed again by the identification.
fn cut_occurrence(id: EdgeUseId, from: Point2, to: Point2) -> CertifiedPlanarCurveOccurrence {
    CertifiedPlanarCurveOccurrence {
        edge_use: id,
        start_vertex: SourceVertexKey::Absent,
        end_vertex: SourceVertexKey::Absent,
        points: vec![from, to],
        route: CertificateRoute::AnalyticCylinderDevelopment,
        endpoint_reconciliation: NonNegativeFinite::new(0.0)
            .expect("zero is a valid nonnegative bound"),
    }
}

/// Cut a certified band open into one planar patch.
///
/// The cycle is, in order: the first component's physical boundary, the right
/// artificial cut side, the second component's physical boundary (which runs
/// in the reverse angular direction, since its homology is the opposite one),
/// and the left artificial cut side. The two cut sides are exactly one deck
/// period apart and are the two lifts of one cut arc — a fact this function
/// establishes by *construction*, not by measuring them afterwards.
///
/// [`jordan_arrangement_of`] and [`bounded_material_region`] then certify the
/// patch simple and select its bounded complementary component, unchanged
/// from the planar slice. That component is the compact strip between the two
/// carriers, so the material region is established **before** any triangle
/// exists.
pub fn cut_open(
    band: &CertifiedCylinderBand,
    plan: &CutOpenDomainPlan,
    outer_bound: OuterBoundStanding,
    tolerance: f64,
) -> Result<PlanarPatch, BandExit> {
    let schema = band.cylinder.schema();
    let radius = schema.radius().get();
    let magnitude = band.period.abs();
    let (first, second) = band.in_source_order();

    let first_shift = plan.first_shift as f64 * magnitude;
    let second_shift = plan.second_shift as f64 * magnitude;

    let first_chain = chain_occurrences(first, first_shift, radius, tolerance);
    let second_chain = chain_occurrences(second, second_shift, radius, tolerance);

    let first_terminal = Point2::new(first.terminal.x, first.terminal.y + first_shift);
    let first_origin = Point2::new(first.starts[0].x, first.starts[0].y + first_shift);
    let second_terminal = Point2::new(second.terminal.x, second.terminal.y + second_shift);
    let second_origin = Point2::new(second.starts[0].x, second.starts[0].y + second_shift);

    let mut occurrences = first_chain;
    let first_chain_segments: usize = occurrences
        .iter()
        .map(|occurrence| occurrence.points.len() - 1)
        .sum();
    occurrences.push(cut_occurrence(RIGHT_CUT, first_terminal, second_origin));
    let second_chain_segments: usize = second_chain
        .iter()
        .map(|occurrence| occurrence.points.len() - 1)
        .sum();
    occurrences.extend(second_chain);
    occurrences.push(cut_occurrence(LEFT_CUT, second_terminal, first_origin));

    let arrangement = jordan_arrangement_of(&occurrences).map_err(BandExit::Patch)?;
    let material = bounded_material_region(arrangement, outer_bound).map_err(BandExit::Patch)?;

    let developed = Rank0DevelopedBoundary {
        displacements: vec![Rank0Displacement; occurrences.len()],
        occurrences,
    };
    let region =
        certified_polygonal_region(material, &developed, tolerance).map_err(BandExit::Patch)?;

    // Cycle layout, by construction: the first chain contributes one vertex
    // per segment starting at index 0, the right cut contributes the first
    // chain's terminal, the second chain follows, and the left cut
    // contributes the second chain's terminal last.
    let first_terminal_index = first_chain_segments;
    let second_origin_index = first_chain_segments + 1;
    let second_terminal_index = first_chain_segments + 1 + second_chain_segments;

    Ok(PlanarPatch {
        region,
        identification: EdgeIdentification {
            // `(left, v) ~ (right, v)`: the left side runs from the second
            // chain's terminal to the first chain's origin, the right side
            // from the first chain's terminal to the second chain's origin.
            // Each pair is two developed lifts of one source vertex.
            pairs: vec![
                (second_terminal_index, second_origin_index),
                (0, first_terminal_index),
            ],
            right_segment: first_terminal_index,
            left_segment: second_terminal_index,
        },
        first_chain_vertices: first_chain_segments + 1,
        direction: first.homology as f64 * band.period.signum(),
    })
}

// ---------------------------------------------------------------------------
// Realization
// ---------------------------------------------------------------------------

/// Triangulate the cut-open patch by walking its two boundary chains.
///
/// The patch is a rectangle-like domain between two chains that are each
/// strictly monotone in the angular coordinate and separated in the axial
/// one, both certified upstream. Advancing whichever chain's next vertex
/// comes first in that shared direction sweeps the patch exactly once, giving
/// `n - 2` triangles over the polygon's own vertices with no Steiner point —
/// the same shape [`final_validity`] checks a reused ear-clipped disk against,
/// which is why that battery is applied to this output unchanged.
///
/// Why not [`super::planar_slice::triangulate`]: see the module docs. Its fan
/// output always carries both developed lifts of one cut vertex in a single
/// triangle, which the identification then collapses.
pub fn triangulate_band_patch(patch: &PlanarPatch) -> Result<TriangulatedRegion, BandExit> {
    let vertices = patch.region.region.boundary.cycle.clone();
    let total = vertices.len();
    let first_count = patch.first_chain_vertices;
    if first_count < 3 || total < first_count + 3 {
        return Err(BandExit::Patch(SliceExit::TriangulationDidNotComplete));
    }

    // The first chain in cycle order; the second chain reversed, so both run
    // in the same angular direction.
    let lower: Vec<usize> = (0..first_count).collect();
    let upper: Vec<usize> = (first_count..total).rev().collect();

    let progress = |index: usize| patch.direction * vertices[index].y;

    let mut triangles = Vec::with_capacity(total - 2);
    let (mut i, mut j) = (0usize, 0usize);
    while i + 1 < lower.len() || j + 1 < upper.len() {
        // The two diagonals the identification cannot survive. `lower[0]` is
        // identified with `lower[nl - 1]` and `upper[0]` with `upper[nu - 1]`,
        // so a triangle edge joining `lower[0]` to `upper[nu - 1]`, or
        // `lower[nl - 1]` to `upper[0]`, becomes a *third* copy of the cut
        // edge once the identification is discharged. Each arises only by
        // running one chain to its end before the other has moved at all, so
        // each is excluded by holding the last step of a chain back until the
        // other chain has taken one. The strip is unaffected everywhere else:
        // the guards can only fire on a patch whose two chains barely overlap
        // in angle, and the reused validity battery still checks the result.
        let lower_would_finish_alone = i + 2 == lower.len() && j == 0 && upper.len() >= 2;
        let upper_would_finish_alone = j + 2 == upper.len() && i == 0 && lower.len() >= 2;
        let advance_lower = if i + 1 >= lower.len() {
            false
        } else if j + 1 >= upper.len() {
            true
        } else if lower_would_finish_alone {
            false
        } else if upper_would_finish_alone {
            true
        } else {
            progress(lower[i + 1]) <= progress(upper[j + 1])
        };
        match advance_lower {
            true => {
                triangles.push([lower[i], lower[i + 1], upper[j]]);
                i += 1;
            }
            false => {
                triangles.push([lower[i], upper[j + 1], upper[j]]);
                j += 1;
            }
        }
    }

    // Match the cycle's own handedness, so the complex is oriented the way
    // the certified region is. The walk emits one consistent handedness; this
    // only chooses which.
    let reference = patch.region.region.signed_area;
    if let Some(first) = triangles.first() {
        let orientation = orient(&vertices, *first);
        if orientation * reference < 0.0 {
            for triangle in &mut triangles {
                triangle.swap(1, 2);
            }
        }
    }

    Ok(TriangulatedRegion {
        vertices,
        triangles,
    })
}

fn orient(vertices: &[Point2], triangle: [usize; 3]) -> f64 {
    let [a, b, c] = triangle;
    let (a, b, c) = (vertices[a], vertices[b], vertices[c]);
    robust::orient2d(
        robust::Coord { x: a.x, y: a.y },
        robust::Coord { x: b.x, y: b.y },
        robust::Coord { x: c.x, y: c.y },
    )
}

/// What the regluded annular complex was proved to be.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandValidityReport {
    /// Triangles.
    pub triangles: usize,
    /// Distinct vertices after the identification.
    pub vertices: usize,
    /// Edges with exactly one incident triangle.
    pub boundary_edges: usize,
    /// Edges with exactly two.
    pub interior_edges: usize,
    /// Connected components of the boundary. Two, for an annulus.
    pub boundary_components: usize,
    /// `V - E + F`. Zero, for an annulus.
    pub euler_characteristic: i64,
}

/// The band's complete product: a validated annular mesh on the cylinder.
#[derive(Debug, Clone)]
pub struct CertifiedBandMesh {
    /// The developed complex, after the identification was discharged.
    pub developed: TriangulatedRegion,
    /// The cut-open patch's own final validity report, before the reglue.
    pub patch_validity: FinalValidityReport,
    /// The annular complex's validity report, after it.
    pub validity: BandValidityReport,
    /// The developed vertices lifted onto the cylinder, in the same order as
    /// `developed.vertices`.
    pub physical_vertices: Vec<Point3>,
}

/// Discharge the identification and validate the annulus.
///
/// The merge is by *identity*: each pair names two cycle positions the
/// construction already knows to be the two developed lifts of one source
/// vertex, and the right-hand one is rewritten to the left-hand one. No
/// coordinate is compared, no proximity threshold exists, and two vertices
/// that happen to coincide are not merged unless the identification says so.
///
/// The artificial sides disappear as a consequence, not as a separate step:
/// once both their endpoint pairs are identified they are the same edge, and
/// that edge then carries two triangles instead of one, so it is interior.
/// The check that it really did is [`BandExit::RegluedCutSurvives`].
pub fn reglue(
    patch: &PlanarPatch,
    developed: TriangulatedRegion,
    schema: &CylinderSchema,
) -> Result<CertifiedBandMesh, BandExit> {
    let patch_validity = final_validity(&developed, &patch.region).map_err(BandExit::Patch)?;

    let total = developed.vertices.len();
    let mut representative: Vec<usize> = (0..total).collect();
    for &(left, right) in &patch.identification.pairs {
        if left >= total || right >= total {
            return Err(BandExit::RegluedCutSurvives);
        }
        let (keep, drop) = (left.min(right), left.max(right));
        representative[drop] = keep;
    }
    // One pass suffices: no pair's target is itself identified away, because
    // the four cut endpoints are four distinct cycle positions forming two
    // disjoint pairs.
    let mut compacted: Vec<Option<usize>> = vec![None; total];
    let mut vertices = Vec::with_capacity(total);
    for index in 0..total {
        if representative[index] == index {
            compacted[index] = Some(vertices.len());
            vertices.push(developed.vertices[index]);
        }
    }
    let remap = |index: usize| compacted[representative[index]].expect("representatives are kept");

    let mut triangles = Vec::with_capacity(developed.triangles.len());
    for triangle in &developed.triangles {
        let [a, b, c] = *triangle;
        let mapped = [remap(a), remap(b), remap(c)];
        if mapped[0] == mapped[1] || mapped[1] == mapped[2] || mapped[0] == mapped[2] {
            return Err(BandExit::RegluedDegenerateTriangle);
        }
        triangles.push(mapped);
    }

    // Directed edge incidence, which decides orientation consistency and
    // manifoldness together: in a consistently oriented complex every
    // directed edge occurs exactly once.
    let mut directed: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut undirected: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for triangle in &triangles {
        let [a, b, c] = *triangle;
        for (p, q) in [(a, b), (b, c), (c, a)] {
            if !directed.insert((p, q)) {
                return Err(BandExit::RegluedOrientationInconsistent);
            }
            *undirected.entry((p.min(q), p.max(q))).or_insert(0) += 1;
        }
    }
    if undirected.values().any(|count| *count > 2) {
        return Err(BandExit::RegluedOrientationInconsistent);
    }

    let boundary: Vec<(usize, usize)> = undirected
        .iter()
        .filter(|(_, count)| **count == 1)
        .map(|(edge, _)| *edge)
        .collect();
    let interior_edges = undirected.values().filter(|count| **count == 2).count();

    // The cut must not have survived. Its two sides became one edge under the
    // identification; if that edge is still a boundary edge, the reglue did
    // not happen.
    let cut_edge = {
        let left = patch.identification.pairs[0];
        let right = patch.identification.pairs[1];
        let p = remap(left.0);
        let q = remap(right.0);
        (p.min(q), p.max(q))
    };
    if undirected.get(&cut_edge) != Some(&2) {
        return Err(BandExit::RegluedCutSurvives);
    }

    if !is_connected(vertices.len(), &triangles) {
        return Err(BandExit::RegluedNotConnected);
    }

    let boundary_components = count_boundary_components(&boundary);
    if boundary_components != 2 {
        return Err(BandExit::RegluedBoundaryComponents {
            components: boundary_components,
        });
    }

    let characteristic = vertices.len() as i64 - (boundary.len() + interior_edges) as i64
        + triangles.len() as i64;
    if characteristic != 0 {
        return Err(BandExit::RegluedEulerCharacteristic { characteristic });
    }

    let developed = TriangulatedRegion {
        vertices,
        triangles,
    };
    let physical_vertices = lift_to_cylinder(&developed, schema);
    if physical_vertices
        .iter()
        .any(|p| !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()))
    {
        return Err(BandExit::LiftNotFinite);
    }

    Ok(CertifiedBandMesh {
        validity: BandValidityReport {
            triangles: developed.triangles.len(),
            vertices: developed.vertices.len(),
            boundary_edges: boundary.len(),
            interior_edges,
            boundary_components,
            euler_characteristic: characteristic,
        },
        developed,
        patch_validity,
        physical_vertices,
    })
}

/// Whether the complex is connected, walking vertex adjacency across
/// triangles.
fn is_connected(vertices: usize, triangles: &[[usize; 3]]) -> bool {
    if vertices == 0 {
        return false;
    }
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); vertices];
    for [a, b, c] in triangles {
        for (p, q) in [(*a, *b), (*b, *c), (*c, *a)] {
            adjacency[p].push(q);
            adjacency[q].push(p);
        }
    }
    let mut seen = vec![false; vertices];
    let mut stack = vec![0usize];
    seen[0] = true;
    let mut count = 1;
    while let Some(current) = stack.pop() {
        for &next in &adjacency[current] {
            if !seen[next] {
                seen[next] = true;
                count += 1;
                stack.push(next);
            }
        }
    }
    count == vertices
}

/// How many connected components the boundary edge set forms.
fn count_boundary_components(boundary: &[(usize, usize)]) -> usize {
    let mut adjacency: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &(a, b) in boundary {
        adjacency.entry(a).or_default().push(b);
        adjacency.entry(b).or_default().push(a);
    }
    let mut seen: BTreeSet<usize> = BTreeSet::new();
    let mut components = 0;
    for &vertex in adjacency.keys() {
        if seen.contains(&vertex) {
            continue;
        }
        components += 1;
        let mut stack = vec![vertex];
        seen.insert(vertex);
        while let Some(current) = stack.pop() {
            for &next in adjacency.get(&current).into_iter().flatten() {
                if seen.insert(next) {
                    stack.push(next);
                }
            }
        }
    }
    components
}

/// The whole band path, composed: certify, plan the cut, cut open,
/// triangulate, reglue and lift.
pub fn run_cylinder_band(
    source_face_id: Option<u64>,
    cylinder: CertifiedEmbeddedCylinder,
    input: &SourceFaceInput,
    outer_bound: OuterBoundStanding,
    curves: &mut impl FnMut(usize) -> CurveSchema,
    vertex_position: &impl Fn(SourceVertexKey) -> Option<Point3>,
    family_of: &impl Fn(EdgeUseId) -> SourceCurveFamily,
    tolerance: f64,
) -> Result<(CertifiedCylinderBand, CertifiedBandMesh), BandExit> {
    let band = certify_cylinder_band(
        source_face_id,
        cylinder,
        input,
        curves,
        vertex_position,
        family_of,
    )?;
    let plan = plan_cut_open(&band)?;
    let patch = cut_open(&band, &plan, outer_bound, tolerance)?;
    let developed = triangulate_band_patch(&patch)?;
    let mesh = reglue(&patch, developed, band.cylinder.schema())?;
    Ok((band, mesh))
}

#[cfg(test)]
mod tests;
