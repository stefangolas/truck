//! Checkpoint 6: lift, deck placement, and terminal holonomy for a rank-1
//! cylinder-disk traversal.
//!
//! # The walk
//!
//! Step 2's traversal is already cyclically continuous by source vertex
//! identity ([`super::planar_slice::traverse_bound`]); this module does not
//! re-check that. What it adds is the developed side of the same claim: two
//! occurrences sharing a source vertex must develop to the *same* point in
//! the universal cover once each is placed on its own deck copy, and the
//! [`super::deck`] solver — never rounding, never re-derived — is what decides
//! which copy.
//!
//! For adjacent occurrences `i` and `i+1` with unplaced (raw) developed
//! endpoints `B_i` (the end of `i`) and `A_{i+1}` (the start of `i+1`), the
//! join is `B_i + n_i g = A_{i+1} + n_{i+1} g`, so `n_{i+1} - n_i` is exactly
//! the deck integer [`solve_axis_aligned`] returns for the point displacement
//! `B_i - A_{i+1}`. The walk fixes `n_0 = 0`, propagates every other `n_i`
//! from that equation, and — critically — does *not* fold the final join back
//! into `n_0`: the candidate `n_0'` it produces, compared against the fixed
//! `n_0 = 0`, *is* the terminal holonomy. Summing the individual developed
//! displacements around the loop would silently discard exactly the
//! information holonomy is: see `FORMAL_SYSTEM.md`'s ban on that shortcut,
//! repeated in this module's own tests.
//!
//! # What is out of scope here
//!
//! Real STEP circular-arc curves have no structural reader yet
//! ([`super::support`] reads only lines and polylines), so which witness class
//! an edge use gets is supplied by the caller via [`WitnessSpec`] rather than
//! inferred from [`super::support::CurveSchema`]. Wiring that inference to the
//! importer is production-integration work (Checkpoint 10), not this
//! checkpoint's concern.

use super::super::source_evidence::{EdgeUseId, SourceVertexKey};
use super::cylinder::CylinderSchema;
use super::curve_witness::{
    axial_line_witness, circumferential_arc_witness, CurveOnCylinderWitness, WitnessFailure,
};
use super::deck::{
    solve_axis_aligned, DeckGenerator, DeckInterval, DeckOperationalFailure, DeckSolveResult,
    DevelopedBox,
};
use super::planar_slice::{RegularClosedTraversal, SliceCategory};
use truck_geometry::prelude::Point3;

/// Which curve family one traversal occurrence's edge use presents.
///
/// Supplied by the caller because the structural curve reader for circular
/// arcs does not exist yet (see the module docs); this type carries exactly
/// the evidence [`circumferential_arc_witness`] needs beyond the endpoints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WitnessSpec {
    /// An axial line edge use.
    AxialLine,
    /// A circumferential arc edge use, with its source curve's own declared
    /// signed angular sweep.
    CircumferentialArc {
        /// See [`circumferential_arc_witness`]'s `declared_sweep`.
        declared_sweep: f64,
    },
}

/// Why the lift, placement or holonomy stage could not certify a zero-
/// holonomy disk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CylinderLiftExit {
    /// An occurrence's start or end source vertex has no certified position.
    MissingVertexPosition {
        /// The occurrence whose endpoint has no certified position.
        edge_use: EdgeUseId,
    },
    /// A curve-on-cylinder witness could not be certified for an occurrence.
    WitnessConstruction {
        /// The occurrence whose witness failed.
        edge_use: EdgeUseId,
        /// Why.
        cause: WitnessFailure,
    },
    /// A join's developed displacement is not compatible with any deck
    /// integer: the shared source vertex is proved inconsistent with the
    /// certified cylinder period.
    JoinNoCompatibleInteger {
        /// Index of the join, between occurrence `join_index` and the next
        /// one in cyclic order.
        join_index: usize,
    },
    /// A join's evidence admits more than one deck integer. For a point
    /// displacement against a nonzero period this is a genuine ambiguity in
    /// the evidence, not a fact about the face.
    JoinMultipleCompatibleIntegers {
        /// See [`Self::JoinNoCompatibleInteger`].
        join_index: usize,
    },
    /// The period is too small, at the displacement's scale, for the
    /// arithmetic enclosure to resolve which deck integer applies.
    JoinIndeterminate {
        /// See [`Self::JoinNoCompatibleInteger`].
        join_index: usize,
    },
    /// The deck solver could not complete the join arithmetically.
    JoinOperationalFailure {
        /// See [`Self::JoinNoCompatibleInteger`].
        join_index: usize,
        /// Why.
        failure: DeckOperationalFailure,
    },
    /// Every join resolved uniquely, but the terminal holonomy is a nonzero
    /// certified deck integer. Outside the initial realization subset (see
    /// the module docs and `FORMAL_SYSTEM.md`); reported rather than
    /// silently reduced to zero.
    NonzeroHolonomy {
        /// The certified nonzero holonomy, in units of the deck generator.
        holonomy: i64,
    },
}

impl CylinderLiftExit {
    /// Which semantic category this exit belongs to.
    pub fn category(self) -> SliceCategory {
        match self {
            Self::MissingVertexPosition { .. } => SliceCategory::Unresolved,
            Self::WitnessConstruction { cause, .. } => cause.category(),
            Self::JoinNoCompatibleInteger { .. } => SliceCategory::Inconsistent,
            Self::JoinMultipleCompatibleIntegers { .. } | Self::JoinIndeterminate { .. } => {
                SliceCategory::Unresolved
            }
            Self::JoinOperationalFailure { .. } => SliceCategory::OperationalFailure,
            Self::NonzeroHolonomy { .. } => SliceCategory::Unsupported,
        }
    }

    /// A short stable tag, for probe records.
    pub fn tag(self) -> &'static str {
        match self {
            Self::MissingVertexPosition { .. } => "lift_missing_vertex_position",
            Self::WitnessConstruction { .. } => "lift_witness_construction_failed",
            Self::JoinNoCompatibleInteger { .. } => "lift_join_no_compatible_integer",
            Self::JoinMultipleCompatibleIntegers { .. } => "lift_join_multiple_compatible_integers",
            Self::JoinIndeterminate { .. } => "lift_join_indeterminate",
            Self::JoinOperationalFailure { .. } => "lift_join_operational_failure",
            Self::NonzeroHolonomy { .. } => "lift_nonzero_holonomy",
        }
    }
}

/// Step 4's rank-1 product: one developed witness per source occurrence, in
/// source cyclic order, each still in its own unplaced (raw) developed frame.
#[derive(Debug, Clone)]
pub struct DevelopedCylinderBoundary {
    /// The edge use each witness represents, in source cyclic order.
    pub edge_uses: Vec<EdgeUseId>,
    /// The raw (unplaced) witnesses, in the same order.
    pub witnesses: Vec<CurveOnCylinderWitness>,
}

/// Step 4. Develop every traversal occurrence into a curve-on-cylinder
/// witness, using the caller's certified vertex positions and witness
/// classification.
///
/// The traversal's own endpoint pairing is reused unchanged:
/// `occurrence.start_vertex` / `occurrence.end_vertex` are exactly what Step
/// 2 already proved cyclically continuous by source identity. This function
/// substitutes one canonical position per source vertex — never each
/// occurrence's own curve endpoint — so that two occurrences sharing a
/// vertex are guaranteed to develop from bit-identical input, the same
/// discipline [`super::planar_slice::certified_planar_curves`] applies to the
/// planar case.
pub fn develop_traversal(
    traversal: &RegularClosedTraversal,
    schema: &CylinderSchema,
    vertex_position: &impl Fn(SourceVertexKey) -> Option<Point3>,
    witness_spec: &impl Fn(EdgeUseId) -> WitnessSpec,
) -> Result<DevelopedCylinderBoundary, CylinderLiftExit> {
    let mut edge_uses = Vec::with_capacity(traversal.occurrences.len());
    let mut witnesses = Vec::with_capacity(traversal.occurrences.len());

    for occurrence in &traversal.occurrences {
        let start = vertex_position(occurrence.start_vertex).ok_or(
            CylinderLiftExit::MissingVertexPosition {
                edge_use: occurrence.edge_use,
            },
        )?;
        let end = vertex_position(occurrence.end_vertex).ok_or(
            CylinderLiftExit::MissingVertexPosition {
                edge_use: occurrence.edge_use,
            },
        )?;

        let witness = match witness_spec(occurrence.edge_use) {
            WitnessSpec::AxialLine => axial_line_witness(schema, start, end),
            WitnessSpec::CircumferentialArc { declared_sweep } => {
                circumferential_arc_witness(schema, start, end, declared_sweep)
            }
        }
        .map_err(|cause| CylinderLiftExit::WitnessConstruction {
            edge_use: occurrence.edge_use,
            cause,
        })?;

        edge_uses.push(occurrence.edge_use);
        witnesses.push(witness);
    }

    Ok(DevelopedCylinderBoundary {
        edge_uses,
        witnesses,
    })
}

/// Step 5's rank-1 product: a certified deck placement for every occurrence,
/// with terminal holonomy proved structurally zero.
#[derive(Debug, Clone)]
pub struct ZeroHolonomyLift {
    /// The certified deck integer for each occurrence, in traversal order,
    /// with `placements[0] == 0` by the walk's fixed starting placement.
    pub placements: Vec<i64>,
    /// How many joins were checked, including the final-to-initial wrap.
    pub joins_checked: usize,
}

/// Solve one join: the deck integer `k` with `A + k g` compatible with `B`,
/// where `B` is the unplaced developed end of one occurrence and `A` the
/// unplaced developed start of the next.
///
/// `tolerance` widens each component into `[value - tolerance, value +
/// tolerance]` rather than passing a bit-exact point to the deck solver. Both
/// sides of a join are evaluated through genuinely different code paths for
/// the same physical vertex — one occurrence's `atan2`-based
/// `angular_coordinate`, the other's `theta_start + declared_sweep`
/// arithmetic — so their difference at a true zero-holonomy join is not
/// exactly `0.0`, only within a few ULPs of it. Without this enclosure the
/// deck solver would correctly, but uselessly, refuse every join as
/// `NoCompatibleInteger`: a zero-width interval demands exact equality to an
/// integer multiple of the period, which floating evaluation of two different
/// expressions never gives even when the underlying claim is true.
fn solve_join(
    generator: &DeckGenerator,
    b: truck_geometry::prelude::Point2,
    a: truck_geometry::prelude::Point2,
    tolerance: f64,
) -> Result<DeckSolveResult, DeckOperationalFailure> {
    // The developed convention is (axial = First, angular = Second); see
    // `super::cylinder`'s module docs.
    let axial = DeckInterval::from_f64(b.x - a.x - tolerance, b.x - a.x + tolerance)
        .map_err(|_| DeckOperationalFailure::ArithmeticOverflow)?;
    let angular = DeckInterval::from_f64(b.y - a.y - tolerance, b.y - a.y + tolerance)
        .map_err(|_| DeckOperationalFailure::ArithmeticOverflow)?;
    let displacement = DevelopedBox {
        first: axial,
        second: angular,
    };
    solve_axis_aligned(generator, &displacement)
}

/// Step 5. Propagate deck placements around the developed boundary from a
/// fixed initial placement, and classify the terminal holonomy.
///
/// `tolerance` is the certified bound on the evaluation discrepancy between
/// two independent developed-coordinate computations of one physical vertex
/// (see [`solve_join`]) — not a geometric approximation of the curve itself,
/// which the witnesses already represent exactly.
///
/// Certifies only `h = 0`: a certified nonzero holonomy is reported as
/// [`CylinderLiftExit::NonzeroHolonomy`] and refused, per the supported
/// subset. Every join is classified before any placement past it is used, so
/// a join that does not resolve uniquely stops the walk with the specific
/// [`CylinderLiftExit`] naming it — this function never guesses a placement
/// to keep going.
pub fn propagate_and_classify_holonomy(
    boundary: &DevelopedCylinderBoundary,
    generator: DeckGenerator,
    tolerance: f64,
) -> Result<ZeroHolonomyLift, CylinderLiftExit> {
    let witnesses = &boundary.witnesses;
    let count = witnesses.len();
    // A degenerate (empty or single-occurrence) boundary has no closed walk
    // to classify; the caller's traversal-continuity check already refuses
    // this shape before reaching here in every real path.
    debug_assert!(count > 0, "traverse_bound refuses an empty traversal");

    let mut placements = vec![0i64; count];

    let classify = |join_index: usize,
                     result: Result<DeckSolveResult, DeckOperationalFailure>|
     -> Result<i64, CylinderLiftExit> {
        match result {
            Ok(DeckSolveResult::Unique(k)) => Ok(k),
            Ok(DeckSolveResult::NoCompatibleInteger) => {
                Err(CylinderLiftExit::JoinNoCompatibleInteger { join_index })
            }
            Ok(DeckSolveResult::MultipleCompatibleIntegers) => {
                Err(CylinderLiftExit::JoinMultipleCompatibleIntegers { join_index })
            }
            Ok(DeckSolveResult::Indeterminate) => {
                Err(CylinderLiftExit::JoinIndeterminate { join_index })
            }
            Err(failure) => Err(CylinderLiftExit::JoinOperationalFailure {
                join_index,
                failure,
            }),
        }
    };

    // Propagate n_1 .. n_{count-1} from the fixed n_0 = 0. Every join here
    // uses the *raw* (unplaced) developed end and start; the running
    // placement is folded in only through the accumulated `k`s, never by
    // re-deriving a coordinate.
    for i in 0..count.saturating_sub(1) {
        let b = witnesses[i].end;
        let a = witnesses[i + 1].start;
        let k = classify(i, solve_join(&generator, b, a, tolerance))?;
        placements[i + 1] = placements[i] + k;
    }

    // The final join closes the cycle back to occurrence 0 without
    // overwriting its fixed placement: the candidate it implies for n_0 is
    // read off directly as the terminal holonomy, and the walk stays "cut
    // open" exactly as the module docs require.
    let last = count - 1;
    let b_last = witnesses[last].end;
    let a_first = witnesses[0].start;
    let k_last = classify(last, solve_join(&generator, b_last, a_first, tolerance))?;
    let holonomy = placements[last] + k_last;

    if holonomy != 0 {
        return Err(CylinderLiftExit::NonzeroHolonomy { holonomy });
    }

    Ok(ZeroHolonomyLift {
        placements,
        joins_checked: count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::source_evidence::{
        BoundId, ErasedOrientationMechanism, OrientationEvidence, OrientationOrigin,
        SourceBoundInput, SourceEdgeOrientationEvidence, SourceEdgeUseInput, SourceFaceInput,
        SourceFaceOrientationEvidence,
    };
    use super::super::cylinder::{identify_cylinder, CylinderIdentification};
    use super::super::cylinder_face::build_cylinder_face;
    use super::super::support::identify_line_segment;
    use truck_geometry::prelude::{Line, RevolutedCurve, Vector3};
    use truck_topology::compress::OuterBoundStanding;

    fn z_cylinder(radius: f64, h: f64) -> super::super::cylinder::CertifiedEmbeddedCylinder {
        let revo = RevolutedCurve::by_revolution(
            Line(Point3::new(radius, 0.0, 0.0), Point3::new(radius, 0.0, h)),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );
        match identify_cylinder(&revo) {
            CylinderIdentification::Cylinder(c) => c,
            other => panic!("expected a certified cylinder, got {other:?}"),
        }
    }

    fn on_cylinder(schema: &CylinderSchema, z: f64, theta: f64) -> Point3 {
        schema.origin()
            + z * schema.axis()
            + schema.radius().get() * theta.cos() * schema.radial_x()
            + schema.radius().get() * theta.sin() * schema.radial_y()
    }

    /// A closed quad face on the cylinder: `points[i] -> points[i+1]`
    /// cyclically. Every edge use runs forward; the reversal hazard is
    /// exercised separately in `curve_witness` and `cylinder_face`. Which
    /// witness class each edge use gets is supplied separately by the caller
    /// to `develop_traversal`.
    fn quad_face(points: &[Point3]) -> (SourceFaceInput, Vec<Point3>) {
        let n = points.len();
        let mut edge_uses = Vec::with_capacity(n);
        for i in 0..n {
            let j = (i + 1) % n;
            edge_uses.push(SourceEdgeUseInput {
                id: EdgeUseId::new(BoundId(0), i),
                source_edge_index: i,
                source_vertices: (
                    SourceVertexKey::ShellVertex(i),
                    SourceVertexKey::ShellVertex(j),
                ),
                use_vertices: (
                    SourceVertexKey::ShellVertex(i),
                    SourceVertexKey::ShellVertex(j),
                ),
                orientation: SourceEdgeOrientationEvidence {
                    bound_times_oriented_edge: OrientationEvidence::Retained {
                        forward: true,
                        origin: OrientationOrigin::BoundTimesOrientedEdge,
                    },
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
        let input = SourceFaceInput {
            source_face_id: Some(99),
            declared_face_index: 0,
            bounds: vec![SourceBoundInput::EdgeUses { id: BoundId(0), edge_uses }],
            orientation: SourceFaceOrientationEvidence {
                face_use_orientation: OrientationEvidence::Missing,
                face_surface_same_sense: OrientationEvidence::Missing,
            },
        };
        (input, points.to_vec())
    }

    fn declared_outer() -> OuterBoundStanding {
        OuterBoundStanding::Declared {
            bound_index: 0,
            declared_count: 1,
        }
    }

    /// The first required end-to-end milestone: traversal -> witnesses ->
    /// unique placements -> h = 0, on a simple non-wrapping quad patch.
    #[test]
    fn a_simple_quad_reaches_zero_holonomy() {
        let cylinder = z_cylinder(2.0, 5.0);
        let schema = cylinder.schema().clone();
        let points = [
            on_cylinder(&schema, 0.0, 0.2),
            on_cylinder(&schema, 0.0, 1.4),
            on_cylinder(&schema, 3.0, 1.4),
            on_cylinder(&schema, 3.0, 0.2),
        ];
        let specs = vec![
            WitnessSpec::CircumferentialArc { declared_sweep: 1.2 },
            WitnessSpec::AxialLine,
            WitnessSpec::CircumferentialArc { declared_sweep: -1.2 },
            WitnessSpec::AxialLine,
        ];
        let (input, positions) = quad_face(&points);
        let curves: Vec<_> = (0..4)
            .map(|i| identify_line_segment(&Line(points[i], points[(i + 1) % 4])))
            .collect();

        let record = build_cylinder_face(
            input.source_face_id,
            cylinder,
            &input,
            declared_outer(),
            &mut |index| curves[index].clone(),
        )
        .expect("regular traversal resolves");

        let vertex_position = |key: SourceVertexKey| match key {
            SourceVertexKey::ShellVertex(index) => positions.get(index).copied(),
            SourceVertexKey::Absent => None,
        };
        let spec_by_use = |edge_use: EdgeUseId| specs[edge_use.index];

        let boundary =
            develop_traversal(&record.traversal, &schema, &vertex_position, &spec_by_use)
                .expect("every occurrence develops to a witness");
        assert_eq!(boundary.witnesses.len(), 4);

        let lift = propagate_and_classify_holonomy(&boundary, schema.deck_generator(), 1e-9)
            .expect("zero holonomy on a non-wrapping quad");
        assert_eq!(lift.placements[0], 0);
        assert_eq!(lift.joins_checked, 4);
    }

    /// The same milestone, but with a seam crossing in the middle of the
    /// walk: two joins solve to nonzero deck integers (`+1` then `-1`) and
    /// still cancel to `h = 0`. This is what exercises the deck solver
    /// nontrivially rather than trivially placing every occurrence at `n =
    /// 0`.
    #[test]
    fn a_seam_crossing_quad_reaches_zero_holonomy_through_nontrivial_placements() {
        let cylinder = z_cylinder(2.0, 5.0);
        let schema = cylinder.schema().clone();
        // theta = 3.0 is inside (-PI, PI]; theta = 3.5 is not, and its
        // physical direction's principal branch is 3.5 - 2*PI.
        let points = [
            on_cylinder(&schema, 0.0, 3.0),
            on_cylinder(&schema, 0.0, 3.5),
            on_cylinder(&schema, 3.0, 3.5),
            on_cylinder(&schema, 3.0, 3.0),
        ];
        let specs = vec![
            WitnessSpec::CircumferentialArc { declared_sweep: 0.5 },
            WitnessSpec::AxialLine,
            WitnessSpec::CircumferentialArc { declared_sweep: -0.5 },
            WitnessSpec::AxialLine,
        ];
        let (input, positions) = quad_face(&points);
        let curves: Vec<_> = (0..4)
            .map(|i| identify_line_segment(&Line(points[i], points[(i + 1) % 4])))
            .collect();

        let record = build_cylinder_face(
            input.source_face_id,
            cylinder,
            &input,
            declared_outer(),
            &mut |index| curves[index].clone(),
        )
        .expect("regular traversal resolves");

        let vertex_position = |key: SourceVertexKey| match key {
            SourceVertexKey::ShellVertex(index) => positions.get(index).copied(),
            SourceVertexKey::Absent => None,
        };
        let spec_by_use = |edge_use: EdgeUseId| specs[edge_use.index];

        let boundary =
            develop_traversal(&record.traversal, &schema, &vertex_position, &spec_by_use)
                .expect("every occurrence develops to a witness");

        let lift = propagate_and_classify_holonomy(&boundary, schema.deck_generator(), 1e-9)
            .expect("zero holonomy: the seam crossings cancel");
        assert_eq!(lift.placements, vec![0, 1, 1, 0]);
    }

    /// A single-occurrence full-turn arc is outside the supported subset —
    /// it is exactly a nonzero-holonomy realization, the case the module docs
    /// and `FORMAL_SYSTEM.md` require to be reported rather than silently
    /// accepted or reduced to zero.
    #[test]
    fn a_full_wrap_is_reported_as_nonzero_holonomy() {
        let cylinder = z_cylinder(2.0, 5.0);
        let schema = cylinder.schema().clone();
        let point = on_cylinder(&schema, 0.0, 0.0);
        let witness = circumferential_arc_witness(&schema, point, point, std::f64::consts::TAU)
            .expect("a full turn from a point back to itself is a valid arc witness");
        let boundary = DevelopedCylinderBoundary {
            edge_uses: vec![EdgeUseId::new(BoundId(0), 0)],
            witnesses: vec![witness],
        };

        let exit = propagate_and_classify_holonomy(&boundary, schema.deck_generator(), 1e-9)
            .expect_err("a full turn must not certify as zero holonomy");
        assert_eq!(exit, CylinderLiftExit::NonzeroHolonomy { holonomy: 1 });
        assert_eq!(exit.category(), SliceCategory::Unsupported);
    }
}
