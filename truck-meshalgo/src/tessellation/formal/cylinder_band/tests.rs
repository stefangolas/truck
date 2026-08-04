//! The two facts this slice exists to establish, and nothing else: a band
//! with two certified-distinct essential circles reaches a validated annular
//! mesh, and two coincident complete circles are classified as one carrier
//! and never reach the realizer.

use super::*;
use crate::tessellation::formal::cylinder::{identify_cylinder, CylinderIdentification};
use crate::tessellation::source_evidence::{
    ErasedOrientationMechanism, OrientationEvidence, OrientationOrigin, SourceEdgeOrientationEvidence,
    SourceEdgeUseInput, SourceFaceOrientationEvidence,
};
use crate::tessellation::formal::support::identify_line_segment;
use truck_geometry::prelude::{InnerSpace, Line, RevolutedCurve, Vector3};

fn z_cylinder(radius: f64, height: f64) -> CertifiedEmbeddedCylinder {
    let revo = RevolutedCurve::by_revolution(
        Line(
            Point3::new(radius, 0.0, 0.0),
            Point3::new(radius, 0.0, height),
        ),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 0.0, 1.0),
    );
    match identify_cylinder(&revo) {
        CylinderIdentification::Cylinder(cylinder) => cylinder,
        other => panic!("expected a certified cylinder, got {other:?}"),
    }
}

fn on_cylinder(schema: &CylinderSchema, z: f64, theta: f64) -> Point3 {
    schema.origin()
        + z * schema.axis()
        + schema.radius().get() * theta.cos() * schema.radial_x()
        + schema.radius().get() * theta.sin() * schema.radial_y()
}

/// One edge use, running forward over its own curve, between two shell
/// vertices. Every fixture here declares the composed sense explicitly rather
/// than leaving it erased, so the traversal stage has the authority it
/// requires and the test exercises the band stages rather than that gate.
fn edge_use(bound: BoundId, index: usize, edge: usize, from: usize, to: usize) -> SourceEdgeUseInput {
    SourceEdgeUseInput {
        id: EdgeUseId::new(bound, index),
        source_edge_index: edge,
        source_vertices: (
            SourceVertexKey::ShellVertex(from),
            SourceVertexKey::ShellVertex(to),
        ),
        use_vertices: (
            SourceVertexKey::ShellVertex(from),
            SourceVertexKey::ShellVertex(to),
        ),
        orientation: SourceEdgeOrientationEvidence {
            bound_times_oriented_edge: OrientationEvidence::Retained {
                forward: true,
                origin: OrientationOrigin::BoundTimesOrientedEdge,
            },
            edge_curve_same_sense: OrientationEvidence::HistoryErased {
                mechanism: ErasedOrientationMechanism::EdgeCurveSenseFoldedIntoConvertedCurve,
            },
            selected_curve_direction: OrientationEvidence::HistoryErased {
                mechanism:
                    ErasedOrientationMechanism::SelectedCurveDirectionFoldedIntoConvertedCurve,
            },
        },
    }
}

fn declared_outer() -> OuterBoundStanding {
    OuterBoundStanding::Declared {
        bound_index: 0,
        declared_count: 1,
    }
}

/// A two-bound face on a cylinder. Each bound is a complete circle cut into
/// two semicircular arcs; `sweeps` gives each edge's own signed parameter
/// interval, so a bound sweeping `+PI` twice winds one way and one sweeping
/// `-PI` twice winds the other.
struct Fixture {
    cylinder: CertifiedEmbeddedCylinder,
    input: SourceFaceInput,
    vertices: Vec<Point3>,
    sweeps: Vec<f64>,
}

impl Fixture {
    /// `levels` are the two circles' axial coordinates; `windings` their two
    /// signed turn directions.
    fn new(radius: f64, levels: [f64; 2], windings: [f64; 2]) -> Self {
        let cylinder = z_cylinder(radius, 5.0);
        let schema = cylinder.schema().clone();
        let mut vertices = Vec::new();
        let mut bounds = Vec::new();
        let mut sweeps = Vec::new();
        for (bound_index, (level, winding)) in levels.iter().zip(windings).enumerate() {
            let bound = BoundId(bound_index);
            let base = vertices.len();
            // Two vertices half a turn apart, in the winding's own direction.
            vertices.push(on_cylinder(&schema, *level, 0.0));
            vertices.push(on_cylinder(
                &schema,
                *level,
                winding * std::f64::consts::PI,
            ));
            let edges = [sweeps.len(), sweeps.len() + 1];
            sweeps.push(winding * std::f64::consts::PI);
            sweeps.push(winding * std::f64::consts::PI);
            bounds.push(SourceBoundInput::EdgeUses {
                id: bound,
                edge_uses: vec![
                    edge_use(bound, 0, edges[0], base, base + 1),
                    edge_use(bound, 1, edges[1], base + 1, base),
                ],
            });
        }
        let input = SourceFaceInput {
            source_face_id: Some(7),
            declared_face_index: 0,
            bounds,
            orientation: SourceFaceOrientationEvidence {
                face_use_orientation: OrientationEvidence::Missing,
                face_surface_same_sense: OrientationEvidence::Missing,
            },
        };
        Self {
            cylinder,
            input,
            vertices,
            sweeps,
        }
    }

    /// Two bounds sharing one circle, written over the *same* shell vertices.
    /// The duplicated representation is what the corroborating source-cycle
    /// check sees; the carrier verdict itself is decided on the two complete
    /// circles, and the test checks that separately against a fixture whose
    /// two coincident circles are written over distinct vertices.
    fn coincident(radius: f64, level: f64) -> Self {
        let cylinder = z_cylinder(radius, 5.0);
        let schema = cylinder.schema().clone();
        let vertices = vec![
            on_cylinder(&schema, level, 0.0),
            on_cylinder(&schema, level, std::f64::consts::PI),
        ];
        let sweeps = vec![
            std::f64::consts::PI,
            std::f64::consts::PI,
            -std::f64::consts::PI,
            -std::f64::consts::PI,
        ];
        let bounds = vec![
            SourceBoundInput::EdgeUses {
                id: BoundId(0),
                edge_uses: vec![
                    edge_use(BoundId(0), 0, 0, 0, 1),
                    edge_use(BoundId(0), 1, 1, 1, 0),
                ],
            },
            // The same circle, traversed the other way round, over the same
            // two vertices.
            SourceBoundInput::EdgeUses {
                id: BoundId(1),
                edge_uses: vec![
                    edge_use(BoundId(1), 0, 2, 0, 1),
                    edge_use(BoundId(1), 1, 3, 1, 0),
                ],
            },
        ];
        let input = SourceFaceInput {
            source_face_id: Some(8),
            declared_face_index: 0,
            bounds,
            orientation: SourceFaceOrientationEvidence {
                face_use_orientation: OrientationEvidence::Missing,
                face_surface_same_sense: OrientationEvidence::Missing,
            },
        };
        Self {
            cylinder,
            input,
            vertices,
            sweeps,
        }
    }

    fn family_of(&self) -> impl Fn(EdgeUseId) -> SourceCurveFamily + '_ {
        move |edge_use| {
            let edge = edge_use.bound.0 * 2 + edge_use.index;
            SourceCurveFamily::CircularArc {
                parameter_interval: (0.0, self.sweeps[edge]),
            }
        }
    }

    fn vertex_position(&self) -> impl Fn(SourceVertexKey) -> Option<Point3> + '_ {
        move |key| match key {
            SourceVertexKey::ShellVertex(index) => self.vertices.get(index).copied(),
            SourceVertexKey::Absent => None,
        }
    }
}

/// A structurally-identified curve schema for every edge. The band path never
/// reads it — [`develop_complete_parallel`] re-derives each occurrence's
/// family from its own source representation — but Step 2's traversal
/// requires one, so the fixture supplies a real identified schema rather than
/// an unidentified stub.
fn curve_schema() -> CurveSchema {
    identify_line_segment(&Line(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)))
}

/// The band vertical slice, end to end: two distinct essential circles on one
/// cylinder become a validated annular mesh, with no artificial cut edge left
/// in it.
#[test]
fn two_distinct_essential_circles_recover_an_annular_mesh() {
    let fixture = Fixture::new(2.0, [0.0, 3.0], [1.0, -1.0]);
    let schema = fixture.cylinder.schema().clone();

    let (band, mesh) = run_cylinder_band(
        fixture.input.source_face_id,
        fixture.cylinder.clone(),
        &fixture.input,
        declared_outer(),
        &mut |_| curve_schema(),
        &fixture.vertex_position(),
        &fixture.family_of(),
        1.0e-3,
    )
    .expect("two distinct oppositely oriented essential circles bound a band");

    // The band itself: primitive, opposite, strictly ordered, disjoint.
    assert_eq!(band.lower_boundary.homology.abs(), 1);
    assert_eq!(
        band.lower_boundary.homology + band.upper_boundary.homology,
        0,
        "the induced homologies are opposite"
    );
    assert!(
        band.lower_carrier.axial_high() < band.upper_carrier.axial_low(),
        "the carriers are strictly ordered and disjoint along the axis"
    );
    assert!(band.separation > 0.0);
    assert_eq!(band.lower_boundary.bound, BoundId(0));
    assert_eq!(band.upper_boundary.bound, BoundId(1));

    // The annulus.
    assert_eq!(mesh.validity.euler_characteristic, 0);
    assert_eq!(mesh.validity.boundary_components, 2);
    assert_eq!(mesh.validity.triangles, mesh.developed.triangles.len());
    assert_eq!(mesh.physical_vertices.len(), mesh.developed.vertices.len());
    assert!(
        mesh.validity.boundary_edges > 0 && mesh.validity.interior_edges > 0,
        "an annulus has both boundary and interior edges"
    );

    // No degenerate triangle survived the identification, and every lifted
    // vertex is genuinely on the certified cylinder.
    for triangle in &mesh.developed.triangles {
        let [a, b, c] = *triangle;
        assert!(a != b && b != c && a != c, "no collapsed triangle");
    }
    let radius = schema.radius().get();
    for vertex in &mesh.physical_vertices {
        let r = *vertex - schema.origin();
        let axial = r.dot(schema.axis());
        let radial = r - axial * schema.axis();
        assert!(
            (radial.magnitude() - radius).abs() < 1.0e-9,
            "lifted vertex {vertex:?} is not on the cylinder"
        );
        assert!(
            axial > -1.0e-9 && axial < 3.0 + 1.0e-9,
            "lifted vertex {vertex:?} left the band's axial extent"
        );
    }

    // The two physical boundaries are the two source circles: every boundary
    // vertex sits at one of the two certified axial levels, and both levels
    // are represented.
    let mut at_lower = 0;
    let mut at_upper = 0;
    for vertex in &mesh.developed.vertices {
        if vertex.x >= band.lower_carrier.axial_low() && vertex.x <= band.lower_carrier.axial_high() {
            at_lower += 1;
        } else if vertex.x >= band.upper_carrier.axial_low()
            && vertex.x <= band.upper_carrier.axial_high()
        {
            at_upper += 1;
        } else {
            panic!("a mesh vertex is on neither certified carrier: {vertex:?}");
        }
    }
    assert!(at_lower >= 3 && at_upper >= 3);
    assert_eq!(at_lower + at_upper, mesh.developed.vertices.len());
}

/// Two complete coincident cylinder circles are the same carrier, and the
/// band realizer is never reached.
#[test]
fn two_coincident_complete_circles_are_the_same_carrier() {
    let fixture = Fixture::coincident(2.0, 1.5);

    // The carriers, classified directly: coincident circles over the same
    // source vertices are certified equal, not merely unseparated.
    let schema = fixture.cylinder.schema().clone();
    let first = develop_complete_parallel(
        &fixture.input.bounds[0],
        &schema,
        &mut |_| curve_schema(),
        &fixture.vertex_position(),
        &fixture.family_of(),
    )
    .expect("the first bound is a complete simple parallel");
    let second = develop_complete_parallel(
        &fixture.input.bounds[1],
        &schema,
        &mut |_| curve_schema(),
        &fixture.vertex_position(),
        &fixture.family_of(),
    )
    .expect("the second bound is a complete simple parallel");
    assert_eq!(
        first.homology + second.homology,
        0,
        "the fixture's two traversals of one circle are opposite, so the pair \
         is refused on the carriers and not on the orientation"
    );

    // The verdict comes from the complete circles' own geometry: same
    // radius, same complete extent, and one certified enclosure covering
    // every axial coordinate either circle presents.
    let (first_carrier, second_carrier) =
        (carrier_of(&first, &schema), carrier_of(&second, &schema));
    assert_eq!(
        classify_carriers(&first_carrier, &second_carrier),
        CarrierRelation::SameCarrier
    );
    assert_eq!(first_carrier.radius, second_carrier.radius);
    assert!(
        first_carrier.observed_high.max(second_carrier.observed_high)
            - first_carrier.observed_low.min(second_carrier.observed_low)
            <= first_carrier.enclosure,
        "one certified enclosure covers both complete circles"
    );

    // The shared source cycle is corroboration, and is reported separately
    // from the physical verdict rather than standing in for it.
    assert!(carriers_share_source_cycle(&first, &second));

    // Two circles the source writes over *different* vertices, at the same
    // level, are still one carrier: the physical test does not depend on the
    // representation coinciding.
    let disjoint_representation = Fixture::new(2.0, [1.5, 1.5], [1.0, -1.0]);
    let disjoint_schema = disjoint_representation.cylinder.schema().clone();
    let a = develop_complete_parallel(
        &disjoint_representation.input.bounds[0],
        &disjoint_schema,
        &mut |_| curve_schema(),
        &disjoint_representation.vertex_position(),
        &disjoint_representation.family_of(),
    )
    .expect("a complete simple parallel");
    let b = develop_complete_parallel(
        &disjoint_representation.input.bounds[1],
        &disjoint_schema,
        &mut |_| curve_schema(),
        &disjoint_representation.vertex_position(),
        &disjoint_representation.family_of(),
    )
    .expect("a complete simple parallel");
    assert!(
        !carriers_share_source_cycle(&a, &b),
        "the two bounds are written over distinct source vertices"
    );
    assert_eq!(
        classify_carriers(&carrier_of(&a, &disjoint_schema), &carrier_of(&b, &disjoint_schema)),
        CarrierRelation::SameCarrier,
        "complete circle equality is decided on the circles, not on the file"
    );

    // And the whole path refuses on exactly that, so nothing downstream —
    // no cut plan, no patch, no triangulation — ever sees this face.
    let exit = run_cylinder_band(
        fixture.input.source_face_id,
        fixture.cylinder.clone(),
        &fixture.input,
        declared_outer(),
        &mut |_| curve_schema(),
        &fixture.vertex_position(),
        &fixture.family_of(),
        1.0e-3,
    )
    .expect_err("two coincident circles do not bound an annulus");
    assert_eq!(exit, BandExit::SameCarrier);
    assert_eq!(exit.category(), SliceCategory::Unsupported);
    assert_eq!(exit.stage(), "carrier");
}
