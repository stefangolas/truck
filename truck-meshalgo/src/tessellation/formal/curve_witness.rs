//! Checkpoint 5: curve-on-cylinder witnesses in the developed universal cover.
//!
//! # Scope
//!
//! Exactly the two curve families the supported subset names: an axial line
//! (constant angular coordinate) and a circumferential circular arc (constant
//! axial coordinate, sweeping the angle). Both project *exactly* into the
//! developed plane once the endpoints are known to lie on the certified
//! cylinder — there is no numerical approximation to bound, unlike the
//! planar slice's polygonal projection.
//!
//! # The developed coordinate convention
//!
//! [`super::cylinder`]'s module docs fix the developed plane as `(axial =
//! First, angular = Second)`, and [`super::cylinder::CylinderSchema::angular_coordinate`]
//! *is* the surface's own periodic parameter `v` — not a value this module
//! re-derives or wraps. A [`Point2`] produced here has `x` the axial
//! coordinate and `y` the developed (possibly non-principal) angular
//! coordinate.
//!
//! # Why the arc witness takes a declared sweep rather than inferring one
//!
//! Two points at the same radius and axial coordinate admit more than one arc
//! between them — short way, long way, or wrapped around one or more extra
//! turns — and nothing about the two endpoints alone disambiguates which the
//! source curve actually traces. The authoritative fact is the source curve's
//! own trimmed parameter interval, which for a circle *is* the signed angular
//! sweep, unwrapped. [`circumferential_arc_witness`] takes that sweep as an
//! input and treats the endpoints as the sample that confirms it, never as
//! the thing that certifies it: this is the same discipline
//! `identify_cylinder`'s `verify_angular_convention` already applies to the
//! `2π` period itself.
//!
//! # Why this does not touch `cylinder.rs`
//!
//! [`super::cylinder::CylinderSchema`] already exposes everything a witness
//! needs — `axial_coordinate`, `angular_coordinate`, `axis`, `radius` — through
//! its public accessors. The on-cylinder check the witnesses need is built
//! from those accessors rather than by exporting `cylinder.rs`'s private
//! `point_on_cylinder`, so the frozen module gains no new surface.

use super::cylinder::CylinderSchema;
use super::numeric::FiniteF64;
use std::f64::consts::TAU;
use truck_geometry::prelude::{InnerSpace, Point2, Point3};

/// Which curve family a witness represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WitnessClass {
    /// A straight segment parallel to the cylinder axis: constant angular
    /// coordinate, varying axial coordinate.
    AxialLine,
    /// A circular arc at constant axial coordinate, sweeping the angular
    /// coordinate. May cross the seam.
    CircumferentialArc,
}

impl WitnessClass {
    /// A short stable tag, for probe records.
    pub fn tag(self) -> &'static str {
        match self {
            Self::AxialLine => "axial_line",
            Self::CircumferentialArc => "circumferential_arc",
        }
    }
}

/// Why a curve could not be certified as a witness in the developed cover.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WitnessFailure {
    /// An input coordinate was `NaN` or infinite.
    NonFiniteInput,
    /// The traversal start point is not on the certified cylinder.
    StartNotOnCylinder,
    /// The traversal end point is not on the certified cylinder.
    EndNotOnCylinder,
    /// An axial-line candidate's endpoints are not separated by a segment
    /// parallel to the axis.
    NotParallelToAxis,
    /// An axial-line candidate's endpoints coincide, so it has no direction.
    DegenerateAxialSegment,
    /// A circumferential-arc candidate's endpoints do not share an axial
    /// coordinate.
    NotConstantAxialCoordinate,
    /// A circumferential-arc candidate declared a zero angular sweep, so it
    /// has no direction.
    ZeroSweep,
    /// The declared sweep, developed from the start angle, does not land on
    /// the declared end point.
    SweepDoesNotReachDeclaredEndpoint,
}

impl WitnessFailure {
    /// A short stable tag, for probe records.
    pub fn tag(self) -> &'static str {
        match self {
            Self::NonFiniteInput => "witness_non_finite_input",
            Self::StartNotOnCylinder => "witness_start_not_on_cylinder",
            Self::EndNotOnCylinder => "witness_end_not_on_cylinder",
            Self::NotParallelToAxis => "witness_not_parallel_to_axis",
            Self::DegenerateAxialSegment => "witness_degenerate_axial_segment",
            Self::NotConstantAxialCoordinate => "witness_not_constant_axial_coordinate",
            Self::ZeroSweep => "witness_zero_sweep",
            Self::SweepDoesNotReachDeclaredEndpoint => "witness_sweep_does_not_reach_endpoint",
        }
    }
}

/// Dimensionless-in-radius tolerance floor shared by the on-cylinder and
/// constant-coordinate checks below. `1e-9` matches the parallelism and
/// convention floors [`super::cylinder`] already certifies at, so a witness is
/// refused only when the discrepancy is orders of magnitude past floating
/// point noise.
const RELATIVE_TOLERANCE: f64 = 1e-9;

fn finite_point(point: Point3) -> Result<Point3, WitnessFailure> {
    for coordinate in [point.x, point.y, point.z] {
        FiniteF64::new(coordinate).map_err(|_| WitnessFailure::NonFiniteInput)?;
    }
    Ok(point)
}

/// Whether `point` lies on the cylinder, within a tolerance scaled by radius.
fn radial_gap(schema: &CylinderSchema, point: Point3) -> f64 {
    let r = point - schema.origin();
    let axial = r.dot(schema.axis());
    let radial = r - axial * schema.axis();
    (radial.magnitude() - schema.radius().get()).abs()
}

fn scaled_tolerance(schema: &CylinderSchema) -> f64 {
    RELATIVE_TOLERANCE * schema.radius().get().max(1.0)
}

/// One curve-on-cylinder witness: a continuous developed segment, exact over
/// its complete interval.
///
/// `start` and `end` are in the developed `(axial, angular)` chart, with the
/// angular coordinate left unwrapped — a seam-crossing arc's `end.y` may lie
/// outside `[0, 2*PI)` or even be negative, and that is precisely how its
/// continuity is represented.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveOnCylinderWitness {
    /// Which curve family this witness represents.
    pub class: WitnessClass,
    /// The developed start point: `(axial_coordinate, angular_coordinate)`.
    pub start: Point2,
    /// The developed end point, continuous with `start` in the universal
    /// cover — not reduced modulo the period.
    pub end: Point2,
}

impl CurveOnCylinderWitness {
    /// The developed displacement `end - start`.
    pub fn displacement(&self) -> Point2 {
        Point2::new(self.end.x - self.start.x, self.end.y - self.start.y)
    }
}

/// Certify an axial-line witness from its physical endpoints.
///
/// `start` and `end` are the traversal-order endpoints — the composed sense
/// already applied exactly once, upstream, the same discipline
/// [`super::planar_slice::certified_planar_curves`] follows. Reversing the
/// traversal is exactly swapping `start` and `end`; nothing else changes.
pub fn axial_line_witness(
    schema: &CylinderSchema,
    start: Point3,
    end: Point3,
) -> Result<CurveOnCylinderWitness, WitnessFailure> {
    let start = finite_point(start)?;
    let end = finite_point(end)?;
    let tolerance = scaled_tolerance(schema);

    if radial_gap(schema, start) > tolerance {
        return Err(WitnessFailure::StartNotOnCylinder);
    }
    if radial_gap(schema, end) > tolerance {
        return Err(WitnessFailure::EndNotOnCylinder);
    }

    let axial_start = schema.axial_coordinate(start);
    let axial_end = schema.axial_coordinate(end);
    if (axial_end - axial_start).abs() <= tolerance {
        return Err(WitnessFailure::DegenerateAxialSegment);
    }

    // Constant angular coordinate: the two points must share a radial
    // direction. `angular_coordinate` is `atan2` in the recorded frame, whose
    // range is `(-PI, PI]`, so two points at the *same* radial direction
    // return the identical value with no branch ambiguity — unlike the arc
    // case, an axial segment never approaches the seam from two sides.
    let theta_start = schema.angular_coordinate(start);
    let theta_end = schema.angular_coordinate(end);
    if (theta_end - theta_start).abs() > RELATIVE_TOLERANCE {
        return Err(WitnessFailure::NotParallelToAxis);
    }

    Ok(CurveOnCylinderWitness {
        class: WitnessClass::AxialLine,
        start: Point2::new(axial_start, theta_start),
        end: Point2::new(axial_end, theta_start),
    })
}

/// Certify a circumferential-arc witness from its physical endpoints and its
/// source curve's own declared signed angular sweep.
///
/// `declared_sweep` is the source curve's trimmed parameter interval, `u2 -
/// u1`, in the surface's own angular parameter — exact and unwrapped. It is
/// what the developed end angle is built *from*; the physical `end` point is
/// only the sample that confirms it lands there, never independently used to
/// pick a branch. A seam-crossing arc is exactly one whose declared sweep
/// carries `theta_start + declared_sweep` outside `[0, 2*PI)`, and it is
/// represented that way, not wrapped back in.
pub fn circumferential_arc_witness(
    schema: &CylinderSchema,
    start: Point3,
    end: Point3,
    declared_sweep: f64,
) -> Result<CurveOnCylinderWitness, WitnessFailure> {
    let start = finite_point(start)?;
    let end = finite_point(end)?;
    let sweep = FiniteF64::new(declared_sweep).map_err(|_| WitnessFailure::NonFiniteInput)?.get();
    let tolerance = scaled_tolerance(schema);

    if radial_gap(schema, start) > tolerance {
        return Err(WitnessFailure::StartNotOnCylinder);
    }
    if radial_gap(schema, end) > tolerance {
        return Err(WitnessFailure::EndNotOnCylinder);
    }

    let axial_start = schema.axial_coordinate(start);
    let axial_end = schema.axial_coordinate(end);
    if (axial_end - axial_start).abs() > tolerance {
        return Err(WitnessFailure::NotConstantAxialCoordinate);
    }

    if sweep == 0.0 {
        return Err(WitnessFailure::ZeroSweep);
    }

    let theta_start = schema.angular_coordinate(start);
    let theta_end_developed = theta_start + sweep;

    // Confirm — never certify — that the declared sweep actually reaches the
    // declared endpoint: the developed angle reduced to the principal branch
    // must agree with the endpoint's own angular coordinate.
    let theta_end_actual = schema.angular_coordinate(end);
    let reduced = principal_branch(theta_end_developed);
    let mut angular_gap = (reduced - theta_end_actual).abs();
    if angular_gap > TAU - angular_gap {
        angular_gap = TAU - angular_gap;
    }
    if angular_gap > RELATIVE_TOLERANCE {
        return Err(WitnessFailure::SweepDoesNotReachDeclaredEndpoint);
    }

    Ok(CurveOnCylinderWitness {
        class: WitnessClass::CircumferentialArc,
        start: Point2::new(axial_start, theta_start),
        end: Point2::new(axial_start, theta_end_developed),
    })
}

/// Reduce an angular coordinate into the surface's own principal branch
/// `(-PI, PI]`, matching `atan2`'s range exactly so a developed value can be
/// compared against `angular_coordinate`'s output.
fn principal_branch(theta: f64) -> f64 {
    let wrapped = theta.rem_euclid(TAU);
    if wrapped > std::f64::consts::PI {
        wrapped - TAU
    } else {
        wrapped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cylinder::{identify_cylinder, CylinderIdentification};
    use truck_geometry::prelude::{Line, RevolutedCurve, Vector3};

    fn z_cylinder(radius: f64, h: f64) -> CylinderSchema {
        let revo = RevolutedCurve::by_revolution(
            Line(Point3::new(radius, 0.0, 0.0), Point3::new(radius, 0.0, h)),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );
        match identify_cylinder(&revo) {
            CylinderIdentification::Cylinder(c) => c.schema().clone(),
            other => panic!("expected a certified cylinder, got {other:?}"),
        }
    }

    /// A point on the cylinder at angle `theta` and axial coordinate `z`.
    fn on_cylinder(schema: &CylinderSchema, z: f64, theta: f64) -> Point3 {
        schema.origin()
            + z * schema.axis()
            + schema.radius().get() * theta.cos() * schema.radial_x()
            + schema.radius().get() * theta.sin() * schema.radial_y()
    }

    // -- axial line -------------------------------------------------------

    #[test]
    fn positive_axial_direction_is_a_witness() {
        let schema = z_cylinder(2.0, 5.0);
        let start = on_cylinder(&schema, 0.0, 0.3);
        let end = on_cylinder(&schema, 3.0, 0.3);
        let w = axial_line_witness(&schema, start, end).expect("axial witness");
        assert_eq!(w.class, WitnessClass::AxialLine);
        assert!((w.start.x - 0.0).abs() < 1e-9);
        assert!((w.end.x - 3.0).abs() < 1e-9);
        assert!((w.start.y - w.end.y).abs() < 1e-9, "constant angle");
        assert!(w.displacement().x > 0.0);
    }

    #[test]
    fn negative_axial_direction_is_a_witness() {
        let schema = z_cylinder(2.0, 5.0);
        let start = on_cylinder(&schema, 3.0, 0.3);
        let end = on_cylinder(&schema, 0.0, 0.3);
        let w = axial_line_witness(&schema, start, end).expect("axial witness");
        assert!(w.displacement().x < 0.0);
        assert!((w.start.y - w.end.y).abs() < 1e-9);
    }

    #[test]
    fn a_segment_off_axis_is_refused() {
        let schema = z_cylinder(2.0, 5.0);
        let start = on_cylinder(&schema, 0.0, 0.3);
        let end = on_cylinder(&schema, 3.0, 0.9); // angle also changes
        assert_eq!(
            axial_line_witness(&schema, start, end).unwrap_err(),
            WitnessFailure::NotParallelToAxis
        );
    }

    #[test]
    fn a_degenerate_axial_segment_is_refused() {
        let schema = z_cylinder(2.0, 5.0);
        let point = on_cylinder(&schema, 1.0, 0.3);
        assert_eq!(
            axial_line_witness(&schema, point, point).unwrap_err(),
            WitnessFailure::DegenerateAxialSegment
        );
    }

    // -- circumferential arc ----------------------------------------------

    #[test]
    fn positive_angular_direction_is_a_witness() {
        let schema = z_cylinder(2.0, 5.0);
        let start = on_cylinder(&schema, 1.0, 0.2);
        let end = on_cylinder(&schema, 1.0, 1.4);
        let w = circumferential_arc_witness(&schema, start, end, 1.2).expect("arc witness");
        assert_eq!(w.class, WitnessClass::CircumferentialArc);
        assert!((w.start.x - w.end.x).abs() < 1e-9, "constant axial coordinate");
        assert!((w.displacement().y - 1.2).abs() < 1e-9);
    }

    #[test]
    fn negative_angular_direction_is_a_witness() {
        let schema = z_cylinder(2.0, 5.0);
        let start = on_cylinder(&schema, 1.0, 1.4);
        let end = on_cylinder(&schema, 1.0, 0.2);
        let w = circumferential_arc_witness(&schema, start, end, -1.2).expect("arc witness");
        assert!((w.displacement().y - (-1.2)).abs() < 1e-9);
    }

    #[test]
    fn seam_crossing_in_the_positive_direction_stays_continuous() {
        let schema = z_cylinder(2.0, 5.0);
        // From just before the seam to just after it, the short way: sweep of
        // +0.4 carries theta from 3.0 past PI (the branch cut) to -2.983...
        // in the principal branch, but the developed value must keep climbing.
        let theta_start = 3.0;
        let sweep = 0.4;
        let start = on_cylinder(&schema, 2.0, theta_start);
        let end = on_cylinder(&schema, 2.0, theta_start + sweep);
        let w = circumferential_arc_witness(&schema, start, end, sweep).expect("seam witness");
        assert!((w.start.y - theta_start).abs() < 1e-9);
        assert!(
            (w.end.y - (theta_start + sweep)).abs() < 1e-9,
            "developed angle is not wrapped back into the principal branch: {}",
            w.end.y
        );
        assert!(w.end.y > TAU / 2.0 - 1e-6 || w.end.y > std::f64::consts::PI);
    }

    #[test]
    fn seam_crossing_in_the_negative_direction_stays_continuous() {
        let schema = z_cylinder(2.0, 5.0);
        let theta_start = -3.0;
        let sweep = -0.4;
        let start = on_cylinder(&schema, 2.0, theta_start);
        let end = on_cylinder(&schema, 2.0, theta_start + sweep);
        let w = circumferential_arc_witness(&schema, start, end, sweep).expect("seam witness");
        assert!((w.end.y - (theta_start + sweep)).abs() < 1e-9);
        assert!(w.end.y < -std::f64::consts::PI);
    }

    #[test]
    fn a_wrong_declared_sweep_is_refused() {
        let schema = z_cylinder(2.0, 5.0);
        let start = on_cylinder(&schema, 1.0, 0.2);
        let end = on_cylinder(&schema, 1.0, 1.4);
        // The physical endpoint sits at theta=1.4; declaring a sweep of 1.5
        // develops to theta=1.7, a different point than the one sampled, so
        // the confirmation must reject it rather than trust the declaration.
        let wrong = circumferential_arc_witness(&schema, start, end, 1.5);
        assert_eq!(
            wrong.unwrap_err(),
            WitnessFailure::SweepDoesNotReachDeclaredEndpoint
        );
    }

    #[test]
    fn selected_source_edge_reversal_swaps_start_and_end_and_negates_sweep() {
        let schema = z_cylinder(2.0, 5.0);
        let a = on_cylinder(&schema, 1.0, 0.2);
        let b = on_cylinder(&schema, 1.0, 1.4);
        let forward = circumferential_arc_witness(&schema, a, b, 1.2).expect("forward");
        let reversed = circumferential_arc_witness(&schema, b, a, -1.2).expect("reversed");
        assert!((forward.start.y - reversed.end.y).abs() < 1e-9);
        assert!((forward.end.y - reversed.start.y).abs() < 1e-9);
        assert!((forward.displacement().y + reversed.displacement().y).abs() < 1e-9);
    }

    #[test]
    fn a_non_constant_axial_coordinate_is_refused_for_an_arc() {
        let schema = z_cylinder(2.0, 5.0);
        let start = on_cylinder(&schema, 0.0, 0.2);
        let end = on_cylinder(&schema, 1.0, 1.4);
        assert_eq!(
            circumferential_arc_witness(&schema, start, end, 1.2).unwrap_err(),
            WitnessFailure::NotConstantAxialCoordinate
        );
    }
}
