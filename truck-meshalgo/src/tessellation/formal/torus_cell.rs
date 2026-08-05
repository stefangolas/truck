//! Certified torus annular atlas cell (rank 2) — B2/B3.
//!
//! Built on [`super::torus::CertifiedRankTwoDeck`]. Admits the largest sound
//! homogeneous torus population the corpus presents: a regular ring torus
//! bounded by **two complete parallel circles** (winding `(±1, 0)` each) that
//! are disjoint and bound a unique annular material region — the
//! `1[Ci1];1[Ci1]` / double-outer-declared population (2,375 faces in the
//! post-cone ABC corpus).
//!
//! # What is proved
//!
//! For each boundary loop:
//! - its plane is perpendicular to the torus axis (a *parallel*, winding
//!   `(±1, 0)`) — a latitude, not a meridian;
//! - its centre lies on the axis (a true torus parallel, not a skew circle);
//! - its radius and axial height are consistent with a single minor angle `v`,
//!   so it lies on the torus.
//!
//! For the pair:
//! - both loops are the same primitive class (homologous);
//! - they are disjoint (distinct minor coordinates, not coincident mod `2π`);
//! - the material side is source-derived from the loop orientation signs:
//!   opposite signs select the annulus *between* them (the induced boundary
//!   orientation agrees with the source); equal signs leave the two
//!   complementary annuli indistinguishable → `AmbiguousMaterialAuthority`.
//!
//! # What is NOT done here
//!
//! No cut-open realization, no triangulation, no production outcome wiring.
//! The cell is a certificate; the realization (B4) and the outcome path (B5)
//! consume it later. Meridian-bounded cells (`(0, ±1)`) and mixed classes are
//! refused, not admitted: the first cell is the parallel-parallel annulus only.

use super::numeric::PositiveFinite;
use super::torus::CertifiedRankTwoDeck;
use std::f64::consts::TAU;
use truck_geometry::prelude::{InnerSpace, Point3, Vector3};

/// Dimensionless floor for the parallel/meridian plane-orientation test and
/// the on-axis / coordinate-consistency tests.
const MINIMUM_TORUS_CELL_PARALLELISM: f64 = 1e-9;

/// A boundary loop's geometric placement on the torus plus the source's
/// orientation sign.
///
/// `orientation_sign` is the loop's directed sense projected onto the certified
/// winding axis: `+1` if the curve runs with the parameter-increasing azimuthal
/// direction at its placement, `-1` against it. The caller (look-side adapter)
/// derives it from the source edge's own parameter direction, never from a
/// visual guess.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryLoopPlacement {
    /// The circle's centre.
    pub center: Point3,
    /// The circle's plane normal (caller-certified unit).
    pub normal: Vector3,
    /// The circle's radius.
    pub radius: f64,
    /// The source orientation sign: `+1` or `-1`.
    pub orientation_sign: i8,
}

/// The primitive homology class of a torus boundary loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveWinding {
    /// A parallel: winding `(±1, 0)`; the circle's plane is perpendicular to
    /// the torus axis (a latitude).
    Parallel,
    /// A meridian: winding `(0, ±1)`; the circle's plane contains the axis.
    Meridian,
}

/// A boundary loop certified as an essential (non-contractible) loop on the
/// torus quotient, carrying its `Z²` winding and the developed coordinate it is
/// constant at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CertifiedEssentialLoop {
    placement: BoundaryLoopPlacement,
    primitive: PrimitiveWinding,
    winding: [i64; 2],
    /// The torus parameter the loop is constant at: the minor angle `v` for a
    /// parallel, the major angle `u` for a meridian, reduced to `[0, 2π)`.
    constant_coordinate: f64,
}

impl CertifiedEssentialLoop {
    /// The certified `Z²` winding of this loop.
    pub fn winding(&self) -> [i64; 2] {
        self.winding
    }
    /// The primitive class.
    pub fn primitive(&self) -> PrimitiveWinding {
        self.primitive
    }
    /// The developed coordinate the loop is constant at.
    pub fn constant_coordinate(&self) -> f64 {
        self.constant_coordinate
    }
}

/// The source-derived material authority for the annular region between two
/// homologous essential loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CertifiedMaterialAuthority {
    /// The two loops carry opposite orientation signs, so the annulus *between*
    /// them is the unique region whose induced boundary orientation agrees with
    /// the source. This is the only authority that does not select by size,
    /// interval length, or visual plausibility.
    OppositeOrientationAnnulus,
}

/// Why a torus annular cell could not be certified.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TorusCellFailure {
    /// A loop's plane is neither parallel nor perpendicular to the axis, or its
    /// centre is off the axis for a parallel: not a primitive torus loop.
    NonprimitiveWinding,
    /// The two loops are not homologous (different primitive classes).
    InhomologousLoops,
    /// The first cell admits parallels only; a meridian-bearing pair is a
    /// different cell.
    MeridianNotAdmitted,
    /// The two loops coincide (same developed coordinate mod `2π`): not disjoint.
    IntersectingBoundaries,
    /// A loop's radius/height are inconsistent with the torus: it does not lie
    /// on the surface.
    SourceContradiction,
    /// Both loops carry the same orientation sign; the two complementary annuli
    /// both satisfy the retained evidence, so the material side is ambiguous.
    AmbiguousMaterialAuthority,
}

/// A certified torus annular atlas cell: a regular torus, a rank-two deck, two
/// homologous essential boundary loops that are disjoint and bound a unique
/// annular material region.
#[derive(Debug, Clone, PartialEq)]
pub struct CertifiedTorusAnnularCell {
    deck: CertifiedRankTwoDeck,
    boundary_a: CertifiedEssentialLoop,
    boundary_b: CertifiedEssentialLoop,
    primitive_class: PrimitiveWinding,
    material_authority: CertifiedMaterialAuthority,
}

impl CertifiedTorusAnnularCell {
    /// The rank-two deck.
    pub fn deck(&self) -> &CertifiedRankTwoDeck {
        &self.deck
    }
    /// The first boundary loop.
    pub fn boundary_a(&self) -> &CertifiedEssentialLoop {
        &self.boundary_a
    }
    /// The second boundary loop.
    pub fn boundary_b(&self) -> &CertifiedEssentialLoop {
        &self.boundary_b
    }
    /// The primitive class shared by both loops.
    pub fn primitive_class(&self) -> PrimitiveWinding {
        self.primitive_class
    }
    /// The source-derived material authority.
    pub fn material_authority(&self) -> CertifiedMaterialAuthority {
        self.material_authority
    }
    /// A short stable tag, for probe records.
    pub fn tag(&self) -> &'static str {
        "torus_annular_cell"
    }
}

/// Certify a torus annular cell from a rank-two deck and two boundary loop
/// placements.
///
/// Admission proves the obligations of B2 (regular torus, rank-two deck, two
/// closed essential loops, certified `Z²` winding, homologous, disjoint, unique
/// annular region) and B3 (material authority source-derived from orientation,
/// never by size). See the module docs.
pub fn certify_torus_annular_cell(
    deck: &CertifiedRankTwoDeck,
    loop_a: BoundaryLoopPlacement,
    loop_b: BoundaryLoopPlacement,
) -> Result<CertifiedTorusAnnularCell, TorusCellFailure> {
    let schema = deck.schema();
    let axis = schema.axis();
    let center = schema.center();
    let large = schema.large_radius().get();
    let small = schema.small_radius().get();
    let scale = large + small;

    let cert_a = certify_loop(loop_a, axis, center, large, small, scale)?;
    let cert_b = certify_loop(loop_b, axis, center, large, small, scale)?;

    // Homologous: same primitive class.
    if cert_a.primitive != cert_b.primitive {
        return Err(TorusCellFailure::InhomologousLoops);
    }
    // First cell: parallels only.
    if cert_a.primitive != PrimitiveWinding::Parallel {
        return Err(TorusCellFailure::MeridianNotAdmitted);
    }
    // Disjoint: distinct minor coordinates (not coincident mod 2π).
    let dv = (cert_a.constant_coordinate - cert_b.constant_coordinate).abs();
    let dv_mod = dv % TAU;
    let dv_wrapped = dv_mod.min(TAU - dv_mod);
    if dv_wrapped < MINIMUM_TORUS_CELL_PARALLELISM * scale.max(1.0) {
        return Err(TorusCellFailure::IntersectingBoundaries);
    }
    // Material authority (B3): opposite orientation signs select the annulus
    // between; equal signs are ambiguous — never pick by size.
    if cert_a.winding[0].signum() == cert_b.winding[0].signum() {
        return Err(TorusCellFailure::AmbiguousMaterialAuthority);
    }
    let authority = CertifiedMaterialAuthority::OppositeOrientationAnnulus;

    Ok(CertifiedTorusAnnularCell {
        deck: deck.clone(),
        boundary_a: cert_a,
        boundary_b: cert_b,
        primitive_class: cert_a.primitive,
        material_authority: authority,
    })
}

/// Certify one boundary loop's primitive class, winding, and constant
/// coordinate.
fn certify_loop(
    lp: BoundaryLoopPlacement,
    axis: Vector3,
    center: Point3,
    large: f64,
    small: f64,
    scale: f64,
) -> Result<CertifiedEssentialLoop, TorusCellFailure> {
    let rel = lp.center - center;
    let height = rel.dot(axis); // signed axial offset from the torus centre
    let radial_offset = rel - height * axis; // must be ~0 for a parallel (centre on axis)

    let dot = lp.normal.dot(axis);
    let abs_dot = dot.abs();
    let tol = MINIMUM_TORUS_CELL_PARALLELISM;

    let primitive = if abs_dot > 1.0 - tol {
        PrimitiveWinding::Parallel
    } else if abs_dot < tol {
        PrimitiveWinding::Meridian
    } else {
        return Err(TorusCellFailure::NonprimitiveWinding);
    };

    let (winding, constant_coordinate) = match primitive {
        PrimitiveWinding::Parallel => {
            // The centre of a torus parallel lies on the axis.
            if radial_offset.magnitude() > tol * scale.max(1.0) {
                return Err(TorusCellFailure::NonprimitiveWinding);
            }
            // cos v = (r - large)/small,  sin v = height/small.
            let cos_v = (lp.radius - large) / small;
            let sin_v = height / small;
            // The circle must lie on the torus: cos²v + sin²v = 1.
            if (cos_v * cos_v + sin_v * sin_v - 1.0).abs() > 1e-9 {
                return Err(TorusCellFailure::SourceContradiction);
            }
            let v = sin_v.atan2(cos_v).rem_euclid(TAU);
            ([lp.orientation_sign as i64, 0], v)
        }
        PrimitiveWinding::Meridian => {
            // The meridian's azimuth around the axis is its constant `u`.
            // radial_offset is the centre's offset from the axis (radius large).
            let u = radial_offset.y.atan2(radial_offset.x).rem_euclid(TAU);
            ([0, lp.orientation_sign as i64], u)
        }
    };

    Ok(CertifiedEssentialLoop {
        placement: lp,
        primitive,
        winding,
        constant_coordinate,
    })
}

#[cfg(test)]
mod tests {
    use super::super::torus::{identify_torus, TorusIdentification};
    use super::*;
    use truck_geometry::prelude::Torus;

    fn deck() -> CertifiedRankTwoDeck {
        let t = Torus::new(Point3::new(0.0, 0.0, 0.0), 5.0, 1.0);
        match identify_torus(&t) {
            TorusIdentification::Torus(d) => d,
            other => panic!("need a deck, got {other:?}"),
        }
    }

    /// A parallel circle at minor angle `v`, orientation `sign`, on the
    /// canonical z-axis torus (large=5, small=1, centre origin).
    fn parallel(v: f64, sign: i8) -> BoundaryLoopPlacement {
        let r = 5.0 + 1.0 * v.cos();
        let z = 1.0 * v.sin();
        BoundaryLoopPlacement {
            center: Point3::new(0.0, 0.0, z),
            normal: Vector3::new(0.0, 0.0, 1.0),
            radius: r,
            orientation_sign: sign,
        }
    }

    #[test]
    fn two_opposite_parallels_certify_an_annulus() {
        let d = deck();
        let cell = certify_torus_annular_cell(&d, parallel(0.0, 1), parallel(1.2, -1));
        assert!(cell.is_ok());
        let cell = cell.unwrap();
        assert_eq!(cell.primitive_class(), PrimitiveWinding::Parallel);
        assert_eq!(
            cell.material_authority(),
            CertifiedMaterialAuthority::OppositeOrientationAnnulus
        );
        let [wa, wb] = [cell.boundary_a().winding(), cell.boundary_b().winding()];
        assert_eq!(wa, [1, 0]);
        assert_eq!(wb, [-1, 0]);
    }

    #[test]
    fn two_same_orientation_parallels_are_ambiguous() {
        let d = deck();
        let err = certify_torus_annular_cell(&d, parallel(0.0, 1), parallel(1.2, 1));
        assert_eq!(err, Err(TorusCellFailure::AmbiguousMaterialAuthority));
    }

    #[test]
    fn coincident_parallels_are_intersecting() {
        let d = deck();
        let err = certify_torus_annular_cell(&d, parallel(0.7, 1), parallel(0.7, -1));
        assert_eq!(err, Err(TorusCellFailure::IntersectingBoundaries));
    }

    #[test]
    fn a_full_turn_apart_parallels_are_coincident_mod_two_pi() {
        let d = deck();
        let err = certify_torus_annular_cell(&d, parallel(0.4, 1), parallel(0.4 + TAU, -1));
        assert_eq!(err, Err(TorusCellFailure::IntersectingBoundaries));
    }

    #[test]
    fn an_off_axis_circle_is_not_a_primitive_parallel() {
        let d = deck();
        // Centre off the axis: not a torus parallel.
        let bad = BoundaryLoopPlacement {
            center: Point3::new(2.0, 0.0, 0.0),
            normal: Vector3::new(0.0, 0.0, 1.0),
            radius: 5.0,
            orientation_sign: 1,
        };
        assert_eq!(
            certify_torus_annular_cell(&d, parallel(0.0, 1), bad),
            Err(TorusCellFailure::NonprimitiveWinding)
        );
    }

    #[test]
    fn a_circle_not_on_the_torus_is_a_source_contradiction() {
        let d = deck();
        // Centre on axis, normal along axis, but radius/height inconsistent
        // with any minor angle (radius 5, height 1 -> cos v = 0, sin v = 1 -> v = pi/2,
        // but then radius should be 5 + 0 = 5; height 1 is fine; so pick a
        // genuinely inconsistent radius).
        let bad = BoundaryLoopPlacement {
            center: Point3::new(0.0, 0.0, 1.0),
            normal: Vector3::new(0.0, 0.0, 1.0),
            radius: 7.0, // would need cos v = 2 (impossible)
            orientation_sign: 1,
        };
        assert_eq!(
            certify_torus_annular_cell(&d, parallel(0.0, 1), bad),
            Err(TorusCellFailure::SourceContradiction)
        );
    }

    #[test]
    fn a_tilted_circle_is_nonprimitive() {
        let d = deck();
        let tilted = BoundaryLoopPlacement {
            center: Point3::new(0.0, 0.0, 0.0),
            normal: Vector3::new(0.0, 1.0, 1.0).normalize(),
            radius: 5.0,
            orientation_sign: 1,
        };
        assert_eq!(
            certify_torus_annular_cell(&d, parallel(0.0, 1), tilted),
            Err(TorusCellFailure::NonprimitiveWinding)
        );
    }

    #[test]
    fn meridian_pair_is_not_admitted_in_the_first_cell() {
        let d = deck();
        // A meridian: plane contains the axis (normal ⊥ axis), centre off axis
        // at radius large, radius small.
        let meridian = BoundaryLoopPlacement {
            center: Point3::new(5.0, 0.0, 0.0),
            normal: Vector3::new(0.0, 1.0, 0.0),
            radius: 1.0,
            orientation_sign: 1,
        };
        let meridian2 = BoundaryLoopPlacement {
            center: Point3::new(0.0, 5.0, 0.0),
            normal: Vector3::new(1.0, 0.0, 0.0),
            radius: 1.0,
            orientation_sign: -1,
        };
        assert_eq!(
            certify_torus_annular_cell(&d, meridian, meridian2),
            Err(TorusCellFailure::MeridianNotAdmitted)
        );
    }
}
