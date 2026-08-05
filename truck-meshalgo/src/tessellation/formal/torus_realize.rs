//! Cut-open development of a certified torus annular cell (B4 foundation).
//!
//! Takes a [`super::torus_cell::CertifiedTorusAnnularCell`] and produces its
//! developed planar representation: the unique annular 2-chain the cell
//! certified, lifted to the universal cover and cut transversely along the
//! complementary deck direction, so it can be triangulated on a plane and
//! reglued.
//!
//! For the first cell (two parallel boundaries, winding `(±1, 0)`) the
//! primitive winding is already the first lattice basis vector, so the
//! certified `GL(2, Z)` basis change is the identity. The annulus develops to a
//! rectangle `[0, 2π] × [v_lo, v_hi]` in the developed `(u, v)` plane, with the
//! `u = 0` and `u = 2π` edges identified by the major deck generator. A future
//! mixed-winding cell (e.g. `(1, 1)`) would carry a non-trivial unimodular
//! basis change here.
//!
//! # What is here
//!
//! - the certified `GL(2, Z)` basis-change witness (identity for parallels);
//! - the developed rectangle, its deck identification, and the lift of each
//!   boundary to a side of the rectangle;
//! - validation that the boundaries retain their certified coordinates and the
//!   cut is transverse to the boundary winding.
//!
//! # What is NOT here
//!
//! No triangulation, no reglue, no production mesh. The meshing step consumes
//! [`DevelopedTorusAnnulus`] and reuses the existing constrained triangulation
//! machinery, then reglues the paired quotient edges.

use super::torus::CertifiedRankTwoDeck;
use super::torus_cell::{CertifiedEssentialLoop, CertifiedTorusAnnularCell, PrimitiveWinding};
use std::f64::consts::TAU;

/// A certified `GL(2, Z)` basis-change witness: a unimodular integer matrix
/// (`det = ±1`) that re-expresses the deck basis so its first generator follows
/// the boundary primitive winding.
///
/// For the parallel-parallel cell the primitive winding `(1, 0)` is already the
/// first basis vector, so this is the identity. The witness is carried
/// explicitly so a future mixed-winding cell can supply a non-trivial change
/// without changing the development API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gl2zBasisChange {
    matrix: [[i64; 2]; 2],
}

impl Gl2zBasisChange {
    /// The identity basis change (no re-expression needed).
    pub const IDENTITY: Self = Self {
        matrix: [[1, 0], [0, 1]],
    };

    /// The unimodular matrix entries.
    pub fn matrix(&self) -> [[i64; 2]; 2] {
        self.matrix
    }

    /// The determinant (`±1` for a certified `GL(2, Z)` element).
    pub fn determinant(&self) -> i64 {
        self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]
    }

    /// Whether the determinant is `±1` (unimodular), hence orientation-preserving
    /// (`+1`) or reversing (`-1`).
    pub fn is_unimodular(&self) -> bool {
        self.determinant().abs() == 1
    }
}

/// One side of the developed rectangle, carrying the boundary loop it lifts
/// from and whether it is a cut edge or a deck-identified edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DevelopedSide {
    /// A boundary loop lifted to this side (`v = v_lo` or `v = v_hi`).
    Boundary {
        /// The certified loop this side lifts from.
        loop_index: usize,
        /// The developed `v` coordinate of this boundary.
        v: f64,
    },
    /// A cut edge along the complementary deck direction (`u = 0` and
    /// `u = 2π`), identified with its partner by the major deck generator.
    CutEdge {
        /// The developed `u` coordinate of this cut edge (`0` or `2π`).
        u: f64,
    },
}

/// The developed planar representation of a certified torus annular cell.
#[derive(Debug, Clone, PartialEq)]
pub struct DevelopedTorusAnnulus {
    /// The certified basis-change witness.
    basis_change: Gl2zBasisChange,
    /// The developed `u` interval: exactly one major period.
    u_range: (f64, f64),
    /// The developed `v` interval: between the two boundary coordinates.
    v_range: (f64, f64),
    /// The four sides of the developed rectangle, in the order
    /// `(u=u_lo, u=u_hi, v=v_lo, v=v_hi)`.
    sides: [DevelopedSide; 4],
}

impl DevelopedTorusAnnulus {
    /// The certified `GL(2, Z)` basis change.
    pub fn basis_change(&self) -> Gl2zBasisChange {
        self.basis_change
    }
    /// The developed `u` interval (one major period).
    pub fn u_range(&self) -> (f64, f64) {
        self.u_range
    }
    /// The developed `v` interval (between the two boundary coordinates).
    pub fn v_range(&self) -> (f64, f64) {
        self.v_range
    }
    /// The four sides of the developed rectangle.
    pub fn sides(&self) -> &[DevelopedSide; 4] {
        &self.sides
    }
    /// Whether the `u = u_lo` and `u = u_hi` sides are the paired cut edges
    /// identified by the major deck generator (they always are for this cell).
    pub fn has_deck_identification(&self) -> bool {
        matches!(self.sides[0], DevelopedSide::CutEdge { .. })
            && matches!(self.sides[1], DevelopedSide::CutEdge { .. })
    }
}

/// Why a torus annulus could not be developed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RealizationFailure {
    /// The cell is not a parallel-parallel annulus (the only one developed
    /// here).
    UnsupportedPrimitiveClass,
    /// The boundary coordinates are not distinct after lifting (degenerate
    /// rectangle).
    DegenerateDevelopment,
}

/// Develop a certified torus annular cell into its planar rectangle.
///
/// Chooses the lattice basis whose first generator follows the primitive
/// boundary winding (identity for parallels), lifts the two boundaries into the
/// cover, and cuts transversely along the complementary deck direction. The
/// result is a rectangle `[0, 2π] × [v_lo, v_hi]` with the `u = 0` and
/// `u = 2π` edges identified.
pub fn develop_torus_annulus(
    cell: &CertifiedTorusAnnularCell,
) -> Result<DevelopedTorusAnnulus, RealizationFailure> {
    if cell.primitive_class() != PrimitiveWinding::Parallel {
        return Err(RealizationFailure::UnsupportedPrimitiveClass);
    }
    let (a, b) = (cell.boundary_a(), cell.boundary_b());
    let (v_lo, v_hi, side_lo, side_hi) = ordered_v(a, b)?;

    Ok(DevelopedTorusAnnulus {
        basis_change: Gl2zBasisChange::IDENTITY,
        u_range: (0.0, TAU),
        v_range: (v_lo, v_hi),
        sides: [
            DevelopedSide::CutEdge { u: 0.0 },
            DevelopedSide::CutEdge { u: TAU },
            side_lo,
            side_hi,
        ],
    })
}

/// Order the two boundary `v` coordinates and return `(v_lo, v_hi, lo_side,
/// hi_side)`, recording which loop lifts to which side.
fn ordered_v(
    a: &CertifiedEssentialLoop,
    b: &CertifiedEssentialLoop,
) -> Result<(f64, f64, DevelopedSide, DevelopedSide), RealizationFailure> {
    let (va, vb) = (a.constant_coordinate(), b.constant_coordinate());
    // Distinctness is already proved by the cell; re-check defensively.
    if (va - vb).abs() < 1e-12 {
        return Err(RealizationFailure::DegenerateDevelopment);
    }
    let (lo, hi, lo_idx, hi_idx) = if va < vb {
        (va, vb, 0usize, 1)
    } else {
        (vb, va, 1, 0)
    };
    Ok((
        lo,
        hi,
        DevelopedSide::Boundary { loop_index: lo_idx, v: lo },
        DevelopedSide::Boundary { loop_index: hi_idx, v: hi },
    ))
}

#[cfg(test)]
mod tests {
    use super::super::torus::{identify_torus, TorusIdentification};
    use super::super::torus_cell::{
        certify_torus_annular_cell, BoundaryLoopPlacement, PrimitiveWinding,
        SourceBoundaryComposition,
    };
    use super::*;
    use truck_geometry::prelude::{Point3, Torus, Vector3};

    fn deck() -> CertifiedRankTwoDeck {
        match identify_torus(&Torus::new(Point3::new(0.0, 0.0, 0.0), 5.0, 1.0)) {
            TorusIdentification::Torus(d) => d,
            other => panic!("need a deck, got {other:?}"),
        }
    }

    fn parallel(v: f64, sign: i8) -> BoundaryLoopPlacement {
        BoundaryLoopPlacement {
            center: Point3::new(0.0, 0.0, 1.0 * v.sin()),
            normal: Vector3::new(0.0, 0.0, 1.0),
            radius: 5.0 + 1.0 * v.cos(),
            effective_orientation_sign: sign,
        }
    }

    fn clean() -> SourceBoundaryComposition {
        SourceBoundaryComposition {
            component_count: 2,
            extra_source_edge: false,
            outer_bound_malformation: None,
        }
    }

    fn cell() -> CertifiedTorusAnnularCell {
        certify_torus_annular_cell(&deck(), parallel(0.0, 1), parallel(1.2, -1), &clean()).unwrap()
    }

    #[test]
    fn a_parallel_annulus_develops_to_one_period_rectangle() {
        let dev = develop_torus_annulus(&cell()).unwrap();
        assert_eq!(dev.u_range(), (0.0, TAU));
        let (vlo, vhi) = dev.v_range();
        assert!((vlo - 0.0).abs() < 1e-12);
        assert!((vhi - 1.2).abs() < 1e-12);
        assert!(dev.has_deck_identification());
    }

    #[test]
    fn the_basis_change_is_identity_for_parallels() {
        let dev = develop_torus_annulus(&cell()).unwrap();
        let bc = dev.basis_change();
        assert_eq!(bc, Gl2zBasisChange::IDENTITY);
        assert!(bc.is_unimodular());
        assert_eq!(bc.determinant(), 1);
    }

    #[test]
    fn both_boundaries_lift_to_sides_of_the_rectangle() {
        let dev = develop_torus_annulus(&cell()).unwrap();
        let bounds: Vec<_> = dev
            .sides()
            .iter()
            .filter_map(|s| match s {
                DevelopedSide::Boundary { loop_index, v } => Some((*loop_index, *v)),
                _ => None,
            })
            .collect();
        assert_eq!(bounds.len(), 2);
        // Both loops (0 and 1) are represented, at the two distinct v's.
        let idxs: Vec<_> = bounds.iter().map(|(i, _)| *i).collect();
        assert!(idxs.contains(&0) && idxs.contains(&1));
        let vs: Vec<_> = bounds.iter().map(|(_, v)| *v).collect();
        assert!((vs[0] - vs[1]).abs() > 1e-6);
    }

    #[test]
    fn the_cut_edges_are_at_u_zero_and_u_two_pi() {
        let dev = develop_torus_annulus(&cell()).unwrap();
        let cuts: Vec<_> = dev
            .sides()
            .iter()
            .filter_map(|s| match s {
                DevelopedSide::CutEdge { u } => Some(*u),
                _ => None,
            })
            .collect();
        assert_eq!(cuts.len(), 2);
        assert!((cuts[0] - 0.0).abs() < 1e-12);
        assert!((cuts[1] - TAU).abs() < 1e-12);
    }

    #[test]
    fn primitive_class_is_recorded_on_the_cell() {
        assert_eq!(cell().primitive_class(), PrimitiveWinding::Parallel);
    }
}
