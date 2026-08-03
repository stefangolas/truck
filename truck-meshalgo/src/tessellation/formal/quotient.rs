//! Rank-parametric deck labels for the quotient substrate.
//!
//! GEN-001A introduces only the label types every arrangement-facing record
//! carries: the lattice rank and the integer deck displacement. The lattice,
//! the period matrix, the translated-copy enumeration (`FORMAL_SYSTEM.md` §VIII
//! Definition 16 / Lemma 1), and the quotient-domain/stratum types land in
//! GEN-001D, reusing the rank-1 machinery already in [`super::deck`].
//!
//! `FORMAL_SYSTEM.md` §VII–VIII state deck displacement `δ ∈ Z^r` and candidate
//! translation sets `K_ij` in terms of the ambient lattice `Λ = LZ^r`,
//! `0 ≤ r ≤ 2`. Rank 0 is the ordinary nonperiodic developed plane and carries
//! only the zero label. Periodicity is represented by deck labels and
//! translated lifts, never by wrapping coordinates modulo a period before the
//! topology is solved.

/// The lattice rank of a developed chart: 0, 1 or 2.
///
/// Carried by the quotient domain (GEN-001D). A rank-0 chart has no deck
/// identifications; rank 1 has one periodic axis (cylinder/cone away from the
/// apex); rank 2 has two (torus).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DeckRank {
    /// The ordinary nonperiodic developed plane.
    Rank0,
    /// One periodic axis (e.g. an embedded cylinder).
    Rank1,
    /// Two periodic axes (e.g. a torus).
    Rank2,
}

impl DeckRank {
    /// A short stable tag, for diagnostics.
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Rank0 => "deck_rank0",
            Self::Rank1 => "deck_rank1",
            Self::Rank2 => "deck_rank2",
        }
    }
}

/// An integer deck displacement of at most two components.
///
/// The zero label identifies the base copy; a nonzero component translates the
/// piece by that multiple of the corresponding period generator. Rank-0 charts
/// carry only [`DeckLabel::ZERO`]. Identity is the integer pair, never a rounded
/// coordinate: two events identified by a certified deck translation carry the
/// label that translates one to the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeckLabel {
    /// Displacement along the first period generator.
    pub u: i64,
    /// Displacement along the second period generator (0 for rank ≤ 1).
    pub v: i64,
}

impl DeckLabel {
    /// The zero displacement: the base copy. The only label a rank-0 chart
    /// carries.
    pub const ZERO: Self = Self { u: 0, v: 0 };

    /// A rank-0 label (always zero).
    pub const fn rank0() -> Self {
        Self::ZERO
    }

    /// A rank-1 label: displacement `u` along the single period generator.
    pub const fn rank1(u: i64) -> Self {
        Self { u, v: 0 }
    }

    /// A rank-2 label: displacements along both period generators.
    pub const fn rank2(u: i64, v: i64) -> Self {
        Self { u, v }
    }

    /// Whether this is the zero (base-copy) label.
    pub const fn is_zero(self) -> bool {
        self.u == 0 && self.v == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank0_label_is_zero() {
        assert_eq!(DeckLabel::rank0(), DeckLabel::ZERO);
        assert!(DeckLabel::ZERO.is_zero());
    }

    #[test]
    fn rank1_label_carries_one_component() {
        let l = DeckLabel::rank1(3);
        assert_eq!(l, DeckLabel { u: 3, v: 0 });
        assert!(!l.is_zero());
    }

    #[test]
    fn rank2_label_carries_two_components() {
        let l = DeckLabel::rank2(-2, 5);
        assert_eq!(l, DeckLabel { u: -2, v: 5 });
    }

    #[test]
    fn labels_equal_by_integer_pair_not_coordinates() {
        assert_eq!(DeckLabel::rank1(1), DeckLabel { u: 1, v: 0 });
        assert_ne!(DeckLabel::rank1(1), DeckLabel::rank1(2));
    }
}
