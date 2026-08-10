//! Establish the parameter traversal a topological edge denotes on its
//! underlying curve, before tessellation samples it.
//!
//! The theorem this module implements is:
//!
//! > A topological STEP edge is not identified by a curve's evaluator domain.
//! > Its geometric traversal must be established from source topology, curve
//! > semantics, orientation, and sufficient evidence.
//!
//! `evaluation_range()` is the curve's *safe evaluator domain*; for ordinary
//! P1 edges it coincides with the source edge interval, and for a closed loop
//! whose source vertices sit at interior parameters it does not. This module
//! decides which of three claims the evidence supports:
//!
//! - [`SourceEdgeTraversal::CanonicalByEvalRange`] — the edge's traversal is
//!   the whole evaluator domain. For the ordinary population the evaluator
//!   endpoints realize the distinct source vertices. For a topologically
//!   closed edge (`edge.vertices.0 == edge.vertices.1`) the full-loop legacy
//!   handling is preserved: topological closure is decided by vertex identity,
//!   never by positional coincidence, and the evaluator seam is a
//!   parameterization seam that need not coincide with the shared source
//!   vertex. The caller samples `evaluation_range()` exactly as it did before,
//!   including its existing closed-edge period and partition-of-unity
//!   extensions.
//! - [`SourceEdgeTraversal::CanonicalBySourceInterval`] — the evaluator
//!   endpoints do not realize the source vertices, but both vertices can be
//!   located at unique interior parameters of the evaluator domain, with
//!   source-consistent residuals. The traversal follows the curve's natural
//!   increasing-parameter direction (the source direction, see below) from the
//!   start root to the end root. When the start root precedes the end root the
//!   traversal is a simple interval, valid on an *open* carrier as much as on a
//!   closed one (NIST STEP contains ordinary open B-spline `EDGE_CURVE`s whose
//!   source start vertex sits at an interior carrier parameter). The evaluator
//!   seam wrap is used only when the start root lies after the end root, which
//!   is geometric only on a carrier established closed (`C(lo) ~= C(hi)`); an
//!   open carrier that would need a wrap has no certified traversal.
//! - [`SourceEdgeTraversal::Unresolved`] — no traversal could be established.
//!   This is **not** a licence to sample the evaluator domain: sampling the
//!   full loop of a closed source crescent would re-emit the malformed
//!   boundary. The caller must propagate a no-renderable-traversal outcome
//!   through the tessellation outcome mechanism instead.
//!
//! # Orientation invariant
//!
//! `truck-stepio` already folds `EDGE_CURVE.same_sense` into the stored curve:
//! `sub_parse_curve3d` applies `curve.invert()` when `same_sense = .F.`
//! (`truck-stepio/src/in/mod.rs`). Therefore the stored curve's
//! increasing-parameter direction is *always* the source edge direction, from
//! `edge_start` to `edge_end`, and `CompressedEdge.vertices = (front, back)`
//! preserves that order. The traversal this module establishes is: follow the
//! curve's natural increasing-parameter direction from the front vertex's
//! parameter root to the back vertex's parameter root, wrapping through the
//! evaluator seam when the closed domain puts the front root after the back
//! root.
//!
//! This is not a "wrap when `t_start > t_end`" heuristic: it is the
//! consequence of the orientation invariant, and it handles `same_sense = .F.`
//! correctly because the importer has already reversed the curve.
//!
//! # Incidence and uniqueness evidence
//!
//! Source incidence is judged against the `source_tolerance` the caller
//! supplies: the geometric uncertainty declared by the source's representation
//! context when one exists, and the existing `truck_base` tolerance
//! ([`SOURCE_INCIDENCE_TOLERANCE`]) as the fallback. It is deliberately not the
//! tessellation chord tolerance `tol`: `tol` bounds geometric mesh error, it is
//! not evidence that a vertex lies on a curve. The residuals for the
//! closed-spline population are ~1e-12, so the 1e-6 incidence threshold is far
//! above numerical noise and far below the smallest feature the tessellator
//! resolves.
//!
//! The incidence tolerance is **acceptance only**. It never participates in
//! localizing a root (the sample-dip scan and golden-section refinement are
//! tolerance-free), never merges distinct candidates, and never widens the
//! class of closed carriers. The lo≈hi evaluator-seam equivalence is restricted
//! to carriers that are closed to the fixed numerical tolerance; a large
//! declared source uncertainty must not turn an open fitted curve into a
//! wrappable loop. Separating these is what keeps a tolerance as large as a
//! model's declared 0.1-unit connectivity accuracy from erasing parameter
//! uniqueness.
//!
//! A [`SourceEdgeTraversal::CanonicalBySourceInterval`] verdict additionally
//! requires the root of each source vertex to be *uniquely established* by a
//! bounded, deterministic isolation: a fine sample-dip scan whose candidate set
//! is verified stable when the grid resolution is doubled, then golden-section
//! refinement of every surviving candidate, then exactly one distinct
//! candidate modulo the closed-domain seam. Any observed ambiguity — zero
//! candidates, several candidates, or a candidate set that changes between the
//! two resolutions — yields [`SourceEdgeTraversal::Unresolved`]. Failure to
//! prove uniqueness is not source invalidity, and it is also not permission to
//! render a full loop that was never certified.

use super::*;

/// The source-incidence tolerance: how close a curve point must be to a source
/// vertex to count as realizing it.
///
/// This is `truck_base`'s own [`TOLERANCE`] (1e-6), the existing geometric
/// consistency tolerance the tessellator already uses as its absolute floor.
/// It is deliberately independent of the tessellation chord tolerance `tol`.
///
/// It is the *fallback* incidence tolerance, used when the source declares no
/// geometric uncertainty of its own. A STEP model that declares one carries it
/// through on the shell, and the tessellator passes it here instead: the
/// source's own asserted-connectivity accuracy is the authority, this constant
/// is what remains when the source supplies none.
///
/// Regardless of which value is in force, it is **acceptance-only**. It never
/// participates in root localization, candidate merging, or the carrier-closure
/// (lo≈hi seam) decision, which stay at numerically sharp thresholds.
pub const SOURCE_INCIDENCE_TOLERANCE: f64 = TOLERANCE;

/// The parameter interval of the underlying curve that a source edge denotes.
///
/// `Simple` is an ordinary arc; `Wrapped` crosses the closed evaluator-domain
/// seam and is represented as two adjacent pieces `[start → domain_end]` and
/// `[domain_start → end]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParamTraversal {
    /// A single increasing-parameter interval within the evaluator domain.
    Simple {
        /// The start parameter.
        start: f64,
        /// The end parameter.
        end: f64,
    },
    /// Two pieces joined across the evaluator-domain seam of a closed curve.
    Wrapped {
        /// The start parameter.
        start: f64,
        /// The evaluator domain's high end (`start → domain_end`).
        domain_end: f64,
        /// The evaluator domain's low end (`domain_start → end`).
        domain_start: f64,
        /// The end parameter.
        end: f64,
    },
}

/// The evidence that established a source-interval traversal.
///
/// The residuals are the incidence evidence and the candidate counts are the
/// uniqueness evidence: a `CanonicalBySourceInterval` verdict is only as
/// strong as these two numbers say it is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceEdgeTraversalWitness {
    /// The traversal the witness certifies.
    pub traversal: ParamTraversal,
    /// Parameter of the start vertex root.
    pub start_parameter: f64,
    /// Parameter of the end vertex root.
    pub end_parameter: f64,
    /// `|C(start_parameter) - start_vertex|`.
    pub start_residual: f64,
    /// `|C(end_parameter) - end_vertex|`.
    pub end_residual: f64,
    /// Distinct source-consistent candidate roots found for the start vertex.
    pub start_candidates: usize,
    /// Distinct source-consistent candidate roots found for the end vertex.
    pub end_candidates: usize,
}

/// The established source-edge traversal.
#[derive(Clone, Debug, PartialEq)]
pub enum SourceEdgeTraversal {
    /// The evaluator domain is the source edge's traversal.
    CanonicalByEvalRange {
        /// The evaluator range to sample.
        range: (f64, f64),
    },
    /// The source edge denotes a source-determined interval (possibly wrapped).
    CanonicalBySourceInterval {
        /// The parameter interval to sample.
        traversal: ParamTraversal,
        /// The evidence that established it.
        witness: SourceEdgeTraversalWitness,
    },
    /// No traversal could be established. Not an invalid edge, and not a
    /// licence to sample the evaluator domain: the caller propagates a
    /// no-renderable-traversal outcome through the tessellation outcome
    /// mechanism.
    Unresolved {
        /// A short stable reason tag.
        reason: &'static str,
    },
}

/// Establish which portion of `curve` the topological edge between the source
/// vertices `start_pos` and `end_pos` denotes.
///
/// `start_pos` / `end_pos` are the shell positions of `edge.vertices.0` /
/// `edge.vertices.1` (the EDGE_CURVE start / end), and the curve's
/// increasing-parameter direction is the source direction (see the module-level
/// orientation invariant).
///
/// `topologically_closed` is `edge.vertices.0 == edge.vertices.1`: the source
/// claims the edge starts and ends at the *same vertex entity*. Positional
/// coincidence is deliberately not used for this: two distinct vertex entities
/// may share a position, and that is not a closed edge. A genuinely
/// topologically closed edge is a full-loop edge and keeps the legacy
/// full-loop handling; the evaluator seam is a parameterization seam and need
/// not coincide with the shared source vertex.
///
/// `source_tolerance` is the accepted residual for a source vertex lying on
/// the curve: the geometric uncertainty declared by the source's representation
/// context, or [`SOURCE_INCIDENCE_TOLERANCE`] when the source declares none.
/// It is acceptance-only: it does not localize roots, merge candidates, or
/// decide carrier closure.
pub fn establish_source_edge_traversal<C>(
    curve: &C,
    start_pos: Point3,
    end_pos: Point3,
    topologically_closed: bool,
    source_tolerance: f64,
) -> SourceEdgeTraversal
where
    C: PolylineableCurve,
{
    let (lo, hi) = curve.evaluation_range();
    if !source_tolerance.is_finite() || source_tolerance <= 0.0 {
        return SourceEdgeTraversal::Unresolved {
            reason: "non_positive_source_tolerance",
        };
    }

    // A topologically closed edge (`edge.vertices.0 == edge.vertices.1`) is a
    // full-loop edge: its traversal is the whole evaluator loop, and the
    // legacy full-loop handling (period and partition-of-unity extensions) is
    // preserved. Topological closure is decided by vertex identity, never by
    // positional coincidence, and the evaluator seam is a parameterization
    // seam that may lie anywhere on the closed carrier.
    if topologically_closed {
        return SourceEdgeTraversal::CanonicalByEvalRange { range: (lo, hi) };
    }

    // For the distinct-vertex paths below the evaluator domain must be
    // genuine.
    if !lo.is_finite() || !hi.is_finite() || !(hi > lo) {
        return SourceEdgeTraversal::Unresolved {
            reason: "degenerate_evaluation_range",
        };
    }
    let subs_lo = curve.subs(lo);
    let subs_hi = curve.subs(hi);

    // The evaluator endpoints realize the distinct source vertices: the
    // ordinary P1 population. `CanonicalByEvalRange` preserves its sampling
    // exactly.
    if subs_lo.distance(start_pos) <= source_tolerance
        && subs_hi.distance(end_pos) <= source_tolerance
    {
        return SourceEdgeTraversal::CanonicalByEvalRange { range: (lo, hi) };
    }

    // Distinct source vertices not realized by the evaluator endpoints. The
    // source vertices must then be located at parameters of the genuine
    // evaluator domain, *regardless of whether the carrier is closed*. An open
    // carrier is a legitimate simple source interval: NIST STEP files contain
    // ordinary open B-spline `EDGE_CURVE`s whose start vertex sits at an
    // interior carrier parameter while the end vertex is realized at the
    // evaluator end. Requiring a closed carrier before attempting root
    // isolation rejected that ordinary population. The closed carrier is only
    // needed to decide *wrapping* (see below), not to decide whether
    // source-interval recovery is attempted at all.
    //
    // Carrier closure is judged against the fixed numerical tolerance, never
    // against `source_tolerance`. The lo≈hi seam equivalence is a
    // *parameterization* fact -- the curve returns to the same point -- and it
    // is what licenses a wrap through the evaluator seam and the cyclic
    // candidate handling. The STEP source uncertainty is an *incidence* fact:
    // it says how far a source vertex may sit from its carrier and still count
    // as realized. A large declared uncertainty must not widen the class of
    // "closed" carriers: an open fitted spline whose endpoints happen to be
    // within the uncertainty of each other is still an open curve, and wrapping
    // it would invent geometry across the gap.
    let carrier_closed = subs_lo.distance(subs_hi) <= SOURCE_INCIDENCE_TOLERANCE;

    // Locate each source vertex at a unique parameter root of the evaluator
    // domain. Each is required to be uniquely established and to hold a
    // source-consistent residual; any ambiguity or failure is `Unresolved`.
    let Some((t_start, r_start, n_start)) =
        isolate_vertex_root(curve, start_pos, lo, hi, source_tolerance, carrier_closed)
    else {
        return SourceEdgeTraversal::Unresolved {
            reason: "start_vertex_root_not_uniquely_established",
        };
    };
    let Some((t_end, r_end, n_end)) =
        isolate_vertex_root(curve, end_pos, lo, hi, source_tolerance, carrier_closed)
    else {
        return SourceEdgeTraversal::Unresolved {
            reason: "end_vertex_root_not_uniquely_established",
        };
    };
    if r_start > source_tolerance || r_end > source_tolerance {
        return SourceEdgeTraversal::Unresolved {
            reason: "root_residual_exceeds_source_tolerance",
        };
    }
    if (t_end - t_start).abs() <= f64::EPSILON {
        return SourceEdgeTraversal::Unresolved {
            reason: "degenerate_traversal",
        };
    }

    // The traversal follows the curve's natural increasing-parameter direction
    // (the source direction) from the start root to the end root. When the
    // start root precedes the end root the interval is simple, and that holds
    // on an open carrier as much as on a closed one. When the start root lies
    // after the end root the source direction crosses the evaluator seam: that
    // wrap is only geometric on a *closed* carrier, where `C(lo) ~= C(hi)` and
    // the two pieces join at the seam. An open carrier's two ends are distinct
    // points, so a wrapped traversal would invent geometry across the gap; that
    // case is `Unresolved`.
    let traversal = if t_start <= t_end {
        ParamTraversal::Simple {
            start: t_start,
            end: t_end,
        }
    } else if carrier_closed {
        ParamTraversal::Wrapped {
            start: t_start,
            domain_end: hi,
            domain_start: lo,
            end: t_end,
        }
    } else {
        return SourceEdgeTraversal::Unresolved {
            reason: "open_carrier_would_wrap",
        };
    };
    let witness = SourceEdgeTraversalWitness {
        traversal,
        start_parameter: t_start,
        end_parameter: t_end,
        start_residual: r_start,
        end_residual: r_end,
        start_candidates: n_start,
        end_candidates: n_end,
    };
    SourceEdgeTraversal::CanonicalBySourceInterval { traversal, witness }
}

/// Sample a traversal into a polyline, joining a wrapped interval's two pieces
/// across the evaluator seam without duplicating the closure sample.
pub fn sample_traversal<C>(curve: &C, traversal: &ParamTraversal, tol: f64) -> PolylineCurve
where C: PolylineableCurve {
    match traversal {
        ParamTraversal::Simple { start, end } => {
            PolylineCurve::from_curve(curve, (*start, *end), tol)
        }
        ParamTraversal::Wrapped {
            start,
            domain_end,
            domain_start,
            end,
        } => {
            let (_, mut points) = curve.parameter_division((*start, *domain_end), tol);
            let (_, second) = curve.parameter_division((*domain_start, *end), tol);
            // The seam sample `C(domain_end)` equals `C(domain_start)`; drop
            // the duplicate so the joined boundary carries no repeated point.
            points.pop();
            points.extend(second);
            PolylineCurve::from(points)
        }
    }
}

/// The base grid resolution of the candidate scan.
///
/// A true root always produces a sample dip at the grid vertex nearest to it
/// (the nearest sample to a minimum of a smooth distance function is that
/// minimum's own dip), so a fine uniform scan does not need a sample to land
/// on the root. The scan is repeated at twice this resolution as a stability
/// certificate: a second candidate narrower than the base grid would separate
/// into its own dip on refinement, changing the candidate count and forcing
/// `Unresolved` instead of a false uniqueness claim.
const ROOT_SCAN_N: usize = 1 << 16;

/// The agreement bound for the two-resolution stability certificate, as a
/// fraction of the domain span. Both resolutions refine the *same* root by
/// golden-section, so their parameters agree to far tighter than this; a
/// disagreement at this scale means the two resolutions resolved different
/// candidates.
const STABILITY_EPS: f64 = 1.0e-5;

/// Deterministically isolate every source-consistent candidate root of
/// `d(t) = |C(t) - vertex|²` on `[lo, hi]` at one grid resolution.
///
/// A uniform grid is scanned for *sample dips* — vertices whose squared
/// distance is no greater than both neighbours' — and each dip is refined to a
/// local minimum by golden-section search. Every root of `d` at or below
/// `source_tolerance` produces such a dip (at the grid vertex nearest to it),
/// and every refined dip whose residual exceeds the tolerance is discarded, so
/// the returned list is exactly the source-consistent candidate set at this
/// resolution.
///
/// The neighbour access is cyclic across the seam (`lo ~ hi`) **only for a
/// closed carrier**, so a root sitting on the closed seam is a dip at both
/// ends and is deduplicated downstream. On an open carrier the evaluator
/// endpoints are genuine boundaries: the first and last samples are not
/// neighbours of each other, and a root exactly at an endpoint is a valid
/// boundary root, not a seam duplicate.
fn isolate_candidates_at<C>(
    curve: &C,
    vertex: Point3,
    lo: f64,
    hi: f64,
    source_tolerance: f64,
    n: usize,
    carrier_closed: bool,
) -> Vec<(f64, f64)>
where
    C: PolylineableCurve,
{
    let step = (hi - lo) / n as f64;
    let at = |i: usize| lo + step * i as f64;
    let d2: Vec<f64> = (0..=n)
        .map(|i| curve.subs(at(i)).distance2(vertex))
        .collect();
    let mut out: Vec<(f64, f64)> = Vec::new();
    for i in 0..=n {
        // A root at the evaluator endpoint of an open carrier is a dip against
        // its single interior neighbour. On a closed carrier the seam wraps so
        // the two ends are neighbours of each other.
        let (left, right) = if carrier_closed {
            (
                if i == 0 { n } else { i - 1 },
                if i == n { 0 } else { i + 1 },
            )
        } else if i == 0 {
            (0, 1)
        } else if i == n {
            (n - 1, n)
        } else {
            (i - 1, i + 1)
        };
        if d2[i] <= d2[left] && d2[i] <= d2[right] {
            let (t, res) = golden_section_min(curve, vertex, at(left), at(right));
            if res <= source_tolerance {
                out.push((t, res));
            }
        }
    }
    out
}

/// Deduplicate a candidate list. On a closed carrier `lo` and `hi` are the
/// same geometric point, so a root near `lo` and one near `hi` are one root
/// modulo the period. On an open carrier the evaluator endpoints are distinct
/// and no such equivalence holds: roots near `lo` and near `hi` are genuinely
/// distinct candidates.
fn deduplicate_candidates(
    lo: f64,
    hi: f64,
    candidates: Vec<(f64, f64)>,
    carrier_closed: bool,
) -> Vec<(f64, f64)> {
    let period = hi - lo;
    let dedup_eps = period * 1.0e-9;
    let mut distinct: Vec<(f64, f64)> = Vec::new();
    for (t, res) in candidates {
        let duplicate = if carrier_closed {
            distinct.iter().any(|(u, _)| {
                (t - u).abs() <= dedup_eps || (period - (t - u).abs()).abs() <= dedup_eps
            })
        } else {
            distinct.iter().any(|(u, _)| (t - u).abs() <= dedup_eps)
        };
        if !duplicate {
            distinct.push((t, res));
        }
    }
    distinct
}

/// Locate the parameter of `vertex` on the curve over `[lo, hi]`, requiring
/// exactly one distinct source-consistent root.
///
/// Returns `(t, residual, candidate_count)` when exactly one distinct root is
/// found and that result is *stable* across the two-resolution certificate:
/// the same single candidate must appear, refined to the same parameter, at
/// both the base and double resolutions. `None` when there are no candidates,
/// several, or the two resolutions disagree — all of which are "not uniquely
/// established" and must not certify a traversal.
fn isolate_vertex_root<C>(
    curve: &C,
    vertex: Point3,
    lo: f64,
    hi: f64,
    source_tolerance: f64,
    carrier_closed: bool,
) -> Option<(f64, f64, usize)>
where
    C: PolylineableCurve,
{
    let base = deduplicate_candidates(
        lo,
        hi,
        isolate_candidates_at(
            curve,
            vertex,
            lo,
            hi,
            source_tolerance,
            ROOT_SCAN_N,
            carrier_closed,
        ),
        carrier_closed,
    );
    let fine = deduplicate_candidates(
        lo,
        hi,
        isolate_candidates_at(
            curve,
            vertex,
            lo,
            hi,
            source_tolerance,
            ROOT_SCAN_N * 2,
            carrier_closed,
        ),
        carrier_closed,
    );
    let stable = base.len() == 1
        && fine.len() == 1
        && (base[0].0 - fine[0].0).abs() <= STABILITY_EPS * (hi - lo);
    if !stable {
        return None;
    }
    let (t, res) = base[0];
    if res > source_tolerance {
        return None;
    }
    Some((t, res, base.len()))
}

/// Golden-section minimum of `|C(t) - vertex|²` on `[a, b]`.
fn golden_section_min<C>(curve: &C, vertex: Point3, a: f64, b: f64) -> (f64, f64)
where C: PolylineableCurve {
    const PHI: f64 = 1.618033988749895;
    let mut lo = a;
    let mut hi = b;
    let mut c = hi - (hi - lo) / PHI;
    let mut d = lo + (hi - lo) / PHI;
    for _ in 0..200 {
        if (hi - lo).abs() < 1.0e-14 {
            break;
        }
        let fc = curve.subs(c).distance2(vertex);
        let fd = curve.subs(d).distance2(vertex);
        if fc < fd {
            hi = d;
        } else {
            lo = c;
        }
        c = hi - (hi - lo) / PHI;
        d = lo + (hi - lo) / PHI;
    }
    let t = (lo + hi) * 0.5;
    (t, curve.subs(t).distance(vertex))
}

#[cfg(test)]
mod tests {
    use super::*;
    use truck_geometry::prelude::{BSplineCurve, KnotVec, Vector3};

    /// An ordinary open cubic Bezier over `[0, 1]` whose evaluator endpoints
    /// are exactly its source endpoints: the P1 population's canonical shape.
    fn open_cubic() -> BSplineCurve<Point3> {
        BSplineCurve::new(
            KnotVec::bezier_knot(3),
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(2.0, -1.0, 0.0),
                Point3::new(3.0, 0.0, 0.0),
            ],
        )
    }

    /// A clamped, closed cubic spline over `[0, 1]` whose last control point
    /// equals its first, so `C(0) = C(1)` exactly. The vertices for the source
    /// interval test are interior curve points, which is the `00007667`
    /// edge-#30 shape: a closed evaluator loop whose source vertices are not
    /// at the evaluator endpoints.
    fn closed_cubic() -> BSplineCurve<Point3> {
        BSplineCurve::new(
            KnotVec::uniform_knot(3, 4),
            vec![
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(0.0, 1.5, 0.0),
                Point3::new(-1.0, 0.0, 0.0),
                Point3::new(0.0, -1.5, 0.0),
                Point3::new(1.0, -1.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
            ],
        )
    }

    #[test]
    fn evaluator_endpoints_realizing_the_source_vertices_yield_canonical_eval_range() {
        let curve = open_cubic();
        let (lo, hi) = curve.evaluation_range();
        let start = curve.subs(lo);
        let end = curve.subs(hi);
        // Distinct source vertices realized by the evaluator endpoints.
        assert!(start.distance(end) > SOURCE_INCIDENCE_TOLERANCE);
        let traversal =
            establish_source_edge_traversal(&curve, start, end, false, SOURCE_INCIDENCE_TOLERANCE);
        match traversal {
            SourceEdgeTraversal::CanonicalByEvalRange { range } => {
                assert_eq!(range, (lo, hi));
            }
            other => panic!("expected CanonicalByEvalRange, got {other:?}"),
        }
    }

    #[test]
    fn topologically_closed_edge_keeps_the_full_loop() {
        // A full-loop edge: the source claims one vertex (`vertices.0 ==
        // .1`), and the evaluator seam is a parameterization seam that need
        // not coincide with the shared source vertex. The traversal is the
        // whole loop either way.
        let curve = closed_cubic();
        let (lo, hi) = curve.evaluation_range();
        let seam = curve.subs(lo);
        assert!(seam.distance(curve.subs(hi)) < 1.0e-9);
        for shared in [seam, curve.subs(0.6)] {
            let traversal = establish_source_edge_traversal(
                &curve,
                shared,
                shared,
                true,
                SOURCE_INCIDENCE_TOLERANCE,
            );
            match traversal {
                SourceEdgeTraversal::CanonicalByEvalRange { range } => {
                    assert_eq!(range, (lo, hi));
                }
                other => panic!("expected CanonicalByEvalRange, got {other:?}"),
            }
        }
    }

    #[test]
    fn distinct_coincident_vertices_do_not_license_the_full_loop() {
        // Two *distinct* source vertex entities at the same interior position
        // of a closed loop (`vertices.0 != .1`): positional coincidence is not
        // topological closure, so the whole loop must not be accepted. Both
        // vertices resolve to the same interior root, which is a degenerate
        // traversal, and the honest verdict is `Unresolved`.
        let curve = closed_cubic();
        let interior = curve.subs(0.6);
        let traversal = establish_source_edge_traversal(
            &curve,
            interior,
            interior,
            false,
            SOURCE_INCIDENCE_TOLERANCE,
        );
        match traversal {
            SourceEdgeTraversal::Unresolved { reason } => {
                assert_eq!(reason, "degenerate_traversal");
            }
            other => panic!("expected Unresolved, got {other:?}"),
        }
    }

    #[test]
    fn closed_loop_with_interior_vertices_wraps_through_the_seam() {
        let curve = closed_cubic();
        let (lo, hi) = curve.evaluation_range();
        // The fixture is genuinely closed: C(0) == C(1) to numerical precision.
        assert!(curve.subs(lo).distance(curve.subs(hi)) < 1.0e-9);
        // Source vertices at interior parameters, start root after the end
        // root, so the source-directed traversal crosses the evaluator seam.
        // The two source vertices are distinct entities, so the edge is not
        // topologically closed.
        let t_start = 0.8;
        let t_end = 0.2;
        let start_pos = curve.subs(t_start);
        let end_pos = curve.subs(t_end);
        assert!(start_pos.distance(end_pos) > SOURCE_INCIDENCE_TOLERANCE);
        let traversal = establish_source_edge_traversal(
            &curve,
            start_pos,
            end_pos,
            false,
            SOURCE_INCIDENCE_TOLERANCE,
        );
        match traversal {
            SourceEdgeTraversal::CanonicalBySourceInterval { traversal, witness } => {
                let ParamTraversal::Wrapped {
                    start,
                    domain_end,
                    domain_start,
                    end,
                } = traversal
                else {
                    panic!("expected wrapped traversal, got {traversal:?}");
                };
                assert!(
                    (start - t_start).abs() < 1.0e-4,
                    "start={start} t_start={t_start}"
                );
                assert!((end - t_end).abs() < 1.0e-4, "end={end} t_end={t_end}");
                assert_eq!(domain_end, hi);
                assert_eq!(domain_start, lo);
                assert!(witness.start_residual < SOURCE_INCIDENCE_TOLERANCE);
                assert!(witness.end_residual < SOURCE_INCIDENCE_TOLERANCE);
                assert_eq!(witness.start_candidates, 1);
                assert_eq!(witness.end_candidates, 1);
            }
            other => panic!("expected CanonicalBySourceInterval, got {other:?}"),
        }
    }

    #[test]
    fn open_spline_with_interior_start_vertex_and_evaluator_end_end_vertex() {
        // The NIST regression shape: an ordinary *open* spline carrier
        // (`C(0) != C(1)`) whose source edge's start vertex sits at an
        // interior carrier parameter while its end vertex is realized at the
        // evaluator's high end. R01 must isolate the interior start root on
        // the open domain and return a simple `CanonicalBySourceInterval`;
        // it must NOT reject the edge for lack of a closed carrier, and it
        // must NOT wrap an open carrier.
        let curve = open_cubic();
        let (lo, hi) = curve.evaluation_range();
        // The fixture's evaluator endpoints are genuinely open: C(0) != C(1).
        assert!(curve.subs(lo).distance(curve.subs(hi)) > SOURCE_INCIDENCE_TOLERANCE);
        // Start vertex at an interior parameter, end vertex at the evaluator
        // high end. Distinct source vertices, so not topologically closed.
        let t_start = 0.3;
        let t_end = 1.0;
        let start_pos = curve.subs(t_start);
        let end_pos = curve.subs(t_end);
        assert!(start_pos.distance(end_pos) > SOURCE_INCIDENCE_TOLERANCE);
        let traversal = establish_source_edge_traversal(
            &curve,
            start_pos,
            end_pos,
            false,
            SOURCE_INCIDENCE_TOLERANCE,
        );
        match traversal {
            SourceEdgeTraversal::CanonicalBySourceInterval { traversal, witness } => {
                let ParamTraversal::Simple { start, end } = traversal else {
                    panic!("expected simple traversal, got {traversal:?}");
                };
                assert!(
                    (start - t_start).abs() < 1.0e-4,
                    "start={start} t_start={t_start}"
                );
                assert!((end - t_end).abs() < 1.0e-4, "end={end} t_end={t_end}");
                assert!(witness.start_residual < SOURCE_INCIDENCE_TOLERANCE);
                assert!(witness.end_residual < SOURCE_INCIDENCE_TOLERANCE);
                assert_eq!(witness.start_candidates, 1);
                assert_eq!(witness.end_candidates, 1);
            }
            other => panic!("expected CanonicalBySourceInterval, got {other:?}"),
        }
    }

    /// A degree-6 figure-eight Bezier that passes through the origin three
    /// times: at `t = 0`, `t = 1` (both the seam) and `t = 0.5` (the crossing).
    ///
    /// The three passages are spatially distinct *parameters*: the seam roots
    /// merge modulo the closed domain, leaving two distinct candidate
    /// locations for a vertex near the crossing. Used by the T3 uniqueness test.
    fn figure_eight() -> BSplineCurve<Point3> {
        BSplineCurve::new(
            KnotVec::bezier_knot(6),
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.5, 2.0, 0.0),
                Point3::new(-1.5, 2.0, 0.0),
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.5, -2.0, 0.0),
                Point3::new(-1.5, -2.0, 0.0),
                Point3::new(0.0, 0.0, 0.0),
            ],
        )
    }

    /// T1: the source tolerance admits an approximate incidence that Truck's
    /// fixed `1e-6` rejects.
    ///
    /// The NIST population is a vertex that sits off its edge-curve carrier by
    /// more than `1e-6` but within the geometric uncertainty the source
    /// declares (the fitted-spline / off-plane-circle family). The same edge
    /// must be `Unresolved` under the fixed tolerance and `SourceInterval`
    /// under the source's own tolerance, with the sharp residual accepted and
    /// the parameter realization still unique.
    #[test]
    fn source_tolerance_admits_approximate_incidence() {
        let curve = open_cubic();
        let (lo, _) = curve.evaluation_range();
        // Start vertex lifted off the carrier by 2e-5 (perpendicular to the
        // planar fixture, so the minimum distance is exactly 2e-5 at t=0.3),
        // end vertex exactly at the evaluator high end. 2e-5 > 1e-6 but
        // < the declared 1e-4.
        let t_start = 0.3;
        let t_end = 1.0;
        let start_pos = curve.subs(t_start) + Vector3::new(0.0, 0.0, 2.0e-5);
        let end_pos = curve.subs(t_end);
        let source_tolerance = 1.0e-4;
        // The EvalRange branch must not fire: the off-carrier start vertex is
        // far from C(lo) under either tolerance.
        assert!(curve.subs(lo).distance(start_pos) > source_tolerance);

        let tight = establish_source_edge_traversal(
            &curve,
            start_pos,
            end_pos,
            false,
            SOURCE_INCIDENCE_TOLERANCE,
        );
        assert!(
            matches!(tight, SourceEdgeTraversal::Unresolved { .. }),
            "the 1e-6 gate must reject a 2e-5 off-carrier vertex, got {tight:?}"
        );

        let admitted =
            establish_source_edge_traversal(&curve, start_pos, end_pos, false, source_tolerance);
        match admitted {
            SourceEdgeTraversal::CanonicalBySourceInterval { traversal, witness } => {
                let ParamTraversal::Simple { start, end } = traversal else {
                    panic!("expected simple traversal, got {traversal:?}");
                };
                assert!(
                    (start - t_start).abs() < 1.0e-4,
                    "start={start} t_start={t_start}"
                );
                assert!((end - t_end).abs() < 1.0e-4, "end={end} t_end={t_end}");
                // The residual is the sharp refined distance, 2e-5, admitted by
                // the source tolerance and still one candidate.
                assert!(
                    (witness.start_residual - 2.0e-5).abs() < 1.0e-7,
                    "start_residual={}",
                    witness.start_residual
                );
                assert_eq!(witness.start_candidates, 1);
                assert_eq!(witness.end_candidates, 1);
            }
            other => panic!("expected CanonicalBySourceInterval, got {other:?}"),
        }
    }

    /// T2: a residual outside the source tolerance remains `Unresolved`, with
    /// no evaluator-range fallback.
    #[test]
    fn residual_outside_source_tolerance_stays_unresolved() {
        let curve = open_cubic();
        // Off-carrier by 2e-4, which the 1e-4 source tolerance rejects.
        let t_start = 0.3;
        let t_end = 1.0;
        let start_pos = curve.subs(t_start) + Vector3::new(0.0, 0.0, 2.0e-4);
        let end_pos = curve.subs(t_end);
        let traversal = establish_source_edge_traversal(&curve, start_pos, end_pos, false, 1.0e-4);
        match traversal {
            SourceEdgeTraversal::Unresolved { .. } => {}
            other => panic!("expected Unresolved, got {other:?}"),
        }
    }

    /// T3: a larger source tolerance must not destroy parameter uniqueness.
    ///
    /// The figure-eight passes near a vertex at the crossing in two spatially
    /// distinct parameter locations (the seam, merged modulo the closed domain,
    /// and the crossing itself). A large tolerance admits both; the verdict
    /// must be `Unresolved` -- ambiguity -- never an arbitrary pick of one
    /// candidate as if it were unique.
    #[test]
    fn larger_source_tolerance_does_not_merge_distinct_parameter_locations() {
        let curve = figure_eight();
        let (lo, hi) = curve.evaluation_range();
        // The fixture is closed at the seam: C(0) == C(1) == origin.
        assert!(curve.subs(lo).distance(curve.subs(hi)) < 1.0e-9);
        // The crossing passage: C(0.5) == origin, so a vertex just off the
        // crossing is within a large tolerance of both the crossing parameter
        // and the seam parameter (two distinct parameter realizations).
        let start_pos = Point3::new(0.0, 0.01, 0.0);
        let end_pos = Point3::new(0.5, 0.0, 0.0);
        assert!(start_pos.distance(end_pos) > SOURCE_INCIDENCE_TOLERANCE);
        // A tolerance large enough that the crossing AND the seam are both
        // within it. Each side is a genuinely distinct parameter location; the
        // source interval cannot be certified.
        let traversal = establish_source_edge_traversal(&curve, start_pos, end_pos, false, 0.05);
        assert!(
            matches!(traversal, SourceEdgeTraversal::Unresolved { .. }),
            "two distinct realizations inside the tolerance must be Unresolved, got {traversal:?}"
        );
    }

    /// T5: the wrap authority stays restricted to established closed carriers
    /// no matter how large the source tolerance.
    ///
    /// An open carrier whose start root lies after its end root would need a
    /// wrap; the endpoints of an open carrier are distinct points, so the wrap
    /// would invent geometry across the gap. A large source tolerance must not
    /// reclassify the open carrier as closed (the lo~hi seam equivalence is a
    /// numerical-parameterization fact, not an incidence fact).
    #[test]
    fn open_carrier_never_wraps_regardless_of_source_tolerance() {
        let curve = open_cubic();
        let (lo, hi) = curve.evaluation_range();
        // Genuinely open: C(0) and C(1) are far apart.
        assert!(curve.subs(lo).distance(curve.subs(hi)) > 0.1);
        // Start root after the end root on the open carrier. The endpoints are
        // each realized at their own parameter, so the source direction crosses
        // the evaluator seam -- which is only geometric on a closed carrier.
        let t_start = 0.8;
        let t_end = 0.2;
        let start_pos = curve.subs(t_start);
        let end_pos = curve.subs(t_end);
        assert!(start_pos.distance(end_pos) > SOURCE_INCIDENCE_TOLERANCE);
        let traversal = establish_source_edge_traversal(&curve, start_pos, end_pos, false, 0.5);
        assert!(
            matches!(traversal, SourceEdgeTraversal::Unresolved { .. }),
            "an open carrier that would wrap must stay Unresolved, got {traversal:?}"
        );
    }

    /// T6: the topological self-loop semantics are unchanged by the source
    /// tolerance.
    ///
    /// A genuinely topologically-closed edge (one vertex entity) keeps the full
    /// evaluator loop even under a large source tolerance, and two distinct
    /// coincident vertex entities are still not a closed edge.
    #[test]
    fn topological_self_loop_semantics_are_unchanged_by_source_tolerance() {
        let curve = closed_cubic();
        let (lo, hi) = curve.evaluation_range();
        // Topologically closed: the whole loop, whatever the tolerance.
        let seam = curve.subs(lo);
        let traversal = establish_source_edge_traversal(&curve, seam, seam, true, 0.5);
        match traversal {
            SourceEdgeTraversal::CanonicalByEvalRange { range } => {
                assert_eq!(range, (lo, hi));
            }
            other => panic!("expected CanonicalByEvalRange, got {other:?}"),
        }
        // Distinct entities at the same interior position: not topologically
        // closed, and a degenerate traversal under any tolerance.
        let interior = curve.subs(0.6);
        let traversal = establish_source_edge_traversal(&curve, interior, interior, false, 0.5);
        assert!(
            matches!(traversal, SourceEdgeTraversal::Unresolved { .. }),
            "distinct coincident vertices must not license the full loop, got {traversal:?}"
        );
    }
}
