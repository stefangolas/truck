#![allow(clippy::many_single_char_names)]
// PR 4A.1 adds sidecar diagnostic types and outcome entry points that are not
// yet consumed by callers. Silence their dead-code until the census examples
// wire them up; remove this `allow` when the diagnostic API is finalized.
#![allow(dead_code, unused)]

use super::*;
use crate::filters::NormalFilters;
use crate::Point2;
use array_macro::array;
use handles::{FixedUndirectedEdgeHandle, FixedVertexHandle};
use itertools::Itertools;
use rustc_hash::FxHashMap as HashMap;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

type SPoint2 = spade::Point2<f64>;
type Cdt = ConstrainedDelaunayTriangulation<SPoint2>;
std::thread_local! {
    /// Optional document-local source face id, declared face index, and
    /// parameter-space periodic rank for probes.
    static PROBE_FACE_CONTEXT: std::cell::Cell<(Option<u64>, usize, u8)> =
        const { std::cell::Cell::new((None, usize::MAX, 0)) };
}

type MeshedShell = Shell<Point3, PolylineCurve, Option<PolygonMesh>>;
type MeshedCShell = CompressedShell<Point3, PolylineCurve, Option<PolygonMesh>>;

pub(super) trait SP<S>:
    Fn(&S, Point3, Option<(f64, f64)>) -> Option<(f64, f64)> + Parallelizable
{
}
impl<S, F> SP<S> for F where
    F: Fn(&S, Point3, Option<(f64, f64)>) -> Option<(f64, f64)> + Parallelizable
{
}

/// Print taxonomy summary.
pub fn print_taxonomy_summary() {}

pub(super) fn by_search_parameter<S>(
    surface: &S,
    point: Point3,
    hint: Option<(f64, f64)>,
) -> Option<(f64, f64)>
where
    S: MeshableSurface,
{
    surface
        .search_parameter(point, hint, 100)
        .or_else(|| surface.search_parameter(point, None, 100))
}

pub(super) fn by_search_nearest_parameter<S>(
    surface: &S,
    point: Point3,
    hint: Option<(f64, f64)>,
) -> Option<(f64, f64)>
where
    S: RobustMeshableSurface,
{
    surface
        .search_parameter(point, hint, 100)
        .or_else(|| surface.search_parameter(point, None, 100))
        .or_else(|| surface.search_nearest_parameter(point, hint, 100))
        .or_else(|| surface.search_nearest_parameter(point, None, 100))
}

/// Tessellates faces
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn shell_tessellation<'a, C, S>(
    shell: &Shell<Point3, C, S>,
    tol: f64,
    sp: impl SP<S>,
) -> MeshedShell
where
    C: PolylineableCurve + 'a,
    S: PreMeshableSurface + 'a,
{
    let vmap: HashMap<_, _> = shell
        .vertex_par_iter()
        .map(|v| (v.id(), v.mapped(Point3::clone)))
        .collect();
    let eset: HashMap<_, _> = shell.edge_par_iter().map(move |e| (e.id(), e)).collect();
    let edge_map: HashMap<_, _> = eset
        .into_par_iter()
        .map(move |(id, edge)| {
            let v0 = vmap.get(&edge.absolute_front().id()).unwrap();
            let v1 = vmap.get(&edge.absolute_back().id()).unwrap();
            let curve = edge.curve();
            let poly = PolylineCurve::from_curve(&curve, curve.range_tuple(), tol);
            (id, Edge::debug_new(v0, v1, poly))
        })
        .collect();
    let create_edge = |edge: &Edge<Point3, C>| -> Edge<_, _> {
        let new_edge = edge_map.get(&edge.id()).unwrap();
        match edge.orientation() {
            true => new_edge.clone(),
            false => new_edge.inverse(),
        }
    };
    let create_boundary =
        |wire: &Wire<Point3, C>| -> Wire<_, _> { wire.edge_iter().map(create_edge).collect() };
    let create_face = move |face: &Face<Point3, C, S>| -> Face<_, _, _> {
        let wires: Vec<_> = face
            .absolute_boundaries()
            .iter()
            .map(create_boundary)
            .collect();
        shell_create_polygon(&face.surface(), wires, face.orientation(), tol, &sp)
    };
    shell.face_par_iter().map(create_face).collect()
}

/// Tessellates faces
#[cfg(any(target_arch = "wasm32", test))]
pub(super) fn shell_tessellation_single_thread<'a, C, S>(
    shell: &'a Shell<Point3, C, S>,
    tol: f64,
    sp: impl SP<S>,
) -> MeshedShell
where
    C: PolylineableCurve + 'a,
    S: PreMeshableSurface + 'a,
{
    use truck_base::entry_map::FxEntryMap as EntryMap;
    use truck_topology::Vertex as TVertex;
    let mut vmap = EntryMap::new(
        move |v: &TVertex<Point3>| v.id(),
        move |v| v.mapped(Point3::clone),
    );
    let mut edge_map = EntryMap::new(
        move |edge: &'a Edge<Point3, C>| edge.id(),
        move |edge| {
            let vf = edge.absolute_front();
            let v0 = vmap.entry_or_insert(vf).clone();
            let vb = edge.absolute_back();
            let v1 = vmap.entry_or_insert(vb).clone();
            let curve = edge.curve();
            let poly = PolylineCurve::from_curve(&curve, curve.range_tuple(), tol);
            Edge::debug_new(&v0, &v1, poly)
        },
    );
    let mut create_edge = move |edge: &'a Edge<Point3, C>| -> Edge<_, _> {
        let new_edge = edge_map.entry_or_insert(edge);
        match edge.orientation() {
            true => new_edge.clone(),
            false => new_edge.inverse(),
        }
    };
    let mut create_boundary = move |wire: &'a Wire<Point3, C>| -> Wire<_, _> {
        wire.edge_iter().map(&mut create_edge).collect()
    };
    let create_face = move |face: &'a Face<Point3, C, S>| -> Face<_, _, _> {
        let wires: Vec<_> = face
            .absolute_boundaries()
            .iter()
            .map(&mut create_boundary)
            .collect();
        shell_create_polygon(&face.surface(), wires, face.orientation(), tol, &sp)
    };
    shell.face_iter().map(create_face).collect()
}

/// Tessellates faces
pub(super) fn cshell_tessellation<'a, C, S>(
    shell: &CompressedShell<Point3, C, S>,
    tol: f64,
    sp: impl SP<S>,
) -> MeshedCShell
where
    C: PolylineableCurve + 'a,
    S: PreMeshableSurface + 'a,
{
    let vertices = shell.vertices.clone();
    let edge_probe = std::env::var_os("TRUCK_PROBE_EDGE").is_some();
    let tessellate_edge = |edge: &CompressedEdge<C>| {
        let curve = &edge.curve;
        let range = curve.range_tuple();
        if edge_probe {
            // How much of its own period a curve is asked to cover. An edge
            // whose start and end vertices coincide gives the importer no
            // independent parameter for each end -- they are the same point
            // modulo the period -- so a generic endpoint solver can resolve
            // them into copies two periods apart. The ratio, not the absolute
            // range, is what says so: a shifted circle may legitimately run
            // over [-pi, 3pi], which is still two periods.
            let span = range.1 - range.0;
            let ratio = curve.period().map(|period| span / period);
            // The radius the converted curve actually has. The source file's
            // circles are a short, known inventory, so a converted radius that
            // is not in it means the conversion built the wrong geometry for
            // the right entity; one that is in it exonerates the curve and
            // points at the face-to-surface pairing instead.
            let fitted = {
                let (t0, t1) = range;
                let (a, b, c) = (
                    curve.subs(t0),
                    curve.subs(t0 + (t1 - t0) / 3.0),
                    curve.subs(t0 + 2.0 * (t1 - t0) / 3.0),
                );
                let (ab, ac) = (b - a, c - a);
                let normal = ab.cross(ac);
                let n2 = normal.magnitude2();
                match n2 > f64::EPSILON {
                    true => {
                        let centre = a
                            + (ac.magnitude2() * ab.cross(normal)
                                - ab.magnitude2() * ac.cross(normal))
                                / (2.0 * n2);
                        Some(centre.distance(a))
                    }
                    false => None,
                }
            };
            eprintln!(
                "EDGE range=({:.6},{:.6}) span={span:.6} period={:?} span/period={:?} \
                 same_vertex={} fitted_radius={:?}",
                range.0,
                range.1,
                curve.period(),
                ratio,
                edge.vertices.0 == edge.vertices.1,
                fitted.map(|r| (r * 1.0e5).round() / 1.0e5),
            );
        }
        let mut range = curve.range_tuple();
        if edge.vertices.0 == edge.vertices.1 && (range.1 - range.0).abs() < 1e-4 {
            if let Some(period) = curve.period() {
                if period > 1e-4 {
                    range = (range.0, range.0 + period);
                }
            }
        }
        let mut poly = PolylineCurve::from_curve(curve, range, tol);
        if poly.len() <= 2 && range.1 - range.0 > 1e-4 {
            let mut pts = Vec::new();
            const STEPS: usize = 16;
            for i in 0..=STEPS {
                let t = range.0 + (i as f64 / STEPS as f64) * (range.1 - range.0);
                pts.push(curve.subs(t));
            }
            poly = PolylineCurve::from(pts);
        }
        CompressedEdge {
            vertices: edge.vertices,
            curve: poly,
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    let edges: Vec<_> = shell.edges.par_iter().map(tessellate_edge).collect();
    #[cfg(target_arch = "wasm32")]
    let edges: Vec<_> = shell.edges.iter().map(tessellate_edge).collect();
    // Which surface in this shell does a face's own boundary actually lie on?
    //
    // A residual says the boundary and the surface it was handed are
    // incompatible; it does not say whether the pairing is wrong or one of the
    // entities is built wrong. Testing the boundary against *every* surface in
    // the shell separates them. If another surface fits to near zero, this is
    // an association defect and the correct partner is named. If none fits, one
    // of the two entities is constructed incorrectly.
    if std::env::var_os("TRUCK_PROBE_ASSOC").is_some() {
        const GRID: usize = 60;
        let nearest = |surface: &S, point: Point3| {
            let (urange, vrange) = surface.try_range_tuple();
            let axis = |range: Option<(f64, f64)>, period: Option<f64>| match (range, period) {
                (Some(r), _) => r,
                (None, Some(p)) => (-p, p),
                (None, None) => (-1.0, 1.0),
            };
            let (ulo, uhi) = axis(urange, surface.u_period());
            let (vlo, vhi) = axis(vrange, surface.v_period());
            let mut best = f64::INFINITY;
            for i in 0..=GRID {
                let u = ulo + (uhi - ulo) * i as f64 / GRID as f64;
                for j in 0..=GRID {
                    let v = vlo + (vhi - vlo) * j as f64 / GRID as f64;
                    best = best.min(surface.subs(u, v).distance(point));
                }
            }
            best
        };
        for (index, face) in shell.faces.iter().enumerate() {
            // A handful of boundary samples is enough to decide a pairing.
            let samples: Vec<Point3> = face
                .boundaries
                .iter()
                .flatten()
                .filter_map(|e| edges.get(e.index))
                .flat_map(|e| e.curve.iter().copied())
                .step_by(3)
                .take(8)
                .collect();
            if samples.is_empty() {
                continue;
            }
            let worst = |surface: &S| {
                samples
                    .iter()
                    .fold(0.0_f64, |acc, p| acc.max(nearest(surface, *p)))
            };
            let own = worst(&face.surface);
            if own <= tol * 3.0 {
                continue;
            }
            let mut best = (usize::MAX, f64::INFINITY);
            for (other, candidate) in shell.faces.iter().enumerate() {
                if other == index {
                    continue;
                }
                let d = worst(&candidate.surface);
                if d < best.1 {
                    best = (other, d);
                }
            }
            eprintln!(
                "ASSOC face={index} own={own:.4e} best_other=face{} at {:.4e} tol={tol:.4e}",
                best.0, best.1
            );
        }
    }
    let tessellate_face = |(declared_face_index, face): (usize, &CompressedFace<S>)| {
        let source_face_id = face.provenance.best_id().map(SourceEntityId::get);
        let periodic_rank = u8::from(face.surface.u_period().is_some())
            + u8::from(face.surface.v_period().is_some());
        PROBE_FACE_CONTEXT.with(|context| {
            context.set((source_face_id, declared_face_index, periodic_rank));
        });

        let boundaries = face.boundaries.clone();
        let surface = &face.surface;
        let create_edge = |edge_idx: &CompressedEdgeIndex| match edge_idx.orientation {
            true => Some(edges.get(edge_idx.index)?.curve.clone()),
            false => Some(edges.get(edge_idx.index)?.curve.inverse()),
        };
        let create_boundary = |wire: &Vec<CompressedEdgeIndex>| {
            let wire_iter = wire.iter().filter_map(create_edge);
            PolyBoundaryPiece::try_new(surface, wire_iter, &sp, tol)
        };
        let preboundary: Option<Vec<_>> = boundaries.iter().map(create_boundary).collect();
        let polygon: Option<PolygonMesh> = preboundary.map(|preboundary| {
            let boundary = PolyBoundary::new(preboundary, &surface, tol);
            trimming_tessellation(&surface, &boundary, tol)
        });
        let result = CompressedFace {
            boundaries,
            orientation: face.orientation,
            surface: polygon,
            // Tessellation is the stage most likely to produce nothing, so it
            // is the stage that most needs to say which face produced nothing.
            // `polygon` is `None` on failure, and the identity is then the only
            // thing left that can name what was lost.
            provenance: face.provenance,
        };
        PROBE_FACE_CONTEXT.with(|context| context.set((None, usize::MAX, 0)));
        result
    };
    #[cfg(not(target_arch = "wasm32"))]
    let faces = shell
        .faces
        .par_iter()
        .enumerate()
        .map(tessellate_face)
        .collect();
    #[cfg(target_arch = "wasm32")]
    let faces = shell.faces.iter().enumerate().map(tessellate_face).collect();
    MeshedCShell {
        vertices,
        edges,
        faces,
    }
}

fn shell_create_polygon<S: PreMeshableSurface>(
    surface: &S,
    wires: Vec<Wire<Point3, PolylineCurve>>,
    orientation: bool,
    tol: f64,
    sp: impl SP<S>,
) -> Face<Point3, PolylineCurve, Option<PolygonMesh>> {
    let preboundary = wires
        .iter()
        .map(|wire: &Wire<_, _>| {
            let wire_iter = wire.iter().map(Edge::oriented_curve);
            PolyBoundaryPiece::try_new(surface, wire_iter, &sp, tol)
        })
        .collect::<Option<Vec<_>>>();
    let polygon: Option<PolygonMesh> = preboundary.map(|preboundary| {
        let boundary = PolyBoundary::new(preboundary, &surface, tol);
        trimming_tessellation(surface, &boundary, tol)
    });
    let mut new_face = Face::debug_new(wires, polygon);
    if !orientation {
        new_face.invert();
    }
    new_face
}

#[derive(Clone, Copy, Debug, derive_more::Deref, derive_more::DerefMut)]
pub(in crate::tessellation) struct SurfacePoint {
    pub(in crate::tessellation) point: Point3,
    #[deref]
    #[deref_mut]
    pub(in crate::tessellation) uv: Point2,
}

impl From<(Point2, Point3)> for SurfacePoint {
    fn from((uv, point): (Point2, Point3)) -> Self {
        Self { point, uv }
    }
}


/// Reconciles a UV step entering or leaving a detected collapsed direction.
///
/// A small derivative only proposes a chart substitution. The candidate UV
/// must also evaluate back to the associated 3D point within model tolerance.
/// An outgoing bridge retains the previous singular 3D point; only its UV
/// representative changes to connect the two chart branches.
fn reconcile_singular_transition<S: ParametricSurface3D>(
    surface: &S,
    previous_uv: Point2,
    previous_point: Point3,
    current_uv: &mut Point2,
    current_point: Point3,
    tolerance: f64,
    output: &mut Vec<SurfacePoint>,
) {
    let represents = |uv: Point2, point: Point3| {
        surface.subs(uv.x, uv.y).distance(point) <= tolerance
    };
    if !previous_uv.x.near(&current_uv.x) && surface.uder(current_uv.x, current_uv.y).so_small() {
        let candidate = Point2::new(previous_uv.x, current_uv.y);
        if represents(candidate, current_point) {
            current_uv.x = previous_uv.x;
        }
    }
    if !previous_uv.y.near(&current_uv.y) && surface.vder(current_uv.x, current_uv.y).so_small() {
        let candidate = Point2::new(current_uv.x, previous_uv.y);
        if represents(candidate, current_point) {
            current_uv.y = previous_uv.y;
        }
    }
    if !previous_uv.x.near(&current_uv.x) && surface.uder(previous_uv.x, previous_uv.y).so_small() {
        let candidate = Point2::new(current_uv.x, previous_uv.y);
        if represents(candidate, previous_point) {
            output.push((candidate, previous_point).into());
        }
    }
    if !previous_uv.y.near(&current_uv.y) && surface.vder(previous_uv.x, previous_uv.y).so_small() {
        let candidate = Point2::new(previous_uv.x, current_uv.y);
        if represents(candidate, previous_point) {
            output.push((candidate, previous_point).into());
        }
    }
}

#[derive(Debug, Default, Clone)]
struct PolyBoundaryPiece(Vec<SurfacePoint>);

impl PolyBoundaryPiece {
    fn try_new<S: PreMeshableSurface>(
        surface: &S,
        wire: impl Iterator<Item = PolylineCurve>,
        sp: impl SP<S>,
        tol: f64,
    ) -> Option<Self> {
        let (up, vp) = (surface.u_period(), surface.v_period());
        let (urange, vrange) = surface.try_range_tuple();
        // How many polylines this bound is assembled from, and how long each
        // is. A bound winding twice is either fed two once-winding pieces --
        // assembly -- or one piece that the lift doubles. This separates them.
        let mut piece_lengths: Vec<usize> = Vec::new();
        let mut bdry3d: Vec<Point3> = wire
            .inspect(|poly_edge| piece_lengths.push(poly_edge.len()))
            .flat_map(|poly_edge| {
                if poly_edge.len() == 2 {
                    let p0 = poly_edge[0];
                    let p1 = poly_edge[1];
                    let mut pts = Vec::new();
                    const N: usize = 8;
                    for i in 0..N {
                        let frac = i as f64 / N as f64;
                        pts.push(p0 + (p1 - p0) * frac);
                    }
                    pts
                } else {
                    let n = poly_edge.len().saturating_sub(1);
                    poly_edge.into_iter().take(n).collect()
                }
            })
            .collect();
        // A wire that contributed no points cannot bound a face. This
        // constructor is already fallible, so say so rather than closing the
        // boundary by indexing a vector that is empty. Real exports do produce
        // such wires, and panicking here aborts the whole model.
        if bdry3d.is_empty() {
            return None;
        }
        bdry3d.push(bdry3d[0]);
        let lift_probe = std::env::var_os("TRUCK_PROBE_LIFT").is_some();
        let mut previous: Option<(f64, f64)> = None;
        let mut previous_pt: Option<Point3> = None;
        let mut vec: Vec<SurfacePoint> = Vec::with_capacity(bdry3d.len());
        // Samples still to lift, most recent last. A step whose periodic
        // representative is ambiguous pushes its own chord midpoint and then
        // revisits itself, so density is spent only where the lift is unsafe
        // rather than across every edge in the model.
        // The flag marks a point this refinement invented rather than one the
        // edge supplied.
        let mut pending: Vec<(Point3, bool)> = Vec::new();
        for point in &bdry3d {
            pending.clear();
            pending.push((*point, false));
            let mut refinements = 0usize;
            while let Some((pt, synthetic)) = pending.pop() {
                let projected = sp(surface, pt, previous);
                // A midpoint is only a device for disambiguating the step, and
                // a chord midpoint of a coarse arc does not lie on the surface,
                // so its projection can legitimately fail. Dropping it costs
                // only the refinement; failing the face over it costs the face,
                // which is how this turned 276 surfaceless faces into 391.
                let (mut u, mut v) = match (projected, synthetic) {
                    (Some(uv), _) => uv,
                    (None, true) => continue,
                    (None, false) => return None,
                };
                // A nearest point is not an incidence.
                //
                // `search_nearest_parameter` answers whether or not the query
                // lies on the surface, so a boundary belonging to a different
                // face still yields a plausible parameter, and the uv path
                // built from it is smooth enough to triangulate into a large
                // wrong region. Every symptom chased downstream of this — a
                // doubled periodic winding, bounds landing in different period
                // copies, a domain spanning the whole chart — was a reading of
                // that path as though it meant something.
                //
                // The contract is that a face's boundary lies on its own
                // surface. Check it where the answer is produced, and refuse
                // the boundary rather than pass a fiction downstream. Measured
                // on shell 160039: 0.027 against a 0.003 tolerance, nine times
                // over, with no nearer solution anywhere in the domain.
                if !synthetic {
                    let residual = surface.subs(u, v).distance(pt);
                    if residual > tol * compatibility_factor() {
                        if std::env::var_os("TRUCK_PROBE_COMPAT").is_some() {
                            // The residual as a multiple of tolerance is the
                            // number the sweep needs: it says where this
                            // rejection would land under any other factor, so
                            // one run yields the whole distribution rather
                            // than one point on it.
                            eprintln!(
                                "COMPAT boundary point off surface: residual={residual:.4e} \
                                 permitted={:.4e} ratio={:.4}",
                                tol * compatibility_factor(),
                                residual / tol,
                            );
                        }
                        return None;
                    }
                }
                let raw = (u, v);
                if let (Some(up), Some((u0, _))) = (up, previous) {
                    u = get_mindiff(u, u0, up);
                }
                if let (Some(vp), Some((_, v0))) = (vp, previous) {
                    v = get_mindiff(v, v0, vp);
                }
                if lift_probe {
                    // Each sample's raw projection, the periodic representative
                    // chosen for it, and the step that choice implies. Aliasing
                    // shows up as a step near or beyond half a period, or as a
                    // step that closes a loop which should have stayed open.
                    let (du, dv) = match previous {
                        Some((u0, v0)) => (u - u0, v - v0),
                        None => (0.0, 0.0),
                    };
                    let frac = |d: f64, p: Option<f64>| p.map_or(0.0, |p| d / p);
                    eprintln!(
                        "LIFT raw=({:.6},{:.6}) chosen=({u:.6},{v:.6}) \
                         step=({du:+.6},{dv:+.6}) step/period=({:+.4},{:+.4})",
                        raw.0,
                        raw.1,
                        frac(du, up),
                        frac(dv, vp),
                    );
                }
                // Halve the step rather than guess which copy was meant. The
                // projection of the chord midpoint recovers a point the curve
                // actually passes through, so each half advances by less and
                // the nearest copy becomes unambiguous.
                if let (Some((u0, v0)), Some(previous_point)) = (previous, previous_pt) {
                    let ambiguous = |now: f64, before: f64, period: Option<f64>| {
                        period.is_some_and(|period| {
                            f64::abs(now - before) >= AMBIGUOUS_STEP_FRACTION * period
                        })
                    };
                    if refinements < MAX_LIFT_REFINEMENTS
                        && (ambiguous(u, u0, up) || ambiguous(v, v0, vp))
                    {
                        refinements += 1;
                        pending.push((pt, synthetic));
                        pending.push((previous_point.midpoint(pt), true));
                        continue;
                    }
                }
                vec.push((Point2::new(u, v), pt).into());
                previous = Some((u, v));
                previous_pt = Some(pt);
            }
        }
        if (bdry3d.len() <= 2 || bdry3d[0].distance(bdry3d[bdry3d.len() - 1]) < 1e-4)
            && piece_lengths.iter().all(|&l| l <= 2)
        {
            if let Some(up) = surface.u_period() {
                let p0 = bdry3d[0];
                if let Some((u0, v0)) = sp(surface, p0, None) {
                    let mut dense = Vec::new();
                    const STEPS: usize = 16;
                    for i in 0..=STEPS {
                        let frac = i as f64 / STEPS as f64;
                        let u = u0 + frac * up;
                        let pt = surface.subs(u, v0);
                        dense.push((Point2::new(u, v0), pt).into());
                    }
                    vec = dense;
                }
            } else if let Some(vp) = surface.v_period() {
                let p0 = bdry3d[0];
                if let Some((u0, v0)) = sp(surface, p0, None) {
                    let mut dense = Vec::new();
                    const STEPS: usize = 16;
                    for i in 0..=STEPS {
                        let frac = i as f64 / STEPS as f64;
                        let v = v0 + frac * vp;
                        let pt = surface.subs(u0, v);
                        dense.push((Point2::new(u0, v), pt).into());
                    }
                    vec = dense;
                }
            }
        }
        let grav = vec.iter().fold(Point2::origin(), |g, p| g + p.uv.to_vec()) / vec.len() as f64;
        let mut quot_u = 0.0;
        let mut quot_v = 0.0;
        if let (Some(up), Some((u0, _))) = (up, urange) {
            quot_u = f64::floor((grav.x - u0) / up);
            vec.iter_mut().for_each(|p| p.x -= quot_u * up);
        }
        if let (Some(vp), Some((v0, _))) = (vp, vrange) {
            quot_v = f64::floor((grav.y - v0) / vp);
            vec.iter_mut().for_each(|p| p.y -= quot_v * vp);
        }
        if lift_probe {
            // Which period copy this bound was placed in, and where it ended
            // up. The shift is chosen from this bound's own centroid alone, and
            // `try_new` runs once per wire, so two bounds of the same face are
            // normalized independently and can be placed in different copies.
            // Comparing these lines across the bounds of one face is the test.
            let (mut u_lo, mut u_hi) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut v_lo, mut v_hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for p in &vec {
                u_lo = u_lo.min(p.uv.x);
                u_hi = u_hi.max(p.uv.x);
                v_lo = v_lo.min(p.uv.y);
                v_hi = v_hi.max(p.uv.y);
            }
            let winding = |lo: f64, hi: f64, period: Option<f64>| {
                period.map_or(0.0, |period| (hi - lo) / period)
            };
            // Span conflates two different defects, so report the pair that
            // separates them. `k` is the net winding — how many periods the
            // boundary ends away from where it started — and `V` the total
            // variation, how far it travelled altogether. Circling once gives
            // |k| = 1 with V ~ 1. |k| = 1 with V ~ 2 means it went out and came
            // back, a branch chosen wrongly part way. |k| = 2 with V ~ 2 means
            // it genuinely went round twice, which is a duplicated wire or a
            // seam concatenated in both orientations.
            let (mut travel_u, mut travel_v) = (0.0, 0.0);
            for pair in vec.windows(2) {
                travel_u += f64::abs(pair[1].uv.x - pair[0].uv.x);
                travel_v += f64::abs(pair[1].uv.y - pair[0].uv.y);
            }
            let net = |period: Option<f64>, first: f64, last: f64| {
                period.map_or(0.0, |period| f64::round((last - first) / period))
            };
            let (first, last) = (vec[0].uv, vec[vec.len() - 1].uv);
            // Is the reported period a real period, and is the lift a valid
            // inverse at all? `e_p` and `e_2p` say whether `S` actually repeats
            // after one or two periods; `e_hp` catches a period that is not
            // fundamental. `e_inv` is the reconstruction residual, the distance
            // from each lifted parameter back to the 3D point it came from --
            // small residual with a doubled winding means the chart genuinely
            // takes two parameter circuits per geometric circuit.
            let anchor = vec[0].uv;
            let base = surface.subs(anchor.x, anchor.y);
            let shifted = |dv: f64| surface.subs(anchor.x, anchor.y + dv).distance(base);
            let (e_p, e_2p, e_hp) = match vp {
                Some(period) => (
                    shifted(period),
                    shifted(2.0 * period),
                    shifted(0.5 * period),
                ),
                None => (f64::NAN, f64::NAN, f64::NAN),
            };
            let e_inv = vec.iter().fold(0.0_f64, |worst, s| {
                worst.max(surface.subs(s.uv.x, s.uv.y).distance(s.point))
            });
            // Independent of `sp` entirely: brute-force the true distance from
            // the first boundary point to the surface. This separates the two
            // remaining explanations for a large residual. If the minimum is
            // also far, the point genuinely does not lie on this surface and
            // the edge has been paired with the wrong face. If the minimum is
            // near zero, a valid inverse existed and the projection search
            // failed to find it.
            let target = vec[0].point;
            let anchor_uv = vec[0].uv;
            let axis =
                |range: Option<(f64, f64)>, period: Option<f64>, centre: f64| match (range, period)
                {
                    (Some(r), _) => r,
                    (None, Some(p)) => (centre - p, centre + p),
                    (None, None) => (centre - 1.0, centre + 1.0),
                };
            let (ulo, uhi) = axis(urange, up, anchor_uv.x);
            let (vlo, vhi) = axis(vrange, vp, anchor_uv.y);
            const GRID: usize = 400;
            let mut d_min = f64::INFINITY;
            for i in 0..=GRID {
                let u = ulo + (uhi - ulo) * i as f64 / GRID as f64;
                for j in 0..=GRID {
                    let v = vlo + (vhi - vlo) * j as f64 / GRID as f64;
                    d_min = d_min.min(surface.subs(u, v).distance(target));
                }
            }
            // The structure of the residual, not just its size, says which
            // upstream error produced it. A constant world-space vector is a
            // missing translation. A constant magnitude aligned with the
            // surface normal is a radius or offset error. Neither pattern
            // holding means the entities are unrelated rather than
            // misplaced.
            let residuals: Vec<_> = vec
                .iter()
                .map(|s| s.point - surface.subs(s.uv.x, s.uv.y))
                .collect();
            let mean =
                residuals.iter().fold(Vector3::zero(), |acc, r| acc + r) / residuals.len() as f64;
            let spread = residuals
                .iter()
                .fold(0.0_f64, |worst, r| worst.max((r - mean).magnitude()));
            let (mut mag_lo, mut mag_hi) = (f64::INFINITY, 0.0_f64);
            let mut normal_alignment = 0.0;
            for (r, s) in residuals.iter().zip(&vec) {
                let magnitude = r.magnitude();
                mag_lo = mag_lo.min(magnitude);
                mag_hi = mag_hi.max(magnitude);
                let normal = surface
                    .uder(s.uv.x, s.uv.y)
                    .cross(surface.vder(s.uv.x, s.uv.y));
                if magnitude > 0.0 && normal.magnitude() > 0.0 {
                    normal_alignment += f64::abs(r.dot(normal.normalize()) / magnitude);
                }
            }
            normal_alignment /= residuals.len() as f64;
            eprintln!(
                "PERIOD e_p={e_p:.3e} e_2p={e_2p:.3e} e_hp={e_hp:.3e} e_inv={e_inv:.3e} \
                 d_min={d_min:.3e}"
            );
            eprintln!(
                "RESID |mean|={:.4e} spread={spread:.4e} |r|=[{mag_lo:.4e},{mag_hi:.4e}] \
                 normal_align={normal_alignment:.3}",
                mean.magnitude(),
            );
            // The same incidence certificate the source file satisfies exactly,
            // recomputed on the *converted* geometry. Both frames are recovered
            // from three sampled points, so this needs no knowledge of how the
            // conversion stores them and introduces no grid error. The source
            // gives e_angle = 0, e_axis <= 1e-14, e_radius = 0; whichever of
            // the three is non-zero here names the field the conversion
            // damaged.
            let circle_through = |a: Point3, b: Point3, c: Point3| {
                let (ab, ac) = (b - a, c - a);
                let normal = ab.cross(ac);
                let n2 = normal.magnitude2();
                if n2 < f64::EPSILON {
                    return None;
                }
                let centre = a
                    + (ac.magnitude2() * ab.cross(normal) - ab.magnitude2() * ac.cross(normal))
                        / (2.0 * n2);
                Some((centre, normal.normalize(), centre.distance(a)))
            };
            // Boundary curve: three points spread along it.
            let n = vec.len();
            let boundary_fit = if n >= 3 {
                circle_through(vec[0].point, vec[n / 3].point, vec[2 * n / 3].point)
            } else {
                None
            };
            // Surface cross-section: three points around one periodic axis.
            let anchor = vec[0].uv;
            let cross_fit = match (up, vp) {
                (Some(p), _) => circle_through(
                    surface.subs(anchor.x, anchor.y),
                    surface.subs(anchor.x + p / 3.0, anchor.y),
                    surface.subs(anchor.x + 2.0 * p / 3.0, anchor.y),
                ),
                (_, Some(p)) => circle_through(
                    surface.subs(anchor.x, anchor.y),
                    surface.subs(anchor.x, anchor.y + p / 3.0),
                    surface.subs(anchor.x, anchor.y + 2.0 * p / 3.0),
                ),
                _ => None,
            };
            if let (Some((cc, cn, cr)), Some((sc, sn, sr))) = (boundary_fit, cross_fit) {
                let e_angle = f64::acos(f64::min(1.0, f64::abs(cn.dot(sn)))).to_degrees();
                let offset = cc - sc;
                let e_axis = (offset - sn * offset.dot(sn)).magnitude();
                eprintln!(
                    "INCID e_angle={e_angle:.4}deg e_axis={e_axis:.4e} \
                     e_radius={:.4e} r_curve={cr:.5} r_surf={sr:.5}",
                    f64::abs(cr - sr),
                );
            }
            eprintln!(
                "BOUND pieces={piece_lengths:?} pts={} k=({:+.0},{:+.0}) V=({:.2},{:.2}) \
                 quot=({quot_u:+.0},{quot_v:+.0}) \
                 u=[{u_lo:.4},{u_hi:.4}] v=[{v_lo:.4},{v_hi:.4}] \
                 span/period=({:.3},{:.3})",
                vec.len(),
                net(up, first.x, last.x),
                net(vp, first.y, last.y),
                up.map_or(0.0, |p| travel_u / p),
                vp.map_or(0.0, |p| travel_v / p),
                winding(u_lo, u_hi, up),
                winding(v_lo, v_hi, vp),
            );
        }
        let last = *vec.last().unwrap();
        if !vec[0].near(&last) {
            let Point2 { x: u0, y: v0 } = last.uv;
            if surface.uder(u0, v0).so_small() || surface.vder(u0, v0).so_small() {
                vec.push(vec[0]);
            }
        }
        Some(Self(vec))
    }
}

fn get_mindiff(u: f64, u0: f64, up: f64) -> f64 {
    // The nearest periodic copy outright, rather than the nearest among five.
    // The old search covered only two periods either side, so a boundary that
    // wrapped further was silently pulled back; rounding has no such bound and
    // is cheaper.
    u + f64::round((u0 - u) / up) * up
}

/// How far a boundary point may sit from its own surface, as a multiple of the
/// chord tolerance, before the pairing is refused.
///
/// A face's boundary is required to lie on that face's surface; this is the
/// slack allowed for the chord approximation and for imperfect exports, not a
/// licence to trim a surface with a curve belonging to something else.
///
/// **Off by default, deliberately.** The violation this detects is real —
/// swept on `00009190`, the rejected points sit at a median of 191x tolerance
/// and a maximum of 617x, and loosening the factor twentyfold removes only 62
/// of 315 rejections, so this is a population and not a threshold. But
/// rejecting them repairs nothing visible: the blob shells are byte-identical
/// with the gate on and off, while the gate costs 292 faces and 21,131
/// triangles, a tenth of the model. Deleting real geometry to fix nothing is
/// the wrong default. Set `TRUCK_COMPAT_FACTOR=5` to measure the population;
/// turn it on for real only once something downstream can use the refusal.
const COMPATIBILITY_FACTOR: f64 = f64::INFINITY;

/// The factor in force, overridable by `TRUCK_COMPAT_FACTOR`.
///
/// Sweeping the factor is what established that the gate names a real
/// population rather than a threshold, and a rebuild per sample would have made
/// that a five-build afternoon. Read once — this sits in the per-boundary-point
/// loop, and an env lookup there would be a measurable cost charged to every
/// model.
fn compatibility_factor() -> f64 {
    static FACTOR: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *FACTOR.get_or_init(|| {
        std::env::var("TRUCK_COMPAT_FACTOR")
            .ok()
            .and_then(|raw| raw.trim().parse::<f64>().ok())
            .filter(|factor| *factor > 0.0 && !factor.is_nan())
            .unwrap_or(COMPATIBILITY_FACTOR)
    })
}

/// How far a step may advance, as a fraction of the period, before the periodic
/// representative it implies is treated as ambiguous.
///
/// [`get_mindiff`] takes the copy nearest the previous parameter, which is the
/// right answer only while the true step is under half a period. At exactly
/// half, the two candidates are equidistant and the tie is broken arbitrarily —
/// measured advancing `-0.5` of a period where the curve went `+0.5`, which
/// folds a full turn onto itself and makes a period-wrapping boundary look like
/// a closed loop. The margin below `0.5` keeps numerical noise clear of the tie.
const AMBIGUOUS_STEP_FRACTION: f64 = 0.45;

/// How many times a single step may be halved before refinement gives up.
const MAX_LIFT_REFINEMENTS: usize = 8;

/// Result of boundary loop closure evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryClosure {
    /// Endpoints are within Euclidean UV tolerance.
    EuclideanClosed,
    /// Endpoints close modulo parameter period.
    PeriodicClosed {
        /// Integer winding displacement along [u, v].
        displacement: [i64; 2],
    },
    /// Boundary loop is un-closed (open).
    Open,
}

/// Evaluates periodic displacement winding modulo period if residual <= tolerance.
fn periodic_displacement(start: f64, end: f64, period: f64, tolerance: f64) -> Option<i64> {
    if period <= 1e-6 {
        return None;
    }
    let winding = ((end - start) / period).round() as i64;
    let residual = (end - start) - winding as f64 * period;
    if residual.abs() <= tolerance {
        Some(winding)
    } else {
        None
    }
}

#[derive(Debug, Default, Clone)]
struct PolyBoundary(Vec<Vec<SurfacePoint>>);

fn normalize_range(curve: &mut Vec<SurfacePoint>, compidx: usize, (u0, u1): (f64, f64)) {
    let p = curve[0];
    let q = curve[curve.len() - 1];
    let tmp = f64::min(p[compidx], q[compidx]) + TOLERANCE;
    let del = f64::floor((tmp - u0) / (u1 - u0)) * (u1 - u0);
    curve.iter_mut().for_each(|p| p[compidx] -= del);
    let Some(i) = curve
        .iter()
        .position(|p| (curve[0][compidx] - u1) * (p[compidx] - u1) < 0.0)
    else {
        return;
    };
    let mut curve1 = curve.split_off(i + 1);
    curve1.pop();
    curve1.insert(0, curve[i]);
    match curve[0][compidx] < curve[curve.len() - 1][compidx] {
        true => curve1.iter_mut(),
        false => curve.iter_mut(),
    }
    .for_each(|p| p[compidx] -= u1 - u0);
    curve1.append(curve);
    *curve = curve1;
}

/// Twice the signed area of a closed `uv` loop, by the shoelace formula.
///
/// Diagnostic only. Its *sign* must not be used to decide what a face
/// contains: it negates under an orientation-reversing reparameterization,
/// which no observer of the solid can detect, so any predicate built on it
/// classifies the same face differently depending on how its surface happens
/// to be parameterized. Relative sign between loops is invariant; absolute
/// sign is not.
#[allow(dead_code)]
fn signed_area(curve: &[SurfacePoint]) -> f64 {
    curve
        .iter()
        .circular_tuple_windows()
        .fold(0.0, |sum, (p, q)| sum + (q.x + p.x) * (q.y - p.y))
}

/// The working parameter rectangle of a trimmed face, derived from the face's
/// own bounds.
///
/// **`PAR-RANGE-INHERITANCE-001`.** A surface's declared `parameter_range` is a
/// property of the primitive it was constructed from, not of any face that
/// references it. `Line::parameter_range` is `[0, 1]` unconditionally and
/// `RevolutedCurve` inherits it, so a cone built as a revolved line declares
/// `[0, 1] x [0, 2pi)` — one unit of generatrix starting at the STEP reference
/// radius, chosen by the primitive and unrelated to the face. Stitching an open
/// boundary piece against the edge of that rectangle fabricates trim geometry
/// no source entity describes (`DOM-ARTIFICIAL-CLOSURE-001`), and when the
/// piece already lies on the edge the enclosed area is zero
/// (`DOM-ZERO-AREA-001`).
///
/// Measured: extending that range by a constant instead recovers 348 NIST faces
/// and destroys 268 others, in a disjoint set of models — one part in two
/// encodings loses 148 cone faces under whichever window excludes it. Any
/// fixed-size window trades one population for another, because whether a
/// face's material interval falls inside is decided by where its exporter put
/// the reference circle. The extent has to come from the face.
///
/// **A periodic axis is different in kind**: its extent *is* determined, by the
/// period, and the seam handling below relies on the wrap interval being one
/// full period. Only non-periodic axes are re-derived.
///
/// Returns `None` for an axis the bounds do not determine. A degenerate extent
/// means the material region is not recoverable from the boundary alone, and
/// the caller must refuse rather than invent one — a collapsed single-vertex
/// bound is exactly that case, marking a point the domain must reach while
/// contributing no trim segment (`QUO-005`, `SNG-COLLAPSED-DIRECTION-001`).
fn working_range(
    pieces: &[PolyBoundaryPiece],
    surface: &impl PreMeshableSurface,
) -> (Option<(f64, f64)>, Option<(f64, f64)>) {
    let (udeclared, vdeclared) = surface.try_range_tuple();
    let axis = |idx: usize, period: Option<f64>, declared: Option<(f64, f64)>| {
        // A period determines the extent; the bounds do not get a say.
        if period.is_some() {
            return declared;
        }
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        pieces.iter().for_each(|PolyBoundaryPiece(vec)| {
            vec.iter().for_each(|p| {
                lo = f64::min(lo, p[idx]);
                hi = f64::max(hi, p[idx]);
            })
        });
        (hi - lo > TOLERANCE).then_some((lo, hi))
    };
    (
        axis(0, surface.u_period(), udeclared),
        axis(1, surface.v_period(), vdeclared),
    )
}

impl PolyBoundary {
    fn new(pieces: Vec<PolyBoundaryPiece>, surface: &impl PreMeshableSurface, tol: f64) -> Self {
        let probe = std::env::var_os("TRUCK_PROBE_BOUNDARY").is_some();
        let had_source_pieces = !pieces.is_empty();
        // EXPERIMENT (TRUCK_FACE_DOMAIN): take the working rectangle from the
        // face's own bounds rather than from the supporting primitive's
        // declared range. Default off until swept.
        let range = match std::env::var_os("TRUCK_FACE_DOMAIN").is_some() {
            true => working_range(&pieces, surface),
            false => surface.try_range_tuple(),
        };
        let (mut closed, mut open) = (Vec::new(), Vec::new());
        let u_period = surface.u_period();
        let v_period = surface.v_period();
        pieces.into_iter().for_each(|PolyBoundaryPiece(mut vec)| {
            let p0 = vec[0].uv;
            let p1 = vec[vec.len() - 1].uv;

            let closure = if p0.distance(p1) < 1.0e-3 {
                BoundaryClosure::EuclideanClosed
            } else {
                let ku = u_period
                    .and_then(|up| periodic_displacement(p0.x, p1.x, up, 1e-3))
                    .unwrap_or(0);
                let kv = v_period
                    .and_then(|vp| periodic_displacement(p0.y, p1.y, vp, 1e-3))
                    .unwrap_or(0);
                if (ku != 0 || kv != 0) && vec[0].point.distance(vec[vec.len() - 1].point) < 1e-3 {
                    BoundaryClosure::PeriodicClosed {
                        displacement: [ku, kv],
                    }
                } else {
                    BoundaryClosure::Open
                }
            };

            if probe {
                let perimeter: f64 = vec
                    .windows(2)
                    .map(|w| w[0].uv.distance(w[1].uv))
                    .sum::<f64>();
                eprintln!(
                    "PROBE piece pts={} gap={:.6e} perimeter={perimeter:.6e} \
                     closure={:?}",
                    vec.len(),
                    p0.distance(p1),
                    closure,
                );
            }

            match closure {
                BoundaryClosure::EuclideanClosed => {
                    vec.pop();
                    closed.push(vec);
                }
                BoundaryClosure::PeriodicClosed {
                    displacement: [ku, kv],
                } => {
                    if let Some(up) = u_period {
                        if ku != 0 {
                            for p in &mut vec {
                                p.uv.x -= (ku as f64) * up;
                            }
                            vec.last_mut().unwrap().uv.x = vec[0].uv.x + (ku as f64) * up;
                        }
                    }
                    if let Some(vp) = v_period {
                        if kv != 0 {
                            for p in &mut vec {
                                p.uv.y -= (kv as f64) * vp;
                            }
                            vec.last_mut().unwrap().uv.y = vec[0].uv.y + (kv as f64) * vp;
                        }
                    }
                    closed.push(vec);
                }
                BoundaryClosure::Open => open.push(vec),
            }
        });
        if closed.len() == 2 && (surface.u_period().is_some() || surface.v_period().is_some()) {
            let area0 = signed_area(&closed[0]);
            let area1 = signed_area(&closed[1]);
            if area0.abs() < 1e-4 && area1.abs() < 1e-4 {
                let loop0 = closed.remove(0);
                let mut loop1 = closed.remove(0);
                let u0_mean: f64 = loop0.iter().map(|p| p.uv.x).sum::<f64>() / loop0.len() as f64;
                let u1_mean: f64 = loop1.iter().map(|p| p.uv.x).sum::<f64>() / loop1.len() as f64;
                if let Some(up) = surface.u_period() {
                    let ku = ((u0_mean - u1_mean) / up).round();
                    if ku != 0.0 {
                        for p in &mut loop1 {
                            p.uv.x += ku * up;
                        }
                    }
                }
                let v0_mean: f64 = loop0.iter().map(|p| p.uv.y).sum::<f64>() / loop0.len() as f64;
                let v1_mean: f64 = loop1.iter().map(|p| p.uv.y).sum::<f64>() / loop1.len() as f64;
                if let Some(vp) = surface.v_period() {
                    let kv = ((v0_mean - v1_mean) / vp).round();
                    if kv != 0.0 {
                        for p in &mut loop1 {
                            p.uv.y += kv * vp;
                        }
                    }
                }
                let mut merged = Vec::new();
                merged.extend(loop0.into_iter());
                merged.extend(loop1.into_iter().rev());
                closed.push(merged);
            }
        } else if let Some(pair) =
            CollapsedPeriodicBoundaryPair::try_classify(surface, &closed, &open, range)
        {
            let mut loop0 = closed.remove(0);
            if loop0.len() > 1 && loop0[0].uv.distance(loop0.last().unwrap().uv) < 1e-3 {
                loop0.pop();
            }
            let is_v = surface.v_period().is_some_and(|p| p > 1e-6);
            let period = if is_v {
                surface.v_period().unwrap()
            } else {
                surface.u_period().unwrap()
            };

            let mut loop0_full = loop0.clone();
            let mut last_p = *loop0.last().unwrap();
            if is_v {
                last_p.uv.y = loop0[0].uv.y + period;
            } else {
                last_p.uv.x = loop0[0].uv.x + period;
            }
            loop0_full.push(last_p);

            let loop1_full: Vec<SurfacePoint> = loop0_full
                .iter()
                .map(|p| {
                    let uv = if is_v {
                        Point2::new(pair.apex_u, p.uv.y)
                    } else {
                        Point2::new(p.uv.x, pair.apex_u)
                    };
                    (uv, surface.subs(uv.x, uv.y)).into()
                })
                .collect();

            let end_loop0 = *loop0_full.last().unwrap();
            let loop1_rev: Vec<SurfacePoint> = loop1_full.into_iter().rev().collect();
            let start_loop1 = loop1_rev[0];
            let end_loop1 = *loop1_rev.last().unwrap();
            let start_loop0 = loop0_full[0];

            let seam_down = polyline_on_surface(surface, end_loop0, start_loop1, tol);
            let seam_up = polyline_on_surface(surface, end_loop1, start_loop0, tol);

            let merged = connect_edges([loop0_full, seam_down, loop1_rev, seam_up]);
            closed.push(merged);
        }
        let (n_closed_in, n_open_in) = (closed.len(), open.len());
        fn connect_edges<P>(vecs: impl IntoIterator<Item = Vec<P>>) -> Vec<P> {
            let closure = |vec: Vec<P>| {
                let len = vec.len();
                vec.into_iter().take(len - 1)
            };
            vecs.into_iter().flat_map(closure).collect()
        }
        match open.len() {
            1 => {
                let mut curve = open.pop().unwrap();
                let p = curve[0];
                let q = curve[curve.len() - 1];
                if let (Some((u0, u1)), Some((v0, v1))) = range {
                    if p.x < q.x - TOLERANCE {
                        normalize_range(&mut curve, 0, (u0, u1));
                        let p = curve[0];
                        let q = curve[curve.len() - 1];
                        let x = (Point2::new(u0, v1), surface.subs(u0, v1)).into();
                        let y = (Point2::new(u1, v1), surface.subs(u1, v1)).into();
                        let vec0 = polyline_on_surface(surface, q, y, tol);
                        let vec1 = polyline_on_surface(surface, y, x, tol);
                        let vec2 = polyline_on_surface(surface, x, p, tol);
                        closed.push(connect_edges([vec0, vec1, vec2, curve]));
                    } else if q.x < p.x - TOLERANCE {
                        normalize_range(&mut curve, 0, (u0, u1));
                        let p = curve[0];
                        let q = curve[curve.len() - 1];
                        let x = (Point2::new(u1, v0), surface.subs(u1, v0)).into();
                        let y = (Point2::new(u0, v0), surface.subs(u0, v0)).into();
                        let vec0 = polyline_on_surface(surface, q, y, tol);
                        let vec1 = polyline_on_surface(surface, y, x, tol);
                        let vec2 = polyline_on_surface(surface, x, p, tol);
                        closed.push(connect_edges([vec0, vec1, vec2, curve]));
                    } else if p.y < q.y - TOLERANCE {
                        normalize_range(&mut curve, 1, (v0, v1));
                        let p = curve[0];
                        let q = curve[curve.len() - 1];
                        let x = (Point2::new(u0, v0), surface.subs(u0, v0)).into();
                        let y = (Point2::new(u0, v1), surface.subs(u0, v1)).into();
                        let vec0 = polyline_on_surface(surface, q, y, tol);
                        let vec1 = polyline_on_surface(surface, y, x, tol);
                        let vec2 = polyline_on_surface(surface, x, p, tol);
                        closed.push(connect_edges([vec0, vec1, vec2, curve]));
                    } else if q.y < p.y - TOLERANCE {
                        normalize_range(&mut curve, 1, (v0, v1));
                        let p = curve[0];
                        let q = curve[curve.len() - 1];
                        let x = (Point2::new(u1, v1), surface.subs(u1, v1)).into();
                        let y = (Point2::new(u1, v0), surface.subs(u1, v0)).into();
                        let vec0 = polyline_on_surface(surface, q, y, tol);
                        let vec1 = polyline_on_surface(surface, y, x, tol);
                        let vec2 = polyline_on_surface(surface, x, p, tol);
                        closed.push(connect_edges([vec0, vec1, vec2, curve]));
                    }
                }
            }
            2 => {
                let mut curve1 = open.pop().unwrap();
                let mut curve0 = open.pop().unwrap();
                fn end_pts<T: Copy>(vec: &[T]) -> (T, T) {
                    (vec[0], vec[vec.len() - 1])
                }
                let ((p0, p1), (q0, q1)) = (end_pts(&curve0), end_pts(&curve1));
                if !p0.x.near(&p1.x) && !q0.x.near(&q1.x) {
                    if let (Some(urange), _) = range {
                        normalize_range(&mut curve0, 0, urange);
                        normalize_range(&mut curve1, 0, urange);
                    }
                } else if !p0.y.near(&p1.y) && !q0.y.near(&q1.y) {
                    if let (_, Some(vrange)) = range {
                        normalize_range(&mut curve0, 1, vrange);
                        normalize_range(&mut curve1, 1, vrange);
                    }
                }
                let ((p0, p1), (q0, q1)) = (end_pts(&curve0), end_pts(&curve1));
                let vec0 = polyline_on_surface(surface, p1, q0, tol);
                let vec1 = polyline_on_surface(surface, q1, p0, tol);
                closed.push(connect_edges([curve0, vec0, curve1, vec1]));
            }
            _ => {}
        }
        if probe {
            let areas: Vec<String> = closed
                .iter()
                .map(|c| format!("{:+.4e}", signed_area(c)))
                .collect();
            let range = surface.try_range_tuple();
            let has_rect = matches!(range, (Some(_), Some(_)));
            eprintln!(
                "PROBE in_closed={n_closed_in} in_open={n_open_in} loops={} \
                 areas=[{}] uperiod={:?} vperiod={:?} range={} rect={}",
                closed.len(),
                areas.join(","),
                surface.u_period(),
                surface.v_period(),
                has_rect,
                closed.is_empty() && has_rect,
            );
        }
        // Only a face with no enclosing loop takes its domain from the surface.
        if closed.is_empty() && !had_source_pieces {
            if let (Some((u0, u1)), Some((v0, v1))) = range {
                let p = [
                    (Point2::new(u0, v0), surface.subs(u0, v0)).into(),
                    (Point2::new(u1, v0), surface.subs(u1, v0)).into(),
                    (Point2::new(u1, v1), surface.subs(u1, v1)).into(),
                    (Point2::new(u0, v1), surface.subs(u0, v1)).into(),
                ];
                let vec0 = polyline_on_surface(surface, p[0], p[1], tol);
                let vec1 = polyline_on_surface(surface, p[1], p[2], tol);
                let vec2 = polyline_on_surface(surface, p[2], p[3], tol);
                let vec3 = polyline_on_surface(surface, p[3], p[0], tol);
                closed.push(connect_edges([vec0, vec1, vec2, vec3]));
            }
        }
        Self(closed)
    }

    /// whether `c` is included in the domain with boundary = `self`.
    fn include(&self, c: Point2) -> bool {
        let t = 2.0 * std::f64::consts::PI * HashGen::hash1(c);
        let r = Vector2::new(f64::cos(t), f64::sin(t));
        self.0
            .iter()
            .flat_map(|vec| vec.iter().circular_tuple_windows())
            .try_fold(0_i32, move |crossings, (p0, p1)| {
                let a = **p0 - c;
                let b = **p1 - c;
                let s0 = r.x * a.y - r.y * a.x; // v times a
                let s1 = r.x * b.y - r.y * b.x; // v times b
                let s2 = a.x * b.y - a.y * b.x; // a times b
                let x = s2 / (s1 - s0);
                if x.so_small() && s0 * s1 < 0.0 {
                    None
                } else if x > 0.0 && ((s0 <= 0.0 && s1 > 0.0) || (s0 >= 0.0 && s1 < 0.0)) {
                    Some(crossings + 1)
                } else {
                    Some(crossings)
                }
            })
            .map(|crossings| crossings % 2 == 1)
            .unwrap_or(false)
    }

    /// Inserts points and adds constraint
    fn insert_to(
        &self,
        triangulation: &mut Cdt,
        boundary_map: &mut HashMap<FixedVertexHandle, Point3>,
        roles: &mut ConstraintRoles,
    ) -> bool {
        let mut success = true;
        let probe = std::env::var_os("TRUCK_PROBE_FAIL").is_some();
        let probe_face_id = if probe {
            std::env::var("TRUCK_PROBE_FACE_ID")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
        } else {
            None
        };
        if let Some(target) = probe_face_id {
            let (source_face_id, declared_face_index, periodic_rank) =
                PROBE_FACE_CONTEXT.with(std::cell::Cell::get);
            if source_face_id == Some(target) {
                for (piece_index, piece) in self.0.iter().enumerate() {
                    for (point_index, point) in piece.iter().enumerate() {
                        eprintln!(
                            "WL\tsource_face_id={target}\tdeclared_face_index={declared_face_index}\t\
                             periodic_rank={periodic_rank}\tpost_stitch_piece={piece_index}\t\
                             point_index={point_index}\tuv={:.17e},{:.17e}",
                            point.uv.x, point.uv.y,
                        );
                    }
                }
            }
        }
        let mut probe_point_fail = 0usize;
        let mut probe_degenerate = 0usize;
        let mut probe_already_direct = 0usize;
        let mut probe_refused_with_conflicts = 0usize;
        let mut probe_refused_without_conflicts = 0usize;
        let mut probe_conflicting_edges = 0usize;
        let mut probe_add_returned_false = 0usize;
        // These are post-stitch loop/segment proxies, not source-edge
        // provenance. Keep all direct contributors because duplicates exist.
        let mut installed_origins = probe.then(
            HashMap::<FixedUndirectedEdgeHandle, Vec<(usize, usize)>>::default,
        );
        let mut first_conflict = None;
        for (piece_index, piece) in self.0.iter().enumerate() {
            let poly2tri: Vec<Option<FixedVertexHandle>> = piece
                .iter()
                .map(|pt| {
                    let sp = SPoint2::new(spade_round(pt.uv.x), spade_round(pt.uv.y));
                    if let Some(idx) = triangulation
                        .vertices()
                        .find(|v| sp.distance_2(*v.as_ref()) < 1e-12)
                        .map(|v| v.fix())
                    {
                        boundary_map.insert(idx, pt.point);
                        Some(idx)
                    } else {
                        match triangulation.insert(sp) {
                            Ok(idx) => {
                                boundary_map.insert(idx, pt.point);
                                Some(idx)
                            }
                            Err(_) => None,
                        }
                    }
                })
                .collect();

            if poly2tri.iter().any(|v| v.is_none()) {
                probe_point_fail += 1;
                success = false;
                continue;
            }
            let len = poly2tri.len();
            if len < 3 {
                continue;
            }
            for k in 0..len {
                let i = k;
                let j = (k + 1) % len;
                let vi = poly2tri[i].unwrap();
                let vj = poly2tri[j].unwrap();
                if vi == vj {
                    probe_degenerate += 1;
                    continue;
                }
                let is_already_constraint = triangulation
                    .get_edge_from_neighbors(vi, vj)
                    .map(|e| e.is_constraint_edge())
                    .unwrap_or(false);
                let can_add = !is_already_constraint && triangulation.can_add_constraint(vi, vj);
                if can_add {
                    if triangulation.add_constraint(vi, vj) {
                        if let Some(edge) = triangulation.get_edge_from_neighbors(vi, vj) {
                            let handle = edge.as_undirected().fix();
                            // Audit A1: every segment reaching here comes from a
                            // `PolyBoundary` piece, so it is treated as physical
                            // boundary. That is deliberately *not* the whole
                            // truth — `PolyBoundary::new` also stitches synthetic
                            // closure segments into these same pieces, and those
                            // are `UnresolvedSyntheticClosure` in reality (audit
                            // A6). Distinguishing them needs per-piece
                            // provenance, which would be a second semantic change
                            // in the same experiment. Tagged as it behaves today;
                            // A6 splits the population.
                            roles.record(handle, ConstraintRole::PhysicalBoundary);
                            if let Some(installed_origins) = installed_origins.as_mut() {
                                installed_origins
                                    .entry(handle)
                                    .or_default()
                                    .push((piece_index, k));
                            }
                        }
                    } else {
                        probe_add_returned_false += 1;
                    }
                } else {
                    if is_already_constraint {
                        probe_already_direct += 1;
                    } else if probe {
                        let conflicts: Vec<_> = triangulation
                            .get_conflicting_edges_between_vertices(vi, vj)
                            .map(|edge| {
                                (
                                    edge.as_undirected().fix(),
                                    edge.from().position(),
                                    edge.to().position(),
                                )
                            })
                            .collect();
                        probe_conflicting_edges += conflicts.len();
                        if conflicts.is_empty() {
                            probe_refused_without_conflicts += 1;
                        } else {
                            probe_refused_with_conflicts += 1;
                            if first_conflict.is_none() {
                                let mapped_conflicts = conflicts
                                    .iter()
                                    .filter(|(handle, _, _)| {
                                        installed_origins
                                            .as_ref()
                                            .is_some_and(|origins| origins.contains_key(handle))
                                    })
                                    .count();
                                // Prefer a resolvable direct origin, but make
                                // the selection bias explicit in the record.
                                let selected_conflict_index = conflicts
                                    .iter()
                                    .position(|(handle, _, _)| {
                                        installed_origins
                                            .as_ref()
                                            .is_some_and(|origins| origins.contains_key(handle))
                                    })
                                    .unwrap_or(0);
                                let selected = &conflicts[selected_conflict_index];
                                let existing_origins = installed_origins
                                    .as_ref()
                                    .and_then(|origins| origins.get(&selected.0))
                                    .cloned()
                                    .unwrap_or_default();
                                first_conflict = Some((
                                    piece_index,
                                    k,
                                    piece[i].uv,
                                    piece[j].uv,
                                    selected_conflict_index,
                                    conflicts.len(),
                                    mapped_conflicts,
                                    existing_origins,
                                    selected.1,
                                    selected.2,
                                ));
                            }
                        }
                    }
                    success = false;
                }
            }
        }
        if probe {
            if let Some((
                proposed_piece,
                proposed_segment,
                proposed_a,
                proposed_b,
                selected_conflict_index,
                selected_refusal_conflicts,
                mapped_conflicts,
                existing_origins,
                existing_a,
                existing_b,
            )) = first_conflict
            {
                let (source_face_id, declared_face_index, periodic_rank) =
                    PROBE_FACE_CONTEXT.with(std::cell::Cell::get);
                let existing_first_origin = existing_origins.first().copied();
                let selection = if mapped_conflicts == 0 {
                    "first_reported"
                } else {
                    "first_mappable"
                };
                let origin_resolution = if existing_first_origin.is_some() {
                    "direct"
                } else {
                    "missing"
                };
                eprintln!(
                    "CW\tsource_face_id={source_face_id:?}\tdeclared_face_index={declared_face_index}\t\
                     periodic_rank={periodic_rank}\tselection={selection}\t\
                     selected_conflict_index={selected_conflict_index}\t\
                     selected_refusal_conflicts={selected_refusal_conflicts}\t\
                     mapped_conflicts={mapped_conflicts}\t\
                     proposed_post_stitch_piece={proposed_piece}\t\
                     proposed_post_stitch_segment={proposed_segment}\t\
                     existing_origin_resolution={origin_resolution}\t\
                     existing_direct_contributors={}\t\
                     existing_first_post_stitch_origin={existing_first_origin:?}\t\
                     proposed_uv={:.17e},{:.17e};{:.17e},{:.17e}\t\
                     existing_uv={:.17e},{:.17e};{:.17e},{:.17e}\t\
                     face_conflicting_edges={probe_conflicting_edges}",
                    existing_origins.len(), proposed_a.x, proposed_a.y, proposed_b.x,
                    proposed_b.y, existing_a.x, existing_a.y, existing_b.x, existing_b.y,
                );
            }
        }
        if probe
            && (!success
                || probe_degenerate != 0
                || probe_add_returned_false != 0)
        {
            eprintln!(
                "PF,{},{probe_point_fail},{probe_degenerate},\
                 {probe_already_direct},{probe_refused_with_conflicts},\
                 {probe_refused_without_conflicts},{probe_conflicting_edges},\
                 {probe_add_returned_false}",
                u8::from(success),
            );
        }
        success
    }
}

fn spade_round(x: f64) -> f64 {
    match f64::abs(x) < MIN_ALLOWED_VALUE {
        true => 0.0,
        false => x,
    }
}

/// What a Spade constraint edge *means*, so that material semantics can be
/// derived from it.
///
/// **Audit A1.** The dual-parity flood in [`triangulation_into_polymesh_outcome`]
/// asks `edge.is_constraint_edge()` and toggles material parity on every `true`.
/// That is one bit where the formal system requires a role: `insert_surface`
/// adds constraints across the *interior sampling grid*, wholly inside the
/// material region, and those toggle parity exactly as a trim segment does.
///
/// FORMAL_SYSTEM.md §IX distinguishes `Physical`, `ArtificialCut`,
/// `NativeBoundary` and `SingularLink`; Definition 20 gives them different
/// material constraints — a physical half-edge pins `μ_L = 1, μ_R = 0` while an
/// artificial cut requires `μ_L = μ_R`. This enum is that distinction, carried
/// far enough to make the A1 experiment causal.
///
/// It is deliberately *not* the general cell-constraint solver. Parity is still
/// the mechanism; the only thing that changes is which edges are entitled to
/// flip it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ConstraintRole {
    /// A trim segment carrying source boundary evidence. Toggles material side.
    PhysicalBoundary,
    /// A chart cut or seam introduced to make the domain simply connected.
    /// Both sides are the same material state, so it must not toggle.
    ArtificialCut,
    /// The ambient parameter domain's own edge. Does not itself toggle;
    /// its interpretation is schema-dependent (FORMAL_SYSTEM Definition 20).
    NativeBoundary,
    /// An interior grid constraint inserted by [`insert_surface`] to control
    /// triangle shape. Carries no material meaning whatsoever.
    SurfaceSampling,
    /// A closure segment synthesised by [`PolyBoundary::new`] that no source
    /// entity justifies. Its role is genuinely unknown; it must not be silently
    /// assigned equality semantics (audit A6).
    UnresolvedSyntheticClosure,
}

/// Roles for the constraint edges of one face's triangulation.
///
/// A side table rather than a Spade edge-data parameter: `add_constraint` does
/// not hand back the edges it marked, so a role can only be attached by looking
/// the edge up afterwards. When that lookup fails — Spade may mark a collinear
/// chain rather than the direct edge — the role is genuinely unresolved, and
/// [`Self::role_of`] says so rather than guessing.
#[derive(Debug, Default)]
struct ConstraintRoles {
    roles: HashMap<FixedUndirectedEdgeHandle, ConstraintRole>,
    /// Constraint edges the flood met that no `record` call had claimed.
    /// Counted, not assumed: this is the size of the gap between what we asked
    /// Spade to constrain and what we can name (CDT-001, CDT-002).
    unresolved_at_flood: std::cell::Cell<usize>,
    /// How many edges each role claimed, for the experiment's own report.
    recorded: HashMap<ConstraintRole, usize>,
}

impl ConstraintRoles {
    fn record(&mut self, edge: FixedUndirectedEdgeHandle, role: ConstraintRole) {
        // First claim wins. A later, weaker claim must not overwrite a physical
        // boundary: an interior grid segment that happens to land on an edge a
        // trim segment already constrained is still a trim segment.
        if !self.roles.contains_key(&edge) {
            self.roles.insert(edge, role);
            *self.recorded.entry(role).or_insert(0) += 1;
        }
    }

    fn role_of(&self, edge: FixedUndirectedEdgeHandle) -> Option<ConstraintRole> {
        self.roles.get(&edge).copied()
    }

    /// Whether a constraint edge is entitled to flip material parity.
    ///
    /// **Legacy-preserving by construction.** Everything except
    /// `SurfaceSampling` and `ArtificialCut` toggles, including the unresolved
    /// case. The single semantic change in the A1 experiment is that interior
    /// sampling constraints stop toggling; every other population keeps the
    /// behaviour it had, tagged but unaltered, so a difference in the census is
    /// attributable to that one change and not to a reclassification.
    fn toggles_material(&self, edge: FixedUndirectedEdgeHandle) -> bool {
        match self.role_of(edge) {
            Some(ConstraintRole::SurfaceSampling) => false,
            Some(ConstraintRole::ArtificialCut) => false,
            Some(ConstraintRole::PhysicalBoundary) => true,
            Some(ConstraintRole::NativeBoundary) => true,
            Some(ConstraintRole::UnresolvedSyntheticClosure) => true,
            None => {
                self.unresolved_at_flood
                    .set(self.unresolved_at_flood.get() + 1);
                true
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TessellationFailureReason {
    ContradictoryDualParity,
    NoOddParityRegion,
    ConstraintChainNotClosed,
    ConstraintInsertionIncomplete,
    ConstraintIntersectionUnsupported,
    ConstraintOverlapUnsupported,
    CdtConstructionFailed,
    NonFinitePosition,
    ConstraintRoleMissing,
    DegenerateConstraintChain,
    NoFiniteTrianglesAfterParity,
}

#[derive(Clone, Debug)]
pub struct TessellationFailure {
    pub reason: TessellationFailureReason,
    pub source_bound: Option<usize>,
    pub source_edge_use: Option<usize>,
    pub constraint_ids: Vec<usize>,
    pub uv_location: Option<Point2>,
}

impl From<TessellationFailureReason> for TessellationFailure {
    fn from(reason: TessellationFailureReason) -> Self {
        Self {
            reason,
            source_bound: None,
            source_edge_use: None,
            constraint_ids: Vec::new(),
            uv_location: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VertexGeneration {
    SurfaceEvaluation,
    SourceEdgeSample,
    ConstraintIntersection,
    SingularRealization,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct VertexRoles(pub u16);

impl VertexRoles {
    pub const PHYSICAL_BOUNDARY: u16 = 1 << 0;
    pub const ARTIFICIAL_SEAM: u16 = 1 << 1;
    pub const SINGULAR_COLLAPSE: u16 = 1 << 2;
    pub const CONSTRAINT_INTERSECT: u16 = 1 << 3;

    pub fn contains(&self, role: u16) -> bool {
        (self.0 & role) == role
    }
    pub fn insert(&mut self, role: u16) {
        self.0 |= role;
    }
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }
}

#[derive(Clone, Debug)]
pub struct SeamPair {
    pub first_chain: Vec<usize>,
    pub second_chain: Vec<usize>,
    pub correspondence: Vec<(usize, usize)>,
    pub orientation_reversed: bool,
    pub deck_displacement: [i64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SingularKind {
    Apex,
    Pole,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SingularLinkKind {
    InteriorCycle,
    BoundaryInterval,
}

#[derive(Clone, Debug)]
pub struct SingularGroup {
    pub vertices: Vec<usize>,
    pub canonical_point: Point3,
    pub kind: SingularKind,
    pub link_kind: SingularLinkKind,
}

#[derive(Clone, Debug)]
pub struct VertexMetadata {
    pub uv: Point2,
    pub generation: VertexGeneration,
    pub roles: VertexRoles,
    pub source_edge_use: Option<usize>,
    pub seam_pair: Option<usize>,
    pub singular_group: Option<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct TessellationDiagnostics {
    pub vertex_metadata: Vec<VertexMetadata>,
    pub seam_pairs: Vec<SeamPair>,
    pub singular_groups: Vec<SingularGroup>,
}

#[derive(Clone, Debug)]
pub struct FaceTessellation {
    pub mesh: PolygonMesh,
    pub diagnostics: TessellationDiagnostics,
}

#[derive(Clone, Debug)]
pub enum TessellationOutcome {
    Mesh(FaceTessellation),
    Failed(TessellationFailure),
}

/// Tessellates one surface trimmed by polyline, returning `TessellationOutcome`.
fn trimming_tessellation_with_diagnostics<S>(
    surface: &S,
    polyboundary: &PolyBoundary,
    tol: f64,
) -> TessellationOutcome
where
    S: PreMeshableSurface,
{
    let mut triangulation = Cdt::new();
    let mut boundary_map = HashMap::<FixedVertexHandle, Point3>::default();
    let mut roles = ConstraintRoles::default();
    if !polyboundary.insert_to(&mut triangulation, &mut boundary_map, &mut roles) {
        return TessellationOutcome::Failed(TessellationFailureReason::ConstraintInsertionIncomplete.into());
    }
    insert_surface(&mut triangulation, surface, polyboundary, tol, &mut roles);
    let outcome = triangulation_into_polymesh_outcome(
        &triangulation,
        surface,
        polyboundary,
        &boundary_map,
        &roles,
    );
    if std::env::var_os("TRUCK_PROBE_ROLES").is_some() {
        // The population sizes the A1 comparison rests on. `unresolved` is the
        // honest gap: constraint edges the flood met that no `record` call had
        // claimed, which keep their legacy toggling behaviour. A large number
        // here would mean the experiment is less causal than it looks.
        let count = |role| roles.recorded.get(&role).copied().unwrap_or(0);
        eprintln!(
            "ROLES\tphysical={}\tsampling={}\tartificial={}\tnative={}\t\
             unresolved_synth={}\tunresolved_at_flood={}",
            count(ConstraintRole::PhysicalBoundary),
            count(ConstraintRole::SurfaceSampling),
            count(ConstraintRole::ArtificialCut),
            count(ConstraintRole::NativeBoundary),
            count(ConstraintRole::UnresolvedSyntheticClosure),
            roles.unresolved_at_flood.get(),
        );
    }
    outcome
}

/// Tessellates one surface trimmed by polyline, returning `TessellationOutcome`.
#[allow(dead_code)]
fn trimming_tessellation_with_outcome<S>(
    surface: &S,
    polyboundary: &PolyBoundary,
    tol: f64,
) -> TessellationOutcome
where
    S: PreMeshableSurface,
{
    trimming_tessellation_with_diagnostics(surface, polyboundary, tol)
}

/// Tessellates one surface trimmed by polyline.
fn trimming_tessellation<S>(surface: &S, polyboundary: &PolyBoundary, tol: f64) -> PolygonMesh
where
    S: PreMeshableSurface,
{
    match trimming_tessellation_with_diagnostics(surface, polyboundary, tol) {
        TessellationOutcome::Mesh(ft) => {
            let mut mesh = ft.mesh;
            mesh.make_face_compatible_to_normal();
            mesh
        }
        TessellationOutcome::Failed(f) => {
            if std::env::var_os("TRUCK_PROBE_FAIL").is_some() {
                eprintln!("PROBE_FAIL reason={:?}", f.reason);
            }
            PolygonMesh::default()
        }
    }
}

/// Inserts parameter divisions into triangulation.
fn insert_surface(
    triangulation: &mut Cdt,
    surface: impl PreMeshableSurface,
    polyline: &PolyBoundary,
    tol: f64,
    roles: &mut ConstraintRoles,
) {
    // Audit A1: every constraint added below is an interior sampling edge. It
    // exists to control triangle shape and lies wholly inside the material
    // region — `polyline.include` gated the insertion of both its endpoints.
    // It carries no material meaning and must not toggle parity.
    let mut constrain = |triangulation: &mut Cdt, a: FixedVertexHandle, b: FixedVertexHandle| {
        if triangulation.can_add_constraint(a, b) && triangulation.add_constraint(a, b) {
            if let Some(edge) = triangulation.get_edge_from_neighbors(a, b) {
                roles.record(edge.as_undirected().fix(), ConstraintRole::SurfaceSampling);
            }
        }
    };
    let bdb: BoundingBox<Point2> = polyline
        .0
        .iter()
        .flatten()
        .map(std::ops::Deref::deref)
        .collect();
    let range = ((bdb.min()[0], bdb.max()[0]), (bdb.min()[1], bdb.max()[1]));
    let (udiv, vdiv) = surface.parameter_division(range, tol);
    let insert_res: Vec<Vec<Option<_>>> = udiv
        .into_iter()
        .map(|u| {
            vdiv.iter()
                .map(|v| match polyline.include(Point2::new(u, *v)) {
                    true => triangulation.insert(SPoint2::new(u, *v)).ok(),
                    false => None,
                })
                .collect()
        })
        .collect();
    insert_res.windows(2).for_each(|vec| {
        vec[0].windows(2).zip(&vec[1]).for_each(|(a, z)| {
            if let Some(x) = a[0] {
                if let Some(y) = a[1] {
                    constrain(triangulation, x, y);
                }
                if let Some(z) = z {
                    constrain(triangulation, x, *z);
                }
            }
        });
        let idx = vec[0].len() - 1;
        if let (Some(x), Some(y)) = (vec[0][idx], vec[1][idx]) {
            constrain(triangulation, x, y);
        }
    });
}

/// Converts triangulation into `TessellationOutcome`.
fn triangulation_into_polymesh_outcome<S: ParametricSurface3D>(
    triangulation: &Cdt,
    surface: &S,
    _polyline: &PolyBoundary,
    boundary_map: &HashMap<FixedVertexHandle, Point3>,
    roles: &ConstraintRoles,
) -> TessellationOutcome {
    use std::collections::{HashMap as StdHashMap, VecDeque};

    // 1. Parity-labeled CDT dual traversal across domain-boundary constraint edges
    let mut face_parity = StdHashMap::<usize, u32>::new();
    let mut queue = VecDeque::new();

    let outer = triangulation.outer_face();
    face_parity.insert(outer.index(), 0);
    queue.push_back((outer.fix(), 0));

    let mut contradictory_parity = false;
    while let Some((ffh, current_parity)) = queue.pop_front() {
        let face = triangulation.face(ffh);
        let edges = if let Some(inner) = face.as_inner() {
            inner.adjacent_edges()
        } else {
            let e0 = face.adjacent_edge().unwrap();
            let e1 = e0.next();
            let e2 = e1.next();
            [e0, e1, e2]
        };
        for e in edges {
            // Audit A1. This was `e.is_constraint_edge()` — one bit, so an
            // interior sampling constraint flipped material parity exactly as a
            // trim segment did. A constraint edge is now only a material
            // transition if its role says so.
            let is_domain_boundary =
                e.is_constraint_edge() && roles.toggles_material(e.as_undirected().fix());
            let next_parity = if is_domain_boundary {
                current_parity ^ 1
            } else {
                current_parity
            };
            let adj_face = e.rev().face();
            let adj_idx = adj_face.index();
            if let Some(&existing_parity) = face_parity.get(&adj_idx) {
                if existing_parity != next_parity {
                    contradictory_parity = true;
                }
            } else {
                face_parity.insert(adj_idx, next_parity);
                queue.push_back((adj_face.fix(), next_parity));
            }
        }
    }

    if contradictory_parity {
        return TessellationOutcome::Failed(TessellationFailureReason::ContradictoryDualParity.into());
    }

    // 2. Vertex positions, parameter coordinates, and roles
    let mut positions = Vec::<Point3>::new();
    let mut uv_coords = Vec::<Vector2>::new();
    let mut normals = Vec::<Vector3>::new();
    let mut vertex_metadata = Vec::<VertexMetadata>::new();

    let u_period = surface.u_period();
    let v_period = surface.v_period();

    let vmap: HashMap<_, _> = triangulation
        .vertices()
        .enumerate()
        .map(|(i, v)| {
            let p = *v.as_ref();
            let idx = v.fix();
            let point = match boundary_map.get(&idx) {
                Some(point) => *point,
                None => surface.subs(p.x, p.y),
            };
            if !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite() {
                return (idx, usize::MAX);
            }
            positions.push(point);
            let uv = Vector2::new(p.x, p.y);
            uv_coords.push(uv);

            // Determine vertex roles (bitflags)
            let mut roles = VertexRoles::default();
            if boundary_map.contains_key(&idx) {
                let is_seam = (u_period.is_some() && (p.x <= 1e-4 || (p.x - u_period.unwrap()).abs() <= 1e-4))
                    || (v_period.is_some() && (p.y <= 1e-4 || (p.y - v_period.unwrap()).abs() <= 1e-4));
                if is_seam {
                    roles.insert(VertexRoles::ARTIFICIAL_SEAM);
                } else {
                    roles.insert(VertexRoles::PHYSICAL_BOUNDARY);
                }
            } else {
                // INTERIOR is derived downstream as the absence of any
                // boundary/seam/singular role; it is not stored as an
                // overlapping flag (PR 4A.1 design, sidecar diagnostics).
            }

            let n = surface.normal(p.x, p.y);
            let n_valid =
                if n.x.is_finite() && n.y.is_finite() && n.z.is_finite() && n.magnitude2() > 1e-12 {
                    n.normalize()
                } else {
                    roles.insert(VertexRoles::SINGULAR_COLLAPSE);
                    Vector3::zero()
                };
            normals.push(n_valid);

            vertex_metadata.push(VertexMetadata {
                uv: Point2::new(p.x, p.y),
                generation: if boundary_map.contains_key(&idx) {
                    VertexGeneration::SourceEdgeSample
                } else {
                    VertexGeneration::SurfaceEvaluation
                },
                roles,
                source_edge_use: None,
                seam_pair: None,
                singular_group: None,
            });

            (idx, i)
        })
        .collect();

    if vmap.values().any(|&i| i == usize::MAX) {
        return TessellationOutcome::Failed(TessellationFailureReason::NonFinitePosition.into());
    }

    // 3. Material triangles selection (odd parity = 1)
    let tri_faces_raw: Vec<[usize; 3]> = triangulation
        .inner_faces()
        .filter(|face| face_parity.get(&face.index()) == Some(&1))
        .map(|tri| tri.vertices())
        .filter_map(|tri| {
            let idcs = [
                vmap[&tri[0].fix()],
                vmap[&tri[1].fix()],
                vmap[&tri[2].fix()],
            ];
            if idcs[0] == idcs[1] || idcs[1] == idcs[2] || idcs[0] == idcs[2] {
                return None;
            }
            let p0 = positions[idcs[0]];
            let p1 = positions[idcs[1]];
            let p2 = positions[idcs[2]];
            let cross = (p1 - p0).cross(p2 - p0);
            let area = 0.5 * cross.magnitude();
            if area <= 1e-12 || !area.is_finite() {
                return None;
            }
            Some(idcs)
        })
        .collect();

    if tri_faces_raw.is_empty() {
        return TessellationOutcome::Failed(TessellationFailureReason::NoOddParityRegion.into());
    }

    // 4. Singular Vertex Normal Repair
    let mut vertex_incident_normals = vec![Vector3::zero(); positions.len()];
    for &[i0, i1, i2] in &tri_faces_raw {
        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];
        let cross = (p1 - p0).cross(p2 - p0);
        let mag = cross.magnitude();
        if mag > 1e-12 {
            let fnorm = cross / mag;
            vertex_incident_normals[i0] += fnorm * mag;
            vertex_incident_normals[i1] += fnorm * mag;
            vertex_incident_normals[i2] += fnorm * mag;
        }
    }

    for (i, norm) in normals.iter_mut().enumerate() {
        if norm.so_small() || !norm.x.is_finite() {
            let inc = vertex_incident_normals[i];
            if inc.magnitude2() > 1e-12 {
                *norm = inc.normalize();
            } else {
                *norm = Vector3::unit_z();
            }
        }
    }

    let tri_faces: Vec<[StandardVertex; 3]> = tri_faces_raw
        .into_iter()
        .map(|idcs| array![i => [idcs[i], idcs[i], idcs[i]].into(); 3])
        .collect();

    // PR 4A.1 invariant: sidecar metadata must stay aligned with mesh positions,
    // one record per vertex, through every transformation below. Enforced in the
    // test build; the census (release) relies on it holding by construction here.
    debug_assert_eq!(vertex_metadata.len(), positions.len());

    let mesh = PolygonMesh::debug_new(
        StandardAttributes {
            positions,
            uv_coords,
            normals,
        },
        Faces::from_tri_and_quad_faces(tri_faces, Vec::new()),
    );

    TessellationOutcome::Mesh(FaceTessellation {
        mesh,
        diagnostics: TessellationDiagnostics {
            vertex_metadata,
            seam_pairs: Vec::new(),
            singular_groups: Vec::new(),
        },
    })
}

#[allow(dead_code)]
fn triangulation_into_polymesh<S: ParametricSurface3D>(
    triangulation: &Cdt,
    surface: &S,
    polyline: &PolyBoundary,
    boundary_map: &HashMap<FixedVertexHandle, Point3>,
    roles: &ConstraintRoles,
) -> PolygonMesh {
    match triangulation_into_polymesh_outcome(triangulation, surface, polyline, boundary_map, roles)
    {
        TessellationOutcome::Mesh(ft) => ft.mesh,
        TessellationOutcome::Failed(_) => PolygonMesh::default(),
    }
}

fn polyline_on_surface(
    surface: impl PreMeshableSurface,
    p: SurfacePoint,
    q: SurfacePoint,
    tol: f64,
) -> Vec<SurfacePoint> {
    use truck_geometry::prelude::*;
    let tol = tol.max(TOLERANCE);
    let line = Line(p.uv, q.uv);
    let pcurve = PCurve::new(line, &surface);
    let (vec, _) = pcurve.parameter_division(pcurve.range_tuple(), tol);
    vec.into_iter()
        .map(|t| {
            let uv = line.subs(t);
            (uv, surface.subs(uv.x, uv.y)).into()
        })
        .collect()
}

/// Explicit classification for a conical/revoluted face with one circular boundary and a collapsed apex.
#[allow(dead_code, unused)]
#[derive(Debug, Clone, Copy)]
pub struct CollapsedPeriodicBoundaryPair {
    pub base_u: f64,
    pub apex_u: f64,
}

impl CollapsedPeriodicBoundaryPair {
    /// Classify whether a face's boundary consists of a single regular periodic loop and a collapsed apex.
    pub fn try_classify<S: PreMeshableSurface>(
        surface: &S,
        closed: &[Vec<SurfacePoint>],
        open: &[Vec<SurfacePoint>],
        _range: (Option<(f64, f64)>, Option<(f64, f64)>),
    ) -> Option<Self> {
        // 1. Exactly one regular closed periodic boundary and no open generator/sector curves
        if closed.len() != 1 || !open.is_empty() {
            return None;
        }

        let (period, is_v) = match (surface.v_period(), surface.u_period()) {
            (Some(vp), _) if vp > 1e-6 => (vp, true),
            (_, Some(up)) if up > 1e-6 => (up, false),
            _ => return None,
        };
        let loop0 = &closed[0];

        // 2. Regular loop must span the periodic parameter (winding ±1, span ~ period)
        let (p_min, p_max) =
            loop0
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), p| {
                    let val = if is_v { p.uv.y } else { p.uv.x };
                    (mn.min(val), mx.max(val))
                });
        if (p_max - p_min) < 0.75 * period {
            return None;
        }

        let base_r: f64 = loop0
            .iter()
            .map(|p| if is_v { p.uv.x } else { p.uv.y })
            .sum::<f64>()
            / loop0.len() as f64;

        // 3. Analytically compute exact apex parameter where angular orbit collapses in 3D.
        use cgmath::InnerSpace;
        let w = |r: f64| -> Vector3 {
            let (p0, p_half) = if is_v {
                (surface.subs(r, 0.0), surface.subs(r, 0.5 * period))
            } else {
                (surface.subs(0.0, r), surface.subs(0.5 * period, r))
            };
            p0 - p_half
        };

        let w0 = w(0.0);
        let w1 = w(1.0);
        let dw = w1 - w0;
        let dw2 = dw.magnitude2();

        if dw2 < 1e-12 {
            return None;
        }

        let apex_r = -w0.dot(dw) / dw2;

        // 4. Certificate guard: verify 3D point collapse at apex
        if w(apex_r).magnitude() > 1e-3 {
            return None;
        }

        // 5. Axial separation between base loop and apex must be nonzero
        if (base_r - apex_r).abs() < 1e-6 {
            return None;
        }

        Some(Self {
            base_u: base_r,
            apex_u: apex_r,
        })
    }
}

#[allow(dead_code, unused)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoundaryStratum {
    #[allow(dead_code, unused)]
    RegularCurve,
    #[allow(dead_code, unused)]
    CollapsedToPoint,
    #[allow(dead_code, unused)]
    PeriodicIdentification,
}

#[allow(dead_code, unused)]
#[derive(Clone, Debug)]
pub struct WireAssembledFace {
    pub loops: Vec<Vec<SurfacePoint>>,
    pub strata: Vec<BoundaryStratum>,
}

#[allow(dead_code, unused)]
#[derive(Clone, Debug)]
pub struct QuotientLiftCertificate {
    #[allow(dead_code, unused)]
    pub period_shifts: Vec<(i32, i32)>,
    #[allow(dead_code, unused)]
    pub max_junction_residual: f64,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct QuotientResolvedFace {
    pub resolved_loops: Vec<Vec<SurfacePoint>>,
    pub certificate: QuotientLiftCertificate,
}

#[allow(dead_code)]
pub fn solve_quotient_lift<S: PreMeshableSurface>(
    surface: &S,
    loops: &[Vec<SurfacePoint>],
) -> Option<QuotientResolvedFace> {
    let u_period = surface.u_period();
    let v_period = surface.v_period();

    let mut resolved_loops = Vec::with_capacity(loops.len());
    let mut period_shifts = Vec::with_capacity(loops.len());
    let mut max_junction_residual = 0.0f64;

    for loop_pts in loops {
        if loop_pts.len() < 2 {
            resolved_loops.push(loop_pts.clone());
            period_shifts.push((0, 0));
            continue;
        }

        let p0 = loop_pts[0].uv;
        let p1 = loop_pts[loop_pts.len() - 1].uv;
        let du = p1.x - p0.x;
        let dv = p1.y - p0.y;

        let ku = match u_period {
            Some(pu) if pu > 1e-6 => (du / pu).round() as i32,
            _ => 0,
        };
        let kv = match v_period {
            Some(pv) if pv > 1e-6 => (dv / pv).round() as i32,
            _ => 0,
        };

        period_shifts.push((ku, kv));

        let u_shift = match u_period {
            Some(pu) => (ku as f64) * pu,
            None => 0.0,
        };
        let v_shift = match v_period {
            Some(pv) => (kv as f64) * pv,
            None => 0.0,
        };

        let mut lifted_loop = loop_pts.clone();
        let n = lifted_loop.len();
        for (idx, pt) in lifted_loop.iter_mut().enumerate() {
            let frac = idx as f64 / (n - 1).max(1) as f64;
            pt.uv.x -= frac * u_shift;
            pt.uv.y -= frac * v_shift;
        }

        let residual = lifted_loop[0].uv.distance(lifted_loop[n - 1].uv);
        max_junction_residual = max_junction_residual.max(residual);
        resolved_loops.push(lifted_loop);
    }

    Some(QuotientResolvedFace {
        resolved_loops,
        certificate: QuotientLiftCertificate {
            period_shifts,
            max_junction_residual,
        },
    })
}

#[test]
fn test_global_quotient_lift_solver() {
    use truck_modeling::{BSplineSurface, KnotVec};
    let knot_vec = KnotVec::from(vec![0.0, 0.0, 1.0, 1.0]);
    let ctrl_pts = vec![
        vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
        vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
    ];
    let surface = BSplineSurface::new((knot_vec.clone(), knot_vec), ctrl_pts);
    let loop_pts = vec![
        (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
        (Point2::new(1.0, 1.0), Point3::new(1.0, 1.0, 0.0)).into(),
    ];
    let resolved = solve_quotient_lift(&surface, &[loop_pts]);
    assert!(resolved.is_some());
    let res = resolved.unwrap();
    assert_eq!(res.certificate.period_shifts, vec![(0, 0)]);
}

/*
#[test]
#[ignore]
#[cfg(not(target_arch = "wasm32"))]
fn par_bench() {
    use std::time::Instant;
    use truck_modeling::*;
    const JSON: &str = include_str!("../../resources/shape/bottle.json");
    let solid: Solid = serde_json::from_str(JSON).unwrap();
    let shell = solid.into_boundaries().pop().unwrap();

    let instant = Instant::now();
    (0..100).for_each(|_| {
        let _shell = shell_tessellation(&shell, 0.01, by_search_parameter);
    });
    println!("{}ms", instant.elapsed().as_millis());

    let instant = Instant::now();
    (0..100).for_each(|_| {
        let _shell = shell_tessellation_single_thread(&shell, 0.01, by_search_parameter);
    });
    println!("{}ms", instant.elapsed().as_millis());
}
*/

#[cfg(test)]
mod cone_topology_tests {
    use super::*;
    use std::f64::consts::PI;
    use truck_modeling::{Line, Point2, Point3, RevolutedCurve, Vector3};

    fn make_test_cone(r_base: f64, r_apex: f64, h: f64) -> RevolutedCurve<Line<Point3>> {
        let p0 = Point3::new(r_apex, 0.0, 0.0);
        let p1 = Point3::new(r_base, 0.0, h);
        RevolutedCurve::by_revolution(Line(p0, p1), Point3::origin(), Vector3::unit_z())
    }

    #[test]
    fn test_cone_lateral_face_forward_classified() {
        let cone = make_test_cone(10.0, 0.0, 10.0);
        let loop0: Vec<SurfacePoint> = (0..=10)
            .map(|i| {
                let v = (i as f64 / 10.0) * 2.0 * PI;
                let uv = Point2::new(1.0, v);
                (uv, cone.subs(uv.x, uv.y)).into()
            })
            .collect();
        let range = (Some((0.0, 1.0)), None);
        let res = CollapsedPeriodicBoundaryPair::try_classify(&cone, &[loop0], &[], range);
        assert!(res.is_some());
        let pair = res.unwrap();
        assert!((pair.apex_u - 0.0).abs() < 1e-6);
        assert!((pair.base_u - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_truncated_cone_two_loops_rejected() {
        let cone = make_test_cone(10.0, 5.0, 10.0);
        let loop0: Vec<SurfacePoint> = (0..=10)
            .map(|i| {
                let v = (i as f64 / 10.0) * 2.0 * PI;
                let uv = Point2::new(1.0, v);
                (uv, cone.subs(uv.x, uv.y)).into()
            })
            .collect();
        let loop1: Vec<SurfacePoint> = (0..=10)
            .map(|i| {
                let v = (i as f64 / 10.0) * 2.0 * PI;
                let uv = Point2::new(0.0, v);
                (uv, cone.subs(uv.x, uv.y)).into()
            })
            .collect();
        let range = (Some((0.0, 1.0)), None);
        let res = CollapsedPeriodicBoundaryPair::try_classify(&cone, &[loop0, loop1], &[], range);
        assert!(res.is_none());
    }

    #[test]
    fn test_cone_sector_with_open_generator_edges_rejected() {
        let cone = make_test_cone(10.0, 0.0, 10.0);
        let loop0: Vec<SurfacePoint> = (0..=10)
            .map(|i| {
                let v = (i as f64 / 10.0) * PI;
                let uv = Point2::new(1.0, v);
                (uv, cone.subs(uv.x, uv.y)).into()
            })
            .collect();
        let open_edge: Vec<SurfacePoint> = vec![
            (Point2::new(0.0, 0.0), cone.subs(0.0, 0.0)).into(),
            (Point2::new(1.0, 0.0), cone.subs(1.0, 0.0)).into(),
        ];
        let range = (Some((0.0, 1.0)), None);
        let res = CollapsedPeriodicBoundaryPair::try_classify(&cone, &[loop0], &[open_edge], range);
        assert!(res.is_none());
    }

    #[test]
    fn test_zero_area_non_cone_cylinder_rejected() {
        let cylinder = RevolutedCurve::by_revolution(
            Line(Point3::new(10.0, 0.0, 0.0), Point3::new(10.0, 0.0, 10.0)),
            Point3::origin(),
            Vector3::unit_z(),
        );
        let loop0: Vec<SurfacePoint> = (0..=10)
            .map(|i| {
                let v = (i as f64 / 10.0) * 2.0 * PI;
                let uv = Point2::new(1.0, v);
                (uv, cylinder.subs(uv.x, uv.y)).into()
            })
            .collect();
        let range = (Some((0.0, 1.0)), None);
        let res = CollapsedPeriodicBoundaryPair::try_classify(&cylinder, &[loop0], &[], range);
        assert!(res.is_none());
    }

    #[test]
    fn test_traversal_semantics_periodic_circle() {
        use crate::tessellation::domain::projection::TraversalSemantics;
        use truck_geometry::prelude::*;
        let circle = UnitCircle::<Point3>::new();
        let cone = make_test_cone(10.0, 0.0, 10.0);
        let semantics = TraversalSemantics::resolve(&circle, &cone, 1e-4);
        assert!(matches!(semantics, TraversalSemantics::FullPeriod { .. }));
    }

    #[test]
    fn test_traversal_semantics_degenerate_point() {
        use crate::tessellation::domain::projection::TraversalSemantics;
        use truck_geometry::prelude::*;
        let line = Line(Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0));
        let cone = make_test_cone(10.0, 0.0, 10.0);
        let semantics = TraversalSemantics::resolve(&line, &cone, 1e-4);
        assert_eq!(semantics, TraversalSemantics::DegeneratePoint);
    }

    #[test]
    fn test_shared_boundary_projection_processor_wrapped_cone() {
        use crate::tessellation::domain::projection::{project_boundary_curve, TraversalSemantics};
        use truck_geometry::prelude::*;
        let cone = make_test_cone(10.0, 0.0, 10.0);
        let p_uv = cone.subs(0.5, 0.0);
        let tr = Matrix4::from_translation(Vector3::new(10.0, 20.0, 30.0));
        let proc_cone = Processor::with_transform(cone, tr);
        let circle = Processor::with_transform(TrimmedCurve::new(Line(p_uv, p_uv), (0.0, 1.0)), tr);
        let sem = TraversalSemantics::resolve(&circle, &proc_cone, 1e-4);
        let path = project_boundary_curve(&circle, &proc_cone, sem, 1.0).unwrap();
        assert!(!path.samples.is_empty());
    }

    #[test]
    fn test_periodic_closure_cone_tessellation() {
        let cone = make_test_cone(10.0, 0.0, 10.0);
        let tol = 0.1;
        let n = 32usize;
        let circle_pts: Vec<SurfacePoint> = (0..=n)
            .map(|i| {
                let v = (i as f64 / n as f64) * 2.0 * PI;
                let uv = Point2::new(1.0, v);
                (uv, cone.subs(uv.x, uv.y)).into()
            })
            .collect();
        let piece = PolyBoundaryPiece(circle_pts);
        let boundary = PolyBoundary::new(vec![piece], &cone, tol);
        let mesh = trimming_tessellation(&cone, &boundary, tol);
        assert!(
            !mesh.faces().is_empty(),
            "Cone C-APEX-DISK face must tessellate to non-empty mesh"
        );
    }

    #[test]
    fn test_parity_single_disk() {
        use truck_geometry::prelude::*;
        let plane = Plane::new(Point3::origin(), Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0));
        let tol = 0.01;
        let loop0: Vec<SurfacePoint> = vec![
            (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
            (Point2::new(10.0, 0.0), Point3::new(10.0, 0.0, 0.0)).into(),
            (Point2::new(10.0, 10.0), Point3::new(10.0, 10.0, 0.0)).into(),
            (Point2::new(0.0, 10.0), Point3::new(0.0, 10.0, 0.0)).into(),
            (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
        ];
        let boundary = PolyBoundary::new(vec![PolyBoundaryPiece(loop0)], &plane, tol);
        let mesh = trimming_tessellation(&plane, &boundary, tol);
        assert!(!mesh.faces().is_empty());
    }

    #[test]
    fn test_parity_disk_with_hole() {
        use truck_geometry::prelude::*;
        let plane = Plane::new(Point3::origin(), Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0));
        let tol = 0.01;
        let outer: Vec<SurfacePoint> = vec![
            (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
            (Point2::new(10.0, 0.0), Point3::new(10.0, 0.0, 0.0)).into(),
            (Point2::new(10.0, 10.0), Point3::new(10.0, 10.0, 0.0)).into(),
            (Point2::new(0.0, 10.0), Point3::new(0.0, 10.0, 0.0)).into(),
            (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
        ];
        let hole: Vec<SurfacePoint> = vec![
            (Point2::new(3.0, 3.0), Point3::new(3.0, 3.0, 0.0)).into(),
            (Point2::new(7.0, 3.0), Point3::new(7.0, 3.0, 0.0)).into(),
            (Point2::new(7.0, 7.0), Point3::new(7.0, 7.0, 0.0)).into(),
            (Point2::new(3.0, 7.0), Point3::new(3.0, 7.0, 0.0)).into(),
            (Point2::new(3.0, 3.0), Point3::new(3.0, 3.0, 0.0)).into(),
        ];
        let boundary = PolyBoundary::new(vec![PolyBoundaryPiece(outer), PolyBoundaryPiece(hole)], &plane, tol);
        let mesh = trimming_tessellation(&plane, &boundary, tol);
        assert!(!mesh.faces().is_empty());
        // Verify hole centroid (5.0, 5.0) has no triangle containing it
        for tri in mesh.faces().tri_faces() {
            let p0 = mesh.uv_coords()[tri[0].pos];
            let p1 = mesh.uv_coords()[tri[1].pos];
            let p2 = mesh.uv_coords()[tri[2].pos];
            let center = (p0 + p1 + p2) / 3.0;
            assert!(!(center.x > 3.1 && center.x < 6.9 && center.y > 3.1 && center.y < 6.9), "Hole interior must not contain triangles");
        }
    }

    #[test]
    fn test_parity_concave_outer_loop() {
        use truck_geometry::prelude::*;
        let plane = Plane::new(Point3::origin(), Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0));
        let tol = 0.01;
        // L-shaped concave outer loop
        let loop0: Vec<SurfacePoint> = vec![
            (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
            (Point2::new(10.0, 0.0), Point3::new(10.0, 0.0, 0.0)).into(),
            (Point2::new(10.0, 5.0), Point3::new(10.0, 5.0, 0.0)).into(),
            (Point2::new(5.0, 5.0), Point3::new(5.0, 5.0, 0.0)).into(),
            (Point2::new(5.0, 10.0), Point3::new(5.0, 10.0, 0.0)).into(),
            (Point2::new(0.0, 10.0), Point3::new(0.0, 10.0, 0.0)).into(),
            (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
        ];
        let boundary = PolyBoundary::new(vec![PolyBoundaryPiece(loop0)], &plane, tol);
        let mesh = trimming_tessellation(&plane, &boundary, tol);
        assert!(!mesh.faces().is_empty());
    }

    #[test]
    fn test_parity_intersecting_constraints_rejected() {
        use truck_geometry::prelude::*;
        let plane = Plane::new(Point3::origin(), Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0));
        let tol = 0.01;
        // Self-overlapping loop traversing same segment (0,0)->(10,0) twice in forward direction
        let loop0: Vec<SurfacePoint> = vec![
            (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
            (Point2::new(10.0, 0.0), Point3::new(10.0, 0.0, 0.0)).into(),
            (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
            (Point2::new(10.0, 0.0), Point3::new(10.0, 0.0, 0.0)).into(),
            (Point2::new(10.0, 10.0), Point3::new(10.0, 10.0, 0.0)).into(),
            (Point2::new(0.0, 10.0), Point3::new(0.0, 10.0, 0.0)).into(),
            (Point2::new(0.0, 0.0), Point3::new(0.0, 0.0, 0.0)).into(),
        ];
        let boundary = PolyBoundary::new(vec![PolyBoundaryPiece(loop0)], &plane, tol);
        let mesh = trimming_tessellation(&plane, &boundary, tol);
        assert!(mesh.faces().is_empty(), "Self-overlapping degenerate loop must fail or produce empty mesh");
    }
}

#[cfg(test)]
mod singular_transition_tests {
    use super::*;
    use truck_modeling::{Point2, Point3, Vector3};

    /// `v` collapses at `u = 1`: the whole `u = 1` row maps to the single
    /// point `(0, 1, 0)`, so every `(1, v)` is a legitimate UV representative
    /// of that one singular 3D point. This is the abstract shape of the
    /// representative face #54588 collapse.
    #[derive(Clone, Copy)]
    struct CollapsedPatch;

    impl ParametricSurface for CollapsedPatch {
        type Point = Point3;
        type Vector = Vector3;
        fn subs(&self, u: f64, v: f64) -> Point3 {
            Point3::new((1.0 - u) * v, u, 0.0)
        }
        fn uder(&self, _: f64, v: f64) -> Vector3 {
            Vector3::new(-v, 1.0, 0.0)
        }
        fn vder(&self, u: f64, _: f64) -> Vector3 {
            Vector3::new(1.0 - u, 0.0, 0.0)
        }
        fn uuder(&self, _: f64, _: f64) -> Vector3 {
            Vector3::zero()
        }
        fn uvder(&self, _: f64, _: f64) -> Vector3 {
            Vector3::new(-1.0, 0.0, 0.0)
        }
        fn vvder(&self, _: f64, _: f64) -> Vector3 {
            Vector3::zero()
        }
        fn der_mn(&self, m: usize, n: usize, u: f64, v: f64) -> Vector3 {
            match (m, n) {
                (0, 0) => self.subs(u, v).to_vec(),
                (1, 0) => self.uder(u, v),
                (0, 1) => self.vder(u, v),
                (1, 1) => self.uvder(u, v),
                _ => Vector3::zero(),
            }
        }
    }

    impl ParametricSurface3D for CollapsedPatch {}

    impl ParameterDivision2D for CollapsedPatch {
        fn parameter_division(
            &self,
            ((u0, u1), (v0, v1)): ((f64, f64), (f64, f64)),
            _: f64,
        ) -> (Vec<f64>, Vec<f64>) {
            (vec![u0, u1], vec![v0, v1])
        }
    }

    /// Mirror of `CollapsedPatch` with the axes swapped, so `u` collapses at
    /// `v = 1`. Used to exercise the u-direction enter/leave branches.
    #[derive(Clone, Copy)]
    struct CollapsedPatchU;

    impl ParametricSurface for CollapsedPatchU {
        type Point = Point3;
        type Vector = Vector3;
        fn subs(&self, u: f64, v: f64) -> Point3 {
            Point3::new((1.0 - v) * u, v, 0.0)
        }
        fn uder(&self, _: f64, v: f64) -> Vector3 {
            Vector3::new(1.0 - v, 0.0, 0.0)
        }
        fn vder(&self, u: f64, _: f64) -> Vector3 {
            Vector3::new(-u, 1.0, 0.0)
        }
        fn uuder(&self, _: f64, _: f64) -> Vector3 {
            Vector3::zero()
        }
        fn uvder(&self, _: f64, _: f64) -> Vector3 {
            Vector3::new(-1.0, 0.0, 0.0)
        }
        fn vvder(&self, _: f64, _: f64) -> Vector3 {
            Vector3::zero()
        }
        fn der_mn(&self, m: usize, n: usize, u: f64, v: f64) -> Vector3 {
            match (m, n) {
                (0, 0) => self.subs(u, v).to_vec(),
                (1, 0) => self.uder(u, v),
                (0, 1) => self.vder(u, v),
                (1, 1) => self.uvder(u, v),
                _ => Vector3::zero(),
            }
        }
    }

    impl ParametricSurface3D for CollapsedPatchU {}

    impl ParameterDivision2D for CollapsedPatchU {
        fn parameter_division(
            &self,
            ((u0, u1), (v0, v1)): ((f64, f64), (f64, f64)),
            _: f64,
        ) -> (Vec<f64>, Vec<f64>) {
            (vec![u0, u1], vec![v0, v1])
        }
    }

    /// Items 5-7: at least one triangle, no zero-area triangles, and a single
    /// consistent orientation sign. CollapsedPatch is planar (z = 0), so the
    /// orientation reduces to the sign of the z component of the cross product.
    fn assert_no_zero_area_or_orientation_flip(mesh: &PolygonMesh) {
        assert!(!mesh.faces().is_empty(), "tessellation produced no triangles");
        let mut orientation = 0.0_f64;
        for tri in mesh.faces().tri_faces() {
            let [p0, p1, p2] = tri.map(|v| mesh.positions()[v.pos]);
            let cross = (p1 - p0).cross(p2 - p0);
            assert!(cross.magnitude2() > 1.0e-18, "zero-area triangle");
            if orientation == 0.0 {
                orientation = cross.z.signum();
            }
            assert_eq!(
                cross.z.signum(),
                orientation,
                "inconsistent triangle orientation"
            );
        }
    }

    /// Items 1-8: entering the collapsed row preserves the incoming UV branch,
    /// leaving it appends a bridge paired with the singular 3D point, every
    /// touched point re-evaluates within tolerance, the reconstructed boundary
    /// does not backtrack across the singular row, and the surface tessellates
    /// with consistent triangle orientation. The reverse-traversal shape is
    /// checked at the end.
    #[test]
    fn singular_transition_preserves_incidence_and_tessellates() {
        let surface = CollapsedPatch;
        let tolerance = 1.0e-9;
        let uv = |u, v| Point2::new(u, v);
        let sp = |p: Point2| (p, surface.subs(p.x, p.y)).into();
        // Approach the singular row along v = 0, leave it along v = 1.
        let mut points: Vec<SurfacePoint> =
            [uv(0.0, 1.0), uv(0.0, 0.5), uv(0.0, 0.0), uv(0.5, 0.0)]
                .into_iter()
                .map(sp)
                .collect();

        // Enter the collapse. The raw lift proposes (1, 1.1006...) but the
        // singular row u = 1 evaluates it to (0, 1, 0).
        let previous_uv = uv(0.5, 0.0);
        let previous_point = surface.subs(previous_uv.x, previous_uv.y);
        let mut entering_uv = uv(1.0, 1.100_620_084_301_750_6);
        let singular_point = surface.subs(entering_uv.x, entering_uv.y);
        assert!(singular_point.near(&Point3::new(0.0, 1.0, 0.0)));
        reconcile_singular_transition(
            &surface,
            previous_uv,
            previous_point,
            &mut entering_uv,
            singular_point,
            tolerance,
            &mut points,
        );
        // (1) The incoming v = 0 branch is preserved.
        assert!(entering_uv.near(&uv(1.0, 0.0)));
        points.push((entering_uv, singular_point).into());

        // Leave the collapse toward v = 1.
        let mut leaving_uv = uv(0.5, 1.0);
        let leaving_point = surface.subs(leaving_uv.x, leaving_uv.y);
        reconcile_singular_transition(
            &surface,
            entering_uv,
            singular_point,
            &mut leaving_uv,
            leaving_point,
            tolerance,
            &mut points,
        );
        // (2) A bridge was appended, paired with the singular 3D point.
        let bridge = *points.last().unwrap();
        assert!(bridge.uv.near(&uv(1.0, 1.0)));
        assert!(bridge.point.near(&singular_point));
        points.push((leaving_uv, leaving_point).into());

        // (3) Every modified or inserted point re-evaluates within tolerance.
        assert!(points
            .iter()
            .all(|p| surface.subs(p.uv.x, p.uv.y).distance(p.point) <= tolerance));

        // (4) No improper backtracking crossing. The original defect produced
        // the spike (0.5,0) -> (1,1.1006) -> (1,1); the repaired boundary
        // enters and leaves the singular row u = 1 exactly once each.
        let u_visits: Vec<f64> = points.iter().map(|p| p.uv.x).collect();
        let near_row = |x: f64| (x - 1.0).abs() < 1.0e-7;
        let crossings = u_visits
            .windows(2)
            .filter(|w| near_row(w[0]) ^ near_row(w[1]))
            .count();
        assert_eq!(
            crossings, 2,
            "boundary must enter and leave u = 1 exactly once each"
        );

        // Close the loop and tessellate.
        points.push(sp(uv(0.0, 1.0)));
        let boundary = PolyBoundary::new(vec![PolyBoundaryPiece(points)], &surface, tolerance);
        let mesh = trimming_tessellation(&surface, &boundary, tolerance);
        // (5),(6),(7) at least one triangle, no zero-area, consistent orientation.
        assert_no_zero_area_or_orientation_flip(&mesh);

        // (8) Reverse traversal: walk the same collapse the other way. The
        // singular 3D point must still be preserved on both enter and leave,
        // with a bridge inserted on the leave.
        let mut rev: Vec<SurfacePoint> = [uv(0.5, 1.0)].into_iter().map(sp).collect();
        let prev_uv = uv(0.5, 1.0);
        let prev_pt = surface.subs(prev_uv.x, prev_uv.y);
        let mut enter_uv = uv(1.0, 0.9);
        let singular_pt = surface.subs(enter_uv.x, enter_uv.y);
        assert!(singular_pt.near(&Point3::new(0.0, 1.0, 0.0)));
        reconcile_singular_transition(
            &surface,
            prev_uv,
            prev_pt,
            &mut enter_uv,
            singular_pt,
            tolerance,
            &mut rev,
        );
        assert!(
            enter_uv.near(&uv(1.0, 1.0)),
            "reverse enter preserves the singular row"
        );
        rev.push((enter_uv, singular_pt).into());
        let mut leave_uv = uv(0.5, 0.0);
        let leave_pt = surface.subs(leave_uv.x, leave_uv.y);
        reconcile_singular_transition(
            &surface,
            enter_uv,
            singular_pt,
            &mut leave_uv,
            leave_pt,
            tolerance,
            &mut rev,
        );
        let rev_bridge = *rev.last().unwrap();
        assert!(
            rev_bridge.point.near(&singular_pt),
            "reverse leave bridges with the singular 3D point"
        );
        assert!(
            rev.iter()
                .all(|p| surface.subs(p.uv.x, p.uv.y).distance(p.point) <= tolerance),
            "reverse traversal: every point re-evaluates within tolerance"
        );
    }

    /// Item 9: the u-direction enter/leave branches on a surface whose `u`
    /// collapses at `v = 1`.
    #[test]
    fn singular_transition_both_parameter_directions() {
        let surface = CollapsedPatchU;
        let tolerance = 1.0e-9;
        let uv = |u, v| Point2::new(u, v);
        let sp = |p: Point2| (p, surface.subs(p.x, p.y)).into();
        let mut points: Vec<SurfacePoint> =
            [uv(1.0, 0.0), uv(0.5, 0.0), uv(0.0, 0.0), uv(0.0, 0.5)]
                .into_iter()
                .map(sp)
                .collect();
        let previous_uv = uv(0.0, 0.5);
        let previous_point = surface.subs(previous_uv.x, previous_uv.y);
        let mut entering_uv = uv(1.100_620_084_301_750_6, 1.0);
        let singular_point = surface.subs(entering_uv.x, entering_uv.y);
        assert!(singular_point.near(&Point3::new(0.0, 1.0, 0.0)));
        reconcile_singular_transition(
            &surface,
            previous_uv,
            previous_point,
            &mut entering_uv,
            singular_point,
            tolerance,
            &mut points,
        );
        assert!(
            entering_uv.near(&uv(0.0, 1.0)),
            "u-collapse enter preserves the incoming u branch"
        );
        points.push((entering_uv, singular_point).into());

        let mut leaving_uv = uv(1.0, 0.5);
        let leaving_point = surface.subs(leaving_uv.x, leaving_uv.y);
        reconcile_singular_transition(
            &surface,
            entering_uv,
            singular_point,
            &mut leaving_uv,
            leaving_point,
            tolerance,
            &mut points,
        );
        let bridge = *points.last().unwrap();
        assert!(bridge.uv.near(&uv(1.0, 1.0)));
        assert!(bridge.point.near(&singular_point));
        points.push((leaving_uv, leaving_point).into());
        assert!(
            points
                .iter()
                .all(|p| surface.subs(p.uv.x, p.uv.y).distance(p.point) <= tolerance),
            "u-direction: every point re-evaluates within tolerance"
        );
    }

    /// Item 10: a derivative that is small but whose candidate UV does NOT
    /// reconstruct the associated 3D point must not be substituted.
    #[test]
    fn singular_transition_negative_case_rejects_non_reconstructing_candidate() {
        let surface = CollapsedPatch;
        let tolerance = 1.0e-9;
        // At u = 1 - delta the v-derivative has magnitude delta < TOLERANCE, so
        // `so_small()` proposes a substitution, but the point still genuinely
        // depends on v, so the residual check must reject it.
        let delta = 5.0e-7;
        let previous_uv = Point2::new(0.5, 0.0);
        let previous_point = surface.subs(previous_uv.x, previous_uv.y);
        let raw_uv = Point2::new(1.0 - delta, 0.6);
        let raw_point = surface.subs(raw_uv.x, raw_uv.y);
        // Sanity: the v-derivative really is small here, and the candidate
        // really would move the point beyond tolerance.
        assert!(surface.vder(raw_uv.x, raw_uv.y).so_small());
        let candidate = Point2::new(raw_uv.x, previous_uv.y);
        assert!(
            surface.subs(candidate.x, candidate.y).distance(raw_point) > tolerance,
            "test setup: candidate must fail the residual check"
        );

        let mut current_uv = raw_uv;
        let mut out: Vec<SurfacePoint> = Vec::new();
        reconcile_singular_transition(
            &surface,
            previous_uv,
            previous_point,
            &mut current_uv,
            raw_point,
            tolerance,
            &mut out,
        );
        // The candidate did not reconstruct the point: nothing changed.
        assert!(
            current_uv.near(&raw_uv),
            "helper must not substitute a non-reconstructing candidate"
        );
        assert!(
            out.is_empty(),
            "no bridge may be appended when the residual check fails"
        );
    }
}
