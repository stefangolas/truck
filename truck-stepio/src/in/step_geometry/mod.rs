use crate::{self as truck_stepio};
use derive_more::From;
use serde::{Deserialize, Serialize};
use truck_derivers::{DisplayByStep, StepCurve, StepLength, StepSurface};

/// re-export structs in `truck-geometry` and `truck-polymesh`.
pub mod re_exports {
    pub use truck_geometry::prelude::*;
    pub use truck_polymesh::*;
}
pub use re_exports::*;

/// Errors that occur when converting STEP format
pub type StepConvertingError = Box<dyn std::error::Error>;

/// `ellipse`, realized in `truck`
pub type Ellipse<P, M> = Processor<TrimmedCurve<UnitCircle<P>>, M>;
/// `hyperbola`, realized in `truck`
pub type Hyperbola<P, M> = Processor<TrimmedCurve<UnitHyperbola<P>>, M>;
/// `parabola`, realized in `truck`
pub type Parabola<P, M> = Processor<TrimmedCurve<UnitParabola<P>>, M>;
/// `spherical_surface`, realized in `truck`
pub type SphericalSurface = Processor<Sphere, Matrix4>;
/// `cylindrical_surface`, realized in `truck`
pub type CylindricalSurface = Processor<RevolutedCurve<Line<Point3>>, Matrix4>;
/// `toroidal_surface`, realized in `truck`
pub type ToroidalSurface = Processor<Torus, Matrix4>;
/// `conical_surface`, realized in `truck`
pub type ConicalSurface = Processor<RevolutedCurve<Line<Point3>>, Matrix4>;
/// `surface_of_linear_extrusion`, realized in `truck`
pub type StepExtrudedCurve = ExtrudedCurve<Curve3D, Vector3>;
/// `surface_of_revolution`, realized in `truck`
pub type StepRevolutedCurve = Processor<RevolutedCurve<Curve3D>, Matrix4>;
/// `pcurve`, realized in `truck`
pub type PCurve = truck_geometry::prelude::PCurve<Box<Curve2D>, Box<Surface>>;

/// `conic` in 2D, realized in `truck`
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    From,
    Serialize,
    Deserialize,
    ParametricCurve,
    BoundedCurve,
    Cut,
    Invertible,
    ParameterDivision1D,
    SearchParameterD1,
    SearchNearestParameterD1,
    TransformedM3,
    SelfSameGeometry,
    StepLength,
    DisplayByStep,
    StepCurve,
)]
pub enum Conic2D {
    Ellipse(Ellipse<Point2, Matrix3>),
    Hyperbola(Hyperbola<Point2, Matrix3>),
    Parabola(Parabola<Point2, Matrix3>),
}

/// `curve` in 2D, realized in `truck`
#[derive(
    Clone,
    Debug,
    PartialEq,
    From,
    Serialize,
    Deserialize,
    ParametricCurve,
    BoundedCurve,
    Cut,
    Invertible,
    ParameterDivision1D,
    SearchParameterD1,
    SearchNearestParameterD1,
    TransformedM3,
    SelfSameGeometry,
    StepLength,
    DisplayByStep,
    StepCurve,
)]

pub enum Curve2D {
    Line(Line<Point2>),
    Polyline(PolylineCurve<Point2>),
    Conic(Conic2D),
    BSplineCurve(BSplineCurve<Point2>),
    NurbsCurve(NurbsCurve<Vector3>),
}

/// `conic` in 3D, realized in `truck`
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    From,
    Serialize,
    Deserialize,
    ParametricCurve,
    BoundedCurve,
    Cut,
    Invertible,
    ParameterDivision1D,
    SearchParameterD1,
    SearchNearestParameterD1,
    TransformedM4,
    SelfSameGeometry,
    StepLength,
    DisplayByStep,
    StepCurve,
)]
pub enum Conic3D {
    Ellipse(Ellipse<Point3, Matrix4>),
    Hyperbola(Hyperbola<Point3, Matrix4>),
    Parabola(Parabola<Point3, Matrix4>),
}

/// `curve` in 3D, realized in `truck`
#[derive(
    Clone,
    Debug,
    PartialEq,
    From,
    Serialize,
    Deserialize,
    ParametricCurve,
    BoundedCurve,
    Cut,
    Invertible,
    ParameterDivision1D,
    SearchParameterD1,
    SearchNearestParameterD1,
    TransformedM4,
    SelfSameGeometry,
    StepLength,
    DisplayByStep,
    StepCurve,
)]
pub enum Curve3D {
    Line(Line<Point3>),
    Polyline(PolylineCurve<Point3>),
    Conic(Conic3D),
    BSplineCurve(BSplineCurve<Point3>),
    PCurve(PCurve),
    NurbsCurve(NurbsCurve<Vector4>),
}

/// `elementary_surface`, realized in `truck`
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Serialize,
    Deserialize,
    ParametricSurface3D,
    ParameterDivision2D,
    SearchParameterD2,
    SearchNearestParameterD2,
    Invertible,
    TransformedM4,
    SelfSameGeometry,
    StepLength,
    StepSurface,
)]
pub enum ElementarySurface {
    Plane(Plane),
    Sphere(SphericalSurface),
    CylindricalSurface(CylindricalSurface),
    ToroidalSurface(ToroidalSurface),
    ConicalSurface(ConicalSurface),
}

/// `swept_surface`, realized in `truck`
#[derive(
    Clone,
    Debug,
    From,
    PartialEq,
    Serialize,
    Deserialize,
    ParametricSurface3D,
    ParameterDivision2D,
    SearchParameterD2,
    SearchNearestParameterD2,
    Invertible,
    TransformedM4,
    SelfSameGeometry,
    StepLength,
    DisplayByStep,
    StepSurface,
)]
pub enum SweptCurve {
    ExtrudedCurve(StepExtrudedCurve),
    RevolutedCurve(StepRevolutedCurve),
}

/// `offset_surface`, realized in `truck`
///
/// STEP defines an offset surface as its basis displaced along the basis'
/// own unit normal by a constant distance. `truck` expresses exactly that as
/// an `Offset` over a `NormalField`, which carries the derivatives of the
/// normal that the tessellator needs; this newtype exists only because those
/// types belong to another crate and the traits below are implemented here.
///
/// The basis is held twice, once as the entity being offset and once inside
/// the normal field. That is a copy of the basis geometry per offset face.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepOffsetSurface {
    inner: Offset<Box<Surface>, NormalField<Box<Surface>, f64>>,
    distance: f64,
}

impl StepOffsetSurface {
    /// Offset `basis` by `distance` along its unit normal.
    #[inline(always)]
    pub fn new(basis: Surface, distance: f64) -> Self {
        let basis = Box::new(basis);
        Self {
            inner: Offset::new(basis.clone(), NormalField::new(basis, distance)),
            distance,
        }
    }
    /// The surface this one is offset from.
    #[inline(always)]
    pub fn basis(&self) -> &Surface {
        self.inner.entity()
    }
    /// The signed offset distance.
    #[inline(always)]
    pub fn distance(&self) -> f64 {
        self.distance
    }
}

impl ParametricSurface for StepOffsetSurface {
    type Point = Point3;
    type Vector = Vector3;
    #[inline(always)]
    fn der_mn(&self, m: usize, n: usize, u: f64, v: f64) -> Self::Vector {
        self.inner.der_mn(m, n, u, v)
    }
    #[inline(always)]
    fn subs(&self, u: f64, v: f64) -> Self::Point {
        self.inner.subs(u, v)
    }
    #[inline(always)]
    fn uder(&self, u: f64, v: f64) -> Self::Vector {
        self.inner.uder(u, v)
    }
    #[inline(always)]
    fn vder(&self, u: f64, v: f64) -> Self::Vector {
        self.inner.vder(u, v)
    }
    #[inline(always)]
    fn uuder(&self, u: f64, v: f64) -> Self::Vector {
        self.inner.uuder(u, v)
    }
    #[inline(always)]
    fn uvder(&self, u: f64, v: f64) -> Self::Vector {
        self.inner.uvder(u, v)
    }
    #[inline(always)]
    fn vvder(&self, u: f64, v: f64) -> Self::Vector {
        self.inner.vvder(u, v)
    }
    #[inline(always)]
    fn parameter_range(&self) -> (ParameterRange, ParameterRange) {
        self.inner.entity().parameter_range()
    }
}

impl ParametricSurface3D for StepOffsetSurface {
    #[inline(always)]
    fn normal(&self, u: f64, v: f64) -> Vector3 {
        self.inner.normal(u, v)
    }
}

impl ParameterDivision2D for StepOffsetSurface {
    #[inline(always)]
    fn parameter_division(
        &self,
        range: ((f64, f64), (f64, f64)),
        tol: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        algo::surface::parameter_division(self, range, tol)
    }
}

impl StepOffsetSurface {
    /// Turn a hint into a concrete starting parameter.
    ///
    /// The search must run against this surface rather than the basis: a
    /// boundary point lies on the offset, so the basis would place it a
    /// distance away. Only the starting guess is taken from the basis, whose
    /// parametrisation this surface shares.
    fn resolve_hint(&self, point: Point3, hint: SPHint2D) -> (f64, f64) {
        const DIVISION: usize = 50;
        match hint {
            SPHint2D::Parameter(u, v) => (u, v),
            SPHint2D::Range(urange, vrange) => {
                algo::surface::presearch(self, point, (urange, vrange), DIVISION)
            }
            SPHint2D::None => {
                let (urange, vrange) = self.basis().try_range_tuple();
                let range = (urange.unwrap_or((0.0, 1.0)), vrange.unwrap_or((0.0, 1.0)));
                algo::surface::presearch(self, point, range, DIVISION)
            }
        }
    }
}

impl SearchParameter<D2> for StepOffsetSurface {
    type Point = Point3;
    #[inline(always)]
    fn search_parameter<H: Into<SPHint2D>>(
        &self,
        point: Point3,
        hint: H,
        trials: usize,
    ) -> Option<(f64, f64)> {
        let hint = self.resolve_hint(point, hint.into());
        algo::surface::search_parameter(self, point, hint, trials)
    }
}

impl SearchNearestParameter<D2> for StepOffsetSurface {
    type Point = Point3;
    #[inline(always)]
    fn search_nearest_parameter<H: Into<SPHint2D>>(
        &self,
        point: Point3,
        hint: H,
        trials: usize,
    ) -> Option<(f64, f64)> {
        let hint = self.resolve_hint(point, hint.into());
        algo::surface::search_nearest_parameter(self, point, hint, trials)
    }
}

impl Invertible for StepOffsetSurface {
    /// Inverting flips the basis, which flips the normal the offset is
    /// measured along, so the distance changes sign to describe the same
    /// surface.
    #[inline(always)]
    fn invert(&mut self) {
        let mut basis = self.basis().clone();
        basis.invert();
        *self = Self::new(basis, -self.distance);
    }
}

impl Transformed<Matrix4> for StepOffsetSurface {
    /// The offset distance is a length in model space, so it only survives a
    /// transform unchanged when that transform preserves length. STEP assembly
    /// placements are rigid motions, which do; a scaling transform would need
    /// the distance scaled with it and is not expected here.
    #[inline(always)]
    fn transform_by(&mut self, trans: Matrix4) {
        let mut basis = self.basis().clone();
        basis.transform_by(trans);
        *self = Self::new(basis, self.distance);
    }
}

impl truck_stepio::out::StepLength for StepOffsetSurface {
    #[inline(always)]
    fn step_length(&self) -> usize {
        1
    }
}

impl truck_stepio::out::StepSurface for StepOffsetSurface {}

/// `surface`, realized in `truck`
#[derive(
    Clone,
    Debug,
    From,
    PartialEq,
    Serialize,
    Deserialize,
    ParametricSurface3D,
    ParameterDivision2D,
    SearchParameterD2,
    SearchNearestParameterD2,
    Invertible,
    TransformedM4,
    SelfSameGeometry,
    StepLength,
    StepSurface,
)]
pub enum Surface {
    ElementarySurface(ElementarySurface),
    SweptCurve(SweptCurve),
    BSplineSurface(BSplineSurface<Point3>),
    NurbsSurface(NurbsSurface<Vector4>),
    OffsetSurface(StepOffsetSurface),
}

impl truck_stepio::out::DisplayByStep for Surface {
    fn fmt(&self, idx: usize, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Surface::*;
        match self {
            ElementarySurface(x) => x.fmt(idx, f),
            SweptCurve(x) => x.fmt(idx, f),
            BSplineSurface(x) => x.fmt(idx, f),
            NurbsSurface(x) => x.fmt(idx, f),
            // look never writes STEP; an offset surface reaching the writer
            // would need its basis emitted first and referenced by index.
            OffsetSurface(_) => Err(std::fmt::Error),
        }
    }
}

/// `spherical_surface`, realized in `truck`
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, StepSurface)]
pub struct Sphere(pub truck_geometry::prelude::Sphere);

impl truck_stepio::out::StepSurface for Processor<Sphere, Matrix4> {
    #[inline(always)]
    fn same_sense(&self) -> bool {
        self.orientation()
    }
}

mod sphere;

/// Implementation required to apply a closed surface division to a shape parsed from a STEP file.
mod from_pcurve {
    use super::{Curve2D, Curve3D, Surface};
    use truck_geometry::prelude::*;

    impl From<PCurve<Line<Point2>, Surface>> for Curve3D {
        fn from(value: PCurve<Line<Point2>, Surface>) -> Self {
            let (line, surface) = value.decompose();
            Curve3D::PCurve(PCurve::new(Curve2D::Line(line).into(), surface.into()))
        }
    }
}

/// implementation for trait `truck_modeling::builder`.
mod geom_impls;
/// implementation for output STEP format.
mod stepout_impls;
