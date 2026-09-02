//! README API-surface example — scratch validation, not a fixture.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use truck_base::cgmath64::{Point3, Vector3};
use truck_base::evidence::Budget;
use truck_geometry::arrange::arrange;
use truck_geometry::canonical::{Curve, Surface};
use truck_geometry::prelude::*;
use truck_shapeops::facade::{self, Mode};
use truck_topology::Solid;

fn square() -> Vec<Curve> {
    vec![
        Curve::Line(Line(Point3::new(0.0, 0.0, 0.0), Point3::new(4.0, 0.0, 0.0))),
        Curve::Line(Line(Point3::new(4.0, 0.0, 0.0), Point3::new(4.0, 4.0, 0.0))),
        Curve::Line(Line(Point3::new(4.0, 4.0, 0.0), Point3::new(0.0, 4.0, 0.0))),
        Curve::Line(Line(Point3::new(0.0, 4.0, 0.0), Point3::new(0.0, 0.0, 0.0))),
    ]
}

#[test]
fn readme_example_compiles_and_runs()
-> std::result::Result<(), String> {
    let profile = square();

    let arrangement = arrange(&profile, None).map_err(|e| format!("{e:?}"))?.value;
    let block: Solid<Point3, Curve, Surface> =
        facade::extrude(&profile, &arrangement, 2.0).map_err(|e| format!("{e:?}"))?.value;

    let mut budget = Budget::new(1000, 1000, 1000);
    let moved =
        facade::translate(&block, Vector3::new(1.0, 0.0, 0.0)).map_err(|e| format!("{e:?}"))?.value;
    let _bb = facade::bounding_box(&moved, &mut budget).map_err(|e| format!("{e:?}"))?.value;

    match facade::revolve(&profile, &arrangement, std::f64::consts::TAU) {
        Ok(_certified) => {}
        Err(truck_base::evidence::Refusal::UnsupportedEnvelope(_)) => {}
        Err(_e) => {}
    }

    let mut budget2 = Budget::new(1000, 1000, 1000);
    let inner = vec![
        Curve::Line(Line(Point3::new(1.0, 1.0, 0.0), Point3::new(2.0, 1.0, 0.0))),
        Curve::Line(Line(Point3::new(2.0, 1.0, 0.0), Point3::new(2.0, 2.0, 0.0))),
        Curve::Line(Line(Point3::new(2.0, 2.0, 0.0), Point3::new(1.0, 2.0, 0.0))),
        Curve::Line(Line(Point3::new(1.0, 2.0, 0.0), Point3::new(1.0, 1.0, 0.0))),
    ];
    let arrangement2 = arrange(&inner, None).map_err(|e| format!("{e:?}"))?.value;
    let hole: Solid<Point3, Curve, Surface> =
        facade::extrude(&inner, &arrangement2, 2.0).map_err(|e| format!("{e:?}"))?.value;
    let _cut = facade::boolean_op(&block, Mode::Subtract, &hole, &mut budget2)
        .map_err(|e| format!("{e:?}"))?;
    Ok(())
}
