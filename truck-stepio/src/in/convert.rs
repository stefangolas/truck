use super::*;
use crate::common::PartAttrs;

impl Table {
    fn place_holder_edge_any_to_index_and_edge_curve(
        &self,
        edge: &PlaceHolder<EdgeAnyHolder>,
    ) -> Option<(u64, EdgeCurveHolder)> {
        use PlaceHolder::Ref;
        let Ref(Name::Entity(ref idx)) = edge else {
            return None;
        };
        self.oriented_edge
            .get(idx)
            .and_then(|oriented_edge| {
                Some((
                    oriented_edge.edge_element_idx()?,
                    oriented_edge.edge_element_holder(self)?,
                ))
            })
            .or_else(|| {
                self.edge_curve
                    .get(idx)
                    .map(|edge_curve| (*idx, edge_curve.clone()))
            })
    }
    fn face_any_to_orientation_and_face(
        &self,
        face: Option<FaceAnyHolder>,
    ) -> Option<(bool, FaceSurfaceHolder)> {
        let (orientation, _, face) = self.face_any_resolved(face)?;
        Some((orientation, face))
    }

    /// Resolve a shell's face reference to its orientation, the id of the
    /// `FACE_SURFACE` that *defines* it, and the definition itself.
    ///
    /// The definition id is deliberately kept apart from whatever the shell
    /// named to get here. An `ORIENTED_FACE` is a *use*: it contributes
    /// orientation, and several uses may resolve to one `FACE_SURFACE`.
    /// Reporting the use where the definition is meant — or the reverse — makes
    /// a wrong shell-use orientation indistinguishable from a wrong underlying
    /// face, which is precisely the distinction this stage exists to preserve.
    fn face_any_resolved(
        &self,
        face: Option<FaceAnyHolder>,
    ) -> Option<(bool, FaceReference, FaceSurfaceHolder)> {
        match face? {
            // The shell named the definition directly. There is no use entity
            // in this file at all, and claiming one — by copying the shell's
            // reference into the use slot — would be the very conflation this
            // split exists to remove.
            FaceAnyHolder::FaceSurface(face) => Some((true, FaceReference::Definition, face)),
            FaceAnyHolder::OrientedFace(oriented_face) => {
                let definition_id = match &oriented_face.face_element {
                    PlaceHolder::Ref(Name::Entity(idx)) => Some(*idx),
                    // An inlined definition has no id to report.
                    _ => None,
                };
                let face_element = oriented_face.face_element_holder(self)?;
                Some((
                    oriented_face.orientation,
                    FaceReference::Use { definition_id },
                    face_element,
                ))
            }
        }
    }

    fn shell_vertices(&self, shell: &ShellHolder) -> Arena<VertexKind, Point3> {
        use PlaceHolder::Ref;
        // This carried the reserve-before-convert defect that was already fixed
        // for edges: the position was inserted into the map, and only then was
        // `get_owned` called, which can fail. The point was never pushed but the
        // entry stayed, so every subsequent vertex was addressed one slot past
        // where it sat. The arena removes the ordering question entirely — there
        // is no way to express the claim before the conversion.
        let mut arena = Arena::new();
        let vertex_holders = shell
            .cfs_faces_holder(self)
            .filter_map(move |(_, face)| self.face_any_to_orientation_and_face(face))
            .flat_map(move |(_, face)| face.bounds_holder(self))
            .filter_map(move |bound| match bound?.bound_holder(self)? {
                FaceBoundLoop::Edges(loop_) => Some(loop_),
                // A collapsed bound has no edges to contribute.
                FaceBoundLoop::Collapsed(_) => None,
            })
            .flat_map(move |bound| bound.edge_list)
            .filter_map(move |edge| self.place_holder_edge_any_to_index_and_edge_curve(&edge))
            .flat_map(move |(_, edge)| [edge.edge_start, edge.edge_end]);
        for holder in vertex_holders {
            let Ref(Name::Entity(idx)) = holder else {
                continue;
            };
            arena.get_or_try_insert(VertexPointId::new(idx), || {
                let point = EntityTable::<VertexPointHolder>::get_owned(self, idx)
                    .map_err(|e| eprintln!("{e}"))
                    .ok()?;
                Some(Point3::from(&point.vertex_geometry))
            });
        }
        arena
    }

    /// The `TRUCK_PROBE_IDENTITY` probe used to live here, checking after the
    /// fact that every mapped index addressed the edge it named. It is gone
    /// because the question it asked can no longer have a bad answer:
    /// [`Arena::get_or_try_insert`] converts before it claims a position, so
    /// map and vector cannot disagree (`TOP-002`). That is the promotion from
    /// diagnostic to certificate the plan calls for — the check moved from
    /// runtime to the shape of the code, and its regression lives in
    /// `arena::tests`.
    fn shell_edges(
        &self,
        shell: &ShellHolder,
        vertices: &Arena<VertexKind, Point3>,
    ) -> Arena<EdgeKind, CompressedEdge<Curve3D>> {
        use PlaceHolder::Ref;
        let mut arena = Arena::new();
        let edge_curves = shell
            .cfs_faces_holder(self)
            .filter_map(move |(_, face)| self.face_any_to_orientation_and_face(face))
            .flat_map(move |(_, face)| face.bounds_holder(self))
            .filter_map(move |bound| match bound?.bound_holder(self)? {
                FaceBoundLoop::Edges(loop_) => Some(loop_),
                // A collapsed bound has no edges to contribute.
                FaceBoundLoop::Collapsed(_) => None,
            })
            .flat_map(move |bound| bound.edge_list)
            .filter_map(move |edge| self.place_holder_edge_any_to_index_and_edge_curve(&edge));
        for (idx, edge) in edge_curves {
            arena.get_or_try_insert(EdgeCurveId::new(idx), move || {
                let edge_curve = edge
                    .clone()
                    .into_owned(self)
                    .map_err(|e| eprintln!("{e}"))
                    .ok()?;
                let curve = edge_curve
                    .parse_curve3d()
                    .map_err(|e| eprintln!("{e}"))
                    .ok()?;
                let Ref(Name::Entity(front_idx)) = edge.edge_start else {
                    return None;
                };
                let Ref(Name::Entity(back_idx)) = edge.edge_end else {
                    return None;
                };
                // An edge whose endpoints did not convert is not an edge. The
                // bare `usize` pair is what `CompressedEdge` demands, so this is
                // one of the few places a `VertexIndex` has to be unwrapped, and
                // it happens only after the lookup proved the vertex exists.
                let endpoints = (
                    vertices.index_of(VertexPointId::new(front_idx))?.position(),
                    vertices.index_of(VertexPointId::new(back_idx))?.position(),
                );
                Some(CompressedEdge {
                    vertices: endpoints,
                    curve,
                })
            });
        }
        arena
    }
    /// Follow a source reference to the position of the edge it names, having
    /// checked that the position addresses that edge and no other.
    ///
    /// `TOP-001` at the point of use (`MATHEMATICAL_FOUNDATION.md` §22.2). The
    /// arena's construction already makes the answer right, so this compare
    /// never fires in a correct build; what it buys is that if it ever does,
    /// the model reports which entity was asked for and which one was stored,
    /// instead of rendering a curve from a neighbouring surface as a smooth
    /// wrong region. One integer comparison per edge use — structural tier.
    fn checked_edge_position(
        edges: &Arena<EdgeKind, CompressedEdge<Curve3D>>,
        named: EdgeCurveId,
    ) -> Option<usize> {
        let index = edges.index_of(named)?;
        if let Err(mismatch) = edges.get_checked(index, named) {
            eprintln!("{mismatch}");
            return None;
        }
        Some(index.position())
    }

    fn face_bound_to_edges(
        &self,
        bound: FaceBoundHolder,
        edges: &Arena<EdgeKind, CompressedEdge<Curve3D>>,
    ) -> Result<BoundOutcome, FaceLossReason> {
        use PlaceHolder::Ref;
        let ori = bound.orientation;
        let bound = match bound
            .bound_holder(self)
            .ok_or(FaceLossReason::LoopReferenceUnresolved)?
        {
            FaceBoundLoop::Edges(loop_) => loop_,
            // A collapsed boundary trims nothing. The apex or pole is closed by
            // the surface's own degeneracy, so the honest contribution is no
            // trim segment at all — not a synthesised loop of zero size, which
            // would trim the face by an empty region and delete it.
            FaceBoundLoop::Collapsed(vl) => {
                let pt = match &vl.loop_vertex {
                    PlaceHolder::Ref(Name::Entity(v_idx)) => {
                        EntityTable::<VertexPointHolder>::get_owned(self, *v_idx)
                            .map(|p| Point3::from(&p.vertex_geometry))
                            .unwrap_or_else(|_| Point3::origin())
                    }
                    _ => Point3::origin(),
                };
                return Ok(BoundOutcome::Collapsed(pt));
            }
        };
        // A bound missing an edge is a broken bound, not a shorter one.
        //
        // This collected through `filter_map`, so every `?` below dropped that
        // edge from the wire and let the rest through. The result is a wire that
        // no longer closes: the gap is bridged by whatever the next stage joins
        // it to, and the face is then trimmed by a region its own file never
        // described. Nothing downstream can detect this, because a short wire is
        // indistinguishable from a wire that was always that shape.
        //
        // Collecting into `Option<Vec<_>>` makes the conversion total — every
        // `ORIENTED_EDGE` the bound names resolves, or the bound does not exist.
        // That also discharges the source-use versus resolved-use count check by
        // construction rather than by assertion: the collect yields exactly as
        // many indices as `edge_list` held, or it yields nothing.
        let source_uses = bound.edge_list.len();
        let mut wire: Vec<CompressedEdgeIndex> = bound
            .edge_list
            .into_iter()
            .map(|edge| {
                let Ref(Name::Entity(ref idx)) = edge else {
                    return Err(FaceLossReason::EdgeUseUnresolved);
                };
                let edge_idx = if let Some(oriented_edge) = self.oriented_edge.get(idx) {
                    let named = EdgeCurveId::new(
                        oriented_edge
                            .edge_element_idx()
                            .ok_or(FaceLossReason::EdgeUseUnresolved)?,
                    );
                    CompressedEdgeIndex {
                        index: Self::checked_edge_position(edges, named)
                            .ok_or(FaceLossReason::EdgeCurveConversionFailed)?,
                        orientation: oriented_edge.orientation == ori,
                    }
                } else {
                    CompressedEdgeIndex {
                        index: Self::checked_edge_position(edges, EdgeCurveId::new(*idx))
                            .ok_or(FaceLossReason::EdgeCurveConversionFailed)?,
                        orientation: ori,
                    }
                };
                Ok(edge_idx)
            })
            .collect::<Result<Vec<_>, FaceLossReason>>()?;
        debug_assert!(
            wire.is_empty() || wire.len() == source_uses,
            "a resolved bound must use every edge its source named"
        );
        if !ori {
            wire.reverse();
        }
        // Every edge resolved, but that does not make this a boundary. The wire
        // has to close, and it is checked in traversal order — after the
        // reversal, since that is the order the face will be trimmed in.
        TopologicallyClosedWire::try_new(wire, |position| {
            edges.value_at(position).map(|edge| edge.vertices)
        })
        .map(BoundOutcome::Wire)
        .ok_or(FaceLossReason::WireNotClosed)
    }

    /// The supporting surface of a face, converted once per source entity.
    ///
    /// Surfaces are the third entity kind to resolve through the one generic
    /// [`Arena`], and §51a is the whole reason it is that arena rather than a
    /// third hand-written map: the reserve-before-convert defect was repaired
    /// in the edge path and survived untouched in the vertex path for exactly
    /// as long as the repair was site-local. A contract is discharged when the
    /// invalid transition has one implementation that cannot express the bad
    /// state — not when every site anyone has looked at is correct.
    ///
    /// The arena stores the surface **as the file describes it**. `same_sense`
    /// inversion is a property of the *face*, so it is applied to the copy a
    /// face takes and never to the canonical entity: two faces may share one
    /// `CYLINDRICAL_SURFACE` and disagree about sense, and inverting in place
    /// would let the first face rewrite the second's geometry.
    ///
    /// **Cost** (§29a): one retained surface per distinct source entity for the
    /// duration of one shell, against one conversion per *face* before. Shared
    /// surfaces now convert once; the arena is dropped when the shell's faces
    /// are built. `CompressedFace` still owns its surface by value, so the copy
    /// is unavoidable until §33a item 11 replaces it with a `SurfaceIndex`.
    ///
    /// **Contracts:** `TOP-001`, `TOP-002`, `TOP-007` for surfaces.
    fn face_surface(
        &self,
        face: &FaceSurfaceHolder,
        surfaces: &mut Arena<SurfaceKind, Surface>,
    ) -> Option<(Surface, Option<u64>)> {
        let convert = || {
            let step_surface: SurfaceAny = face
                .face_geometry
                .clone()
                .into_owned(self)
                .map_err(|e| eprintln!("{e}"))
                .ok()?;
            Surface::try_from(&step_surface)
                .map_err(|e| eprintln!("{e}"))
                .ok()
        };
        // An inline owned surface has no entity id, so there is no identity to
        // be canonical about: it belongs to this face alone, and it has no
        // surface provenance to report.
        let PlaceHolder::Ref(Name::Entity(idx)) = &face.face_geometry else {
            return Some((convert()?, None));
        };
        let named = SurfaceId::new(*idx);
        let index = surfaces.get_or_try_insert(named, convert)?;
        let surface = surfaces
            .get_checked(index, named)
            .map_err(|mismatch| eprintln!("{mismatch}"))
            .ok()
            .cloned()?;
        Some((surface, Some(*idx)))
    }

    fn tune_conical_surface(
        surface: &mut Surface,
        wires: &[TopologicallyClosedWire],
        edges: &Arena<EdgeKind, CompressedEdge<Curve3D>>,
        collapsed_pts: &[Point3],
    ) {
        use cgmath::Transform;
        use step_geometry::*;
        let Surface::ElementarySurface(ElementarySurface::ConicalSurface(ref mut processor)) =
            surface
        else {
            return;
        };

        let mut points: Vec<Point3> = collapsed_pts.to_vec();
        for wire in wires {
            for edge_idx in wire.edges() {
                if let Some(edge) = edges.value_at(edge_idx.index) {
                    let (t0, t1) = edge.curve.range_tuple();
                    points.push(edge.curve.subs(t0));
                    points.push(edge.curve.subs(t1));
                    points.push(edge.curve.subs(t0 * 0.5 + t1 * 0.5));
                }
            }
        }

        if points.is_empty() {
            return;
        }

        let inv_mat = match processor.transform().invert() {
            Some(m) => m,
            None => return,
        };

        let rev = processor.entity();
        let line = rev.entity_curve();
        let p0 = line.0;
        let p1 = line.1;
        let dr = p1.x - p0.x;
        let dz = p1.z - p0.z;

        if dz.abs() < 1.0e-12 {
            return;
        }

        let tan = dr / dz;
        let r0 = p0.x - p0.z * tan;
        let z_apex = if tan.abs() > 1.0e-12 { -r0 / tan } else { 0.0 };

        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;

        for p in &points {
            let p_local = inv_mat.transform_point(*p);
            let z = p_local.z;
            z_min = z_min.min(z);
            z_max = z_max.max(z);
        }

        if !collapsed_pts.is_empty() || (z_min.is_finite() && (z_min - z_apex).abs() < 1.0e-3) {
            z_min = z_min.min(z_apex);
        }

        if !z_min.is_finite() || !z_max.is_finite() {
            z_min = z_apex.min(0.0);
            z_max = z_apex.max(0.0) + 1.0;
        }

        let span = (z_max - z_min).max(1.0);
        let pad = (0.2 * span).max(0.1);
        let u_min = z_min - pad;
        let u_max = z_max + pad;

        let new_p0 = Point3::new(r0 + u_min * tan, 0.0, u_min);
        let new_p1 = Point3::new(r0 + u_max * tan, 0.0, u_max);
        let new_rev = RevolutedCurve::by_revolution(
            Line(new_p0, new_p1),
            Point3::origin(),
            Vector3::unit_z(),
        );

        *processor.entity_mut() = new_rev;
    }

    /// Convert a shell's faces, and say why each one that failed did.
    ///
    /// The reasons come from the real conversion, not from a second
    /// reimplementation of it that reports instead of converting. That
    /// distinction is the whole design: a diagnostic which re-derives the
    /// pipeline can disagree with the pipeline, and this project has twice had
    /// a detector carrying the bug it was hunting. There is one path, and it
    /// either produces a face or a reason.
    fn shell_faces(
        &self,
        shell: &ShellHolder,
        edges: &Arena<EdgeKind, CompressedEdge<Curve3D>>,
        losses: &mut Vec<FaceLoss>,
        singular: &mut Vec<FaceProvenance>,
    ) -> Vec<CompressedFace<Surface>> {
        let mut surfaces = Arena::<SurfaceKind, Surface>::new();
        let mut faces = Vec::new();
        // One explicit loop rather than chained `filter_map`s: two closures
        // cannot both hold `&mut losses`, and threading a cell through to keep
        // the iterator style would be ceremony in exchange for nothing.
        for (shell_ref, face) in shell.cfs_faces_holder(self) {
            let Some((orientation, reference, face)) = self.face_any_resolved(face) else {
                // The shell named something that is not a face it can resolve.
                // All that is known is the reference itself.
                losses.push(FaceLoss {
                    provenance: FaceProvenance {
                        use_id: shell_ref.map(SourceEntityId::new),
                        ..FaceProvenance::default()
                    },
                    reason: FaceLossReason::FaceReferenceUnresolved,
                });
                continue;
            };
            // The shell's own reference lands in whichever slot it actually
            // names, and the other stays empty unless the file supplied it.
            let (use_id, definition_id) = match reference {
                FaceReference::Definition => (None, shell_ref),
                FaceReference::Use { definition_id } => (shell_ref, definition_id),
            };
            let partial = FaceProvenance {
                use_id: use_id.map(SourceEntityId::new),
                definition_id: definition_id.map(SourceEntityId::new),
                surface_id: None,
                // Established below, once the bounds have been read.
                outer_bound: OuterBoundStanding::NotRetained,
            };
            let Some((mut surface, surface_id)) = self.face_surface(&face, &mut surfaces) else {
                losses.push(FaceLoss {
                    provenance: partial,
                    reason: FaceLossReason::SurfaceConversionFailed,
                });
                continue;
            };
            let provenance = FaceProvenance {
                surface_id: surface_id.map(SourceEntityId::new),
                ..partial
            };
            if !face.same_sense && std::env::var_os("TRUCK_NO_INVERT").is_none() {
                surface.invert()
            }
            // Same rule one level up: a face missing a bound is a broken face,
            // not a simpler one. Dropping a failed bound here silently rewrites
            // what the solid is -- lose an inner bound and a hole fills in, lose
            // the outer bound and the remaining holes are read as the outline.
            // Both mesh perfectly happily.
            let outcomes = face
                .bounds_holder(self)
                .into_iter()
                .map(|bound| {
                    let bound = bound.ok_or(FaceLossReason::BoundReferenceUnresolved)?;
                    self.face_bound_to_edges(bound, edges)
                })
                .collect::<Result<Vec<BoundOutcome>, FaceLossReason>>();
            let outcomes = match outcomes {
                Ok(outcomes) => outcomes,
                Err(reason) => {
                    losses.push(FaceLoss { provenance, reason });
                    continue;
                }
            };
            let collapsed_pts: Vec<Point3> = outcomes
                .iter()
                .filter_map(|o| match o {
                    BoundOutcome::Collapsed(pt) => Some(*pt),
                    _ => None,
                })
                .collect();
            let collapsed = collapsed_pts.len();
            // The outer-bound standing, computed against the *surviving* wires
            // because `boundaries` below is the filtered list and an index into
            // the unfiltered one would name a different bound. A collapsed
            // outer bound is not silently reassigned to a neighbour: it simply
            // leaves no index, and the face reports `NoneDeclared`.
            let outer_flags = face.bound_outer_flags(self);
            let outer_bound = outer_bound_standing(&outer_flags, &outcomes);
            let provenance = FaceProvenance {
                outer_bound,
                ..provenance
            };
            let wires: Vec<TopologicallyClosedWire> = outcomes
                .into_iter()
                .filter_map(|o| match o {
                    BoundOutcome::Wire(wire) => Some(wire),
                    BoundOutcome::Collapsed(_) => None,
                })
                .collect();
            if wires.is_empty() {
                // Every bound collapsed, so nothing describes where this face
                // ends. A cone that is only an apex is not a face, and trimming
                // by no boundary at all would emit the entire unbounded surface
                // — the blob failure mode this project exists to avoid.
                losses.push(FaceLoss {
                    provenance,
                    reason: FaceLossReason::AllBoundsCollapsed,
                });
                continue;
            }
            if collapsed > 0 {
                // Recorded rather than silently absorbed: the face is now
                // rendered, but its domain has a singular point that nothing
                // downstream is told about. QUO-005 wants a type here.
                singular.push(provenance);
            }
            Self::tune_conical_surface(&mut surface, &wires, edges, &collapsed_pts);
            faces.push(CompressedFace {
                surface,
                // The proof is discharged here and nowhere earlier: truck's
                // `CompressedFace` takes bare index vectors, so this is the
                // boundary at which the guarantee stops travelling. Closing it
                // is §33a item 11 -- `boundaries` becomes
                // `Vec<TopologicallyClosedWire>`, which the owned-fork decision
                // (§31a) now permits.
                boundaries: wires
                    .into_iter()
                    .map(TopologicallyClosedWire::into_edges)
                    .collect(),
                orientation,
                // The whole reference chain, not one collapsed id. Every later
                // complaint about this face can name entities a reader can grep
                // out of the source file, and can say which layer it means.
                provenance,
            });
        }
        faces
    }

    /// Constructs `CompressedShell` of `truck` from `Shell` in STEP file
    /// # Example
    /// ```
    /// use truck_stepio::r#in::{*, step_geometry::*};
    /// // read file
    /// let step_string = include_str!(concat!(
    ///     env!("CARGO_MANIFEST_DIR"),
    ///     "/../resources/step/occt-cube.step",
    /// ));
    /// // parse into Rust structs
    /// let table = Table::from_step(&step_string).unwrap();
    /// // take one shell (this is only one shell)
    /// let step_shell = table.shell.values().next().unwrap();
    /// // convert STEP shell to `CompressedShell`
    /// let cshell = table.to_compressed_shell(step_shell).unwrap();
    /// // The cube has 6 faces!
    /// assert_eq!(cshell.faces.len(), 6);
    /// ```
    pub fn to_compressed_shell(
        &self,
        shell: &impl StepShell,
    ) -> Result<CompressedShell<Point3, Curve3D, Surface>, StepConvertingError> {
        shell.to_compressed_shell(self)
    }

    /// As [`Self::to_compressed_shell`], and also why each lost face was lost.
    pub fn to_compressed_shell_with_losses(
        &self,
        shell: &impl StepShell,
    ) -> Result<(CompressedShell<Point3, Curve3D, Surface>, Vec<FaceLoss>), StepConvertingError>
    {
        shell.to_compressed_shell_with_losses(self)
    }

    /// Constructs `CompressedShell`s of `truck` from `ShellBasedSurfaceModel` in STEP file
    pub fn to_compressed_shells(
        &self,
        shells: &ShellBasedSurfaceModelHolder,
    ) -> Result<Vec<CompressedShell<Point3, Curve3D, Surface>>, StepConvertingError> {
        let mut res = Vec::new();
        for place_holder in &shells.sbsm_boundary {
            let PlaceHolder::Ref(Name::Entity(idx)) = place_holder else {
                return Err("failed to reference an element of `sbsm_boundary`".into());
            };
            if let Some(shell) = self.shell.get(idx) {
                res.push(self.to_compressed_shell(shell)?);
            } else if let Some(oriented_shell) = self.oriented_shell.get(idx) {
                res.push(self.to_compressed_shell(oriented_shell)?);
            } else {
                return Err("failed to reference an element of `sbsm_boundary`".into());
            }
        }
        Ok(res)
    }

    /// Constructs `CompressedSolid` of `truck` from `ManifoldSolidBrep` in STEP file
    /// # Example
    /// ```
    /// use truck_stepio::r#in::{*, step_geometry::*};
    /// truck_topology::prelude!(Point3, Curve3D, Surface);
    /// // read file
    /// let step_string = include_str!(concat!(
    ///     env!("CARGO_MANIFEST_DIR"),
    ///     "/../resources/step/occt-cube.step",
    /// ));
    /// // parse into Rust structs
    /// let table = Table::from_step(&step_string).unwrap();
    /// // take the solid
    /// let step_solid = table.manifold_solid_brep.values().next().unwrap();
    /// // convert STEP shell to `CompressedSolid`
    /// let csolid = table.to_compressed_solid(step_solid).unwrap();
    /// // Convert to truck `Solid`
    /// let solid = Solid::extract(csolid).unwrap();
    /// // The cube has 6 faces!
    /// assert_eq!(solid.boundaries()[0].len(), 6);
    /// ```
    pub fn to_compressed_solid(
        &self,
        solid: &ManifoldSolidBrepHolder,
    ) -> Result<CompressedSolid<Point3, Curve3D, Surface>, StepConvertingError> {
        let PlaceHolder::Ref(Name::Entity(outer_idx)) = &solid.outer else {
            return Err("failed to reference `solid.outer`".into());
        };
        let outer_shell = if let Some(step_shell) = self.shell.get(outer_idx) {
            self.to_compressed_shell(step_shell)
        } else if let Some(step_shell) = self.oriented_shell.get(outer_idx) {
            self.to_compressed_shell(step_shell)
        } else {
            Err("failed to reference `solid.outer`".into())
        }?;
        let mut boundaries = vec![outer_shell];
        for shell in &solid.voids {
            let PlaceHolder::Ref(Name::Entity(outer_idx)) = shell else {
                return Err("failed to reference an element of `solid.voids`".into());
            };
            let Some(oriented_shell) = self.oriented_shell.get(outer_idx) else {
                return Err("failed to reference an element of `solid.voids`".into());
            };
            boundaries.push(self.to_compressed_shell(oriented_shell)?);
        }
        Ok(CompressedSolid { boundaries })
    }
}

#[derive(Clone, Debug, PartialEq, derive_more::From)]
pub enum NodeMatrix {
    Identity,
    Transform(Box<ItemDefinedTransformation>),
}

#[derive(Clone, Debug, PartialEq, derive_more::From)]
pub enum ProductShape {
    Shells(Vec<CompressedShell<Point3, Curve3D, Surface>>),
    Solid(CompressedSolid<Point3, Curve3D, Surface>),
    Matrix(Matrix4),
}

pub type ProductEntity = NodeEntity<Vec<ProductShape>, PartAttrs>;
pub type AssembleEntity = EdgeEntity<NodeMatrix, PartAttrs>;
pub type StepAssembly = Assembly<Vec<ProductShape>, PartAttrs, NodeMatrix, PartAttrs>;

impl TryFrom<&NodeMatrix> for Matrix3 {
    type Error = StepConvertingError;
    fn try_from(value: &NodeMatrix) -> Result<Self, Self::Error> {
        match value {
            NodeMatrix::Identity => Ok(Self::identity()),
            NodeMatrix::Transform(trans) => (&**trans).try_into(),
        }
    }
}

impl TryFrom<&NodeMatrix> for Matrix4 {
    type Error = StepConvertingError;
    fn try_from(value: &NodeMatrix) -> Result<Self, Self::Error> {
        match value {
            NodeMatrix::Identity => Ok(Self::identity()),
            NodeMatrix::Transform(trans) => (&**trans).try_into(),
        }
    }
}

impl ProductShape {
    pub fn try_from_index(idx: u64, table: &Table) -> Result<Self, StepConvertingError> {
        if let Some(step_solid) = table.manifold_solid_brep.get(&idx) {
            table.to_compressed_solid(step_solid).map(Into::into)
        } else if let Some(step_shells) = table.shell_based_surface_model.get(&idx) {
            table.to_compressed_shells(step_shells).map(Into::into)
        } else if table.axis2_placement_3d.contains_key(&idx) {
            let axis = EntityTable::<Axis2Placement3dHolder>::get_owned(table, idx)?;
            Ok(Matrix4::from(&axis).into())
        } else {
            Err("Unknown Shape".into())
        }
    }
}

impl Table {
    fn product_node_entity(
        &self,
        pds_idx: u64,
        pd: &ProductDefinitionHolder,
    ) -> Result<ProductEntity, StepConvertingError> {
        let PlaceHolder::Ref(Name::Entity(pdf_idx)) = &pd.formation else {
            return Err("failed to reference `product_definition.formation`".into());
        };
        let Some(pdf) = self.product_definition_formation.get(pdf_idx) else {
            return Err("failed to reference `prouct_definition_formation`".into());
        };
        let PlaceHolder::Ref(Name::Entity(p_idx)) = &pdf.of_product else {
            return Err("failed to reference `product_definition_formation.of_product`".into());
        };
        let Some(product) = self.product.get(p_idx) else {
            return Err("failed to reference `product`".into());
        };
        let attrs = PartAttrs {
            id: product.id.clone(),
            name: product.name.clone(),
            description: product.description.clone(),
        };

        let Some(sdr) = self.shape_definition_representation.values().find(|sdr| {
            let &PlaceHolder::Ref(Name::Entity(idx)) = &sdr.definition else {
                return false;
            };
            pds_idx == idx
        }) else {
            return Err("failed to find `shape_definition_representation` corresp. to `product_definition_shape`".into());
        };
        let PlaceHolder::Ref(Name::Entity(sr_idx)) = &sdr.used_representation else {
            return Err(
                "failed to reference `shape_definition_representation.used_representation`".into(),
            );
        };
        let Some(sr) = self.shape_representation.get(sr_idx) else {
            return Err("failed to reference `shape_representation`".into());
        };
        let Some(shape) = sr
            .items
            .iter()
            .map(|place_holder| {
                if let &PlaceHolder::Ref(Name::Entity(item_idx)) = place_holder {
                    ProductShape::try_from_index(item_idx, self).ok()
                } else {
                    None
                }
            })
            .collect::<Option<Vec<_>>>()
        else {
            return Err("failed to reference an element of `shape_representation.items`".into());
        };

        Ok(NodeEntity { shape, attrs })
    }

    fn assy_node_entity(
        &self,
        pds_idx: u64,
        next_assy: &NextAssemblyUsageOccurrenceHolder,
    ) -> Result<(AssembleEntity, (u64, u64)), StepConvertingError> {
        let &PlaceHolder::Ref(Name::Entity(parent_idx)) = &next_assy.relating_product_definition
        else {
            return Err("failed to reference the parent node".into());
        };
        let &PlaceHolder::Ref(Name::Entity(child_idx)) = &next_assy.related_product_definition
        else {
            return Err("failed to reference the child node".into());
        };

        let attrs = PartAttrs {
            id: next_assy.id.clone(),
            name: next_assy.name.clone(),
            description: next_assy.description.clone(),
        };

        let Some(cdsr) = self
            .context_dependent_shape_representation
            .values()
            .find(|cdsr| {
                let &PlaceHolder::Ref(Name::Entity(idx)) = &cdsr.represented_product_relation
                else {
                    return false;
                };
                pds_idx == idx
            })
        else {
            return Err("".into());
        };

        let PlaceHolder::Ref(Name::Entity(srrwt_idx)) = &cdsr.representation_relation else {
            return Err("failed to reference `context_dependent_shape_representation.representation_relation`".into());
        };

        let Some(srrwt) = self
            .shape_representation_relationship_with_transformation
            .get(srrwt_idx)
        else {
            return Err("failed to reference `shape_representation_relationship`".into());
        };
        let idtf = srrwt.transformation_operator.clone().into_owned(self)?;

        let entity = AssembleEntity {
            matrix: NodeMatrix::Transform(idtf.into()),
            attrs,
        };

        Ok((entity, (parent_idx, child_idx)))
    }

    pub fn step_assy(&self) -> Result<StepAssembly, StepConvertingError> {
        let mut product_entities = Vec::<ProductEntity>::new();
        let mut indices_map = HashMap::<u64, usize>::new();
        let mut assy_nodes = Vec::<(AssembleEntity, (u64, u64))>::new();
        for (&pds_idx, pds) in &self.product_definition_shape {
            let &PlaceHolder::Ref(Name::Entity(idx)) = &pds.definition else {
                return Err("failed to reference `product_definition_shape.definition`".into());
            };
            if let Some(pd) = self.product_definition.get(&idx) {
                product_entities.push(self.product_node_entity(pds_idx, pd)?);
                indices_map.insert(idx, product_entities.len() - 1);
            } else if let Some(next_assy) = self.next_assembly_usage_occurrence.get(&idx) {
                assy_nodes.push(self.assy_node_entity(pds_idx, next_assy)?);
            }
        }

        let adjacency = assy_nodes
            .into_iter()
            .map(|(entity, (from, to))| {
                let from = *indices_map.get(&from)?;
                let to = *indices_map.get(&to)?;
                Some((from, to, entity))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or::<StepConvertingError>("failed to reference `product_definiion_shape`".into())?;

        StepAssembly::try_from_adjacency(product_entities, adjacency)
            .ok_or("maybe the graph has a cycle.".into())
    }
}

pub trait StepShell {
    fn to_compressed_shell(
        &self,
        table: &Table,
    ) -> Result<CompressedShell<Point3, Curve3D, Surface>, StepConvertingError> {
        self.to_compressed_shell_with_losses(table)
            .map(|(shell, _)| shell)
    }

    /// The shell, plus one record per source face that did not survive.
    ///
    /// The plain conversion delegates to this rather than the reverse, so there
    /// is exactly one conversion path and the census cannot drift away from
    /// what the renderer actually does.
    fn to_compressed_shell_with_losses(
        &self,
        table: &Table,
    ) -> Result<(CompressedShell<Point3, Curve3D, Surface>, Vec<FaceLoss>), StepConvertingError>;
}

impl StepShell for ShellHolder {
    fn to_compressed_shell_with_losses(
        &self,
        table: &Table,
    ) -> Result<(CompressedShell<Point3, Curve3D, Surface>, Vec<FaceLoss>), StepConvertingError>
    {
        let vertices = table.shell_vertices(self);
        let edges = table.shell_edges(self, &vertices);
        // Faces are resolved while the arenas are still arenas, so a face can
        // only name an edge that converted. Only then are they flattened to the
        // bare vectors `CompressedShell` requires.
        let mut losses = Vec::new();
        let mut singular = Vec::new();
        let faces = table.shell_faces(self, &edges, &mut losses, &mut singular);
        if !singular.is_empty() && std::env::var_os("TRUCK_PROBE_SINGULAR").is_some() {
            eprintln!(
                "SINGULAR {} faces have a collapsed bound (apex or pole)",
                singular.len()
            );
        }
        Ok((
            CompressedShell {
                vertices: vertices.into_items(),
                edges: edges.into_items(),
                faces,
            },
            losses,
        ))
    }
}

impl StepShell for OrientedShellHolder {
    fn to_compressed_shell_with_losses(
        &self,
        table: &Table,
    ) -> Result<(CompressedShell<Point3, Curve3D, Surface>, Vec<FaceLoss>), StepConvertingError>
    {
        let PlaceHolder::Ref(Name::Entity(idx)) = &self.shell_element else {
            return Err("failed to reference shell".into());
        };
        let Some(shell) = table.shell.get(idx) else {
            return Err("failed to reference shell".into());
        };
        let (mut res, losses) = shell.to_compressed_shell_with_losses(table)?;
        if !self.orientation {
            for face in &mut res.faces {
                face.orientation = !face.orientation;
            }
        }
        Ok((res, losses))
    }
}

impl StepShell for ShellAnyHolder {
    fn to_compressed_shell_with_losses(
        &self,
        table: &Table,
    ) -> Result<(CompressedShell<Point3, Curve3D, Surface>, Vec<FaceLoss>), StepConvertingError>
    {
        match self {
            ShellAnyHolder::OrientedShell(shell) => shell.to_compressed_shell_with_losses(table),
            ShellAnyHolder::Shell(shell) => shell.to_compressed_shell_with_losses(table),
        }
    }
}

/// Which layer of the face reference chain a shell named directly.
///
/// STEP shells may reference either layer, and the two are not
/// interchangeable: an `ORIENTED_FACE` is a use that contributes orientation
/// and may share its `FACE_SURFACE` with other uses. Recording which one was
/// found keeps `FaceProvenance` from claiming a use entity that the file never
/// wrote.
enum FaceReference {
    /// The shell named a `FACE_SURFACE`. There is no use entity.
    Definition,
    /// The shell named an `ORIENTED_FACE`, which resolved to this definition.
    Use { definition_id: Option<u64> },
}

#[cfg(test)]
mod provenance_tests {
    use super::*;

    /// A `FACE_SURFACE` with no bounds. Only its identity matters here.
    fn face_surface() -> FaceSurfaceHolder {
        FaceSurfaceHolder {
            label: String::new(),
            bounds: Vec::new(),
            face_geometry: PlaceHolder::Ref(Name::Entity(700)),
            same_sense: true,
        }
    }

    /// A shell naming an `ORIENTED_FACE` must yield a use id *and* a separately
    /// resolved definition id.
    ///
    /// This is the branch no file in either corpus exercises — all 33 NIST
    /// models and the ABC models checked name `ADVANCED_FACE` directly — so
    /// without this test the half of the split that motivated it would be
    /// unrun. Compiling is not evidence.
    #[test]
    fn an_oriented_face_reports_use_and_definition_separately() {
        let mut table = Table::default();
        table.face_surface.insert(402, face_surface());
        table.oriented_face.insert(
            811,
            OrientedFaceHolder {
                label: String::new(),
                face_element: PlaceHolder::Ref(Name::Entity(402)),
                orientation: false,
            },
        );

        let holder = table.oriented_face.get(&811).cloned().unwrap();
        let (orientation, reference, _) = table
            .face_any_resolved(Some(FaceAnyHolder::OrientedFace(holder)))
            .expect("the oriented face resolves");

        assert!(!orientation, "the use carries the orientation flag");
        match reference {
            FaceReference::Use { definition_id } => assert_eq!(
                definition_id,
                Some(402),
                "the definition is the FACE_SURFACE, not the use"
            ),
            FaceReference::Definition => panic!("an ORIENTED_FACE is a use, not a definition"),
        }
    }

    /// A shell naming a `FACE_SURFACE` directly has no use entity at all, and
    /// must not invent one by copying the reference into the use slot — which
    /// is exactly what an earlier version did, printing "face use #43172 of
    /// face #43172".
    #[test]
    fn a_direct_face_surface_reference_has_no_use() {
        let table = Table::default();
        let (orientation, reference, _) = table
            .face_any_resolved(Some(FaceAnyHolder::FaceSurface(face_surface())))
            .expect("the face surface resolves");

        assert!(
            orientation,
            "a definition carries no orientation of its own"
        );
        assert!(
            matches!(reference, FaceReference::Definition),
            "a direct reference names the definition"
        );
    }

    /// The surface reference is a third, independent identity: one surface is
    /// commonly shared by many faces, so "which surface did this face name" is
    /// not answered by either of the other two ids.
    #[test]
    fn the_surface_identity_is_recorded_separately() {
        let table = Table::default();
        let provenance = FaceProvenance {
            use_id: Some(SourceEntityId::new(811)),
            definition_id: Some(SourceEntityId::new(402)),
            surface_id: Some(SourceEntityId::new(91)),
        };
        assert_eq!(
            provenance.to_string(),
            "face use #811 of face #402, surface #91"
        );
        assert_eq!(provenance.best_id(), Some(SourceEntityId::new(402)));
        let _ = table;
    }
}

#[cfg(test)]
mod plane_angle_unit_tests {
    use super::*;

    /// A file writing angles in degrees must report the degree factor.
    #[test]
    fn a_degree_file_reports_the_degree_factor() {
        let mut table = Table::default();
        // #20 = PLANE_ANGLE_MEASURE_WITH_UNIT(PLANE_ANGLE_MEASURE(0.0174532925), #18)
        table
            .plane_angle_measures
            .insert(20, (0.0174532925, Some(18)));
        // #18 is the radian SI unit the conversion is expressed in.
        table.plane_angle_units.push((18, PlaneAngleUnit::Radian));
        // #24 = CONVERSION_BASED_UNIT('DEGREE', #20)
        table
            .plane_angle_units
            .push((24, PlaneAngleUnit::Converted { measure: 20 }));

        assert_eq!(table.plane_angle_factor(), 0.0174532925);
    }

    /// The regression for the defect this resolution *introduced*.
    ///
    /// A degree unit is defined as a multiple of a radian unit, so every degree
    /// file necessarily also contains a radian `SI_UNIT`. Counting that base as
    /// a competing declaration made the agreement rule refuse every file it
    /// existed to fix — observed on the first run against `ftc_07`, which
    /// printed "plane angle units disagree (1 vs 0.0174532925)" and left the
    /// blob exactly as it was. The base must be excluded, not compared.
    #[test]
    fn the_base_unit_of_a_conversion_does_not_count_as_disagreement() {
        let mut table = Table::default();
        table
            .plane_angle_measures
            .insert(20, (0.0174532925, Some(18)));
        table.plane_angle_units.push((18, PlaneAngleUnit::Radian));
        table
            .plane_angle_units
            .push((24, PlaneAngleUnit::Converted { measure: 20 }));

        assert_ne!(
            table.plane_angle_factor(),
            1.0,
            "the radian base of the degree unit must not veto the conversion"
        );
    }

    /// A radian file is left alone, and costs nothing.
    #[test]
    fn a_radian_file_needs_no_conversion() {
        let mut table = Table::default();
        table.plane_angle_units.push((18, PlaneAngleUnit::Radian));
        assert_eq!(table.plane_angle_factor(), 1.0);
    }

    /// A file with no unit declarations at all is assumed to be in radians,
    /// which is what the standard says and what every file did before this
    /// existed.
    #[test]
    fn no_declaration_means_radians() {
        assert_eq!(Table::default().plane_angle_factor(), 1.0);
    }

    /// Two *independently assigned* angle units genuinely conflict, and the
    /// resolution refuses rather than picking one.
    ///
    /// Choosing correctly needs the geometry's own
    /// `GEOMETRIC_REPRESENTATION_CONTEXT`, which is not resolved here. Guessing
    /// would convert a file whose geometry is already in radians and break it —
    /// worse than leaving it as found, because the failure would be new.
    #[test]
    fn independently_assigned_conflicting_units_are_refused() {
        let mut table = Table::default();
        // Two conversions with different factors, neither the base of the other.
        table
            .plane_angle_measures
            .insert(20, (0.0174532925, Some(18)));
        table.plane_angle_measures.insert(30, (0.5, Some(28)));
        table
            .plane_angle_units
            .push((24, PlaneAngleUnit::Converted { measure: 20 }));
        table
            .plane_angle_units
            .push((34, PlaneAngleUnit::Converted { measure: 30 }));

        assert_eq!(
            table.plane_angle_factor(),
            1.0,
            "a real conflict leaves the file as found"
        );
    }

    /// A conical surface's semi-angle is converted into radians on import.
    ///
    /// This is the attribute that produced the `ftc_07` blob: a 2° draft cone
    /// read as 2 radians has slope `tan(2) = -2.185` instead of `0.0349` —
    /// wrong by 63x and inverted in sign, which flares the cone backwards into
    /// a fan.
    #[test]
    fn a_cone_semi_angle_is_normalized_into_radians() {
        let mut table = Table::default();
        table
            .plane_angle_measures
            .insert(20, (0.0174532925, Some(18)));
        table.plane_angle_units.push((18, PlaneAngleUnit::Radian));
        table
            .plane_angle_units
            .push((24, PlaneAngleUnit::Converted { measure: 20 }));
        table.conical_surface.insert(
            686,
            ConicalSurfaceHolder {
                label: String::new(),
                position: PlaceHolder::Ref(Name::Entity(685)),
                radius: 0.282184119986423,
                semi_angle: 2.0,
            },
        );

        table.normalize_angle_units();

        let converted = table.conical_surface[&686].semi_angle;
        assert!(
            (converted - 2.0 * 0.0174532925).abs() < 1.0e-15,
            "2 degrees must become {} radians, got {converted}",
            2.0 * 0.0174532925
        );
        assert!(
            f64::tan(converted) > 0.0,
            "the corrected slope must not be negative; that inversion is the blob"
        );
    }
}

/// Why a source face produced no `CompressedFace`.
///
/// Coarse on purpose. The point of a census is to order a repair queue, and a
/// dozen categories that each name a real code path beat forty that require
/// judgement to assign. Split a variant when its count is large enough that the
/// split would change what gets fixed next, not before.
///
/// These cover conversion only. A face that converts and then meshes to nothing
/// is lost later, in tessellation, and is counted there — the two populations
/// have different causes and must not be summed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FaceLossReason {
    /// The shell named something that does not resolve to a face at all.
    FaceReferenceUnresolved,
    /// The face's surface reference did not resolve, or its geometry could not
    /// be converted. `OFFSET_SURFACE` lands here: it parses and then has no
    /// conversion arm.
    SurfaceConversionFailed,
    /// A `FACE_BOUND`/`FACE_OUTER_BOUND` reference did not resolve.
    BoundReferenceUnresolved,
    /// Every bound of the face collapsed to a point, so nothing bounds it.
    AllBoundsCollapsed,
    /// The bound resolved, but the `EDGE_LOOP` it names did not.
    ///
    /// Split from `BoundReferenceUnresolved` once the census showed the pair
    /// accounted for 45% of all lost faces on `00009190`: they are different
    /// defects — a missing bound entity against a missing loop entity — and
    /// deciding what to fix next requires knowing which.
    LoopReferenceUnresolved,
    /// An `ORIENTED_EDGE` in a bound did not resolve to an edge.
    EdgeUseUnresolved,
    /// The edge resolved as a reference but its curve did not convert, so no
    /// arena position exists for it.
    EdgeCurveConversionFailed,
    /// Every edge resolved and the wire still does not close on vertex
    /// identity, so it bounds nothing (`TOP-004`).
    WireNotClosed,
}

impl FaceLossReason {
    /// A short stable tag, for grouping a census.
    pub fn tag(self) -> &'static str {
        match self {
            Self::FaceReferenceUnresolved => "FaceReferenceUnresolved",
            Self::SurfaceConversionFailed => "SurfaceConversionFailed",
            Self::BoundReferenceUnresolved => "BoundReferenceUnresolved",
            Self::LoopReferenceUnresolved => "LoopReferenceUnresolved",
            Self::AllBoundsCollapsed => "AllBoundsCollapsed",
            Self::EdgeUseUnresolved => "EdgeUseUnresolved",
            Self::EdgeCurveConversionFailed => "EdgeCurveConversionFailed",
            Self::WireNotClosed => "WireNotClosed",
        }
    }
}

/// One source face that did not survive conversion, and why.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceLoss {
    /// As much of the reference chain as was resolved before the failure.
    ///
    /// Partial by nature: a face whose surface reference failed has no surface
    /// id to report, and reporting a fabricated one would defeat the purpose.
    pub provenance: FaceProvenance,
    pub reason: FaceLossReason,
}

/// What one face bound contributed to the trimming domain.
/// The face's outer-bound standing, stated against the surviving wires.
///
/// `flags` is per declared bound, in source order: `Some(true)` for a
/// `FACE_OUTER_BOUND`, `Some(false)` for a plain `FACE_BOUND`, `None` for a
/// bound the document inlined, whose entity type this reader cannot recover.
///
/// Any `None` makes the whole face `NotRetained`. That is deliberate: a face
/// with one inlined bound and one referenced `FACE_BOUND` would otherwise
/// report `NoneDeclared`, which is a *claim* about the source, and the reader
/// is not in a position to make it.
fn outer_bound_standing(flags: &[Option<bool>], outcomes: &[BoundOutcome]) -> OuterBoundStanding {
    if flags.len() != outcomes.len() || flags.iter().any(Option::is_none) {
        return OuterBoundStanding::NotRetained;
    }
    let declared_count = flags.iter().filter(|flag| **flag == Some(true)).count();
    // Index among the wires that survive, since those are what `boundaries`
    // will hold. A collapsed outer bound contributes no wire and so no index.
    let mut surviving = 0u32;
    let mut bound_index = None;
    for (flag, outcome) in flags.iter().zip(outcomes) {
        match outcome {
            BoundOutcome::Wire(_) => {
                if *flag == Some(true) && bound_index.is_none() {
                    bound_index = Some(surviving);
                }
                surviving += 1;
            }
            BoundOutcome::Collapsed(_) => {}
        }
    }
    match (declared_count, bound_index) {
        (0, _) => OuterBoundStanding::NoneDeclared,
        (_, None) => OuterBoundStanding::NoneDeclared,
        (count, Some(bound_index)) => OuterBoundStanding::Declared {
            bound_index,
            declared_count: count as u32,
        },
    }
}

enum BoundOutcome {
    /// An ordinary closed wire of edges.
    Wire(TopologicallyClosedWire),
    /// Nothing: the bound is a single vertex, and the surface closes itself
    /// there. See `FaceBoundLoop::Collapsed`. Retains the 3D vertex position.
    Collapsed(Point3),
}

#[cfg(test)]
mod vertex_loop_tests {
    use super::*;

    fn vertex_point() -> VertexPointHolder {
        VertexPointHolder {
            label: String::new(),
            vertex_geometry: PlaceHolder::Ref(Name::Entity(1820)),
        }
    }

    /// A bound naming a `VERTEX_LOOP` resolves as collapsed, not as unresolved.
    ///
    /// This is the regression for the largest single cause of missing faces
    /// found in either corpus: 272 of 604 on ABC `00009190` and 132 across
    /// NIST, with the `VERTEX_LOOP` count matching the failure count exactly in
    /// all eight files containing one. `bound_holder` checked `edge_loop` alone,
    /// so a cone apex took its whole face with it.
    #[test]
    fn a_vertex_loop_bound_resolves_as_collapsed() {
        let mut table = Table::default();
        table.vertex_loop.insert(
            286,
            VertexLoopHolder {
                label: String::new(),
                loop_vertex: PlaceHolder::Owned(vertex_point()),
            },
        );
        let bound = FaceBoundHolder {
            label: String::new(),
            bound: PlaceHolder::Ref(Name::Entity(286)),
            orientation: true,
        };

        assert!(matches!(
            bound.bound_holder(&table),
            Some(FaceBoundLoop::Collapsed(_))
        ));
    }

    /// An `EDGE_LOOP` still resolves as edges, and the two are distinguished
    /// rather than both being "a loop".
    #[test]
    fn an_edge_loop_bound_still_resolves_as_edges() {
        let mut table = Table::default();
        table.edge_loop.insert(
            4930,
            EdgeLoopHolder {
                label: String::new(),
                edge_list: Vec::new(),
            },
        );
        let bound = FaceBoundHolder {
            label: String::new(),
            bound: PlaceHolder::Ref(Name::Entity(4930)),
            orientation: true,
        };

        assert!(matches!(
            bound.bound_holder(&table),
            Some(FaceBoundLoop::Edges(_))
        ));
    }

    /// A bound naming neither is still unresolved, so a genuinely missing loop
    /// is not quietly reclassified as an apex.
    #[test]
    fn an_unknown_loop_reference_is_still_unresolved() {
        let bound = FaceBoundHolder {
            label: String::new(),
            bound: PlaceHolder::Ref(Name::Entity(99999)),
            orientation: true,
        };
        assert!(bound.bound_holder(&Table::default()).is_none());
    }
}
