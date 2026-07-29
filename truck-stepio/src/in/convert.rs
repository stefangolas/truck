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
        match face? {
            FaceAnyHolder::FaceSurface(face) => Some((true, face)),
            FaceAnyHolder::OrientedFace(oriented_face) => {
                let face_element = oriented_face.face_element_holder(self)?;
                Some((oriented_face.orientation, face_element))
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
            .filter_map(move |face| self.face_any_to_orientation_and_face(face))
            .flat_map(move |(_, face)| face.bounds_holder(self))
            .filter_map(move |bound| bound?.bound_holder(self))
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
            .filter_map(move |face| self.face_any_to_orientation_and_face(face))
            .flat_map(move |(_, face)| face.bounds_holder(self))
            .filter_map(move |bound| bound?.bound_holder(self))
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
    ) -> Option<TopologicallyClosedWire> {
        use PlaceHolder::Ref;
        let ori = bound.orientation;
        let bound = bound.bound_holder(self)?;
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
                    return None;
                };
                let edge_idx = if let Some(oriented_edge) = self.oriented_edge.get(idx) {
                    let named = EdgeCurveId::new(oriented_edge.edge_element_idx()?);
                    CompressedEdgeIndex {
                        index: Self::checked_edge_position(edges, named)?,
                        orientation: oriented_edge.orientation == ori,
                    }
                } else {
                    CompressedEdgeIndex {
                        index: Self::checked_edge_position(edges, EdgeCurveId::new(*idx))?,
                        orientation: ori,
                    }
                };
                Some(edge_idx)
            })
            .collect::<Option<Vec<_>>>()?;
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
    ) -> Option<Surface> {
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
        // be canonical about: it belongs to this face alone.
        let PlaceHolder::Ref(Name::Entity(idx)) = &face.face_geometry else {
            return convert();
        };
        let named = SurfaceId::new(*idx);
        let index = surfaces.get_or_try_insert(named, convert)?;
        surfaces
            .get_checked(index, named)
            .map_err(|mismatch| eprintln!("{mismatch}"))
            .ok()
            .cloned()
    }

    fn shell_faces(
        &self,
        shell: &ShellHolder,
        edges: &Arena<EdgeKind, CompressedEdge<Curve3D>>,
    ) -> Vec<CompressedFace<Surface>> {
        let mut surfaces = Arena::<SurfaceKind, Surface>::new();
        shell
            .cfs_faces_holder(self)
            .filter_map(|face| self.face_any_to_orientation_and_face(face))
            .filter_map(|(orientation, face)| {
                let mut surface = self.face_surface(&face, &mut surfaces)?;
                if !face.same_sense && std::env::var_os("TRUCK_NO_INVERT").is_none() {
                    surface.invert()
                }
                // Same rule one level up: a face missing a bound is a broken
                // face, not a simpler one. Dropping a failed bound here silently
                // rewrites what the solid is — lose an inner bound and a hole
                // fills in, lose the outer bound and the remaining holes are
                // read as the outline. Both mesh perfectly happily.
                let boundaries: Vec<_> = face
                    .bounds_holder(self)
                    .into_iter()
                    .map(|bound| self.face_bound_to_edges(bound?, edges))
                    .collect::<Option<Vec<TopologicallyClosedWire>>>()?
                    .into_iter()
                    // The proof is discharged here and nowhere earlier: truck's
                    // `CompressedFace` takes bare index vectors, so this is the
                    // boundary at which the guarantee stops travelling. Closing
                    // it is §33a item 11 — `boundaries` becomes
                    // `Vec<TopologicallyClosedWire>`, which the owned-fork
                    // decision (§31a) now permits.
                    .map(TopologicallyClosedWire::into_edges)
                    .collect();
                Some(CompressedFace {
                    surface,
                    boundaries,
                    orientation,
                })
            })
            .collect()
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
    ) -> Result<CompressedShell<Point3, Curve3D, Surface>, StepConvertingError>;
}

impl StepShell for ShellHolder {
    fn to_compressed_shell(
        &self,
        table: &Table,
    ) -> Result<CompressedShell<Point3, Curve3D, Surface>, StepConvertingError> {
        let vertices = table.shell_vertices(self);
        let edges = table.shell_edges(self, &vertices);
        // Faces are resolved while the arenas are still arenas, so a face can
        // only name an edge that converted. Only then are they flattened to the
        // bare vectors `CompressedShell` requires.
        let faces = table.shell_faces(self, &edges);
        Ok(CompressedShell {
            vertices: vertices.into_items(),
            edges: edges.into_items(),
            faces,
        })
    }
}

impl StepShell for OrientedShellHolder {
    fn to_compressed_shell(
        &self,
        table: &Table,
    ) -> Result<CompressedShell<Point3, Curve3D, Surface>, StepConvertingError> {
        let PlaceHolder::Ref(Name::Entity(idx)) = &self.shell_element else {
            return Err("failed to reference shell".into());
        };
        let Some(shell) = table.shell.get(idx) else {
            return Err("failed to reference shell".into());
        };
        let mut res = shell.to_compressed_shell(table)?;
        if !self.orientation {
            for face in &mut res.faces {
                face.orientation = !face.orientation;
            }
        }
        Ok(res)
    }
}

impl StepShell for ShellAnyHolder {
    fn to_compressed_shell(
        &self,
        table: &Table,
    ) -> Result<CompressedShell<Point3, Curve3D, Surface>, StepConvertingError> {
        match self {
            ShellAnyHolder::OrientedShell(shell) => shell.to_compressed_shell(table),
            ShellAnyHolder::Shell(shell) => shell.to_compressed_shell(table),
        }
    }
}
