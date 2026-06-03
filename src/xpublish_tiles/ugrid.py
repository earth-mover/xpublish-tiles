from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import xarray as xr


@dataclass
class MeshTopology:
    """Parsed UGRID mesh topology."""

    name: str
    node_dim: str
    face_dim: str
    vertex_dim: str
    node_coordinates: tuple[str, str]
    face_coordinates: tuple[str, str] | None
    face_node_connectivity: str
    start_index: int


def detect_mesh(ds: xr.Dataset) -> MeshTopology | None:
    """Return a MeshTopology for the first mesh_topology variable in ``ds``,
    or None if no valid triangular UGRID mesh is found."""
    topology_vars = ds.cf.cf_roles.get("mesh_topology", [])
    if not topology_vars:
        return None

    topology_name = str(topology_vars[0])
    topo = ds[topology_name]

    conn_name = topo.attrs.get("face_node_connectivity")
    if conn_name is None or conn_name not in ds:
        return None

    conn = ds[conn_name]

    explicit_face_dim = topo.attrs.get("face_dimension")
    if explicit_face_dim is not None:
        if explicit_face_dim not in conn.dims:
            return None
        face_axis = list(conn.dims).index(explicit_face_dim)
    else:
        # Infer: if the first dim has size 3 and the second does not, it is
        # (vertex, face) layout (FVCOM style); otherwise assume (face, vertex).
        if conn.ndim == 2 and conn.shape[0] == 3 and conn.shape[1] != 3:
            face_axis = 1
        else:
            face_axis = 0

    vertex_axis = 1 - face_axis
    if conn.ndim != 2 or conn.shape[vertex_axis] != 3:
        return None

    face_dim = str(conn.dims[face_axis])
    vertex_dim = str(conn.dims[vertex_axis])

    node_coords_str = topo.attrs.get("node_coordinates")
    if not node_coords_str:
        return None
    nc = node_coords_str.split()
    if len(nc) < 2:
        return None
    node_X, node_Y = nc[0], nc[1]
    if node_X not in ds:
        return None
    node_dim_seq = ds[node_X].dims
    if len(node_dim_seq) != 1:
        return None
    node_dim = str(node_dim_seq[0])

    face_coords_str = topo.attrs.get("face_coordinates")
    face_coordinates: tuple[str, str] | None = None
    if face_coords_str:
        fc = face_coords_str.split()
        if len(fc) >= 2:
            face_coordinates = (fc[0], fc[1])

    start_index = int(conn.attrs.get("start_index", 0))

    return MeshTopology(
        name=topology_name,
        node_dim=node_dim,
        face_dim=face_dim,
        vertex_dim=vertex_dim,
        node_coordinates=(node_X, node_Y),
        face_coordinates=face_coordinates,
        face_node_connectivity=conn_name,
        start_index=start_index,
    )


def load_connectivity(ds: xr.Dataset, mesh: MeshTopology) -> np.ndarray:
    """Return a zero-indexed ``(n_faces, 3)`` int64 connectivity array."""
    conn = ds[mesh.face_node_connectivity]
    faces = conn.values.astype(np.int64)
    face_axis = list(conn.dims).index(mesh.face_dim)
    if face_axis != 0:
        faces = faces.T
    faces -= mesh.start_index
    return faces
