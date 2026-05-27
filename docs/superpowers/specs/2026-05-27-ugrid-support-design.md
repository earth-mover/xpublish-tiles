# UGRID Support for Triangular Grids — Design Spec

**Goal:** Read face-node connectivity directly from UGRID-convention datasets instead of running Delaunay triangulation, fixing incorrect mesh topology for coastal unstructured grids (FVCOM, SCHISM, ADCIRC, etc.).

**Architecture:** Add two helper functions in `grids.py` that detect and read UGRID topology, then call them in `Triangular.from_dataset()` before the existing Delaunay fallback. No new runtime dependencies; follows the existing SGRID pattern.

**Tech Stack:** xarray, cf-xarray (`cf.cf_roles`), numpy, existing `triangle` + `numba-celltree` (unchanged)

---

## Problem

`Triangular.from_dataset()` always constructs faces via Delaunay triangulation from node coordinates alone. For coastal ocean models that provide explicit mesh topology via UGRID conventions, this produces incorrect triangulations — particularly near coastlines — because Delaunay ignores land boundaries encoded in the original mesh.

## UGRID Convention Summary (triangular meshes)

A UGRID-compliant dataset has:

- A scalar variable with `cf_role=mesh_topology` (the topology variable)
- Topology variable attributes:
  - `face_node_connectivity`: name of the connectivity variable
  - `node_coordinates`: space-separated names of lon/lat node coordinate variables
  - `face_dimension` (optional): name of the dimension that indexes faces in the connectivity array
- Connectivity variable: shape `(n_faces, 3)` or `(3, n_faces)`, with:
  - `start_index`: 0 or 1 (default 0; FVCOM uses 1)

## Changes to `grids.py`

### New helper: `_ugrid_topology_var()`

Mirrors the existing `_sgrid_topology_var()` at line 371.

```python
def _ugrid_topology_var(ds: xr.Dataset) -> str | None:
    topology_vars = ds.cf.cf_roles.get("mesh_topology", [])
    if not topology_vars:
        return None
    assert len(topology_vars) == 1, (
        f"expected at most one cf_role=mesh_topology variable, found {topology_vars}"
    )
    return str(topology_vars[0])
```

### New helper: `_ugrid_face_node_connectivity()`

Returns a zero-indexed `(n_faces, 3)` int64 array, or `None` if the dataset has no UGRID topology or is not triangular.

```python
def _ugrid_face_node_connectivity(ds: xr.Dataset) -> np.ndarray | None:
    topology_name = _ugrid_topology_var(ds)
    if topology_name is None:
        return None
    var = ds[topology_name]
    conn_name = var.attrs.get("face_node_connectivity")
    if conn_name is None or conn_name not in ds:
        return None
    conn = ds[conn_name]

    face_dim_name = var.attrs.get("face_dimension")
    if face_dim_name is not None:
        face_axis = conn.dims.index(face_dim_name)
    else:
        face_axis = 0  # UGRID convention: first dim is face dim

    if conn.shape[1 - face_axis] != 3:
        return None  # not triangular

    faces = conn.values.astype(np.int64)
    if face_axis == 1:
        faces = faces.T  # normalize to (n_faces, 3)

    start_index = int(conn.attrs.get("start_index", 0))
    faces -= start_index
    return faces
```

Placement: immediately after `_sgrid_node_for_face()` and `_resolve_corner_name()` (around line 420).

### Modified: `Triangular.from_dataset()`

Call `_ugrid_face_node_connectivity()` at the top. If it returns faces, skip Delaunay. Remove the `# FIXME: detect UGRID here` comment.

```python
@classmethod
def from_dataset(cls, ds: xr.Dataset, crs: CRS, Xname: str, Yname: str) -> Self:
    faces = _ugrid_face_node_connectivity(ds)

    vertices = (
        ds.reset_coords()[[Xname, Yname]]
        .to_dataarray("variable")
        .transpose(..., "variable")
        .data
    )
    assert vertices.shape[-1] == 2, (
        f"Attempting to triangulate vertices with shape={vertices.shape}. "
        f"Expected (n_points, 2)"
    )
    if crs.is_geographic:
        vertices[:, 0] = ((vertices[:, 0] + 180) % 360) - 180

    (dim_,) = ds[Xname].dims
    dim = str(dim_)

    if faces is None:
        with log_duration("Triangulating", "🔺"):
            if numbagg.anynan(vertices):
                raise ValueError(
                    f"Triangulation failed. Variables {Xname!r} or {Yname!r} contain NaNs."
                )
            try:
                faces = triangle.delaunay(vertices)
            except Exception as e:
                raise ValueError(
                    f"Triangulation failed. This may indicate bad data in variables "
                    f"{Xname!r}, {Yname!r}. Please check whether all values are the same. "
                    f"Original exception: {e!r}"
                ) from None

    return cls(
        vertices=vertices,
        faces=faces,
        crs=crs,
        Xname=Xname,
        Yname=Yname,
        dim=dim,
        fill_value=-1,
    )
```

## Test Fixture

**File:** `src/xpublish_tiles/testing/grids/machias_bay_fvcom.nc`

Source: Machias Bay FVCOM GOM4 hindcast subset (Icechunk on S3), clipped to `lon [-67.6, -67.08], lat [44.5, 44.8]`, 2 timesteps. 184 nodes, 266 triangular elements.

UGRID structure in this file:
- `mesh_topology`: scalar, `cf_role=mesh_topology`, `face_dimension='nele'`, `face_node_connectivity='nv'`, `node_coordinates='lon lat'`
- `nv (nele, mesh_topology_nMax_face_nodes)`: shape `(266, 3)`, `start_index=1`
- `lon (node)`, `lat (node)`: node coordinates

Test variable: `h (node)` — bathymetry, no time dimension, simple case.

## New Dataset in `testing/datasets.py`

```python
def create_fvcom_ugrid(
    *, dims: tuple[Dim, ...], dtype: npt.DTypeLike, attrs: dict[str, Any]
) -> xr.Dataset:
    nc = Path(__file__).parent / "grids" / "machias_bay_fvcom.nc"
    ds = xr.open_dataset(nc, chunks={})
    return ds[["h", "mesh_topology", "nv"]].assign_attrs(attrs)


FVCOM_MACHIAS_BAY = Dataset(
    name="fvcom_machias_bay",
    dims=(Dim(name="node", size=184, chunk_size=184),),
    setup=create_fvcom_ugrid,
    dtype=np.float32,
    tiles=[
        # computed at implementation time via morecantile for lon=-67.3, lat=44.65
        # one tile each at zoom 9, 10, 11 covering Machias Bay
    ],
)
```

## Tests in `tests/test_grids.py`

1. **`test_ugrid_topology_var`** — `_ugrid_topology_var()` returns `"mesh_topology"` for the FVCOM dataset; returns `None` for a dataset without `cf_role=mesh_topology`.

2. **`test_ugrid_face_node_connectivity`** — `_ugrid_face_node_connectivity()` returns a `(266, 3)` zero-indexed int64 array; verify `faces.min() == 0` and `faces.max() == 183` (valid node indices for 184 nodes).

3. **`test_triangular_from_dataset_uses_ugrid`** — `Triangular.from_dataset()` on the FVCOM dataset produces a `Triangular` grid whose `CellTreeIndex` has `faces.shape == (266, 3)`; confirm Delaunay was NOT run (mock `triangle.delaunay` and assert not called).

4. **`test_triangular_from_dataset_fallback`** — a dataset without `mesh_topology` still runs Delaunay as before (existing behavior unchanged).

5. **Tile render tests** — parametrized over `FVCOM_MACHIAS_BAY.tiles`, verify pipeline returns a valid raster tile without error.
