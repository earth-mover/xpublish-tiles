# UGRID Support for Triangular Grids — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read face-node connectivity from UGRID-convention datasets in `Triangular.from_dataset()`, skipping Delaunay triangulation when the mesh topology is already provided.

**Architecture:** Add `_ugrid_topology_var()` and `_ugrid_face_node_connectivity()` helpers in `grids.py` following the SGRID pattern. Call them at the top of `Triangular.from_dataset()`. Add a real FVCOM fixture and tests.

**Tech Stack:** xarray, cf-xarray, numpy, pytest, unittest.mock

---

## File Map

- Modify: `src/xpublish_tiles/grids.py` — two new helpers + update `Triangular.from_dataset()`
- Copy: `src/xpublish_tiles/testing/grids/machias_bay_fvcom.nc` — FVCOM test fixture
- Modify: `src/xpublish_tiles/testing/datasets.py` — add `create_fvcom_ugrid` and `FVCOM_MACHIAS_BAY`
- Modify: `tests/test_grids.py` — four new tests

---

### Task 1: Copy the FVCOM test fixture

**Files:**
- Copy: `src/xpublish_tiles/testing/grids/machias_bay_fvcom.nc`

- [ ] **Step 1: Copy the file**

```bash
cp /mnt/c/users/rsign/Downloads/machias_bay_v2_lastweek_2ts.nc \
   src/xpublish_tiles/testing/grids/machias_bay_fvcom.nc
```

- [ ] **Step 2: Verify the file is correct**

```bash
uv run python -c "
import xarray as xr
ds = xr.open_dataset('src/xpublish_tiles/testing/grids/machias_bay_fvcom.nc', chunks={})
topo = next(v for v in ds.variables.values() if v.attrs.get('cf_role') == 'mesh_topology')
print('topology attrs:', topo.attrs)
nv = ds[topo.attrs['face_node_connectivity']]
print('nv shape:', nv.shape, 'start_index:', nv.attrs.get('start_index'))
"
```

Expected: `nv shape: (266, 3)  start_index: 1`

- [ ] **Step 3: Commit**

```bash
git add src/xpublish_tiles/testing/grids/machias_bay_fvcom.nc
git commit -m "Add FVCOM Machias Bay UGRID test fixture"
```

---

### Task 2: Write failing tests for the UGRID helpers

**Files:**
- Modify: `tests/test_grids.py`

- [ ] **Step 1: Add imports to `tests/test_grids.py`**

Find the block that imports from `xpublish_tiles.grids` (around line 21) and add the two new helpers:

```python
from xpublish_tiles.grids import (
    ...  # existing imports
    _ugrid_face_node_connectivity,
    _ugrid_topology_var,
)
```

Find the block that imports test datasets (around line 58) and add:

```python
from xpublish_tiles.testing.datasets import (
    ...  # existing imports
    FVCOM_MACHIAS_BAY,
)
```

- [ ] **Step 2: Add the four helper tests**

Add these tests after `test_detect_orca_tripole_fold_row` (near the end of the file):

```python
def test_ugrid_topology_var_found():
    ds = FVCOM_MACHIAS_BAY.create()
    assert _ugrid_topology_var(ds) == "mesh_topology"


def test_ugrid_topology_var_none_for_non_ugrid():
    ds = REDGAUSS_N320.create()
    assert _ugrid_topology_var(ds) is None


def test_ugrid_face_node_connectivity():
    ds = FVCOM_MACHIAS_BAY.create()
    faces = _ugrid_face_node_connectivity(ds)
    assert faces is not None
    assert faces.shape == (266, 3)
    assert faces.dtype == np.int64
    assert faces.min() == 0
    assert faces.max() == 183  # 184 nodes, zero-indexed


def test_ugrid_face_node_connectivity_none_for_non_ugrid():
    ds = REDGAUSS_N320.create()
    assert _ugrid_face_node_connectivity(ds) is None
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
uv run pytest tests/test_grids.py::test_ugrid_topology_var_found \
              tests/test_grids.py::test_ugrid_topology_var_none_for_non_ugrid \
              tests/test_grids.py::test_ugrid_face_node_connectivity \
              tests/test_grids.py::test_ugrid_face_node_connectivity_none_for_non_ugrid \
              -v
```

Expected: 4 failures — `ImportError` on `_ugrid_topology_var` / `_ugrid_face_node_connectivity`, and `ImportError` on `FVCOM_MACHIAS_BAY`.

---

### Task 3: Implement the UGRID helpers in `grids.py`

**Files:**
- Modify: `src/xpublish_tiles/grids.py:420` — insert after `_resolve_corner_name()`

- [ ] **Step 1: Add `_ugrid_topology_var()` after `_resolve_corner_name()` (around line 420)**

```python
def _ugrid_topology_var(ds: xr.Dataset) -> str | None:
    """Return the name of the UGRID mesh topology variable in ``ds``, or
    ``None`` if there isn't one. Mirrors ``_sgrid_topology_var``."""
    topology_vars = ds.cf.cf_roles.get("mesh_topology", [])
    if not topology_vars:
        return None
    assert len(topology_vars) == 1, (
        f"expected at most one cf_role=mesh_topology variable, found {topology_vars}"
    )
    return str(topology_vars[0])


def _ugrid_face_node_connectivity(ds: xr.Dataset) -> np.ndarray | None:
    """Read face_node_connectivity from a UGRID topology variable.

    Returns a zero-indexed ``(n_faces, 3)`` int64 array, or ``None`` if the
    dataset has no UGRID mesh topology or is not triangular.
    """
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

- [ ] **Step 2: Run the helper tests**

```bash
uv run pytest tests/test_grids.py::test_ugrid_topology_var_found \
              tests/test_grids.py::test_ugrid_topology_var_none_for_non_ugrid \
              tests/test_grids.py::test_ugrid_face_node_connectivity \
              tests/test_grids.py::test_ugrid_face_node_connectivity_none_for_non_ugrid \
              -v
```

Expected: 2 pass (`_none_for_non_ugrid` tests), 2 fail (FVCOM tests — `FVCOM_MACHIAS_BAY` not yet defined). Once `FVCOM_MACHIAS_BAY` is added in Task 4, all 4 will pass.

---

### Task 4: Add `FVCOM_MACHIAS_BAY` to `testing/datasets.py`

**Files:**
- Modify: `src/xpublish_tiles/testing/datasets.py`

- [ ] **Step 1: Add `create_fvcom_ugrid` and `FVCOM_MACHIAS_BAY` after the `REDGAUSS_N320` block (around line 1594)**

```python
def create_fvcom_ugrid(
    *, dims: tuple[Dim, ...], dtype: npt.DTypeLike, attrs: dict[str, Any]
) -> xr.Dataset:
    """Load the Machias Bay FVCOM UGRID fixture and expose ``h`` as ``foo``."""
    nc = Path(__file__).parent / "grids" / "machias_bay_fvcom.nc"
    ds = xr.open_dataset(nc, chunks={})
    ds = ds[["h", "mesh_topology", "nv"]].rename({"h": "foo"})
    # CF disambiguation: two longitude coords (lon, lonc) exist; declare which
    # belongs to foo so _guess_coordinates_for_mapping picks lon/lat over lonc/latc.
    ds["foo"].attrs["coordinates"] = "lon lat"
    return ds.assign_attrs(attrs)


FVCOM_MACHIAS_BAY = Dataset(
    name="fvcom_machias_bay",
    dims=(Dim(name="node", size=184, chunk_size=184),),
    setup=create_fvcom_ugrid,
    dtype=np.float32,
    tiles=[
        TileTestParam(tile="9/160/184", tms="WebMercatorQuad"),
        TileTestParam(tile="10/320/369", tms="WebMercatorQuad"),
        TileTestParam(tile="11/641/739", tms="WebMercatorQuad"),
    ],
)
```

- [ ] **Step 2: Add `FVCOM_MACHIAS_BAY` to `DATASET_LOOKUP` (around line 1758)**

```python
DATASET_LOOKUP = {
    ...  # existing entries
    "fvcom_machias_bay": FVCOM_MACHIAS_BAY,
}
```

- [ ] **Step 3: Run the helper tests again — all 4 should now pass**

```bash
uv run pytest tests/test_grids.py::test_ugrid_topology_var_found \
              tests/test_grids.py::test_ugrid_topology_var_none_for_non_ugrid \
              tests/test_grids.py::test_ugrid_face_node_connectivity \
              tests/test_grids.py::test_ugrid_face_node_connectivity_none_for_non_ugrid \
              -v
```

Expected: 4 PASS

- [ ] **Step 4: Commit**

```bash
git add src/xpublish_tiles/grids.py src/xpublish_tiles/testing/datasets.py tests/test_grids.py
git commit -m "feat: add UGRID detection helpers and FVCOM test dataset"
```

---

### Task 5: Write failing test for `Triangular.from_dataset()` UGRID path

**Files:**
- Modify: `tests/test_grids.py`

- [ ] **Step 1: Add the UGRID from_dataset test after the helper tests**

```python
def test_triangular_from_dataset_uses_ugrid_connectivity():
    """from_dataset on a UGRID dataset skips Delaunay and uses provided faces."""
    ds = FVCOM_MACHIAS_BAY.create()
    with patch("xpublish_tiles.grids.triangle.delaunay") as mock_delaunay:
        grid = Triangular.from_dataset(ds, CRS.from_epsg(4326), "lon", "lat")

    mock_delaunay.assert_not_called()
    assert isinstance(grid, Triangular)
    assert grid.indexes[0].tree.faces.shape == (266, 3)
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_grids.py::test_triangular_from_dataset_uses_ugrid_connectivity -v
```

Expected: FAIL — Delaunay IS called because `from_dataset` doesn't check UGRID yet.

---

### Task 6: Modify `Triangular.from_dataset()` to use UGRID connectivity

**Files:**
- Modify: `src/xpublish_tiles/grids.py:2350`

- [ ] **Step 1: Replace `Triangular.from_dataset()` with the UGRID-aware version**

Find the method starting at line 2350 (the `# FIXME: detect UGRID here` line is at ~2357). Replace the entire `from_dataset` body:

```python
@classmethod
def from_dataset(
    cls,
    ds: xr.Dataset,
    crs: CRS,
    Xname: str,
    Yname: str,
) -> Self:
    faces = _ugrid_face_node_connectivity(ds)

    vertices = (
        ds.reset_coords()[[Xname, Yname]]
        .to_dataarray("variable")
        .transpose(..., "variable")
        .data
    )
    assert vertices.shape[-1] == 2, (
        f"Attempting to triangulate vertices with shape={vertices.shape}. Expected (n_points, 2)"
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
                    f"Triangulation failed. This may indicate bad data in variables {Xname!r}, {Yname!r}."
                    f"Please check whether all values are the same. "
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

- [ ] **Step 2: Run the from_dataset test**

```bash
uv run pytest tests/test_grids.py::test_triangular_from_dataset_uses_ugrid_connectivity -v
```

Expected: PASS

- [ ] **Step 3: Run the full test suite to confirm no regressions**

```bash
uv run pytest tests/test_grids.py -x -q --where=local
```

Expected: all existing tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/xpublish_tiles/grids.py tests/test_grids.py
git commit -m "feat: use UGRID face_node_connectivity in Triangular.from_dataset"
```

---

### Task 7: Write and run tile render tests for the FVCOM dataset

**Files:**
- Modify: `tests/test_grids.py`

- [ ] **Step 1: Add tile render tests after the from_dataset test**

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tile,tms_id",
    [
        ("9/160/184", "WebMercatorQuad"),
        ("10/320/369", "WebMercatorQuad"),
        ("11/641/739", "WebMercatorQuad"),
    ],
)
async def test_pipeline_fvcom_ugrid(tile, tms_id):
    """Full pipeline render for the FVCOM UGRID dataset at multiple zoom levels."""
    ds = FVCOM_MACHIAS_BAY.create()
    tms = morecantile.tms.get(tms_id)
    z, x, y = (int(v) for v in tile.split("/"))
    tile_obj = morecantile.Tile(x=x, y=y, z=z)
    query_params = create_query_params(tile_obj, tms)
    result = await pipeline(ds, query_params)
    assert result is not None
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_grids.py::test_pipeline_fvcom_ugrid -v
```

Expected: FAIL — `FVCOM_MACHIAS_BAY` now exists but `pipeline` looks for variable `"foo"` while our dataset has `"foo"` (already renamed in Task 4) — check whether it actually fails or passes.

If it fails because `"foo"` is not in the dataset, double-check that `create_fvcom_ugrid` renames `h` to `foo`. If the rename was done correctly, re-run and expect PASS.

- [ ] **Step 3: Run all new tests together**

```bash
uv run pytest tests/test_grids.py::test_ugrid_topology_var_found \
              tests/test_grids.py::test_ugrid_topology_var_none_for_non_ugrid \
              tests/test_grids.py::test_ugrid_face_node_connectivity \
              tests/test_grids.py::test_ugrid_face_node_connectivity_none_for_non_ugrid \
              tests/test_grids.py::test_triangular_from_dataset_uses_ugrid_connectivity \
              "tests/test_grids.py::test_pipeline_fvcom_ugrid[9/160/184-WebMercatorQuad]" \
              "tests/test_grids.py::test_pipeline_fvcom_ugrid[10/320/369-WebMercatorQuad]" \
              "tests/test_grids.py::test_pipeline_fvcom_ugrid[11/641/739-WebMercatorQuad]" \
              -v
```

Expected: 8 PASS

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest tests/test_grids.py -q --where=local
```

Expected: all pass.

- [ ] **Step 5: Run pre-commit checks**

```bash
pre-commit run --all-files
```

Fix any lint/format issues reported.

- [ ] **Step 6: Commit**

```bash
git add tests/test_grids.py
git commit -m "test: add tile render tests for FVCOM UGRID dataset"
```
