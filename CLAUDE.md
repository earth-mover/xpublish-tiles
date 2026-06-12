# xpublish-tiles

## Project Overview
This project contains a set of web mapping plugins for Xpublish - a framework for serving xarray datasets via HTTP APIs.

The goal of this project is to transform xarray datasets to raster, vector and other types of tiles, which can then be served via HTTP APIs. To do this, the package implements a set of xpublish plugins:
* `xpublish_tiles.xpublish.tiles.TilesPlugin`: An [OGC Tiles](https://www.ogc.org/standards/ogcapi-tiles/) conformant plugin for serving raster, vector and other types of tiles.
* `xpublish_tiles.xpublish.wms.WMSPlugin`: An [OGC Web Map Service](https://www.ogc.org/standards/wms/) conformant plugin for serving raster, vector and other types of tiles.

### Background Information

The WMS and Tiles specifications are available in in the `docs` directory for reference.

## Development Workflow

### Key Commands
- **Environment sync**: `uv sync --dev`
- **Type check**: `uv run ty check src/ tests/` (only checks src/ and tests/ directories)
- **Run unit tests**: `uv run pytest tests` (defaults to --where=local)
- **Run tests with coverage**: `uv run pytest tests --cov=src/xpublish_tiles --cov-report=term-missing`
- **Run pre-commit checks**: `pre-commit run --all-files`

### Dependency Groups
- **dev**: All development dependencies (includes testing, linting, type checking, debugging)
- **testing**: Testing-only dependencies (pytest, syrupy, hypothesis, matplotlib, etc.)

### Adding a new synthetic dataset
Synthetic datasets live in `src/xpublish_tiles/testing/datasets.py` and are the
project-standard way to lock in behavior for a grid type. When adding one:
1. Write a `setup(*, dims, dtype, attrs) -> xr.Dataset` function (model it on
   `radar_polar_grid` / `geostationary_grid`) and a `Dataset(...)` instance.
2. Register it in `DATASET_LOOKUP` (bottom of the file) — this exposes it to the
   CLI (`uv run xpublish-tiles --dataset local://<name>`) and benchmarks.
3. Add render snapshot coverage: a tile list in `testing/tiles.py` and a
   parametrized `test_*_data` in `tests/test_pipeline.py` using
   `assert_render_matches_snapshot`.
4. Add it to the `test_tiles_endpoint_snapshot` parametrization in
   `tests/test_xpublish/test_tiles/test_tiles_metadata.py` to cover the `/tiles/`
   metadata endpoint (bounds, styles, extents).
5. Generate the new snapshots once with `pytest <path> --snapshot-update`, then
   eyeball the resulting PNGs/JSON before committing.

### Adding a new grid system
Grid systems live in `src/xpublish_tiles/grids.py`. When adding one:
1. Subclass the closest existing `GridSystem` (e.g. `Geostationary(Rectilinear)`);
   reuse the parent's index/render machinery and only override what differs.
2. Route detection to it in `_detect_grid_metadata` (the primary path used by
   `guess_grid_metadata`). Verify with `guess_grid_metadata(ds).grid_cls`.
3. Remember both render paths read coordinates differently: the polygons path
   uses `grid.cell_corners()` (index-derived), while the raster path
   (`_transform_raster_patch`) reads `subset[grid.X]` — i.e. whatever
   `assign_index()` puts on the array. Coordinate rewrites must happen in
   `assign_index` to reach the raster path.
4. Override `transform_bbox` if the projection-plane bbox doesn't transform
   cleanly to geographic (e.g. geostationary full-disk corners → `inf`).
5. Add a synthetic dataset (above) so the grid gets render + metadata snapshots.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
ALWAYS run pre-commit checks before committing.
ALWAYS put imports at the top of the file unless you need to avoid circular import issues.
Do not add obvious or silly comments. Code should be self-explanatory.
For pytest fixtures, prefer separate independent parametrized inputs over using itertools.product() for cleaner test combinations.
Do not recreate snapshots by default.
Do not add unnecessary comments.
Add imports to the top of the file unless necessary to avoid circular imports.
Never add try/except clauses that catch Exceptions in a test.
Never remove test cases without confirming with me first.
