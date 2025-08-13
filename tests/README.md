# Tests

## test_arraylake.py

Configurable test for creating sample datasets in different storage backends.

### Usage

```bash
# Required arguments
--where {local,arraylake,arraylake-dev}    # Storage backend

# Optional arguments
--setup                      # Enable dataset creation (checks existing first)
--setup=force                # Force recreate datasets (overwrites existing)
--prefix PATH                # Custom storage path (has defaults)
```

### Storage Backends

| Backend | Default Prefix | Description |
|---------|----------------|-------------|
| `local` | `/tmp/tiles-icechunk/` | Local filesystem using icechunk |
| `arraylake` | `earthmover-integration/tiles-icechunk/` | Arraylake prod deployment |
| `arraylake-dev` | `earthmover-integration/tiles-icechunk/` | Arraylake dev deployment |

### Examples

```bash
# Create datasets locally (checks existing first)
uv run pytest tests/test_arraylake.py --where=local --setup

# Force recreate datasets locally (overwrites existing)
uv run pytest tests/test_arraylake.py --where=local --setup=force

# Create datasets in Arraylake prod deployment
uv run pytest tests/test_arraylake.py --where=arraylake --setup

# Create datasets in Arraylake dev deployment
uv run pytest tests/test_arraylake.py --where=arraylake-dev --setup

# Use custom prefix
uv run pytest tests/test_arraylake.py --where=local --prefix=/tmp/my-data --setup

# Skip setup tests (default behavior)
uv run pytest tests/test_arraylake.py --where=local  # All tests skipped
```

### Setup Modes

- **`--setup`** (default mode): Checks if datasets already exist in the repository. Only creates datasets that are missing. This is efficient for repeated test runs.
- **`--setup=force`**: Always recreates all datasets from scratch, overwriting any existing data. Useful when you need fresh data or schema changes.
- **No `--setup` flag**: Skips all dataset creation tests.

### Datasets

Creates 8 sample datasets with different characteristics:
- `ifs` - IFS weather data (4D: time, step, lat, lon)
- `sentinel2-nocoords` - Sentinel-2 imagery without coordinates (4D: time, lat, lon, band)
- `global_6km` - Global 6km resolution data
- `para` - Land use classification (3D: x, y, time)
- `para_hires` - High resolution land use classification
- `hrrr` - HRRR weather model data with Lambert Conformal Conic projection
- `eu3035` - European data in ETRS89/LAEA projection
- `eu3035_hires` - High resolution European data

### Automated Dataset Creation

A GitHub Actions workflow (`.github/workflows/dataset-creation.yml`) is configured to:
- **Run nightly** at 2 AM UTC to ensure datasets are up-to-date
- **Allow manual triggers** with an option to force recreate datasets
- **Target Arraylake dev environment** with proper authentication

To manually trigger the workflow:
1. Go to Actions tab in GitHub
2. Select "Dataset Creation" workflow
3. Click "Run workflow"
4. Optionally check "Force create datasets" to overwrite existing data

### Test Execution

The `test_create` function uses `@pytest.mark.xdist_group(name="repo_creation")` to ensure all dataset creation tests run sequentially in the same worker process. This prevents concurrent write conflicts when using local filesystem storage. The `--dist=loadgroup` option is set as default in `pyproject.toml`.
