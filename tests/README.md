# Tests

## test_arraylake.py

Configurable test for creating sample datasets in different storage backends.

### Usage

```bash
# Required arguments
--where {local,arraylake,arraylake-dev}    # Storage backend
--setup                      # Enable dataset creation tests

# Optional arguments
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
# Create datasets locally (default prefix)
uv run pytest tests/test_arraylake.py --where=local --setup

# Create datasets in Arraylake prod deployment (default prefix)
uv run pytest tests/test_arraylake.py --where=arraylake --setup

# Create datasets in Arraylake dev deployment (default prefix)
uv run pytest tests/test_arraylake.py --where=arraylake-dev --setup

# Use custom prefix
uv run pytest tests/test_arraylake.py --where=local --prefix=/tmp/my-data --setup

# Skip setup tests (default behavior)
uv run pytest tests/test_arraylake.py --where=local  # All tests skipped
```

### Datasets

Creates 4 sample datasets with different characteristics:
- `ifs` - IFS weather data (4D: time, step, lat, lon)
- `sentinel2-nocoords` - Sentinel-2 imagery without coordinates (4D: time, lat, lon, band)
- `helios` - Solar irradiance data (4D: time, lat, lon, band)
- `para` - Land use classification (3D: x, y, time)
