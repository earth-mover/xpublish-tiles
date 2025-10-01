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
