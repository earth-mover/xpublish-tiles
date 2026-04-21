"""Minimal reproducer for HEALPix polygon-seam transparency in datashader.

When xpublish-tiles rasterizes HEALPix cells as polygons, columns exactly at
HEALPix base-cell longitude boundaries (0°, ±45°, ±90°, ±135°, ±180°) come
out transparent in the output PNG — especially on polar-cap tiles.

This script reproduces the behaviour with the real L1 HEALPix cell vertices,
fed directly into ``datashader.Canvas.polygons`` the same way
``xpublish_tiles.render.polygons`` does.

Run:
    uv run python datashader_polygons_reproducer.py
"""

import datashader as ds
import healpix_geo.nested as hpn
import numpy as np
import spatialpandas
import xdggs

from xpublish_tiles.lib import polygons_from_rings

LEVEL = 1
N_CELLS = 12 * 4**LEVEL
# Full-globe canvas approximating WebMercatorQuad z=0 (projected to ±180/±90
# in geographic coords for clarity — the aliasing is independent of CRS).
CANVAS_W, CANVAS_H = 256, 256


def build_healpix_rings() -> tuple[np.ndarray, np.ndarray]:
    """Return (rings (N, 5, 2), values (N,)) for all L1 HEALPix cells."""
    info = xdggs.HealpixInfo(level=LEVEL, indexing_scheme="nested")
    ellipsoid = info._format_ellipsoid()
    cell_ids = np.arange(N_CELLS, dtype=np.uint64)

    lon, lat = hpn.vertices(cell_ids, depth=LEVEL, ellipsoid=ellipsoid)
    lon = ((np.asarray(lon) + 180) % 360) - 180  # normalize to [-180, 180]
    lat = np.asarray(lat)

    # Shape (N, 4, 2) → (N, 5, 2) with closing vertex.
    corners = np.stack([lon, lat], axis=-1)
    rings = np.concatenate([corners, corners[:, :1, :]], axis=1)
    values = np.arange(N_CELLS, dtype=np.float64)
    return rings, values


def main() -> None:
    rings, values = build_healpix_rings()
    df = spatialpandas.GeoDataFrame(
        {"geometry": polygons_from_rings(rings), "value": values}
    )
    canvas = ds.Canvas(
        plot_width=CANVAS_W,
        plot_height=CANVAS_H,
        x_range=(-180.0, 180.0),
        y_range=(-90.0, 90.0),
    )
    agg = canvas.polygons(df, geometry="geometry", agg=ds.mean("value"))
    arr = agg.values  # (H, W); NaN == no coverage

    total = arr.size
    missing = int(np.isnan(arr).sum())
    print(f"canvas: {arr.shape[1]}x{arr.shape[0]} px, HEALPix L{LEVEL} ({N_CELLS} cells)")
    print(f"total pixels:    {total}")
    print(f"missing pixels:  {missing} ({100 * missing / total:.2f}%)")
    if missing:
        rows, cols = np.where(np.isnan(arr))
        print(f"unique missing cols: {np.unique(cols)}")
        print(f"unique missing rows: {np.unique(rows)}")


if __name__ == "__main__":
    main()
