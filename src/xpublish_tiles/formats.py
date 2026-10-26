"""JSON encodings of extracted point data (see ``extract.extract_points``)."""

import math
from collections.abc import Hashable, Sequence
from typing import Any

import cftime
import numpy as np
import pandas as pd

import xarray as xr
from xpublish_tiles.extract import POINT_DIM
from xpublish_tiles.lib import timedelta_to_iso8601

CRS84 = "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
COVJSON_MEDIA_TYPE = "application/prs.coverage+json"


def json_values(values: np.ndarray) -> list[Any]:
    """Flatten ``values`` to JSON-serializable Python values; NaN/NaT become None."""
    values = np.asarray(values).ravel()
    match values.dtype.kind:
        case "M":
            return [
                None if pd.isna(t) else t.isoformat() for t in pd.DatetimeIndex(values)
            ]
        case "m":
            return [None if pd.isna(t) else timedelta_to_iso8601(t) for t in values]
        case "f":
            return [v if math.isfinite(v) else None for v in values.tolist()]
        case "O":
            return [
                v.isoformat() if isinstance(v, cftime.datetime) else v for v in values
            ]
        case _:
            return values.tolist()


def points_to_geojson(
    result: xr.Dataset, *, lon: Sequence[float], lat: Sequence[float]
) -> dict[str, Any]:
    """A GeoJSON FeatureCollection with one Feature per point in ``result``.

    ``lon``/``lat`` are the requested points, indexed by the ``point`` coordinate.
    Variables and coordinates become properties; object-typed ones (e.g. geometries)
    are skipped. Every non-point dim must already be reduced.
    """
    features = []
    for point in result[POINT_DIM].values:
        values = result.sel({POINT_DIM: point})
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon[point], lat[point]]},
                "properties": {
                    str(name): json_values(var.values)[0]
                    for name, var in values.variables.items()
                    if name != POINT_DIM and var.dtype.kind != "O"
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _axis_name(result: xr.Dataset, name: Hashable) -> str:
    """The CoverageJSON axis for coordinate ``name``: ``t``, ``z``, or itself."""
    axes = result.cf.axes
    if name in axes.get("T", ()) or (
        name in result.coords and result[name].dtype.kind == "M"
    ):
        return "t"
    if name in axes.get("Z", ()):
        return "z"
    return str(name)


def _domain_type(npoints: int, varying: set[str]) -> str | None:
    if npoints == 1:
        if varying <= {"t"}:
            return "PointSeries" if "t" in varying else "Point"
        if varying == {"z"}:
            return "VerticalProfile"
    elif varying <= {"t"}:
        return "MultiPointSeries" if "t" in varying else "MultiPoint"
    return None


def _referencing(result: xr.Dataset, names: dict[Hashable, str]) -> list[dict]:
    referencing: list[dict] = [
        {
            "coordinates": ["x", "y"],
            "system": {"type": "GeographicCRS", "id": CRS84},
        }
    ]
    by_axis = {axis: name for name, axis in names.items()}
    if "t" in by_axis:
        referencing.append(
            {
                "coordinates": ["t"],
                "system": {"type": "TemporalRS", "calendar": "Gregorian"},
            }
        )
    if "z" in by_axis:
        z = result[by_axis["z"]]
        referencing.append(
            {
                "coordinates": ["z"],
                "system": {
                    "type": "VerticalCRS",
                    "cs": {
                        "csAxes": [
                            {
                                "name": {"en": z.attrs.get("long_name", str(z.name))},
                                "direction": (
                                    "down" if z.attrs.get("positive") == "down" else "up"
                                ),
                                **(
                                    {"unit": {"symbol": z.attrs["units"]}}
                                    if "units" in z.attrs
                                    else {}
                                ),
                            }
                        ]
                    },
                },
            }
        )
    return referencing


def _parameter(da: xr.DataArray) -> dict[str, Any]:
    label = da.attrs.get("long_name", str(da.name))
    parameter: dict[str, Any] = {
        "type": "Parameter",
        "description": {"en": label},
        "observedProperty": {"label": {"en": label}},
    }
    if "standard_name" in da.attrs:
        parameter["observedProperty"]["id"] = da.attrs["standard_name"]
    if "units" in da.attrs:
        parameter["unit"] = {"symbol": da.attrs["units"]}
    return parameter


def _data_type(da: xr.DataArray) -> str:
    match da.dtype.kind:
        case "f":
            return "float"
        case "i" | "u":
            return "integer"
        case _:
            return "string"


def points_to_covjson(
    result: xr.Dataset, *, lon: Sequence[float], lat: Sequence[float]
) -> dict[str, Any]:
    """A CoverageJSON Coverage of the points in ``result``.

    One point is a ``Point``, ``PointSeries`` or ``VerticalProfile``; several share
    a ``composite`` axis as a ``MultiPoint`` or ``MultiPointSeries``. Other
    combinations of dims get axes named after them and no ``domainType``. With no
    points, this is an empty CoverageCollection.
    """
    points = result[POINT_DIM].values
    if points.size == 0:
        return {"type": "CoverageCollection", "coverages": []}

    single = points.size == 1
    if single:
        result = result.squeeze(POINT_DIM)
        axes: dict[str, dict] = {
            "x": {"values": [lon[points[0]]]},
            "y": {"values": [lat[points[0]]]},
        }
    else:
        axes = {
            "composite": {
                "dataType": "tuple",
                "coordinates": ["x", "y"],
                "values": [[lon[p], lat[p]] for p in points],
            }
        }

    dims = [d for d in result.dims if d != POINT_DIM]
    names: dict[Hashable, str] = {d: _axis_name(result, d) for d in dims}
    for name, coord in result.coords.items():
        if coord.ndim == 0 and (axis := _axis_name(result, name)) in ("t", "z"):
            names[name] = axis
    for name, axis in names.items():
        values = (
            result[name].values
            if name in result.coords
            else np.arange(result.sizes[name])
        )
        axes[axis] = {"values": json_values(values)}

    ranges = {}
    for var_name, da in result.data_vars.items():
        ranges[str(var_name)] = {
            "type": "NdArray",
            "dataType": _data_type(da),
            "axisNames": ["composite" if d == POINT_DIM else names[d] for d in da.dims],
            "shape": list(da.shape),
            "values": json_values(da.values),
        }

    domain: dict[str, Any] = {"type": "Domain"}
    varying = {names[d] for d in dims}
    if domain_type := _domain_type(points.size, varying):
        domain["domainType"] = domain_type
    domain |= {"axes": axes, "referencing": _referencing(result, names)}
    return {
        "type": "Coverage",
        "domain": domain,
        "parameters": {str(n): _parameter(da) for n, da in result.data_vars.items()},
        "ranges": ranges,
    }
