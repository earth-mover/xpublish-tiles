# Odd-Integer Window Coarsening Plan

## Problem Statement

Currently, the coarsening logic only allows **even** coarsening factors. When using xarray's `.coarsen().mean()`:
- **Data values**: Correctly averaged over the window
- **Coordinates**: Also averaged, which is problematic for 2D coordinates (curvilinear grids)

## Key Insight

With an **odd** coarsening factor `f`, there is always a **center element** at index `f // 2` within each window. Instead of averaging coordinates, we can simply **subselect the center values**.

## Implementation Summary

### Changes Made

1. **`_get_indexer_size`** (pipeline.py):
   - Fixed to use `slice.indices(dim_size)` to correctly handle negative indices and open-ended slices

2. **`get_coarsen_factors`** (pipeline.py):
   - Changed from `largest_even_le` to `largest_odd_ge` for computing factors
   - Minimum factor is now 3 (odd)
   - Only computes factors for dimensions that exceed max size
   - For global datasets (lon_spans_globe), pads longitude with wraparound to make divisible by factor

3. **`coarsen`** (pipeline.py):
   - Uses `boundary="pad"` to handle incomplete windows (NaN-padded at edges)
   - Asserts longitude is divisible by factor for global datasets (pre-padded via slicers)
   - Drops coordinates before coarsening to avoid extra work
   - Subselects coordinates at window centers rather than averaging
   - Clamps indices to valid range for incomplete last window

4. **`pad_slicers`** (lib.py):
   - Fixed wraparound padding to only add slices when `left_pad > 0` or `right_pad > 0`

### Strategy by Dataset Type

1. **Global datasets** (`grid.lon_spans_globe`):
   - Longitude: padded via slicers (wraparound) to be exactly divisible by factor
   - Latitude: uses `boundary="pad"` - incomplete windows at poles get NaN-padded

2. **Regional datasets**:
   - Uses `boundary="pad"` to include all data (incomplete windows get NaN-padded)
   - Coordinates are clamped to valid indices for last window

### Key Design Decisions

- **`boundary="pad"` instead of `"trim"`**: Ensures no data is lost at edges, especially important for latitude at poles
- **Coordinate subselection with clamping**: For incomplete last window, the center index is clamped to the last valid index
- **Assert on longitude divisibility**: Catches bugs where the slicer padding didn't work correctly

## Benefits

1. **Simpler coordinate handling**: No averaging needed, just subselection
2. **Exact coordinate values**: Coordinates are actual values from the original grid
3. **Works naturally with 2D coordinates**: No NaN issues or discontinuity problems from averaging
4. **No data loss at edges**: Using `boundary="pad"` preserves all data

## Test Status

- ✅ Type checking passes
- ✅ Basic unit tests pass
- ✅ Property test `test_property_global_render_no_transparent_tile` passes (500 examples)
