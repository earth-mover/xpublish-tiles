#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "datashader",
#     "matplotlib",
#     "numpy",
#     "xarray",
#     "zarr",
#     "snakeviz",
#     "gilknocker",
#     "distributed",
#     "icechunk",
#     "pooch",
#     "austin-python",
#     "yappi",
# ]
# ///
"""
Reproducible async rendering benchmark script.

This benchmark demonstrates async I/O concurrency patterns by:
1. Reading multiple time slices from a Zarr array concurrently
2. Rendering each slice with datashader to PNG format
3. Measuring the interleaving of I/O-bound reads vs CPU-bound rendering

Performance Characteristics:
- Zarr reads: ~300ms each (I/O bound, benefits from async concurrency)
- Datashader rendering: ~9ms each (CPU bound, 33x faster than reads)
- Overall workload: I/O bound - 97% time spent in zarr reads
- Async benefit: Overlaps slow I/O operations for better throughput

Visualization Features:
- Timeline view shows task interleaving with colored blocks
- Blue squares (■): Zarr reading phase
- Green squares (■): Datashader rendering phase
- Gray dots (·): Idle time between phases

To run this script reproducibly with uv:

1. Install uv if not already installed:
   curl -LsSf https://astral.sh/uv/install.sh | sh

2. Run the script (all dependencies auto-installed):
   uv run renderbench.py --ntasks=100

3. With profiling enabled:
   uv run renderbench.py --ntasks=100 --cprofile
   uv run snakeviz renderbench.prof

4. With yappi profiling (async-aware):
   uv run renderbench.py --ntasks=100 --yappi
   uv run python -c "import yappi; yappi.get_func_stats().print_all()"

5. With GIL contention monitoring:
   uv run renderbench.py --ntasks=100 --gil

6. With visual timeline:
   uv run renderbench.py --ntasks=10 --visual

7. Dataset setup:
   uv run renderbench.py --setup --format=zarr
   uv run renderbench.py --setup --format=icechunk

Usage examples:
- Setup zarr dataset: uv run renderbench.py --setup --format=zarr
- Setup icechunk dataset: uv run renderbench.py --setup --format=icechunk
- Basic async benchmark: uv run renderbench.py --ntasks=100
- Benchmark icechunk: uv run renderbench.py --ntasks=100 --format=icechunk
- Sync vs async comparison: uv run renderbench.py --ntasks=10 --visual --sync
- With profiling: uv run renderbench.py --ntasks=100 --cprofile
- With yappi profiling: uv run renderbench.py --ntasks=100 --yappi
- With GIL monitoring: uv run renderbench.py --ntasks=100 --gil
- With timeline viz: uv run renderbench.py --ntasks=10 --visual --debug
- All options: uv run renderbench.py --ntasks=100 --cprofile --gil --visual
"""
import argparse
import asyncio
import cProfile
import io
import logging
import threading
import time
from collections import defaultdict

import datashader as dsh
import matplotlib as mpl
import numpy as np
import xarray as xr
import zarr


def render_data(data: np.ndarray):
    buffer = io.BytesIO()

    cvs = dsh.Canvas(
        plot_height=256,
        plot_width=256,
        x_range=(0, data.shape[1]),
        y_range=(0, data.shape[0]),
    )

    mesh = cvs.quadmesh(data)
    shaded = dsh.transfer_functions.shade(
        mesh,
        cmap=mpl.colormaps.get_cmap("jet"),
        how="linear",
        span=(290, 330),
    )

    im = shaded.to_pil()
    im.save(buffer, format="png")
    buffer.seek(0)
    return buffer


# Global timeline tracking for visualization
timeline_events = []

async def render_time(array, itime: int, task_id: int = None, track_timeline: bool = False, use_sync: bool = False) -> io.BytesIO:
    start_time = time.perf_counter()
    
    # Log start of zarr read
    if task_id is not None:
        logging.info(f"Task {task_id}: Starting zarr read for itime={itime}")
    
    if track_timeline and task_id is not None:
        timeline_events.append(('read_start', task_id, itime, start_time))
    
    read_start = time.perf_counter()
    if use_sync:
        # Use synchronous zarr API
        data = array[itime, :, :]
    else:
        # Use asynchronous zarr API
        data = await array._async_array.getitem(selection=(itime, slice(None), slice(None)))
    read_end = time.perf_counter()
    
    # Log completion of zarr read, start of rendering
    if task_id is not None:
        logging.info(f"Task {task_id}: Zarr read complete for itime={itime} ({read_end - read_start:.3f}s), starting render")
    
    if track_timeline and task_id is not None:
        timeline_events.append(('read_end', task_id, itime, read_end))
        timeline_events.append(('render_start', task_id, itime, read_end))
    
    render_start = time.perf_counter()
    result = render_data(
        xr.DataArray(
            data,
            dims=("y", "x"),
            coords={"x": np.arange(data.shape[1]), "y": np.arange(data.shape[0])},
            name="foo",
        )
    )
    render_end = time.perf_counter()
    
    # Log completion
    if task_id is not None:
        total_time = render_end - start_time
        logging.info(f"Task {task_id}: Render complete for itime={itime} ({render_end - render_start:.3f}s), total: {total_time:.3f}s")
    
    if track_timeline and task_id is not None:
        timeline_events.append(('render_end', task_id, itime, render_end))
    
    return result


def get_icechunk_repository(path: str, mode: str = "r"):
    """Get an icechunk repository for reading or writing"""
    from icechunk import Repository, local_filesystem_storage
    
    storage = local_filesystem_storage(path)
    
    if mode == "w":
        # Create new repository
        return Repository.create(storage)
    else:
        # Open existing repository
        return Repository.open(storage)


def setup_dataset(format_type: str = "zarr"):
    """Set up the benchmark dataset using either zarr or icechunk format"""
    import distributed
    import numpy as np
    import xarray as xr
    
    print(f"Setting up dataset in {format_type} format...")
    
    # Create dask client
    client = distributed.Client()
    print(f"Dask client: {client}")
    
    try:
        # Load and process the dataset
        ds = (
            xr.tutorial.open_dataset("air_temperature", chunks={"time": 1})
            .isel(time=slice(100))
            .interp(lon=np.linspace(200, 330, 2400), lat=np.linspace(75, 15, 3600))
        )
        
        if format_type == "icechunk":
            from icechunk.xarray import to_icechunk
            filename = "airt4.icechunk"
            print(f"Writing to {filename} using icechunk...")
            
            # Create repository and writable session
            repo = get_icechunk_repository(filename, mode="w")
            session = repo.writable_session("main")
            to_icechunk(ds, session)
            session.commit("Initial dataset creation")
        else:
            filename = "airt4.zarr"
            print(f"Writing to {filename} using zarr...")
            ds.to_zarr(filename, encoding={"air": dict(chunks=(1, 200, 200))}, mode="w")
        
        print(f"Dataset created successfully: {filename}")
        print(f"Shape: {ds.air.shape}")
        print(f"Chunks: {ds.air.chunks}")
        
    finally:
        client.close()


def visualize_timeline():
    """Create a visual timeline using colored blocks"""
    if not timeline_events:
        return
    
    # Get timeline bounds
    start_time = min(event[3] for event in timeline_events)
    end_time = max(event[3] for event in timeline_events)
    duration = end_time - start_time
    
    # Get unique tasks
    tasks = sorted(set(event[1] for event in timeline_events))
    
    # ANSI color codes for different phases
    colors = {
        'read': '\033[94m',     # Blue
        'render': '\033[92m',   # Green
        'idle': '\033[90m',     # Gray
        'reset': '\033[0m'      # Reset
    }
    
    print(f"\n📊 Timeline Visualization (Total: {duration:.3f}s)")
    print(f"{'Task':<6} {'itime':<6} Timeline (■ = read, ■ = render, · = idle)")
    print("-" * 80)
    
    # Create timeline for each task
    timeline_width = 60
    for task_id in tasks:
        # Get events for this task
        task_events = [e for e in timeline_events if e[1] == task_id]
        if not task_events:
            continue
            
        # Get itime for this task
        itime = task_events[0][2]
        
        # Create timeline array
        timeline = [' '] * timeline_width
        
        # Track current state
        current_state = 'idle'
        last_time = start_time
        
        for event_type, _, _, event_time in sorted(task_events, key=lambda x: x[3]):
            # Fill previous state up to this event
            start_pos = int((last_time - start_time) / duration * timeline_width)
            end_pos = int((event_time - start_time) / duration * timeline_width)
            
            char = '■' if current_state == 'read' else ('■' if current_state == 'render' else '·')
            color = colors.get(current_state, colors['reset'])
            
            for i in range(start_pos, min(end_pos, timeline_width)):
                if i >= 0:
                    timeline[i] = f"{color}{char}{colors['reset']}"
            
            # Update state
            if event_type == 'read_start':
                current_state = 'read'
            elif event_type == 'read_end':
                current_state = 'idle'
            elif event_type == 'render_start':
                current_state = 'render'
            elif event_type == 'render_end':
                current_state = 'idle'
                
            last_time = event_time
        
        # Print the timeline
        timeline_str = ''.join(timeline)
        print(f"T{task_id:<5} {itime:<6} {timeline_str}")
    
    print("-" * 80)
    print(f"Legend: {colors['read']}■{colors['reset']} Reading  {colors['render']}■{colors['reset']} Rendering  · Idle")


async def main():
    zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1})
    parser = argparse.ArgumentParser(description="Benchmark async rendering tasks")
    parser.add_argument("--ntasks", type=int, default=1, help="Number of tasks to run")
    parser.add_argument("--gil", action="store_true", help="Enable GIL contention monitoring")
    parser.add_argument("--cprofile", action="store_true", help="Enable cProfile profiling")
    parser.add_argument("--yappi", action="store_true", help="Enable yappi profiling (async-aware)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to trace task interleaving")
    parser.add_argument("--visual", action="store_true", help="Show visual timeline with colored blocks")
    parser.add_argument("--sync", action="store_true", help="Use synchronous zarr API instead of async")
    parser.add_argument("--setup", action="store_true", help="Set up the benchmark dataset and exit")
    parser.add_argument("--format", choices=["zarr", "icechunk"], default="zarr", help="Storage format to use (default: zarr)")
    args = parser.parse_args()
    
    # Handle setup mode
    if args.setup:
        setup_dataset(args.format)
        return
    
    # Configure logging if debug mode enabled
    if args.debug:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s.%(msecs)03d - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    knocker = None
    if args.gil:
        from gilknocker import KnockKnock
        knocker = KnockKnock(1_000)

    # Open dataset based on format
    if args.format == "icechunk":
        filename = "airt4.icechunk"
        repo = get_icechunk_repository(filename, mode="r")
        session = repo.readonly_session("main")
        group = zarr.open_group(session.store, mode="r")
    else:
        filename = "airt4.zarr"
        group = zarr.open_group(filename)
    
    array = group["air"]
    print(f"Loaded dataset: {filename} (shape: {array.shape})")

    # pre-compile
    await render_time(array, 0)

    # Start profiling after pre-compilation if requested
    pr = None
    if args.cprofile:
        pr = cProfile.Profile()
        pr.enable()
    
    if args.yappi:
        import yappi
        yappi.set_clock_type("WALL")
        yappi.start(builtins=True)

    if knocker:
        knocker.start()
    
    # Start timing the main benchmark
    benchmark_start = time.perf_counter()
        
    # Create ntasks coroutines with task IDs for logging/visualization
    if args.debug or args.visual:
        tasks = [render_time(array, i % array.shape[0], task_id=i, track_timeline=args.visual, use_sync=args.sync) for i in range(args.ntasks)]
    else:
        tasks = [render_time(array, i % array.shape[0], use_sync=args.sync) for i in range(args.ntasks)]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Stop timing
    benchmark_end = time.perf_counter()
    total_time = benchmark_end - benchmark_start
    
    # Stop profiling if enabled
    if pr:
        pr.disable()
        pr.dump_stats('renderbench.prof')
    
    if args.yappi:
        import yappi
        yappi.stop()
        
        # Save function stats
        func_stats = yappi.get_func_stats()
        func_stats.save('renderbench_func.prof', type='pstat')
        
        # Save thread stats  
        thread_stats = yappi.get_thread_stats()
        with open('renderbench_thread.txt', 'w') as f:
            thread_stats.print_all(out=f)
        
        print("Yappi profiling data saved to:")
        print("  - renderbench_func.prof (function stats)")
        print("  - renderbench_thread.txt (thread stats)")
        
        # Print top functions summary
        print("\nTop 10 functions by total time:")
        func_stats.sort("ttot", "desc")
        for i, stat in enumerate(func_stats):
            if i >= 10:
                break
            print(f"{stat.name:<60} {stat.ttot:.4f}s {stat.ncall:>8} calls")
        
        yappi.clear_stats()
    
    print(f"Completed {args.ntasks} render tasks")
    print(f"Total execution time: {total_time:.3f}s")
    print(f"Average time per task: {total_time / args.ntasks:.3f}s")
    print(f"Average image buffer size: {sum(len(r.getvalue()) for r in results) / len(results):.0f} bytes")
    print(f"Active threads: {threading.active_count()}")
    
    if knocker:
        print(f"GIL contention: {knocker.contention_metric}")
    
    # Show visual timeline if requested
    if args.visual:
        visualize_timeline()
        
        # Show timing statistics
        if timeline_events:
            read_times = []
            render_times = []
            
            # Group events by task
            task_events = defaultdict(list)
            for event in timeline_events:
                task_events[event[1]].append(event)
            
            for task_id, events in task_events.items():
                events = sorted(events, key=lambda x: x[3])
                read_start = next(e[3] for e in events if e[0] == 'read_start')
                read_end = next(e[3] for e in events if e[0] == 'read_end')
                render_start = next(e[3] for e in events if e[0] == 'render_start')
                render_end = next(e[3] for e in events if e[0] == 'render_end')
                
                read_times.append(read_end - read_start)
                render_times.append(render_end - render_start)
            
            avg_read = sum(read_times) / len(read_times)
            avg_render = sum(render_times) / len(render_times)
            ratio = avg_read / avg_render if avg_render > 0 else 0
            
            print(f"\n⏱️  Timing Analysis:")
            print(f"Average read time:   {avg_read:.3f}s")
            print(f"Average render time: {avg_render:.3f}s")
            print(f"Read/Render ratio:   {ratio:.1f}:1")
            print(f"Render is {ratio:.1f}x faster than read")

if __name__ == "__main__":
    asyncio.run(main())
