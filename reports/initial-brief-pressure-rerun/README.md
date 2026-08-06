# Initial-brief controlled-cache pressure rerun

The predeclared replacement for the failed pressure criterion is
[`cluster/initial-brief-pressure-rerun-manifest.json`](../../cluster/initial-brief-pressure-rerun-manifest.json).
The executable GPU procedure is
[`NUS_GPU_PRESSURE_RERUN_INSTRUCTIONS.md`](../../NUS_GPU_PRESSURE_RERUN_INSTRUCTIONS.md).

This directory is reserved for a compact, independently validated GPU
handover. The original 24 low-occupancy runs remain preserved under
`reports/initial-brief-closure/20260805-222246-gpu-executor/`; they are not
overwritten or relabelled.

A result subdirectory is accepted only when
`scripts/summarize_pressure_replays.py` confirms all 24 regime-runs, all
predeclared cache and counter checks, at least 90% sampled occupancy, and a
positive sampled eviction count in every regime-run. Raw per-request outputs,
server logs, and metrics remain outside Git in a checksummed archive.
