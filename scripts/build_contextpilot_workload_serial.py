#!/usr/bin/env python
"""Run build_contextpilot_workload.py with ContextPilot's distance batch made serial.

At the pinned commit (1fa0a143), `compute_distances_batch` switches to a
`multiprocessing.Pool` once a query meets 1,000+ tree nodes, and its worker is a
function local to that call, which Python cannot pickle. The planner therefore
crashes on long request streams (here, ~4,700 planning events). The pool only
parallelises a loop in which every pair computes
`compute_distance_single(query, target, alpha)`, the same computation the
function already runs serially for small inputs. This wrapper swaps that one
function for its serial form at runtime, leaving the pinned checkout
unmodified, then runs the builder with the same arguments.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import contextpilot.server.live_index as live_index
from contextpilot.context_index.compute_distance_cpu import compute_distance_single


def serial_distances(queries, targets, alpha=0.001, num_workers=None):
    distances = np.ones((len(queries), len(targets)), dtype=np.float32)
    for i, query in enumerate(queries):
        for j, target in enumerate(targets):
            distances[i, j] = compute_distance_single(query, target, alpha)
    return distances


live_index.compute_distances_batch = serial_distances
sys.argv[0] = str(Path(__file__).with_name("build_contextpilot_workload.py"))
runpy.run_path(sys.argv[0], run_name="__main__")
