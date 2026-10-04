"""Worker pools that are safe to start from a process that already has threads.

Python's default on Linux is to fork workers. Forking a process whose
thread pool is live (numba after a QUESTS call, OpenMP/MKL after a large
numpy solve) can deadlock inside fork() itself: seen on Dane compute nodes
as an `al-batch` job that wrote nothing and ran out its walltime. Spawned
workers start from a clean interpreter, at the cost of ~1 s of imports
each, which is nothing next to the fits and cluster enumerations they run.
Worker functions must be importable module-level functions.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def process_pool(max_workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=max(1, int(max_workers)), mp_context=multiprocessing.get_context("spawn"))
