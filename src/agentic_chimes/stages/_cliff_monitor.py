"""Parses DLARS solver logs to detect the ill-conditioning "cliff" near the
end of the LARS path and decide when to stop and finalize from the last
stable iteration, formalizing a previously hand-rolled bash workaround.

Log format confirmed against
codes/chimes_lsq-LLfork/contrib/dlars/src/dlars.C:
  "Finished iteration <j>"
  "Time for iteration <j> = <sec> seconds"
  "L1 norm of solution: <v> RMS Error: <v> Objective fn: <v> Number of vars: <n>"
Cliff failure signatures (dlars.C ~1119-1398):
  "Non-incremental Cholesky failed"
  "Failed to add a row to the Cholesky decomposition"
  "Failed to remove a row from the Cholesky decomposition"
  "Cholesky solution test failed"
  "Iteration failed: continuing"

Pure log parsing, no subprocess/Slurm dependency, so it's unit-testable
against a synthetic log fixture without a live DLARS run (see
tests/unit/test_cliff_monitor.py). Wired into a live solve in the HPC-layer
phase (stages/solve.py's dlars/dlasso path), once hpc/slurm.py can stream a
running job's log for `feed()` to consume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

FAILURE_SIGNATURES = (
    "Non-incremental Cholesky failed",
    "Failed to add a row to the Cholesky decomposition",
    "Failed to remove a row from the Cholesky decomposition",
    "Cholesky solution test failed",
    "Iteration failed: continuing",
)

_ITER_RE = re.compile(r"Finished iteration (\d+)")
_STOP_RE = re.compile(r"^Stopping: (.+)$")


@dataclass
class CliffDecision:
    action: str  # "continue" | "finalize_from_restart" | "abort" | "converged"
    target_iteration: Optional[int] = None
    evidence: list = field(default_factory=list)


class CliffMonitor:
    """Stateful line-by-line log parser. Feed lines via `feed()`/`feed_lines()`;
    call `poll()` once per polling window to get a CliffDecision.

    Detection is iteration-rate collapse while a failure signature is
    present in the tail (not just "N failures seen"), matching the
    hand-validated fix: a hard hang never happens on the observed cliff,
    DLARS limps at ~1 iter/min instead of ~4/sec while emitting Cholesky
    failures. `finalize_margin` iterations are subtracted from the
    iteration at first failure, reproducing the pre-cliff solution (LARS's
    path is deterministic, so a fresh run capped at `--iterations=target`
    reproduces it exactly).
    """

    def __init__(self, *, min_iter_advance: int = 10, max_stall_polls: int = 3, finalize_margin: int = 3):
        self.min_iter_advance = min_iter_advance
        self.max_stall_polls = max_stall_polls
        self.finalize_margin = finalize_margin

        self.last_iteration = 0
        self.failure_lines: list = []
        self.first_failure_iteration: Optional[int] = None
        self.converged = False
        self.stall_polls = 0
        self._iteration_at_last_poll = 0

    def feed(self, line: str) -> None:
        m = _ITER_RE.search(line)
        if m:
            self.last_iteration = int(m.group(1))
        for sig in FAILURE_SIGNATURES:
            if sig in line:
                self.failure_lines.append(line.strip())
                if self.first_failure_iteration is None:
                    self.first_failure_iteration = self.last_iteration
        if _STOP_RE.search(line):
            self.converged = True

    def feed_lines(self, lines) -> None:
        for line in lines:
            self.feed(line)

    def poll(self) -> CliffDecision:
        if self.converged:
            return CliffDecision(action="converged", evidence=["solver emitted a 'Stopping: ...' line"])

        advance = self.last_iteration - self._iteration_at_last_poll
        self._iteration_at_last_poll = self.last_iteration

        stalling = bool(self.failure_lines) and advance < self.min_iter_advance
        self.stall_polls = self.stall_polls + 1 if stalling else 0

        if self.stall_polls >= self.max_stall_polls:
            base = self.first_failure_iteration if self.first_failure_iteration is not None else self.last_iteration
            target = max(base - self.finalize_margin, 0)
            return CliffDecision(
                action="finalize_from_restart",
                target_iteration=target,
                evidence=list(self.failure_lines[-5:]),
            )

        return CliffDecision(action="continue")
