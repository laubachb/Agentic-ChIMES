"""CliffMonitor tested against synthetic DLARS log text -- no live solve
needed, since the log grammar is fully specified from dlars.C."""

from agentic_chimes.stages._cliff_monitor import CliffMonitor


def _clean_iterations(n_start, n_end):
    return [f"Finished iteration {i}\nL1 norm of solution: 1.0 RMS Error: 1.0 Objective fn: 1.0 Number of vars: {i}" for i in range(n_start, n_end)]


def test_clean_run_never_flags_cliff():
    mon = CliffMonitor()
    for batch_start in range(0, 200, 20):
        mon.feed_lines(_clean_iterations(batch_start, batch_start + 20))
        decision = mon.poll()
        assert decision.action == "continue"


def test_converged_run_reports_converged():
    mon = CliffMonitor()
    mon.feed_lines(_clean_iterations(0, 50))
    mon.feed("Stopping: no more iterations possible")
    decision = mon.poll()
    assert decision.action == "converged"


def test_cliff_rate_collapse_triggers_finalize():
    mon = CliffMonitor(min_iter_advance=10, max_stall_polls=3)
    # healthy run up to iteration 200
    mon.feed_lines(_clean_iterations(0, 200))
    assert mon.poll().action == "continue"

    # cliff hits at iteration 200: failures with almost no iteration advance
    stuck_lines = [
        "Non-incremental Cholesky failed",
        "Finished iteration 201",
        "Cholesky solution test failed",
        "Finished iteration 202",
        "Failed to add a row to the Cholesky decomposition",
    ]
    mon.feed_lines(stuck_lines)
    d1 = mon.poll()
    assert d1.action == "continue"  # first stall poll, not yet at threshold

    mon.feed_lines(["Iteration failed: continuing", "Finished iteration 203"])
    d2 = mon.poll()
    assert d2.action == "continue"  # second stall poll

    mon.feed_lines(["Cholesky solution test failed", "Finished iteration 203"])
    d3 = mon.poll()
    assert d3.action == "finalize_from_restart"
    assert d3.target_iteration is not None
    assert d3.target_iteration < 201  # margin subtracted from first failure iteration
    assert len(d3.evidence) > 0


def test_finalize_margin_subtracted_from_first_failure():
    mon = CliffMonitor(min_iter_advance=10, max_stall_polls=1, finalize_margin=3)
    mon.feed_lines(_clean_iterations(0, 100))
    mon.feed("Finished iteration 100")
    assert mon.poll().action == "continue"  # establish the iteration-count baseline

    mon.feed("Cholesky solution test failed")
    # no new "Finished iteration" fed since the last poll -> advance is 0 -> stalls immediately
    decision = mon.poll()
    assert decision.action == "finalize_from_restart"
    assert decision.target_iteration == 100 - 3
