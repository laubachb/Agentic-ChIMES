"""Regression-tests the Dane 1-CPU sbatch gotcha without spending any
allocation: every rendered script must carry an explicit --ntasks-per-node,
and the correct account/partition per machine."""

import pytest

from agentic_chimes import machines
from agentic_chimes.hpc import dry_run


def test_dane_default_ntasks_per_node_is_112():
    profile = machines.load_profile("dane")
    assert profile.default_ntasks_per_node == 112


def test_dane_script_never_bare_dash_n_1():
    profile = machines.load_profile("dane")
    rendered = dry_run.render_sbatch_script(
        profile,
        job_name="test",
        commands=["echo hi"],
        nodes=1,
        ntasks_per_node=profile.default_ntasks_per_node,
        walltime_hours=1,
        queue="debug",
    )
    assert "--ntasks-per-node 112" in rendered.script
    assert "-A pls2" in rendered.script
    assert "-p pdebug" in rendered.script
    # never a bare "-N 1" sbatch directive without an ntasks-per-node line
    lines = rendered.script.splitlines()
    n_lines = [ln for ln in lines if ln.startswith("#SBATCH -N")]
    assert n_lines and any("--ntasks-per-node" in ln for ln in lines)


def test_render_refuses_missing_ntasks_per_node():
    profile = machines.load_profile("dane")
    with pytest.raises(ValueError):
        dry_run.render_sbatch_script(
            profile,
            job_name="test",
            commands=["echo hi"],
            nodes=1,
            ntasks_per_node=0,
            walltime_hours=1,
            queue="debug",
        )


def test_stampede3_queue_translation():
    profile = machines.load_profile("stampede3")
    rendered = dry_run.render_sbatch_script(
        profile,
        job_name="test",
        commands=["echo hi"],
        nodes=1,
        ntasks_per_node=profile.default_ntasks_per_node,
        walltime_hours=1,
        queue="debug",
    )
    assert "-p skx-dev" in rendered.script
    assert "-A TG-CHM250118" in rendered.script


def test_all_builtin_profiles_set_ntasks_default():
    for name in machines.available_profiles():
        profile = machines.load_profile(name)
        assert profile.default_ntasks_per_node, f"{name} profile has no default_ntasks_per_node"
