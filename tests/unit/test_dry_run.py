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


def test_hours_to_slurm_time_conversion():
    assert dry_run.hours_to_slurm_time(1.0) == "01:00:00"
    assert dry_run.hours_to_slurm_time(0.5) == "00:30:00"
    assert dry_run.hours_to_slurm_time(4.0) == "04:00:00"
    assert dry_run.hours_to_slurm_time(1.0 / 60) == "00:01:00"
    assert dry_run.hours_to_slurm_time(36.0) == "36:00:00"


def test_hours_to_slurm_time_rejects_non_positive():
    with pytest.raises(ValueError):
        dry_run.hours_to_slurm_time(0)
    with pytest.raises(ValueError):
        dry_run.hours_to_slurm_time(-1)


def test_rendered_script_never_has_a_bare_decimal_walltime():
    # regression test for a real bug: sbatch -t wants HH:MM:SS, not a bare
    # decimal hour count like "1.0" -- Slurm parses a bare number as
    # *minutes* and rejects the decimal point outright. This was live
    # (never caught by earlier tests, which only checked for substrings
    # like "--ntasks-per-node 112") until the first real (non-dry-run)
    # submission was attempted.
    profile = machines.load_profile("dane")
    rendered = dry_run.render_sbatch_script(
        profile, job_name="test", commands=["echo hi"], nodes=1,
        ntasks_per_node=112, walltime_hours=1.5, queue="debug",
    )
    assert "-t 1.5" not in rendered.script
    assert "-t 01:30:00" in rendered.script
