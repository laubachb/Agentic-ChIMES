"""amat-build's --hpc dry-run path (no Slurm/chimes_lsq binary needed)."""

from types import SimpleNamespace

from agentic_chimes.stages import amat_build


def test_hpc_dry_run_renders_srun_command(tmp_path):
    fm_setup_in = tmp_path / "fm_setup.in"
    fm_setup_in.write_text("# fake fm_setup.in\n")

    args = SimpleNamespace(
        fm_setup_in=str(fm_setup_in),
        chimes_lsq_bin="/fake/path/chimes_lsq",
        machine="dane",
        queue="debug",
        walltime_hours=1.0,
        nodes=1,
        ntasks_per_node=None,
        dry_run=True,
        output_dir=str(tmp_path / "run"),
    )
    result = amat_build.run(args)

    assert result["dry_run"] is True
    script = (tmp_path / "run" / "run.cmd").read_text()
    assert "srun -n 112" in script
    assert "chimes_lsq" in script
    assert "--ntasks-per-node 112" in script  # the Dane guardrail, same as every other HPC stage
