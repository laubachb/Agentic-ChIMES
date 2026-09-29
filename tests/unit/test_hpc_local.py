"""An MPI binary launched as a child of a Slurm step must not inherit the
step's PMI variables (MVAPICH2 segfaults on the dangling PMI_FD)."""

from agentic_chimes.hpc.local import singleton_env


def test_singleton_env_strips_launcher_variables(monkeypatch):
    for k in ("PMI_FD", "PMI_RANK", "PMIX_NAMESPACE", "MV2_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"):
        monkeypatch.setenv(k, "1")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("PATH_KEEP_TEST", "x")
    env = singleton_env()
    assert not any(k.startswith(("PMI", "MV2_", "OMPI_")) for k in env)
    assert env["SLURM_JOB_ID"] == "123" and env["PATH_KEEP_TEST"] == "x"
