import pytest


@pytest.fixture(autouse=True)
def _allow_local_job_dirs(monkeypatch):
    """Dry-run tests render job scripts under pytest's /tmp; the shared-
    filesystem guard (hpc.slurm.check_shared_dir) is tested explicitly."""
    monkeypatch.setenv("CHIMES_AGENT_ALLOW_LOCAL_JOB_DIRS", "1")
