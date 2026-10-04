"""Running MPI-built binaries (chimes_lsq, LAMMPS) as plain local processes."""

from __future__ import annotations

import os

_STEP_PREFIXES = ("PMI_", "PMIX_", "MPIRUN_", "MV2_", "OMPI_", "I_MPI_", "HYDRA_")


def singleton_env() -> dict:
    """The current environment minus MPI launcher/PMI variables.

    Inside a Slurm step (srun, or a batch job's own step), an MPI binary
    started as a child process inherits the step's PMI_FD and similar
    variables but not the file descriptors behind them. MVAPICH2 then tries
    to join the step, writes to a closed descriptor and segfaults
    ("Unable to write to PMI_fd"). This was seen with chimes_lsq on a Dane
    pdebug node. Without these variables it starts as a singleton.
    """
    return {k: v for k, v in os.environ.items() if not k.startswith(_STEP_PREFIXES)}


def no_core_dumps() -> None:
    """Call before starting a native code: a crash must not drop a core file
    into the study directory (they count against the Lustre file quota).

    The limit is set on this process and inherited by its children. It is
    deliberately not a `preexec_fn`: that runs Python between fork and exec,
    which deadlocks when this process already has threads (numba after
    QUESTS, OpenMP after a solve). Seen as an `al-batch` job that hung
    starting chimes_lsq for its committee.
    """
    import resource

    try:
        hard = resource.getrlimit(resource.RLIMIT_CORE)[1]
        resource.setrlimit(resource.RLIMIT_CORE, (0, hard))
    except (ValueError, OSError):
        pass
