"""Machine-profile abstraction: one declarative YAML per HPC target.

Built-in profiles (`dane`, `stampede3`) ship under `machines/profiles/`.
`load_profile` also accepts a path to a user-supplied YAML file, so a new
cluster (e.g. UM-ARC) can be added without touching package code -- just
point `--hpc /path/to/my_cluster.yaml` at a file following the same schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"

_BUILTIN = {p.stem: p for p in _PROFILES_DIR.glob("*.yaml")}


@dataclass
class MachineProfile:
    name: str
    job_system: str
    launcher: str
    account: str
    hosttype: str
    partitions: dict
    default_ntasks_per_node: int
    poll_interval_s: int = 60
    require_ntasks_per_node_or_exclusive: bool = True
    modules: list = field(default_factory=list)
    conda_env: Optional[str] = None
    filesystem: dict = field(default_factory=dict)
    qe: dict = field(default_factory=dict)
    source_path: Optional[Path] = None

    def queue_for(self, queue: str) -> str:
        """Translate a logical queue name ("debug"/"batch") to this machine's
        actual partition string, so callers never hardcode "pdebug" vs
        "skx-dev" themselves."""
        if queue in self.partitions:
            return self.partitions[queue]
        # allow passing a raw partition name directly (e.g. a queue this
        # profile doesn't have a logical alias for)
        return queue


_REQUIRED_KEYS = ("name", "job_system", "launcher", "account", "hosttype", "partitions", "default_ntasks_per_node")


def _load_yaml(path: Path) -> MachineProfile:
    with open(path) as f:
        data = yaml.safe_load(f)
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"machine profile {path} is missing required key(s): {missing}")
    return MachineProfile(
        name=data["name"],
        job_system=data["job_system"],
        launcher=data["launcher"],
        account=data["account"],
        hosttype=data["hosttype"],
        partitions=data["partitions"],
        default_ntasks_per_node=int(data["default_ntasks_per_node"]),
        poll_interval_s=int(data.get("poll_interval_s", 60)),
        require_ntasks_per_node_or_exclusive=bool(data.get("require_ntasks_per_node_or_exclusive", True)),
        modules=list(data.get("modules", [])),
        conda_env=data.get("conda_env"),
        filesystem=dict(data.get("filesystem", {})),
        qe=dict(data.get("qe", {})),
        source_path=path,
    )


def available_profiles() -> list[str]:
    return sorted(_BUILTIN)


def load_profile(name_or_path: str) -> MachineProfile:
    """Load a built-in profile by name (e.g. "dane"), or a user-supplied YAML
    file by path."""
    if name_or_path in _BUILTIN:
        return _load_yaml(_BUILTIN[name_or_path])
    path = Path(name_or_path)
    if path.is_file():
        return _load_yaml(path)
    raise FileNotFoundError(
        f"no built-in machine profile named {name_or_path!r} (available: "
        f"{available_profiles()}) and no such file exists to load as a "
        "custom profile"
    )
