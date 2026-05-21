from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List

from .config import Settings


def _torchrun_cmd(settings: Settings, job_json: Path, rdzv_id: str) -> List[str]:
    repo = Path(settings.repo_root).resolve()
    script = repo / "generate_job.py"
    return [
        settings.torchrun_bin,
        f"--nnodes={settings.nnodes}",
        f"--nproc_per_node={settings.nproc_per_node}",
        f"--node_rank={settings.node_rank}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={settings.master_addr}:{settings.master_port}",
        f"--rdzv_id={rdzv_id}",
        str(script),
        "--job_json",
        str(job_json.resolve()),
    ]


def _env_for_child(settings: Settings) -> dict:
    env = os.environ.copy()
    repo = str(Path(settings.repo_root).resolve())
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo}:{prev}" if prev else repo
    return env


def launch_generate_job(
    settings: Settings,
    job_json: Path,
    rdzv_id: str,
) -> int:
    cmd = _torchrun_cmd(settings, job_json, rdzv_id)
    env = _env_for_child(settings)
    repo = Path(settings.repo_root).resolve()

    proc = subprocess.run(
        cmd,
        cwd=str(repo),
        env=env,
    )
    return int(proc.returncode)