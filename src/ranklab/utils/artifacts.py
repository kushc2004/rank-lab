from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def atomic_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)
    return path


def sha256_files(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def environment_snapshot() -> str:
    packages = sorted(
        {
            f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
        },
        key=str.lower,
    )
    return "\n".join(
        [
            f"python={sys.version.replace(chr(10), ' ')}",
            f"platform={platform.platform()}",
            "packages:",
            *packages,
        ]
    ) + "\n"
