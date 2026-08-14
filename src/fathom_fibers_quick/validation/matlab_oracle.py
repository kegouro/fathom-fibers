from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy.io import loadmat

# Never hardcode a machine-specific MATLAB install path in source; the
# executable is provided through the FATHOM_MATLAB_EXECUTABLE environment
# variable by the operator or validation runner.
DEFAULT_MATLAB: Path | None = None
CANONICAL_SOURCE_SHA256 = "6cbb827ebfa92a2f951d3fd06cb3561d81854ddd8fc4fc9f8f7bb1151ad1f446"


@dataclass(frozen=True, slots=True)
class MatlabOracle:
    executable: Path
    harness_dir: Path

    @classmethod
    def discover(cls, repo: Path) -> MatlabOracle | None:
        configured = os.environ.get("FATHOM_MATLAB_EXECUTABLE")
        candidate = Path(configured) if configured else DEFAULT_MATLAB
        if candidate is None:
            return None
        resolved = shutil.which(str(candidate))
        if not resolved:
            return None
        return cls(Path(resolved), repo / ".validation/matlab-oracle/src")

    def batch(self, expression: str, *, timeout: float = 1800) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.executable), "-batch", expression],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

    def check(self, *, timeout: float = 180) -> dict[str, Any]:
        expression = (
            "fprintf('MATLAB_VERSION=%s\\n',version);"
            "fprintf('MATLAB_RELEASE=%s\\n',version('-release'));disp('MATLAB_BATCH_OK');"
        )
        completed = self.batch(expression, timeout=timeout)
        return {
            "available": completed.returncode == 0 and "MATLAB_BATCH_OK" in completed.stdout,
            "executable": str(self.executable),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }


def read_environment_report(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_intermediates(path: str | Path) -> dict[str, Any]:
    return {key: value for key, value in loadmat(path).items() if not key.startswith("__")}


def oracle_cache_key(
    *,
    source_tiff_sha256: str,
    matlab_release: str,
    matlab_source_sha256: str,
    profile: str,
    conversion_ratio: float | None,
    pipeline_version: str,
) -> str:
    payload = json.dumps(
        {
            "source_tiff_sha256": source_tiff_sha256,
            "matlab_release": matlab_release,
            "matlab_source_sha256": matlab_source_sha256,
            "profile": profile,
            "conversion_ratio": conversion_ratio,
            "pipeline_version": pipeline_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()
