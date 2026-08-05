from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_matlab_oracle(image_path: str | Path, output_json_path: str | Path) -> dict[str, Any]:
    """Executes MATLAB SIMPoly oracle if MATLAB is installed; otherwise handles gracefully."""
    image_path = Path(image_path).resolve()
    output_json_path = Path(output_json_path).resolve()

    matlab_bin = shutil.which("matlab")
    if not matlab_bin:
        result = {
            "run_id": f"RUN_{image_path.stem}",
            "oracle_id": "SIMPOLY_MATLAB_ORIGINAL",
            "oracle_version": "1.0.0",
            "image_id": image_path.name,
            "status": "SKIPPED_MATLAB_ABSENT",
            "message": "MATLAB binary is not found in system PATH.",
        }
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    script_dir = Path(__file__).parent.resolve()
    cmd = [
        matlab_bin,
        "-batch",
        f"addpath('{script_dir}'); run_simpoly_oracle('{image_path}', '{output_json_path}');",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        if proc.returncode == 0 and output_json_path.exists():
            return json.loads(output_json_path.read_text(encoding="utf-8"))
        else:
            result = {
                "run_id": f"RUN_{image_path.stem}",
                "oracle_id": "SIMPOLY_MATLAB_ORIGINAL",
                "oracle_version": "1.0.0",
                "image_id": image_path.name,
                "status": "FAILED_MATLAB_RUNNER",
                "message": proc.stderr or proc.stdout or "MATLAB execution failed.",
            }
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            output_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result
    except Exception as exc:
        result = {
            "run_id": f"RUN_{image_path.stem}",
            "oracle_id": "SIMPOLY_MATLAB_ORIGINAL",
            "oracle_version": "1.0.0",
            "image_id": image_path.name,
            "status": "FAILED_MATLAB_RUNNER",
            "message": str(exc),
        }
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_simpoly_oracle.py <image_path> <output_json_path>")
        sys.exit(1)

    res = run_matlab_oracle(sys.argv[1], sys.argv[2])
    print(json.dumps(res, indent=2))
