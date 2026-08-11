#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
RUFF_BIN="${REPO_ROOT}/.venv/bin/ruff"

cd "${REPO_ROOT}"
QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}" "${PYTHON_BIN}" -m pytest -q
"${PYTHON_BIN}" -m compileall -q src tests
"${RUFF_BIN}" check .
git diff --check
