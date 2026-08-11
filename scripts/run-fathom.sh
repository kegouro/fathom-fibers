#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
CHECK_SCRIPT="${REPO_ROOT}/scripts/check.sh"

RUN_CHECKS=0

usage() {
    cat <<'HELP'
Uso:
  ./scripts/run-fathom.sh
  ./scripts/run-fathom.sh --check
  ./scripts/run-fathom.sh --help

Opciones:
  --check   Ejecuta la suite de verificaciones antes de abrir Fathom.
  --help    Muestra esta ayuda.

Los logs quedan bajo:
  /tmp/fathom-fibers-<usuario>/
HELP
}

while (($# > 0)); do
    case "$1" in
        --check)
            RUN_CHECKS=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Opción desconocida: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: no existe el Python del entorno virtual:" >&2
    echo "  ${PYTHON}" >&2
    echo >&2
    echo "Crea o repara .venv antes de iniciar Fathom." >&2
    exit 1
fi

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

LOG_DIR="/tmp/fathom-fibers-${USER:-user}"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/fathom-${TIMESTAMP}.log"
LATEST_LOG="${LOG_DIR}/latest.log"

ln -sfn "${LOG_FILE}" "${LATEST_LOG}"

echo "Repositorio : ${REPO_ROOT}"
echo "Python      : ${PYTHON}"
echo "Log         : ${LOG_FILE}"
echo

"${PYTHON}" -c \
    "import fathom_fibers_quick; print('Importación de fathom_fibers_quick: OK')"

if ((RUN_CHECKS)); then
    echo
    echo "Ejecutando verificaciones..."

    if [[ -x "${CHECK_SCRIPT}" ]]; then
        "${CHECK_SCRIPT}"
    else
        "${PYTHON}" -m pytest -q
        "${PYTHON}" -m compileall -q src tests
    fi

    echo "Verificaciones terminadas."
fi

echo
echo "Iniciando Fathom Fibers Quick..."
echo "Para revisar el último log:"
echo "  less ${LATEST_LOG}"
echo

set +e
"${PYTHON}" -m fathom_fibers_quick gui \
    > >(tee -a "${LOG_FILE}") \
    2> >(tee -a "${LOG_FILE}" >&2)
EXIT_CODE=$?
set -e

echo
echo "Fathom terminó con código: ${EXIT_CODE}"
echo "Log: ${LOG_FILE}"

exit "${EXIT_CODE}"
