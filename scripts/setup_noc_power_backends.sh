#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/setup_noc_power_backends.sh [--force] [--skip-smoke]

Build the external NoC power backends used by TSIM:
  - DSENT 0.91 checkout at external/dsent0.91
  - VNoC/ORION checkout at external/vnoc20
  - TSIM ORION link probe at tools/tsim_orion_probe

Environment overrides:
  MAKE_JOBS      parallel make jobs, default: nproc
  CC             C compiler for the ORION probe, default: gcc
  DSENT_URL/REV  DSENT checkout override
  ORION_URL/REV  VNoC/ORION checkout override
EOF
}

FORCE=0
SMOKE=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;;
    --skip-smoke) SMOKE=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAKE_JOBS="${MAKE_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
CC_BIN="${CC:-gcc}"
DSENT_URL="${DSENT_URL:-https://github.com/zzcnb1/dsent0.91.git}"
DSENT_REV="${DSENT_REV:-73ad328557e4bbc04c00d9fa76f6c8d7ecb9c53d}"
ORION_URL="${ORION_URL:-https://github.com/eigenpi/vnoc20.git}"
ORION_REV="${ORION_REV:-62a974a22265d7f970cfe5793217002699a72648}"

log() {
  printf '[setup-noc] %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_cmd git
require_cmd make
require_cmd "${CC_BIN}"
require_cmd python3

cd "${REPO_DIR}"

init_checkout() {
  local relative_path="$1"
  local url="$2"
  local revision="$3"
  local checkout="${REPO_DIR}/${relative_path}"

  if [[ -d "${checkout}" ]]; then
    log "using existing checkout ${relative_path}"
    return
  fi
  if git ls-files --stage -- "${relative_path}" | grep -q "^160000 "; then
    log "initializing submodule ${relative_path}"
    git submodule update --init "${relative_path}"
    return
  fi
  log "cloning ${relative_path} at ${revision}"
  git clone "${url}" "${checkout}"
  git -C "${checkout}" checkout --detach "${revision}"
}

init_checkout external/dsent0.91 "${DSENT_URL}" "${DSENT_REV}"
init_checkout external/vnoc20 "${ORION_URL}" "${ORION_REV}"

DSENT_DIR="${REPO_DIR}/external/dsent0.91/OENOC/dsent0.91"
ORION_DIR="${REPO_DIR}/external/vnoc20/orion3"
PROBE="${REPO_DIR}/tools/tsim_orion_probe"

if [[ ! -f "${DSENT_DIR}/Makefile" ]]; then
  echo "DSENT Makefile not found under ${DSENT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${ORION_DIR}/Makefile" ]]; then
  echo "ORION Makefile not found under ${ORION_DIR}" >&2
  exit 1
fi

if [[ "${FORCE}" -eq 1 ]]; then
  log "cleaning DSENT and ORION build products"
  make -C "${DSENT_DIR}" clean >/dev/null || true
  make -C "${ORION_DIR}" clean >/dev/null || true
  rm -f "${PROBE}"
fi

if [[ ! -x "${DSENT_DIR}/dsent" || "${FORCE}" -eq 1 ]]; then
  log "building DSENT"
  make -C "${DSENT_DIR}" -j"${MAKE_JOBS}"
else
  log "DSENT already built"
fi

if [[ ! -x "${ORION_DIR}/orion_router" || ! -f "${ORION_DIR}/libpower.a" || "${FORCE}" -eq 1 ]]; then
  log "cleaning stale VNoC/ORION objects"
  make -C "${ORION_DIR}" clean >/dev/null
  log "building VNoC/ORION library"
  make -C "${ORION_DIR}" -j"${MAKE_JOBS}" CC="${CC_BIN} -no-pie" orion_lib
  log "building VNoC/ORION router executable"
  make -C "${ORION_DIR}" -j"${MAKE_JOBS}" CC="${CC_BIN} -no-pie" orion_router
else
  log "ORION already built"
fi
chmod u+x "${ORION_DIR}/orion_router"

if [[ ! -x "${PROBE}" || "${FORCE}" -eq 1 ]]; then
  log "building TSIM ORION link probe"
  "${CC_BIN}" -no-pie -I "${ORION_DIR}" -DTECHNEW \
    "${REPO_DIR}/tools/tsim_orion_probe.c" "${ORION_DIR}/libpower.a" -lm \
    -o "${PROBE}"
else
  log "TSIM ORION link probe already built"
fi

if [[ "${SMOKE}" -eq 1 ]]; then
  log "running NoC backend smoke test"
  TSIM_REPO_DIR="${REPO_DIR}" PYTHONPATH="${REPO_DIR}" python3 - <<'PY'
import os
from pathlib import Path

from tsim_components.noc_power import NoCPowerConfig, describe
repo_root = Path(os.environ["TSIM_REPO_DIR"])
for backend in ("tsim_simple", "dsent", "orion"):
    meta = describe(NoCPowerConfig(backend=backend, frequency_hz=1.5e9), repo_root)
    print(f"{backend}: energy={meta['total_dynamic_energy_j_per_flit']:.6e} scale={meta['scale_vs_tsim_simple']:.6f}")
PY
fi

log "NoC power backends are ready"
