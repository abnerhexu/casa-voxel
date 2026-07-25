#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/setup_thermal_backends.sh [--force] [--skip-hotspot] [--skip-3dice] [--skip-smoke]

Install/build external thermal simulators used by TSIM:
  - HotSpot under external/hotspot-7.0
  - 3D-ICE under external/3d-ice-src

The script downloads upstream GitHub archives unless the matching local archive
already exists under external/.

Environment overrides:
  MAKE_JOBS      parallel make jobs, default: nproc
  HOTSPOT_URL    default: https://github.com/uvahotspot/HotSpot/archive/9f92256.tar.gz
  THREEDICE_REV  pinned 3D-ICE revision used by the thermal artifact
  THREEDICE_URL  archive URL derived from THREEDICE_REV
  THREEDICE_SHA256 expected SHA-256 of the pinned archive
EOF
}

FORCE=0
INSTALL_HOTSPOT=1
INSTALL_THREEDICE=1
SMOKE=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;;
    --skip-hotspot) INSTALL_HOTSPOT=0 ;;
    --skip-3dice) INSTALL_THREEDICE=0 ;;
    --skip-smoke) SMOKE=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTERNAL_DIR="${REPO_DIR}/external"
MAKE_JOBS="${MAKE_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
HOTSPOT_URL="${HOTSPOT_URL:-https://github.com/uvahotspot/HotSpot/archive/9f92256.tar.gz}"
THREEDICE_REV="${THREEDICE_REV:-2021ff93d88fabd67d5d5a2ad963ed5e51b79f5d}"
THREEDICE_URL="${THREEDICE_URL:-https://github.com/esl-epfl/3d-ice/archive/${THREEDICE_REV}.tar.gz}"
THREEDICE_SHA256="${THREEDICE_SHA256:-9bcf05a69b890603a35795c817dbc251da4c6b27b649d13c09018328ec27e97b}"
HOTSPOT_ARCHIVE="${EXTERNAL_DIR}/hotspot-v7.0.tar.gz"
THREEDICE_ARCHIVE="${EXTERNAL_DIR}/3d-ice-${THREEDICE_REV}.tar.gz"
HOTSPOT_DIR="${EXTERNAL_DIR}/hotspot-7.0"
THREEDICE_DIR="${EXTERNAL_DIR}/3d-ice-src"
THREEDICE_MARKER="${THREEDICE_DIR}/.tsim-pinned-revision"

log() {
  printf '[setup-thermal] %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

download_archive() {
  local url="$1"
  local out="$2"
  if [[ -f "${out}" && "${FORCE}" -eq 0 ]]; then
    log "using existing archive ${out}"
    return
  fi
  log "downloading ${url}"
  curl -L --fail --retry 3 --output "${out}.tmp" "${url}"
  mv "${out}.tmp" "${out}"
}

verify_sha256() {
  local file="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum "${file}" | awk '{print $1}')"
  if [[ "${observed}" != "${expected}" ]]; then
    echo "checksum mismatch for ${file}: expected ${expected}, observed ${observed}" >&2
    exit 1
  fi
}

extract_single_root_archive() {
  local archive="$1"
  local dest="$2"
  local tmp
  tmp="$(mktemp -d "${EXTERNAL_DIR}/extract.XXXXXX")"
  tar -xzf "${archive}" -C "${tmp}"
  local roots=("${tmp}"/*)
  if [[ "${#roots[@]}" -ne 1 || ! -d "${roots[0]}" ]]; then
    echo "archive ${archive} did not contain one top-level directory" >&2
    rm -rf "${tmp}"
    exit 1
  fi
  if [[ -e "${dest}" ]]; then
    if [[ "${FORCE}" -ne 1 ]]; then
      echo "${dest} already exists; use --force to replace it" >&2
      rm -rf "${tmp}"
      exit 1
    fi
    rm -rf "${dest}"
  fi
  mv "${roots[0]}" "${dest}"
  rmdir "${tmp}"
}

if [[ "${INSTALL_HOTSPOT}" -eq 1 || "${INSTALL_THREEDICE}" -eq 1 ]]; then
  require_cmd curl
  require_cmd make
  require_cmd tar
fi
if [[ "${INSTALL_THREEDICE}" -eq 1 ]]; then
  require_cmd unzip
  require_cmd sha256sum
fi

mkdir -p "${EXTERNAL_DIR}"

if [[ "${INSTALL_HOTSPOT}" -eq 1 ]]; then
  if [[ ! -x "${HOTSPOT_DIR}/hotspot" || "${FORCE}" -eq 1 ]]; then
    download_archive "${HOTSPOT_URL}" "${HOTSPOT_ARCHIVE}"
    extract_single_root_archive "${HOTSPOT_ARCHIVE}" "${HOTSPOT_DIR}"
    log "building HotSpot"
    make -C "${HOTSPOT_DIR}" -j"${MAKE_JOBS}"
  else
    log "HotSpot already built"
  fi
fi

if [[ "${INSTALL_THREEDICE}" -eq 1 ]]; then
  if [[ "${FORCE}" -eq 1 || ! -d "${THREEDICE_DIR}" ]]; then
    download_archive "${THREEDICE_URL}" "${THREEDICE_ARCHIVE}"
    verify_sha256 "${THREEDICE_ARCHIVE}" "${THREEDICE_SHA256}"
    extract_single_root_archive "${THREEDICE_ARCHIVE}" "${THREEDICE_DIR}"
    printf '%s\n' "${THREEDICE_REV}" > "${THREEDICE_MARKER}"
  elif [[ ! -f "${THREEDICE_MARKER}" || "$(<"${THREEDICE_MARKER}")" != "${THREEDICE_REV}" ]]; then
    echo "existing 3D-ICE tree has no matching revision marker; rerun with --force" >&2
    exit 1
  fi

  if [[ ! -x "${THREEDICE_DIR}/bin/3D-ICE-Emulator" || "${FORCE}" -eq 1 ]]; then
    if [[ "${FORCE}" -eq 0 ]]; then
      log "resuming pinned 3D-ICE build at ${THREEDICE_REV}"
    fi
    if [[ ! -f "${THREEDICE_DIR}/superlu_mt-4.0.0/lib/libsuperlu_mt_OPENMP.a" ]]; then
      log "building bundled SuperLU_MT"
      (
        cd "${THREEDICE_DIR}"
        bash ./install-superlumt.sh
      )
    fi
    log "building 3D-ICE"
    make -C "${THREEDICE_DIR}" -j"${MAKE_JOBS}"
  else
    log "3D-ICE already built at pinned revision ${THREEDICE_REV}"
  fi
fi

if [[ "${SMOKE}" -eq 1 ]]; then
  if [[ "${INSTALL_HOTSPOT}" -eq 1 ]]; then
    test -x "${HOTSPOT_DIR}/hotspot"
    log "HotSpot binary: ${HOTSPOT_DIR}/hotspot"
  fi
  if [[ "${INSTALL_THREEDICE}" -eq 1 ]]; then
    test -x "${THREEDICE_DIR}/bin/3D-ICE-Emulator"
    log "3D-ICE binary: ${THREEDICE_DIR}/bin/3D-ICE-Emulator"
  fi
fi

log "thermal backends are ready"