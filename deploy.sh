#!/usr/bin/env bash
set -Eeuo pipefail

# ----------------------------
# Config
# ----------------------------
BRANCH="master"
PM2_NAME="emergensee-facerecognition"
VENV_DIR=".venv"
DEPLOY_BRANCH=""

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use sudo only when needed
if [[ "${EUID}" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

# ----------------------------
# Helpers
# ----------------------------
log() {
  echo "[deploy] $*"
}

fail() {
  echo "[deploy][error] $*" >&2
  exit 1
}

on_error() {
  local exit_code=$?
  local line_no=$1
  echo "[deploy][error] Failed at line ${line_no} (exit code: ${exit_code})" >&2
  exit "${exit_code}"
}
trap 'on_error $LINENO' ERR

get_env_value() {
  local env_file="$1"
  local key="$2"
  local line

  line="$(grep -E "^${key}=" "${env_file}" | tail -n1 || true)"
  if [[ -z "${line}" ]]; then
    echo ""
    return
  fi

  echo "${line#*=}"
}

validate_required_env_var() {
  local env_file="$1"
  local key="$2"
  local value

  value="$(get_env_value "${env_file}" "${key}")"
  if [[ -z "${value}" ]]; then
    fail "Missing required env var ${key} in ${env_file}"
  fi
}

validate_env_files() {
  local env_file="${REPO_ROOT}/.env"

  [[ -f "${env_file}" ]] || fail "Missing ${env_file}"

  validate_required_env_var "${env_file}" "MONGO_URI"
  validate_required_env_var "${env_file}" "MONGO_DB_NAME"
}

detect_deploy_branch() {
  local configured_branch="$1"

  if git ls-remote --exit-code --heads origin "${configured_branch}" >/dev/null 2>&1; then
    echo "${configured_branch}"
    return
  fi

  for fallback_branch in master main; do
    if git ls-remote --exit-code --heads origin "${fallback_branch}" >/dev/null 2>&1; then
      echo "${fallback_branch}"
      return
    fi
  done

  fail "Could not detect a deploy branch on origin. Set BRANCH to an existing remote branch."
}

# ----------------------------
# Preflight checks
# ----------------------------
command -v git     >/dev/null 2>&1 || fail "git is not installed"
command -v python3 >/dev/null 2>&1 || fail "python3 is not installed"
command -v pm2     >/dev/null 2>&1 || fail "pm2 is not installed"

python3 -c "import sys; assert sys.version_info >= (3,11), 'Python 3.11+ required'" \
  || fail "Python 3.11 or higher is required"

if [[ -n "${SUDO}" ]]; then
  command -v sudo >/dev/null 2>&1 || fail "sudo is not installed"
  sudo -n true >/dev/null 2>&1 || fail "sudo requires a password. Run as root or configure passwordless sudo for deploy user."
fi

# ----------------------------
# 1) Git pull
# ----------------------------
cd "${REPO_ROOT}"
DEPLOY_BRANCH="$(detect_deploy_branch "${BRANCH}")"
log "Force syncing repository to origin/${DEPLOY_BRANCH}..."
git fetch origin "${DEPLOY_BRANCH}"
git reset --hard "origin/${DEPLOY_BRANCH}"

log "Validating environment files..."
validate_env_files

# ----------------------------
# 2) Python venv + dependencies
# ----------------------------
mkdir -p "${REPO_ROOT}/logs"

log "Setting up Python virtual environment at ${VENV_DIR}..."
if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

log "Installing/updating dependencies..."
"${VENV_DIR}/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/bin/pip" install -r requirements.txt --quiet

# ----------------------------
# 3) Start or restart via PM2
# ----------------------------
PYTHON="${REPO_ROOT}/${VENV_DIR}/bin/python3"
[[ -f "${PYTHON}" ]] || fail "python3 not found in venv after install"

log "Deploying service via PM2 (${PM2_NAME})..."
if pm2 describe "${PM2_NAME}" >/dev/null 2>&1; then
  log "Deleting existing PM2 process: ${PM2_NAME}..."
  pm2 delete "${PM2_NAME}"
fi

ENV_FILE="${REPO_ROOT}/.env"
MONGO_URI="$(get_env_value "${ENV_FILE}" "MONGO_URI")"
MONGO_DB_NAME="$(get_env_value "${ENV_FILE}" "MONGO_DB_NAME")"

log "Starting PM2 process: ${PM2_NAME}..."
pm2 start "${PYTHON}" \
  --name "${PM2_NAME}" \
  --cwd "${REPO_ROOT}" \
  --output "${REPO_ROOT}/logs/out.log" \
  --error "${REPO_ROOT}/logs/err.log" \
  --log-date-format "YYYY-MM-DD HH:mm:ss" \
  --env PYTHONPATH="${REPO_ROOT}/src" \
  --env MONGO_URI="${MONGO_URI}" \
  --env MONGO_DB_NAME="${MONGO_DB_NAME}" \
  --env ENVIRONMENT=production \
  --env HOST=0.0.0.0 \
  --env PORT=8000 \
  -- -m uvicorn main:app --app-dir src --host 0.0.0.0 --port 8000 --root-path /face

pm2 save

# ----------------------------
# 4) Done
# ----------------------------
log "Deployment completed successfully."
echo
pm2 status
