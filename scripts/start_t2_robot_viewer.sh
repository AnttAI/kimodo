#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

KIMODO_CONDA_ENV="${KIMODO_CONDA_ENV:-kimodo}"
VIEWER_HOST="${VIEWER_HOST:-127.0.0.1}"
VIEWER_PORT="${VIEWER_PORT:-7876}"
DEFAULT_MOTIONS_ROOT="${DEFAULT_MOTIONS_ROOT:-/home/jony/important/soma-retargeter/assets/motions}"
export ROS_DOMAIN_ID="${KIMODO_ROS_DOMAIN_ID:-10}"
export ROS_LOCALHOST_ONLY="${KIMODO_ROS_LOCALHOST_ONLY:-0}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$KIMODO_CONDA_ENV"

cd "$REPO_ROOT"

if [[ "$#" -gt 0 ]]; then
  exec kimodo_view "$@"
fi

exec kimodo_view \
  --motions-root "$DEFAULT_MOTIONS_ROOT" \
  --host "$VIEWER_HOST" \
  --port "$VIEWER_PORT" \
  --t2-arms-only
