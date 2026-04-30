#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export ROS_DOMAIN_ID="${KIMODO_ROS_DOMAIN_ID:-10}"
export ROS_LOCALHOST_ONLY="${KIMODO_ROS_LOCALHOST_ONLY:-0}"

set +u
if [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi

if [[ -f "$HOME/catkin_ws/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/catkin_ws/install/setup.bash"
fi

if [[ -f /home/jony/agx_arm_ws/install/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /home/jony/agx_arm_ws/install/setup.bash
fi
set -u

ROS_PYTHON_BIN="${ROS_PYTHON:-/usr/bin/python3}"
SUBSCRIBER_WAIT="${KIMODO_ROBOT_SUBSCRIBER_WAIT:-5}"
EXTRA_ARGS=()
if [[ "${KIMODO_ROBOT_DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi
"$ROS_PYTHON_BIN" "$REPO_ROOT/kimodo/scripts/t2_robot_stream_publisher.py" \
  "${EXTRA_ARGS[@]}" \
  --wait-for-subscribers "$SUBSCRIBER_WAIT" \
  "$@"
