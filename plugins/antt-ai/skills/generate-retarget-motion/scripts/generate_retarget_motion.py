#!/usr/bin/env python3
"""Generate, save, and retarget a Kimodo robot motion through the app control API."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def plugin_root() -> Path:
    return Path(__file__).resolve().parents[3]


def configured_controller_url() -> str:
    env_url = os.environ.get("KIMODO_CONTROL_URL")
    if env_url:
        return env_url

    config_path = plugin_root() / "config" / "kimodo-controller.json"
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Missing ANTT_AI controller config at {config_path}. "
            "Create config/kimodo-controller.json or set KIMODO_CONTROL_URL."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid ANTT_AI controller config at {config_path}: {exc}") from exc

    controller_url = str(config.get("controllerUrl") or "").strip()
    if not controller_url:
        raise RuntimeError(
            f"Missing controllerUrl in ANTT_AI controller config at {config_path}. "
            "Set controllerUrl or use KIMODO_CONTROL_URL."
        )
    return controller_url


DEFAULT_CONTROLLER_URL = configured_controller_url()
RACK_PROMPT_RE = re.compile(r"\brack[\s_-]*(\d+)\b", re.IGNORECASE)
PICK_PROMPT_RE = re.compile(r"\bpick\s+object\s+\d+\s+from\s+rack[\s_-]*\d+\s+shelf\s+\d+\b", re.IGNORECASE)
RACK_WIDTH_M = 0.34
RACK_HUMAN_APPROACH_CLEARANCE_M = 0.45
RACK_HUMAN_SLOW_WALK_SPEED_M_S = 0.60
RACK_HUMAN_FAST_WALK_SPEED_M_S = 0.90
WORK_AREA_GRID_SECTION_M = 0.60
WORK_AREA_GRID_SHAPE = (4, 6)
WORK_AREA_SIDE_SHIFT_M = 0.60
RACK_MAP_POSITIONS = {
    1: (-1.63, 0.0, -1.20),
    2: (-1.63, 0.0, -2.40),
    3: (-1.10, 0.0, -3.43),
    4: (0.00, 0.0, -3.43),
}
RACK_MAP_YAWS_RAD = {
    3: -math.pi / 2.0,
    4: -math.pi / 2.0,
}


def repo_root() -> Path:
    return plugin_root().parents[1]


def _planner_module():
    planner_path = repo_root() / "kimodo" / "demo" / "warehouse_ui" / "rack_motion_planner.py"
    spec = importlib.util.spec_from_file_location("kimodo_rack_motion_planner", planner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load rack motion planner from {planner_path}")
    planner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = planner
    spec.loader.exec_module(planner)
    return (
        planner.cardinal_rack_route_required_seconds,
        planner.rack_width_side_approach_pose,
    )


def minimum_prompt_duration_seconds(prompt: str) -> float | None:
    if PICK_PROMPT_RE.search(prompt):
        return 7.0

    rack_match = RACK_PROMPT_RE.search(prompt)
    if rack_match is None:
        return None
    if not re.search(r"\b(move|walk|go|navigate)\b", prompt, re.IGNORECASE):
        return None

    rack_number = int(rack_match.group(1))
    if rack_number not in RACK_MAP_POSITIONS:
        return None

    cardinal_rack_route_required_seconds, rack_width_side_approach_pose = _planner_module()
    x_near = WORK_AREA_SIDE_SHIFT_M
    x_far = x_near - WORK_AREA_GRID_SHAPE[0] * WORK_AREA_GRID_SECTION_M
    z_far = -WORK_AREA_GRID_SHAPE[1] * WORK_AREA_GRID_SECTION_M
    rack_target, rack_facing_heading = rack_width_side_approach_pose(
        RACK_MAP_POSITIONS[rack_number],
        RACK_MAP_YAWS_RAD.get(rack_number, 0.0),
        RACK_WIDTH_M,
        RACK_HUMAN_APPROACH_CLEARANCE_M,
        (x_far, x_near),
        (z_far, 0.0),
    )
    required_seconds = cardinal_rack_route_required_seconds(
        approach_position=rack_target,
        final_heading=rack_facing_heading,
        fps=30.0,
        walk_speed_m_s=rack_human_walk_speed_m_s(rack_number),
        first_axis="z" if rack_number in {1, 2, 3} else "x",
    )
    return required_seconds + 1.0


def rack_human_walk_speed_m_s(rack_number: int) -> float:
    if rack_number in {3, 4}:
        return RACK_HUMAN_FAST_WALK_SPEED_M_S
    return RACK_HUMAN_SLOW_WALK_SPEED_M_S


def resolved_duration_seconds(prompt: str, requested_duration: float) -> float:
    tool_duration = minimum_prompt_duration_seconds(prompt)
    if tool_duration is None:
        return requested_duration
    return tool_duration


def controller_json(
    controller_url: str,
    endpoint: str,
    payload: dict[str, object] | None,
    timeout_sec: float,
) -> dict[str, object]:
    url = controller_url.rstrip("/") + endpoint
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    method = "GET" if payload is None else "POST"
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Kimodo control API returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Kimodo control API at {controller_url}. "
            "Make sure the Kimodo laptop is running robot_app.py with "
            "`--control-host 0.0.0.0 --control-port 8787`, both laptops are on the "
            "same network, and firewall rules allow port 8787. If this command runs "
            "inside Codex CLI with network sandboxing, rerun it with escalated/host "
            "network permissions."
        ) from exc


def job_endpoint(job_id: str) -> str:
    return "/job?" + urllib.parse.urlencode({"job_id": job_id})


def print_job(job: dict[str, object]) -> None:
    print(f"job:{job.get('job_id')}")
    print(f"status:{job.get('status')}")
    for key in (
        "stem",
        "retarget_target",
        "bvh_path",
        "csv_path",
        "t3_csv_path",
        "wheel_csv_path",
        "log_path",
        "playback_started",
        "playback_mode",
        "error",
    ):
        value = job.get(key)
        if value:
            print(f"{key}:{value}")


def run(args: argparse.Namespace) -> int:
    duration = resolved_duration_seconds(args.prompt, args.duration)
    payload: dict[str, object] = {
        "prompt": args.prompt,
        "duration_seconds": duration,
        "seed": args.seed,
        "diffusion_steps": args.diffusion_steps,
    }
    if args.stem:
        payload["stem"] = args.stem
    if args.output_root:
        payload["output_root"] = args.output_root

    response = controller_json(args.controller_url, "/generate-retarget", payload, args.timeout)
    if not response.get("ok"):
        raise RuntimeError(f"Kimodo control API failed: {response}")

    job_id = str(response.get("job_id") or "")
    if not job_id:
        raise RuntimeError(f"Kimodo control API did not return a job_id: {response}")
    print(f"started:{job_id}")

    if args.no_wait:
        return 0

    deadline = time.time() + args.wait_timeout
    while time.time() < deadline:
        job_response = controller_json(args.controller_url, job_endpoint(job_id), None, args.timeout)
        job = job_response.get("job")
        if not isinstance(job, dict):
            raise RuntimeError(f"Kimodo control API returned invalid job status: {job_response}")
        status = str(job.get("status") or "")
        if status in {"done", "error"}:
            print_job(job)
            return 0 if status == "done" else 1
        if args.verbose:
            print(f"status:{status}")
        time.sleep(args.poll_interval)

    raise RuntimeError(f"Timed out waiting for job {job_id}")


def status(args: argparse.Namespace) -> int:
    response = controller_json(args.controller_url, job_endpoint(args.job_id), None, args.timeout)
    job = response.get("job")
    if not isinstance(job, dict):
        raise RuntimeError(f"Kimodo control API returned invalid job status: {response}")
    print_job(job)
    return 0 if job.get("status") != "error" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller-url",
        default=DEFAULT_CONTROLLER_URL,
        help=f"Kimodo control API URL. Default: {DEFAULT_CONTROLLER_URL}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for each Kimodo control API request.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Generate, save, and retarget a motion prompt.")
    run_parser.add_argument("prompt", help="Text prompt to generate.")
    run_parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Motion duration in seconds. Rack/pick prompts automatically use the tool-computed duration.",
    )
    run_parser.add_argument("--seed", type=int, default=42, help="Generation seed.")
    run_parser.add_argument("--diffusion-steps", type=int, default=100, help="Denoising steps.")
    run_parser.add_argument("--stem", help="Optional memory stem to save under.")
    run_parser.add_argument("--output-root", help="Optional memories output root.")
    run_parser.add_argument("--no-wait", action="store_true", help="Start the job and return immediately.")
    run_parser.add_argument("--wait-timeout", type=float, default=900.0, help="Seconds to wait for completion.")
    run_parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between status polls.")
    run_parser.add_argument("--verbose", action="store_true", help="Print intermediate job states.")
    run_parser.set_defaults(func=run)

    status_parser = subparsers.add_parser("status", help="Check a generate-retarget job.")
    status_parser.add_argument("job_id")
    status_parser.set_defaults(func=status)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
