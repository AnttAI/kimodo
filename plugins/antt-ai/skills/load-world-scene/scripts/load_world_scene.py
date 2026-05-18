#!/usr/bin/env python3
"""Load a Kimodo robot demo world through the app control API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
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

ALIASES = {
    "office": "office_world.ply",
    "office_world": "office_world.ply",
    "office-world": "office_world.ply",
    "home": "home.ply",
    "scene": "scene.ply",
    "default": "scene.ply",
}


def resolve_world_name(requested: str) -> str:
    normalized = requested.strip()
    alias = ALIASES.get(normalized.lower(), normalized)
    if "/" in alias or alias.endswith(".ply"):
        return alias
    return f"{alias}.ply"


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


def list_worlds(args: argparse.Namespace) -> int:
    response = controller_json(args.controller_url, "/worlds", None, args.timeout)
    if not response.get("ok"):
        raise RuntimeError(f"Kimodo control API failed: {response}")
    for world in response.get("worlds", []):
        print(world)
    return 0


def load_world(args: argparse.Namespace) -> int:
    world = resolve_world_name(args.world)
    response = controller_json(
        args.controller_url,
        "/load-world",
        {"world": world},
        args.timeout,
    )
    if not response.get("ok"):
        raise RuntimeError(f"Kimodo control API failed: {response}")
    print(f"loaded:{response.get('world', world)}")
    return 0


def status(args: argparse.Namespace) -> int:
    response = controller_json(args.controller_url, "/status", None, args.timeout)
    print(json.dumps(response, sort_keys=True))
    return 0


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
        help="Seconds to wait for the Kimodo control API.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available worlds.")
    list_parser.set_defaults(func=list_worlds)

    load_parser = subparsers.add_parser("load", help="Load a world by alias or name.")
    load_parser.add_argument("world", help="World alias, file name, or app world label.")
    load_parser.set_defaults(func=load_world)

    status_parser = subparsers.add_parser("status", help="Check the Kimodo control API.")
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
