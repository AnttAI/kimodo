---
name: load-world-scene
description: Load Kimodo robot demo world scenes by name through the ANTT_AI Kimodo control API. Use when the user says things like "load office world", "switch to home world", "load scene.ply", or asks Codex CLI on another laptop to control which Kimodo world is loaded.
---

# Load World Scene

## Purpose

Use this skill to ask the Kimodo app's lightweight control API to load a world
scene into connected Viser browser sessions.

Default app assumptions:
- Controller URL comes from the ANTT_AI plugin config at
  `../../config/kimodo-controller.json`.
- `KIMODO_CONTROL_URL` or `--controller-url` can override the plugin config.
- `robot_app.py` owns the worlds root via `WORLD_SCENES_ROOT`.
- Office world file: `office_world.ply`.
- No Viser WebSocket, browser automation, Chrome address-bar script, or `xdotool` is used.

The Kimodo laptop must run the app with the control API reachable on the LAN:

```bash
TEXT_ENCODER_MODE=local TEXT_ENCODER_DEVICE=cpu python -m kimodo.demo.robot_app --model kimodo-soma-rp --control-host 0.0.0.0 --control-port 8787
```

## Command

Run the helper from the plugin root:

```bash
python skills/load-world-scene/scripts/load_world_scene.py load office
```

Common examples:

```bash
python skills/load-world-scene/scripts/load_world_scene.py list
python skills/load-world-scene/scripts/load_world_scene.py status
python skills/load-world-scene/scripts/load_world_scene.py load office
python skills/load-world-scene/scripts/load_world_scene.py load home
python skills/load-world-scene/scripts/load_world_scene.py load scene.ply
```

## Workflow

1. Resolve the user's requested world name to an alias or `.ply` label.
2. Prefer aliases:
   - `office` -> `office_world.ply`
   - `home` -> `home.ply`
   - `scene` -> `scene.ply`
3. Read the controller URL from `KIMODO_CONTROL_URL`, `--controller-url`, or the plugin config.
4. POST the resolved world to `/load-world`.
5. The app-side handler resolves the world under its own `WORLD_SCENES_ROOT` and loads it into connected Viser sessions.
6. Report the exact loaded label or the exact failure.

## Codex CLI Requirement

When invoking this skill from Codex CLI, request host/escalated network
permissions if the sandbox cannot reach the Kimodo laptop IP.
