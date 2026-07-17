---
name: load-memory
description: Load Kimodo robot demo motion memories by memory-root search text, dropdown label, or generated stem through the ANTT_AI Kimodo control API. Use when the user says things like "load switch on memory", "load shake hand", "load shelf pick", "load generated/kimodo_b8557cb50f", or asks Codex CLI on another laptop to load a Kimodo memory.
---

# Load Memory

## Purpose

Use this skill to ask the Kimodo app's lightweight control API to load a motion
memory into every connected Viser browser session. The app searches each
client's current memories root, resolves the requested text to an actual BVH
memory stem, and then handles playback itself: after a memory is loaded,
`robot_app.py` sets the active session's `playing` state to `True`, and the
existing Space shortcut still toggles play/pause.

Default app assumptions:
- Controller URL comes from the ANTT_AI plugin config at
  `../../config/kimodo-controller.json`.
- `KIMODO_CONTROL_URL` or `--controller-url` can override the plugin config.
- `robot_app.py` owns the memories root, searches available BVH memories, and resolves BVH/CSV files on the Kimodo laptop.
- No local memory-folder scan is required on the Codex CLI laptop.
- No Viser WebSocket, browser automation, Chrome address-bar script, or `xdotool` is used.

The Kimodo laptop must run the app with the control API reachable on the LAN:

```bash
TEXT_ENCODER_MODE=local TEXT_ENCODER_DEVICE=cpu python -m kimodo.demo.robot_app --model kimodo-soma-rp --control-host 0.0.0.0 --control-port 8787
```

## Command

Run the helper from the plugin root:

```bash
python skills/load-memory/scripts/load_memory.py load "switch on"
```

Common examples:

```bash
python skills/load-memory/scripts/load_memory.py list
python skills/load-memory/scripts/load_memory.py status
python skills/load-memory/scripts/load_memory.py load generated/kimodo_b8557cb50f
```

## Workflow

1. Send the requested text, dropdown label, or generated stem to the controller without local alias mapping.
2. Read the controller URL from `KIMODO_CONTROL_URL`, `--controller-url`, or the plugin config.
3. POST the memory request to `/load-memory`.
4. The app-side handler searches each connected client's memory root, loads the BVH/T3 preview into connected browser sessions, falls back to T2 only when no T3 CSV exists, and starts playback.
5. Report the exact loaded stem or exact failure.

## Codex CLI Requirement

When invoking this skill from Codex CLI, request host/escalated network
permissions if the sandbox cannot reach the Kimodo laptop IP.
