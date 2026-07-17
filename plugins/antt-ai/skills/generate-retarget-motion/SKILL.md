---
name: generate-retarget-motion
description: Generate a Kimodo SOMA motion from a user prompt, save it into the app memory root, and retarget it to the ANTT/T3 robot through the ANTT_AI Kimodo control API. Use when the user asks to create/generate a new motion from text and then save or retarget it for the robot.
---

# Generate And Retarget Motion

## Purpose

Use this skill to ask the Kimodo app to run the full robot asset pipeline:

1. Generate a SOMA motion from a text prompt.
2. Save the generated motion as BVH/NPZ under the active memories root.
3. Run soma-retargeter to create the T3 robot CSV and wheel CSV.
4. Load the saved memory/T3 preview back into connected Viser sessions.

The Kimodo app must be running a SOMA model with the control API reachable on
the LAN:

```bash
TEXT_ENCODER_MODE=local TEXT_ENCODER_DEVICE=cpu python -m kimodo.demo.robot_app --model kimodo-soma-rp --control-host 0.0.0.0 --control-port 8787
```

## Command

Run the helper from the plugin root:

```bash
python skills/generate-retarget-motion/scripts/generate_retarget_motion.py run "a person waves hello"
```

Common options:

```bash
python skills/generate-retarget-motion/scripts/generate_retarget_motion.py run "pick item from the shelf" --duration 6 --seed 42
python skills/generate-retarget-motion/scripts/generate_retarget_motion.py run "shake hands with a person" --stem generated/kimodo_shake_hands_custom
python skills/generate-retarget-motion/scripts/generate_retarget_motion.py status generate_retarget_abc123def456
```

The helper automatically sets prompt durations to avoid controller conflicts:

- `pick object N from rack M shelf S`: the shelf-pick minimum plus 3 seconds.
- `move/walk/go/navigate to rack N`: the route planner's required walking time plus 1 second.

## Workflow

1. Read the controller URL from `KIMODO_CONTROL_URL`, `--controller-url`, or the plugin config.
2. POST the prompt and optional generation settings to `/generate-retarget`.
3. Poll `/job?job_id=...` until the app reports `done` or `error`.
4. Report the exact saved BVH path, retarget CSV path, and log path.

## Codex CLI Requirement

When invoking this skill from Codex CLI, request host/escalated network
permissions if the sandbox cannot reach the Kimodo laptop IP.
