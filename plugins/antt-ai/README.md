# antt.ai Codex Plugin

antt.ai bundles Kimodo robot control skills:

- `load-memory`: loads robot demo motion memories.
- `load-world-scene`: loads Kimodo world scenes.
- `generate-retarget-motion`: generates a motion from a text prompt, saves it,
  and retargets it to the robot.

Both skills use the shared controller URL in `config/kimodo-controller.json`.
You can override the URL at runtime with `KIMODO_CONTROL_URL` or the helper
script `--controller-url` option.

The bundled skill descriptions tell Codex which skill to use. Requests about
memories, motions, or generated Kimodo stems use `load-memory`; requests about
worlds, scenes, or `.ply` files use `load-world-scene`; requests to generate a
new motion from a prompt and retarget it use `generate-retarget-motion`.
