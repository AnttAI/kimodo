# Kimodo Robot Demo ANTT T2 Testing

This guide is for the new robot demo app:

```bash
python -m kimodo.demo.robot_app
```

It covers Kimodo motion generation, motion memories, SOMA BVH export, SOMA-to-T2 CSV retargeting, side-by-side SOMA/T2 preview, dry-run streaming, and optional sync to the real ANTT T2 Nero arms.

Use visual preview and dry-run first. Connect the real robot only after the SOMA and T2 preview look correct.

## 1. Motion Memories

The robot demo uses this default memories folder:

```bash
/home/jony/Downloads/soma-retargeter/assets/motions
```

Expected layout:

```text
assets/motions/
  bvh/
    clip_name.bvh
    generated/kimodo_xxxxx.bvh
  t2_csv/
    clip_name.csv
    generated/kimodo_xxxxx.csv
  kimodo_npz/
  metadata/
```

The app treats a BVH and CSV with the same relative stem as one memory:

```text
bvh/generated/kimodo_abc123.bvh
t2_csv/generated/kimodo_abc123.csv
```

If the matching CSV exists, loading the memory loads both SOMA and T2. If the CSV is missing, the UI shows the memory as needing retargeting.

## 2. Start The Robot Demo

From the Kimodo repo:

```bash
cd /home/jony/Downloads/kimodo
conda activate kimodo
TEXT_ENCODER_DEVICE=cpu python -m kimodo.demo.robot_app --model kimodo-soma-rp
```

`TEXT_ENCODER_DEVICE=cpu` keeps the local text encoder on CPU so more GPU memory is available for motion generation. `--model kimodo-soma-rp` selects the SOMA model needed by the BVH and T2 retargeting workflow.

Open:

```text
http://pop-os1:7860/
```

If port `7860` is busy:

```bash
SERVER_PORT=7861 TEXT_ENCODER_DEVICE=cpu python -m kimodo.demo.robot_app --model kimodo-soma-rp
```

## 3. Load Existing Memories

In the UI:

1. Open `Memories`.
2. Set `Root` to `/home/jony/Downloads/soma-retargeter/assets/motions`.
3. Click `Refresh Memories`.
4. Pick a memory.
5. Click `Load Memory`.

Expected result:

- The SOMA BVH plays in the visualizer.
- If `t2_csv/<same memory>.csv` exists, the T2 robot preview loads beside SOMA.
- If the CSV is missing, click `Retarget Memory to T2`.

Use the space bar or playback controls to verify motion speed and frame sync.

## 4. Retarget A Missing T2 CSV

For a selected memory with BVH but no CSV:

1. Confirm `Robot Pipeline` has the correct `soma-retargeter` path.
2. Confirm `Conda Env` is `soma-retargeter`.
3. Click `Retarget Memory to T2`.

The app runs the headless soma-retargeter flow equivalent to:

```bash
cd /home/jony/Downloads/soma-retargeter
conda run -n soma-retargeter python ./app/bvh_to_csv_converter.py \
  --config ./assets/default_bvh_to_csv_converter_config.json \
  --viewer null
```

For app-driven retargeting, the selected BVH is staged as a one-clip batch and the result is copied back under:

```text
/home/jony/Downloads/soma-retargeter/assets/motions/t2_csv/<same relative stem>.csv
```

After retargeting completes, the memory should refresh and load with the new T2 CSV.

## 5. Build Memories From Generated Motion

In the robot demo:

1. Generate a motion with the Kimodo controls.
2. Inspect the SOMA motion in the visualizer.
3. Click `Save Generated as Memory`.

The app writes a new random memory:

```text
bvh/generated/kimodo_<random>.bvh
kimodo_npz/generated/kimodo_<random>.npz
metadata/generated/kimodo_<random>.json
```

Then:

1. Select the new `[needs csv] generated/...` memory.
2. Click `Retarget Memory to T2`.
3. Load the memory and verify SOMA and T2 side by side.

Generated BVH export uses the model's native FPS so generated memories should not be accidentally sped up by a previously loaded high-FPS BVH.

## 6. Dry-Run Robot Sync

Dry-run is enabled by default in `Sync to Real Robot`.

To test the streaming path without ROS commands reaching the robot:

1. Load a memory that has a T2 CSV.
2. Keep `Dry run` checked.
3. Click `Connect`.
4. Click `Send Current Frame`, or press play.

Expected result:

- UI status changes to connected.
- The terminal prints streamed right and left arm joint values.
- No real robot movement occurs.

The dry-run path uses:

```bash
scripts/stream_t2_robot_sync.sh
```

with:

```bash
KIMODO_ROBOT_DRY_RUN=1
```

## 7. Start The Robot Controllers

Open a separate terminal and launch the dual AGX arm controller:

```bash
cd ~/catkin_ws/src/agx_arm_ros/scripts

ros2 launch agx_arm_ctrl start_double_agx_arm.launch.py \
  left_can_port:=can_arm1 \
  right_can_port:=can_arm2 \
  left_arm_type:=nero \
  right_arm_type:=nero \
  left_speed_percent:=20 \
  right_speed_percent:=20
```

Wait until both arms report that all joints are enabled and healthy.

The Kimodo wrappers default to `ROS_DOMAIN_ID=10`. If your robot is on another ROS domain:

```bash
KIMODO_ROS_DOMAIN_ID=<domain_id> python -m kimodo.demo.robot_app
```

## 8. Connect The Real Robot

In the Kimodo UI:

1. Load a memory with a verified T2 CSV.
2. Open `Sync to Real Robot`.
3. Uncheck `Dry run`.
4. Click `Connect`.
5. Click `Send Current Frame` first.
6. If the frame is safe, use playback or `Play on Robot`.

The app publishes only the seven arm joints per side:

```text
joint1, joint2, joint3, joint4, joint5, joint6, joint7
```

Default topics:

```text
/right_arm/control/move_j
/left_arm/control/move_j
```

The app does not send T2 root, body, legs, head, waist, or gripper data to the physical robot.

## 9. Useful ROS Checks

Check that the robot topics are visible:

```bash
ROS_DOMAIN_ID=10 ros2 topic info -v /right_arm/control/move_j --no-daemon
ROS_DOMAIN_ID=10 ros2 topic info -v /left_arm/control/move_j --no-daemon
```

Check feedback:

```bash
ROS_DOMAIN_ID=10 ros2 topic echo --once /right_arm/feedback/joint_states
ROS_DOMAIN_ID=10 ros2 topic echo --once /left_arm/feedback/joint_states
```

Check arm status:

```bash
ROS_DOMAIN_ID=10 ros2 topic echo --once /right_arm/feedback/arm_status
ROS_DOMAIN_ID=10 ros2 topic echo --once /left_arm/feedback/arm_status
```

Healthy status should show:

```text
arm_status: 0
err_status: 0
```

## 10. Troubleshooting

If the memory list is empty:

- Confirm BVH files exist under `/home/jony/Downloads/soma-retargeter/assets/motions/bvh`.
- Click `Refresh Memories`.
- Confirm `Root` points to the `assets/motions` folder, not directly to `bvh`.

If T2 preview does not move:

- Confirm the loaded CSV has changing `right_joint*_dof` or `left_joint*_dof` columns.
- Try disabling `Freeze T2 root/body` if you expect body motion.
- Load a known-good memory and compare the CSV summary in `Robot Pipeline`.

If playback crashes or is too fast:

- Restart the app and load the memory again.
- Confirm the BVH frame time is valid.
- Generated memories should be saved at model-native FPS; existing BVHs keep their own FPS.

If real robot connection fails:

- Confirm the dual Nero controller is running.
- Confirm Kimodo and the robot are on the same `ROS_DOMAIN_ID`.
- Try dry-run first to verify Kimodo is producing arm frames.
- Check that both `/control/move_j` topics have subscribers.

If the robot moves unexpectedly:

- Stop playback.
- Click `Stop Robot`.
- Re-check the T2 preview and test one frame at a time with `Send Current Frame`.
