# Kimodo View T2 Robot Playback

This guide explains how to run `kimodo_view` with the T2 motion browser and optionally sync the visualized arm motion to the real dual Nero robot.

## 1. Start the robot controllers

Open a terminal and launch the dual AGX arm controller:

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

Wait until both arms print that all joints are enabled.

The Kimodo wrappers default to `ROS_DOMAIN_ID=10`. If your robot is on another ROS domain, set it when starting Kimodo:

```bash
KIMODO_ROS_DOMAIN_ID=<domain_id> scripts/start_t2_robot_viewer.sh
```

## 2. Start the Kimodo viewer

Open a second terminal:

```bash
cd /home/jony/important/kimodo
scripts/start_t2_robot_viewer.sh
```

Open the viewer in a browser:

```text
http://127.0.0.1:7876
```

The launcher uses:

```text
/home/jony/important/soma-retargeter/assets/motions
```

as the motion browser folder. It expects matching files under:

```text
bvh/<clip>.bvh
t2_csv/<clip>.csv
```

## 3. Choose a motion

In the right panel:

1. Open `Motion Browser`.
2. Pick a `Motion Clip`.
3. Click `Load Motion Set`.

Kimodo loads the BVH for visualization and the matching T2 CSV for robot arm commands.

## 4. Play the visualizer

Use the `Playback` controls:

1. `Frame` moves to an exact frame.
2. `Speed` controls playback speed.
3. `Play` starts or pauses playback.

Speed is a multiplier over the viewer FPS. The default viewer FPS is 30.

Examples:

```text
Speed 0.1 = about 3 motion frames per second
Speed 1.0 = about 30 motion frames per second
Speed 5.0 = about 150 motion frames per second
```

## 5. Connect the robot

Tick the `Connect Robot` checkbox in `Playback`.

When checked, Kimodo streams the same frame shown in the visualizer to the robot:

```text
/right_arm/control/move_j
/left_arm/control/move_j
```

Only the seven arm joints per side are sent:

```text
joint1, joint2, joint3, joint4, joint5, joint6, joint7
```

No body, root, leg, head, waist, or gripper columns are sent to the physical robot.

The robot stays synchronized with the visualizer. If you move the frame slider, press play, pause, or change speed, the robot receives the corresponding visualizer frame.

## 6. Dry-run without the robot

To test the same stream path without publishing to ROS:

```bash
cd /home/jony/important/kimodo
KIMODO_ROBOT_DRY_RUN=1 scripts/start_t2_robot_viewer.sh
```

Then tick `Connect Robot` in the UI. The terminal prints the streamed right and left arm joint values instead of commanding the robot.

## 7. Run in the background

Start the viewer detached:

```bash
cd /home/jony/important/kimodo
setsid -f bash -lc 'scripts/start_t2_robot_viewer.sh </dev/null >/tmp/kimodo_t2_viewer.log 2>&1'
```

Check that it is running:

```bash
pgrep -x kimodo_view
tail -f /tmp/kimodo_t2_viewer.log
```

Stop it:

```bash
pkill -x kimodo_view
```

## 8. Useful checks

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

## 9. Common problems

If the robot does not move:

1. Confirm the AGX controller is still running.
2. Confirm Kimodo and the robot are on the same `ROS_DOMAIN_ID`.
3. Confirm both `/control/move_j` topics have one subscriber.
4. Try dry-run mode to verify Kimodo is producing arm frames.
5. Use a slow speed first, then increase speed only after the motion looks safe.

If the viewer does not start:

```bash
cd /home/jony/important/kimodo
tail -100 /tmp/kimodo_t2_viewer.log
```

If port `7876` is busy:

```bash
VIEWER_PORT=7877 scripts/start_t2_robot_viewer.sh
```
