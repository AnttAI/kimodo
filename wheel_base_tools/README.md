# T2 CSV To Differential Base Commands

This folder contains a small post-processor that converts a retargeted T2 CSV root trajectory into differential-drive base commands.

The T2 CSV stores pose per frame, not wheel commands. This tool derives:

- rounded left and right motor RPM commands for the physical base
- accumulated root ground distance

## Assumptions

For the retargeted T2 CSV used by the Kimodo robot demo:

- `root_translateX/Y/Z` are in centimeters.
- `root_rotateX/Y/Z` are Euler angles in degrees.
- The raw CSV is treated as MuJoCo-style: `Z` is up, `X/Y` are ground-plane axes.
- Robot forward is the root local `+X` direction.

For your differential base:

- wheel diameter: `0.20 m`
- wheel radius: `0.10 m`
- wheel separation / track width: `0.38 m`

## Usage

```bash
python wheel_base_tools/t2_csv_to_diff_drive.py \
  robot_demo_outputs/t2_csv/kimodo_motion.csv \
  --output robot_demo_outputs/t2_csv/kimodo_motion_diff_drive.csv
```

Optional safety limits:

```bash
python wheel_base_tools/t2_csv_to_diff_drive.py \
  robot_demo_outputs/t2_csv/kimodo_motion.csv \
  --max-forward-speed 0.4 \
  --max-yaw-rate 1.0
```

Hardware-ready RPM output:

```bash
python wheel_base_tools/t2_csv_to_diff_drive.py \
  robot_demo_outputs/t2_csv/kimodo_motion.csv \
  --output robot_demo_outputs/t2_csv/kimodo_motion_diff_drive.csv \
  --rpm-output robot_demo_outputs/t2_csv/kimodo_motion_motor_rpm.csv
```

The full diff-drive CSV includes the displayed root path plus rounded integer
command columns (`left_motor_rpm`, `right_motor_rpm`). The optional
`--rpm-output` file writes only:

```text
Frame,time_s,left_motor_rpm,right_motor_rpm
```

By default, motor RPM commands are clamped to `-3000..3000` RPM for the real
base command range. Change this with `--max-motor-rpm`.

The wheel-base simulation can also load RPM command CSVs directly. When
`left_motor_rpm/right_motor_rpm` are present, the viewer uses those RPM commands
for wheel animation and derives its internal forward velocity/yaw rate from the
configured wheel radius and separation.

## Main Equations

Given forward velocity `v` and yaw rate `omega`:

```text
left_linear  = v - omega * wheel_separation / 2
right_linear = v + omega * wheel_separation / 2

left_rad_s  = left_linear / wheel_radius
right_rad_s = right_linear / wheel_radius
```

If `omega` is positive, the right wheel becomes faster than the left wheel, producing a left turn under the standard differential-drive sign convention.
