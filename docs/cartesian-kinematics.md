# Maker Arm Cartesian kinematics and camera calibration

There are two independent coordinate problems:

1. **Arm FK/IK:** calibrated joint radians ↔ robot-base tool pose. This requires the
   physical six-axis arm geometry and tool-center point in a URDF.
2. **Camera calibration:** image pixels ↔ robot-base table X/Y. This requires at least four
   matching points after FK is trustworthy.

Four arm poses cannot identify a general six-axis kinematic chain. They calibrate the
camera plane only.

## Current status

The adjacent local `maker-arm-sdk` checkout contains:

- URDF serial-chain parsing;
- forward kinematics;
- bounded damped-least-squares inverse kinematics;
- offline FK and IK commands; and
- a read-only hardware command that reads encoders and emits a camera/robot calibration
  pair.

Everest adds LeRobot-frame reconciliation, so the seven joint values in the Hugging Face
dataset can be converted into Maker Arm radians before FK.

The repository intentionally does not contain a fabricated Maker Arm URDF. The upstream
SO-101 URDF is not suitable: it has five motion joints, while Maker Arm has six, including
`wrist_yaw`.

## Hardware/model information required

Obtain a production URDF or CAD-derived kinematic description containing:

- the origin and axis of each joint relative to its parent;
- the chain `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_yaw`,
  `wrist_roll` in that order;
- the SDK calibrated zero pose and positive direction for every joint;
- the transform from the final wrist to the grasp center between the fingers; and
- a documented robot base frame.

Place the URDF at an absolute path and configure:

```bash
export EVEREST_ARM_URDF=/absolute/path/to/maker_arm_v1.urdf
export EVEREST_ARM_BASE_LINK=base_link
export EVEREST_ARM_TOOL_LINK=gripper_frame_link
```

Until the SDK changes are merged upstream, install the adjacent checkout into Everest's
environment after the hardware extra:

```bash
uv sync --extra hardware
uv pip install -e ../maker-arm-sdk
```

## Convert recorded Hugging Face joints into X/Y

The input order is the dataset's own order, in degrees:

```text
shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_yaw wrist_roll gripper
```

For example:

```bash
uv run robot-kinematics dataset-fk --joints-deg \
  4.3395 -3.0 2.0 0.8587 2.5136 0.0 -28.1673
```

The command first applies `lerobot_frame` from `config/maker_arm_v1.yaml`, then evaluates
FK. It prints `tool_xyz_m`; the first two numbers are robot-base X/Y. It opens no camera,
bus, or motor connection.

## Capture camera/robot point pairs

1. Secure the camera and robot base permanently.
2. Put a visible mark in the pickup plane.
3. With motors disabled, move the tool-center point exactly over the mark at the configured
   calibration height.
4. Record the mark's full-frame camera pixel `(u, v)`.
5. Run:

```bash
uv run robot-kinematics capture --camera-u U --camera-v V
```

The command claims the robot, connects read-only, refreshes all seven encoders, calculates
FK, prints a JSON calibration record, and disconnects. It never enables or commands a
motor. Copy `image_point_px` and `robot_point_m` into `pickup_config.json`.

Four non-collinear points are the mathematical minimum. Use eight to twelve points spread
across the usable table and keep several additional points out of the fit. At every held-out
point, compare FK/camera-predicted X/Y with its independently measured X/Y.

## Solve IK without moving

Use the current or another nearby measured joint pose as the seed:

```bash
uv run robot-kinematics ik \
  --seed-dataset-deg Q1 Q2 Q3 Q4 Q5 Q6 GRIPPER \
  --xyz X Y Z --rpy-deg ROLL PITCH YAW
```

The result includes both Maker Arm radians and LeRobot degrees, convergence, position
error, and orientation error. The gripper value is preserved from the seed because it is
not part of arm IK. This command calculates only; it sends no target.

## Physical acceptance before motion

1. With torque off, compare FK against independently measured XYZ at several poses.
2. Check each positive joint direction and the URDF zero pose.
3. Verify the tool link is the midpoint between the fingers, not the wrist housing.
4. For every proposed IK result, run FK on the result and require the configured residual.
5. Reject results on a joint limit or a different elbow/wrist branch.
6. Test the first powered target well above the table at reduced speed, with the workspace
   clear and an operator on the e-stop.
7. Validate pregrasp-only targets across the entire pickup zone before permitting descent.

The numerical solver proves geometric consistency and respects the driver's limits. It does
not provide collision avoidance; accepted pickup paths still require physical validation.
