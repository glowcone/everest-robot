# Backup non-RL carabiner solution

This branch preserves a deterministic alternative to reinforcement learning or imitation
learning for locating and picking up the carabiner. It uses marker vision, measured
camera-to-robot calibration, URDF forward/inverse kinematics, and the existing bounded
joint-motion safety layer.

The branch is a backup implementation. It does not authorize powered pickup on an
unvalidated robot.

## Data flow

```text
two white image points + black gate point
                  |
                  v
camera-pixel homography (measured point pairs)
                  |
                  v
robot-base carabiner X/Y + directed yaw
                  |
                  v
pregrasp / grasp / lift Cartesian poses
                  |
                  v
URDF-backed bounded numerical IK
                  |
                  v
six arm joint targets; gripper handled separately
```

The midpoint of the white points provides the pickup location. The black point resolves
the 180-degree ambiguity of the white-point axis. A measured grasp yaw offset relates the
directed carabiner pose to the gripper pose.

## What is implemented

- normalized least-squares planar homography with four or more point pairs;
- camera-pixel to robot-base X/Y command;
- two-white-points plus black-side-point orientation disambiguation;
- pregrasp, grasp, lift, and optional canonical Cartesian target generation;
- LeRobot dataset-degree to Maker Arm calibrated-radian conversion;
- offline dataset joint frame to robot-base tool XYZ conversion through FK;
- offline Cartesian pose to bounded joint solution through IK;
- read-only encoder capture of camera-pixel/robot-X/Y calibration pairs;
- Maker Arm adapter integration with explicit URDF, base-link, and tool-link settings; and
- deterministic tests for calibration, orientation, frame conversion, FK/IK integration,
  and missing encoder feedback.

No command in the calibration or offline kinematics tools enables motors or sends a
target. Powered execution remains behind the existing robot session and motion safety
boundaries.

## Required physical inputs

The repository deliberately does not invent arm geometry. Before the output is physically
meaningful, provide a production Maker Arm URDF containing:

- the origin and axis of all six motion joints in driver order;
- the calibrated zero pose and positive direction for every joint;
- the robot base frame; and
- the transform from the last wrist to the grasp center between the fingers.

Fix the overhead camera and robot base, then capture at least four non-collinear
camera-pixel/robot-X/Y pairs. Eight to twelve points across the pickup area plus held-out
validation points are preferred.

Four calibration poses solve the camera-plane mapping. They are not sufficient to infer a
general six-axis robot model.

## Setup

The adjacent `maker-arm-sdk` backup branch contains the URDF parser and FK/IK solver.
Install it into this project after the hardware environment:

```bash
uv sync --extra hardware
uv pip install -e ../maker-arm-sdk

export EVEREST_ARM_URDF=/absolute/path/to/maker_arm_v1.urdf
export EVEREST_ARM_BASE_LINK=base_link
export EVEREST_ARM_TOOL_LINK=gripper_frame_link
```

For a serial-CAN arm on macOS:

```bash
export EVEREST_CAN_BACKEND=slcan
export EVEREST_CAN_PORT=/dev/cu.usbmodem-YOUR_DEVICE
export EVEREST_LEASE_BACKEND=file
```

## Capture camera/robot calibration pairs

With motors disabled, place the tool center over a visible point in the pickup plane and
record its full-frame camera pixel:

```bash
uv run robot-kinematics capture --camera-u U --camera-v V
```

The JSON output includes `image_point_px`, `robot_point_m`, the seven measured joints, and
the complete base-to-tool transform. Put matching point arrays into `pickup_config.json`.

Convert a new pixel after calibration:

```bash
uv run robot-camera-to-xy \
  --config pickup_config.json \
  --point U V \
  --show-matrix
```

## Convert a recorded dataset frame to X/Y

The expected order is `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
`wrist_yaw`, `wrist_roll`, `gripper`, expressed in LeRobot degrees:

```bash
uv run robot-kinematics dataset-fk --joints-deg Q1 Q2 Q3 Q4 Q5 Q6 GRIPPER
```

Read X/Y from the first two values of `tool_xyz_m`. This command is offline and does not
construct a CAN backend.

## Solve IK without moving

Use the current or another nearby joint state as the seed:

```bash
uv run robot-kinematics ik \
  --seed-dataset-deg Q1 Q2 Q3 Q4 Q5 Q6 GRIPPER \
  --xyz X Y Z \
  --rpy-deg ROLL PITCH YAW
```

The result reports convergence, both joint coordinate conventions, and Cartesian
residuals. The gripper value is preserved because the six-axis arm solver does not treat
finger opening as a kinematic joint.

## Acceptance boundary

Before any powered Cartesian move:

1. Compare FK with independently measured XYZ at several poses.
2. Verify each joint's zero and positive direction.
3. Verify the configured tool link is the finger midpoint.
4. Require FK of each IK result to reproduce its target within the chosen tolerance.
5. Reject joint-limit, wrong-branch, and non-converged solutions.
6. Test pregrasp-only targets above a clear table at reduced speed with an operator on the
   emergency stop.
7. Validate held-out camera calibration error across the entire pickup area.

The numerical IK solver enforces joint limits but does not implement collision avoidance.

## Verification at backup time

- Everest lint: passed.
- Everest tests: 257 passed, 3 skipped.
- Maker Arm SDK tests: 113 passed, 4 skipped.
- Offline seven-joint dataset FK smoke test: passed without a CAN configuration.
- Offline IK/FK round-trip smoke test: converged with zero position error using a synthetic
  test model. The synthetic model is not deployable geometry.

