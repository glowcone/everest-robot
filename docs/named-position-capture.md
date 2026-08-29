# Capturing and approving a named position

A named position is a set of joint angles the workflow will drive the arm to without any
perception in the loop. Nothing validates that the pose is collision-free except this
procedure. Joint values are never authored by hand, estimated, or copied between arms.

Prerequisites: the arm is running the calibration named in `config/maker_arm_v1.yaml`
(`robot.calibration_id`), the work fixture and attachment point are in their production
positions, and one operator has a hand on the e-stop for every step below.

## 1. Measure the pose

1. Bring the arm up and read joint feedback without enabling motion:
   `uv run maker-arm check` (or `Arm.connect()` followed by `get_joint_positions()`).
2. Move the arm to the target pose by hand, with motors disabled, or by teleoperation at
   reduced speed. Do not type the pose in.
3. Read `get_joint_positions()` again. These are the values to record, in the order given
   by `robot.joint_order`, in radians, in calibrated joint coordinates.
4. Repeat the measurement three times, re-approaching the pose from a different direction
   each time. If any joint varies by more than the configured `tolerance_rad`, the pose is
   not repeatable: fix the fixture or the calibration before continuing.

## 2. Record it

Add an entry under `named_positions` with the measured values and its provenance:

```yaml
named_positions:
  clip_attachment_ready:
    joints: [...]                              # one value per joint in joint_order
    calibration_id: maker-arm-02-2026-08-20    # must equal robot.calibration_id
    approved_by: "<operator name or team>"
    captured_at: 2026-08-29
    notes: "measured with the clip fixture at station 2"
    # Optional per-preset overrides, same fields as motion_defaults:
    max_velocity_rad_s: 0.3
```

The loader rejects a preset whose `calibration_id` does not match the file's robot, and
rejects unknown fields outright. The parameters file's content digest is stored in every
motion, policy and replay result, so a run can be traced back to the exact configuration.

## 3. Validate the approach at reduced speed

Each *transition* is approved separately from each *pose*. A pose being safe says nothing
about the straight line in joint space that reaches it.

1. Dry-run first: `go_to_known_position(name, dry_run=True)` validates the preset against
   the active hardware limits and reports the planned motion without commanding it.
2. Execute the move at `speed_scale=0.25` from every pose the workflow can start it from,
   with a hand on the e-stop. Watch for contact with the fixture, cable pull, and gripper
   payload swing.
3. Repeat at `speed_scale=0.5`, then at the approved `max_velocity_rad_s`.
4. If any approach cannot be made safe as a direct interpolation, do not raise the speed
   or widen the tolerance. Capture the intermediate clearance poses and record the path as
   a `named_transitions` entry instead. The runtime uses the transition when one exists.

## 4. Re-approval

Recapture every preset and re-run step 3 whenever:

- the arm is re-zeroed or recalibrated (which changes `robot.calibration_id`);
- a joint, belt, gripper or end effector is replaced or mechanically adjusted;
- the arm base or work fixture moves;
- the payload changes enough to alter settling behavior.

Presets from a previous calibration are not migrated. Delete them.
