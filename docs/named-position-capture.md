# Capturing and approving a named position

A named position is a set of joint angles the workflow will drive the arm to without any
perception in the loop. Nothing validates that the pose is collision-free except this
procedure. Joint values are never authored by hand, estimated, or copied between arms.

Prerequisites: the arm is running the calibration named in `config/maker_arm_v1.yaml`
(`robot.calibration_id`), the work fixture and attachment point are in their production
positions, and one operator has a hand on the e-stop for every step below.

## 1. Measure the pose

1. Set `EVEREST_STAR_PORT` to the Star 102 leader's UART port and run `just monitor`.
   This one process owns the follower lease, follows the leader at a conservative velocity,
   and renders the follower's measured encoders. It is a POWERED operation.
2. Move the leader until the follower reaches the target pose. Press space to pause following
   and hold the follower at the measured pose. Do not type the pose in. `?` opens an
   on-screen guide to the columns, the joint states, and this procedure.
3. Press `p` to capture the current follower state, then `q`. Once the TUI restores the
   terminal it prints the pose in canonical radians with robot, calibration, and
   configuration identity, then offers to save it -- see step 2. Support the arm while the
   session releases torque.
4. Repeat the measurement three times, re-approaching the pose from a different direction
   each time. If any joint varies by more than the configured `tolerance_rad`, the pose is
   not repeatable: fix the fixture or the calibration before continuing.

For hand positioning with motors disabled, use `just monitor-read-only`. `just monitor-once`
remains a single non-powered snapshot. A separate monitor cannot run alongside teleoperation:
both would participate on the same CAN bus, which is why powered following and display are
combined in `just monitor`.

## 2. Record it

The save prompt runs after the session has released the arm, so nothing is claimed or
energized while you type:

```
Save this pose to config/maker_arm_v1.yaml as a named position?
  name (blank to skip) > clip-attachment-ready
  approved by > <operator name or team>
  notes (optional) > measured with the clip fixture at station 2
```

A blank name skips it. It refuses to save a pose from `--fake`, one with a joint reporting
no feedback, one outside the driver's soft limits, or one measured while the arm was
faulted, because none of those describe a place the arm actually was. Re-saving an existing
name is a re-approval and has to be confirmed with `REPLACE`. The write is atomic and
verified: the file is re-read through the strict loader afterwards and rolled back to its
original bytes unless the preset that comes back is the one that went in. `--no-save` prints
the pose and offers nothing.

Hand-editing remains available for anything the prompt does not cover -- per-preset motion
overrides, or reordering. The entry it writes has this shape:

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
rejects unknown fields outright -- the `robot_id` and `config_digest` the monitor prints
beside a capture are identity for you to check, not preset fields, which is why the saved
entry does not carry them. The parameters file's content digest is stored in every
motion, policy and replay result, so a run can be traced back to the exact configuration.

## 3. Validate the approach at reduced speed

Each *transition* is approved separately from each *pose*. A pose being safe says nothing
about the straight line in joint space that reaches it.

1. Dry-run first: `just goto-dry <name>` validates the preset against the active hardware
   limits and reports the planned motion without commanding it. `just goto-list` shows
   which presets and transitions the file currently offers.
2. Execute the move with `just goto <name> 0.25` from every pose the workflow can start it
   from, with a hand on the e-stop. Watch for contact with the fixture, cable pull, and
   gripper payload swing. A move that swings any joint more than 0.35 rad asks for
   confirmation before it energizes anything.
3. Repeat with `just goto <name> 0.5`, then at the approved `max_velocity_rad_s`
   (`just goto <name> 1.0`).
4. If any approach cannot be made safe as a direct interpolation, do not raise the speed
   or widen the tolerance. Capture the intermediate clearance poses and record the path as
   a `named_transitions` entry instead. The runtime uses the transition when one exists,
   and so does `just goto`: once a transition ends at a pose, that is how the command
   reaches it, with no flag to go straight there. Where two transitions end at the same
   pose it refuses and asks which, since that is an operator's choice.

## 4. Re-approval

Recapture every preset and re-run step 3 whenever:

- the arm is re-zeroed or recalibrated (which changes `robot.calibration_id`);
- a joint, belt, gripper or end effector is replaced or mechanically adjusted;
- the arm base or work fixture moves;
- the payload changes enough to alter settling behavior.

Presets from a previous calibration are not migrated. Delete them.
