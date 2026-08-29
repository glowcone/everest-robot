# Reconciling the LeRobot and maker-arm joint frames

Everest drives the arm through `maker-arm-sdk`, which works in calibrated joint **radians**.
Every LeRobot artefact — a recorded dataset, a trained checkpoint — is expressed in
`MakerFollower`'s **degrees**, against a different zero pose. The two are related by a
per-joint constant offset, recorded in `config/maker_arm_v1.yaml` under `lerobot_frame`:

```text
lerobot_degrees = everest_radians * 180/pi + offset_deg
```

Getting this wrong is not a subtle error. On most joints the offset is over 100 degrees, so
an unreconciled replay either commands a pose nowhere near the recorded one or — more
likely, and more safely — fails preflight because every action lands outside the soft
limits.

## Where the shipped values come from

They were derived, not measured. Both drivers publish soft limits for the same seven
joints:

| Joint | maker-arm-sdk (rad) | as degrees | MakerFollower (deg) | offset |
| --- | --- | ---: | --- | ---: |
| `shoulder_pan` | -0.668 … 4.818 | -38.27 … 276.05 | -158.2 … 156.1 | -119.94 |
| `shoulder_lift` | -2.024 … 0.979 | -115.97 … 56.09 | -175.3 … -3.2 | -59.31 |
| `elbow_flex` | 3.882 … 7.955 | 222.42 … 455.78 | 2.8 … 236.1 | -219.66 |
| `wrist_flex` | -0.832 … 2.122 | -47.67 … 121.58 | -65.2 … 104.0 | -17.56 |
| `wrist_yaw` | 0.577 … 3.641 | 33.06 … 208.61 | -97.6 … 78.0 | -130.64 |
| `wrist_roll` | 0.966 … 6.292 | 55.35 … 360.51 | -153.2 … 152.0 | -208.53 |
| `gripper` | -2.092 … -0.039 | -119.86 … -2.23 | -120.1 … -2.5 | -0.25 |

Two independent consistency checks support a pure per-joint offset:

1. The two ranges have the **same width** on every joint, to within 0.07 degrees. A
   different scale or a flipped direction would show up here immediately.
2. The offset computed from the lower limit and the offset computed from the upper limit
   **agree** on every joint, again to within 0.07 degrees. The table above uses their mean.

This is strong evidence that the mapping is right, and it is still not a measurement. The
limit tables could share a common error; `MakerFollower`'s comment says its values were
re-expressed from the same vendor capture, so they are not fully independent sources.

## The hardware check to run before powered replay

1. Bring the arm up through `maker-arm-sdk` and park it at a known, safe pose.
2. Read `Arm.get_joint_positions()` (radians, Everest frame).
3. Read the same physical pose through `MakerFollower`'s convention — either by connecting
   that driver (needs MIT-mode motors and a power cycle, so usually not on the same day) or
   by comparing against a dataset frame recorded at a visually identical pose.
4. For each joint, check `lerobot_deg - degrees(everest_rad)` matches the offset above.
   Disagreements larger than the replay's `initial_pose_tolerance_deg` must be resolved
   before any powered replay.
5. Record the result: set `lerobot_frame.approved_by` and `captured_at` to the person and
   date that performed this check, and replace the derivation note.

Until that is done, `lerobot_frame.approved_by` says the values are unverified, and the
replay preflight report restates the offsets so they are visible in every dry run.

## When it changes

The offsets are a property of the zero pose, so recapture them whenever the arm is
re-zeroed or recalibrated — the same trigger that invalidates every named position
(`docs/named-position-capture.md`). A frame from a previous calibration is not migrated.
