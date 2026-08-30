# 0001. Production motor protocol

- Status: Superseded by [0002-mit-protocol-motor-operation.md](0002-mit-protocol-motor-operation.md)
- Date: 2026-08-29

The driver code analysis below remains accurate and is referenced by ADR-0002; the
decision itself was reversed after local rollouts on maker-arm-02 established that its
motors are provisioned in MIT mode and that the MIT stack, with Everest-owned
compensating controls, operates the arm correctly.

## Context

The Maker Arm v1 uses seven RobStride joints (six arm joints plus a gripper) on a single
classic-CAN bus. Two mutually incompatible drivers currently exist in this project's
dependency set, and each requires the motors to be running in a different RobStride firmware
protocol mode:

- `maker-arm-sdk` (`maker_arm.Arm`) drives the motors in RobStride's **private** protocol.
- LeRobot's `MakerFollower` (`lerobot/src/lerobot/robots/maker_follower/maker_follower.py`)
  drives the same motors through `RobstrideMotorsBus`
  (`lerobot/src/lerobot/motors/robstride/robstride.py`) in RobStride's **MIT** protocol.

A motor's protocol mode is persistent firmware state, not a per-session setting. Switching a
motor between private and MIT mode is a deliberate, documented procedure that requires a motor
power cycle to take effect (`MakerFollowerConfigBase` docstring,
`lerobot/src/lerobot/robots/maker_follower/config_maker_follower.py:27-35`: "The motors must
run RobStride's MIT protocol (see the Maker docs for the one-time switch from the factory
private protocol)"). There is no supported way to address a motor in both protocols
concurrently, and only one process may own the CAN interface at a time — two independent
drivers issuing commands or watchdog writes to the same bus is a collision, not a
fallback path. Production workflows must therefore commit to exactly one protocol/driver
per arm and must never attempt to switch protocols dynamically at runtime.

### Safety features actually implemented by each driver

**`maker-arm-sdk` (`maker_arm.Arm`, private protocol):**

- Explicit state machine — `DISCONNECTED` / `CONNECTED` / `ENABLED` / `FAULT`
  (`maker_arm/arm.py:19-23`).
- Fresh-feedback validation before enabling: `connect()` probes each motor sequentially and
  refuses to proceed until every motor has replied (`maker_arm/arm.py:71-90`); `enable()`
  additionally re-probes with `refresh(wait=True)` and refuses to enable on stale feedback
  (`maker_arm/arm.py:206-211`).
- Absolute-coordinate enable gate: `enable()` refuses to arm if the current measured position
  falls outside `[lo - ENABLE_LIMIT_GRACE, hi + ENABLE_LIMIT_GRACE]` for any joint
  (`maker_arm/arm.py:218-223`).
- Per-session full-turn (2π) encoder correction computed once at `connect()`, applied
  transparently in both coordinate-conversion directions (`maker_arm/arm.py:91-107`,
  `_wrap`, `_to_motor`/`_to_joint` at `maker_arm/arm.py:143-149`).
- Host-side feedback watchdog: `_check_health()` faults the arm if `m.feedback_age >
  self.config.feedback_timeout` (`maker_arm/arm.py:352-353`, default `feedback_timeout: 0.2`
  in `ArmConfig`, `maker_arm/config.py:27`).
- Motor-side CAN watchdog: `enable()` writes `protocol.ParamIndex.CAN_TIMEOUT` (`0x7028`) in
  50 µs counts, computed from `config.motor_can_timeout_ms` (`maker_arm/arm.py:174-178`,
  `maker_arm/protocol.py:54,64`), and **reads back** both `RUN_MODE` and `CAN_TIMEOUT` to
  confirm the motor actually accepted them before proceeding (`maker_arm/arm.py:180-203`).
  `ArmConfig._validate()` refuses a config with `motor_can_timeout_ms <= 0`
  (`maker_arm/config.py:51-52`): the motor-side watchdog can never be disabled by
  configuration.
- Velocity-limited joint commands: the control loop steps `_internal` toward `_user_targets`
  by at most `config.max_velocity * dt` per tick, inside a margin-shrunk soft-limit band
  (`maker_arm/arm.py:303-311`, `_tick`).
- Fault detection and fault-hold: `_check_health()` checks feedback age, motor `fault_bits`,
  and MIT-mode persistence (`mode != 2` for `MODE_FAULT_TICKS` consecutive ticks) and calls
  `_enter_fault()` (`maker_arm/arm.py:348-369`). With `hold_on_fault: True` (the default,
  `maker_arm/config.py:29`), `_enter_fault()` freezes and continues sending the last readable
  position under `kp`/`kd` rather than releasing torque (`maker_arm/arm.py:371-387`).
- `estop()` is callable from any state and unconditionally disables every motor
  (`maker_arm/arm.py:254-266`).
- Fault codes are decoded from the 6-bit fault field per `FAULT_NAMES`
  (`maker_arm/errors.py:22-29`).
- Command boundary is a single method, `Arm.set_joint_targets()`, which rejects wrong-length
  or non-finite targets outright (`maker_arm/arm.py:268-277`); it operates in radians in
  calibrated joint coordinates via `_to_joint`/`_to_motor`, so direction, offset, and the
  per-session wrap correction never leak outside `arm.py`.

**LeRobot `MakerFollower` (`RobstrideMotorsBus`, MIT protocol):**

- Per-session full-turn (±360°) correction, structurally the same idea as `maker_arm`'s 2π
  correction: `_detect_full_turn_offsets()` shifts a joint reading found a whole turn outside
  its limits (`lerobot/src/lerobot/robots/maker_follower/maker_follower.py:132-164`,
  constants `_FULL_TURN_DEG`/`_WRAP_GRACE_DEG` at lines 58-59). If no whole-turn shift
  explains the reading, the joint is recorded in `self._stale_zero` and `send_action()`
  refuses to move any joint until `calibrate()` is rerun
  (`maker_follower.py:333-339`).
- Soft-limit clamping in `send_action()`, per motor including the gripper
  (`maker_follower.py:342-351`).
- Fault querying via `_query_status_via_clear_fault()`, used at handshake and per-tick state
  update in `RobstrideMotorsBus.update_motor_state()`
  (`lerobot/src/lerobot/motors/robstride/robstride.py:219-275`); a detected fault raises
  `RuntimeError` rather than entering a driver-level fault-hold state.
- `_handshake()` at connect probes every configured motor and reports missing/faulted motors
  (`robstride.py:280-312`).
- Optional relative-target capping via `ensure_safe_goal_position()`
  (`lerobot/src/lerobot/robots/utils.py:99-131`), gated on `config.max_relative_target`,
  which defaults to `None` (disabled) (`config_maker_follower.py:104`).
- Optional slow startup sync to avoid snapping a follower to a distant leader target, gated
  on `config.startup_sync_speed_deg` (`maker_follower.py:356-392`,
  `config_maker_follower.py:100-101`).
- `disable_torque_on_disconnect: bool = True` releases torque on `disconnect()`
  (`config_maker_follower.py:106-108`, used at `maker_follower.py:408`).

What `RobstrideMotorsBus` does **not** implement, as of this checkout: any write of a
motor-side CAN/communication watchdog parameter. `robstride.py` has no `CAN_TIMEOUT`-equivalent
constant or write path (grep of the module and of `motors/robstride/tables.py` finds none), and
the module carries `# TODO(Virgile): Robustify mode control, only the MIT protocol is
implemented for now` (`robstride.py:15`). There is also no host-side feedback-age watchdog
comparable to `maker_arm`'s `feedback_timeout`/`_check_health()`, and no fault-hold behavior —
a detected fault is surfaced as a raised exception (`update_motor_state`,
`robstride.py:268-275`), not a locked, actively-held pose. Velocity limiting is likewise
optional and off by default (`max_relative_target: None`), whereas `maker_arm`'s velocity limit
is unconditional and applied inside the control loop itself.

## Decision

Retain the RobStride **private** protocol, with `maker-arm-sdk` (`maker_arm.Arm`) as the
production motor driver and the hardware safety boundary for all Maker Arm operation. Everest
implements LeRobot's `Robot` contract (`connect()` / `get_observation()` / `send_action()` /
`disconnect()`) on top of `maker_arm.Arm`, so LeRobot's policy inference, `LeRobotDataset`,
camera abstractions, and replay machinery continue to work unmodified against that adapter.

Everest will not depend on `MakerFollower` or `RobstrideMotorsBus` for production motor control.
`MakerFollower` remains useful only as a reference implementation and for any bench work
explicitly conducted with the motors switched into MIT mode; it must never run against arms
whose motors are provisioned for production (private-protocol) operation.

The rejected alternative — standardizing deployed arms on MIT mode and using `MakerFollower`
directly — was given fair consideration. It would remove the adapter-writing work below and
let Everest depend on a driver LeRobot's own maintainers already own. It was rejected because,
on the current checkout, `RobstrideMotorsBus` has no motor-side CAN watchdog, no host-side
feedback-age watchdog, and no fault-hold behavior (see Context), it is explicitly flagged
incomplete by its own author (`robstride.py:15`), and it has not completed real-hardware
validation as a production hardware safety boundary. Making it the production safety boundary
would mean shipping without the arm's two strongest existing protections (the motor-side
watchdog and fault-hold) until that work is done, which the project owner is not willing to
accept for a physical arm that can carry a payload.

## Ownership table

| Concern | Owner | Concrete symbol / config key |
|---|---|---|
| Calibration (direction/offset/zero) | maker-arm-sdk | `JointConfig.direction`, `JointConfig.offset` (`maker_arm/config.py:13-14`); zero set via `Motor.set_zero()` / `maker-arm zero` CLI |
| Soft limits | maker-arm-sdk | `JointConfig.lo`/`JointConfig.hi` (`maker_arm/config.py:15-16`), enforced in `Arm._tick()` (`maker_arm/arm.py:308-309`) and gated at `Arm.enable()` (`maker_arm/arm.py:218-223`); values shipped in `maker_arm/profiles/maker_arm_v1.yaml` |
| Host watchdog | maker-arm-sdk | `ArmConfig.feedback_timeout` (`maker_arm/config.py:27`), checked in `Arm._check_health()` (`maker_arm/arm.py:352-353`) |
| Motor-side CAN watchdog | maker-arm-sdk | `ArmConfig.motor_can_timeout_ms` (`maker_arm/config.py:28`, must be `>0`, `maker_arm/config.py:51-52`), written via `protocol.ParamIndex.CAN_TIMEOUT` and read back in `Arm.enable()` (`maker_arm/arm.py:174-203`) |
| Full-turn (2π) encoder correction | maker-arm-sdk | `Arm._wrap`, computed once per `connect()` (`maker_arm/arm.py:91-107`), applied in `_to_motor`/`_to_joint` (`maker_arm/arm.py:143-149`) |
| Fault detection and fault-hold | maker-arm-sdk | `Arm._check_health()` / `Arm._enter_fault()` (`maker_arm/arm.py:348-395`), gated by `ArmConfig.hold_on_fault` (`maker_arm/config.py:29`); fault text from `errors.fault_text()`/`FAULT_NAMES` (`maker_arm/errors.py:22-35`) |
| E-stop | maker-arm-sdk | `Arm.estop()` (`maker_arm/arm.py:254-266`), callable from any state; Everest SDK layer must expose an equivalent operator/workflow-level e-stop entry point that calls it |
| Velocity limiting | maker-arm-sdk | `ArmConfig.max_velocity` (`maker_arm/config.py:25`), applied unconditionally in `Arm._tick()` (`maker_arm/arm.py:306-311`) |
| Joint <-> motor coordinate conversion | maker-arm-sdk | `Arm._to_motor()` / `Arm._to_joint()` (`maker_arm/arm.py:143-149`) — the only place direction/offset/wrap are applied; everything outside `arm.py` sees radians in calibrated joint coordinates via `Arm.get_joint_positions()` / `Arm.set_joint_targets()` |
| Feature naming/units for policies | Everest SDK layer (LeRobot contract, Everest-owned adapter) | The Everest `Robot`-contract adapter around `maker_arm.Arm` must define `observation_features`/`action_features` (LeRobot's `{joint}.pos` naming convention, mirroring `MakerFollower._motors_ft`, `maker_follower.py:193-195`) and perform the radians-to-adapter-units conversion; LeRobot itself only consumes whatever feature dict the adapter produces |

Calibration identity/versioning at the Everest workflow level (matching a preset or dataset
to the arm that produced it) is Everest's responsibility, per the `initial-sdk-plan.md`
`config/maker_arm_v1.yaml` proposal; it is listed under "feature naming/units for policies"
above because it is implemented in the same adapter layer, not inside `maker-arm-sdk`.

## Consequences

**Positive**

- The arm's strongest currently-implemented safety behaviors — the motor-side CAN watchdog
  with read-back verification, the unconditional host feedback watchdog, fault-hold, and the
  unconditional velocity limit — remain in force for every production motion, with no
  dependency on a driver still marked incomplete by its own maintainers.
- `maker-arm-sdk`'s command boundary (`Arm.set_joint_targets()`) and its coordinate-conversion
  functions are small, private, and already exercised by this repository's unit and vcan
  integration tests (`maker-arm-sdk/tests/unit/test_arm_safety.py`,
  `tests/integration/test_vcan_integration.py`), so Everest inherits a safety boundary that
  has direct test coverage rather than one still awaiting hardware validation.
- LeRobot's policy, dataset, camera, and replay machinery remain fully usable, since only the
  `Robot`-contract adapter is Everest-specific; everything above that contract is standard
  LeRobot.

**Negative**

- Everest now owns writing and maintaining a `Robot`-contract adapter over `maker_arm.Arm` —
  `connect()`/`get_observation()`/`send_action()`/`disconnect()`, `observation_features`,
  `action_features`, and a `calibrate()`/`is_calibrated` story — that `MakerFollower` would
  otherwise have supplied for free. This is delegatable work item 7 in
  `planning/initial-sdk-plan.md` ("LeRobot compatibility bridge").
- **Unit mismatch.** `maker_arm.Arm` works in radians, in calibrated joint coordinates:
  `Arm.get_joint_positions()`/`Arm.set_joint_targets()` (`maker_arm/arm.py:152-154,268-277`).
  `MakerFollower` normalizes in degrees via `Motor(..., MotorNormMode.DEGREES)`
  (`maker_follower.py:99`, `MotorNormMode.DEGREES` defined at
  `lerobot/src/lerobot/motors/motors_bus.py:172`), and exposes per-joint features named
  `{joint}.pos` in degrees (`MakerFollower._motors_ft`, `maker_follower.py:193-195`, e.g.
  `shoulder_pan.pos`). Everest's adapter must perform an explicit radians<->degrees conversion
  (and the joint-order mapping) at the adapter boundary; nothing upstream does this for it.
- **No implicit dataset/policy reuse.** Any `LeRobotDataset` episode, policy checkpoint, or
  normalization statistics produced against `MakerFollower` (degrees, `{joint}.pos` feature
  names, `MakerFollower`'s own soft-limit table in `config_maker_follower.py:69-79`, which is
  independently re-expressed from `maker_arm/profiles/maker_arm_v1.yaml` rather than shared
  with it) is not directly consumable by an Everest/`maker_arm.Arm`-backed rollout or replay.
  It requires an explicit, validated unit and feature-name conversion, and a check that the
  two soft-limit tables and calibration/zero pose actually agree, before any such data or
  checkpoint is used against the private-protocol arm.
- Two independent RobStride drivers with diverging safety feature sets now exist in the
  dependency graph (`maker-arm-sdk` and `lerobot`'s `robstride` motor bus). Nothing prevents a
  future contributor from wiring `MakerFollower` into a production path by mistake; this must
  be caught in review, since nothing in the code itself enforces the choice made here.

## Deployment and recovery implications

- **Only one process may hold the CAN interface at a time.** `Arm.connect()` opens the
  configured backend (`maker_arm/arm.py:65-69`) and expects exclusive access; a second process
  (an accidental second worker, a leftover `MakerFollower` bench script, or a duplicate Absurd
  task) attached to the same bus will corrupt both drivers' feedback and command state. Everest
  must serialize access with an exclusive per-robot lease (`initial-sdk-plan.md`, "Durable
  execution concerns" / delegatable item 11) before this ADR's adapter goes into a durable
  workflow.
- **A motor left in the wrong protocol mode will not respond, and must not be worked around in
  software.** If an arm's motors are left in MIT mode (e.g., after bench validation of
  `MakerFollower`), `maker_arm.Arm.connect()` will report missing feedback for those motors
  (`ConnectError` at `maker_arm/arm.py:87-90`) rather than silently mis-decoding MIT frames as
  private-protocol frames. Recovery is the documented one-time protocol switch back to private
  mode plus the accompanying motor power cycle; this is an operator procedure, not something
  the adapter should attempt to detect-and-fix automatically at runtime.
- **On worker crash**, the motor-side CAN watchdog (`motor_can_timeout_ms`, non-zero and
  read-back verified per joint) is what actually protects the arm: if the host process dies
  without disabling the motors, each motor stops receiving MIT command frames and drops out of
  control mode on its own after the configured timeout, independent of any host-side cleanup.
  The host-side `feedback_timeout` watchdog and `hold_on_fault` behavior only apply while a
  host process is alive and running the control loop; they do not protect against total process
  loss by themselves. Everest's lease/reconnect logic (delegatable item 11) must still assume
  the arm may be in an unknown last-commanded pose after a crash and re-establish state through
  `Arm.connect()`'s fresh-feedback and absolute-limit checks rather than trusting any
  in-process cache.
- Because `hold_on_fault: True` is the profile default (`maker_arm/profiles/maker_arm_v1.yaml`
  inherits `ArmConfig.hold_on_fault` default `True`, `maker_arm/config.py:29`), a fault leaves
  the arm actively holding its last readable pose under torque rather than dropping. Recovery
  from `ArmState.FAULT` requires an explicit `Arm.clear_faults()` call
  (`maker_arm/arm.py:397-403`), which itself requires the caller to be in `FAULT` state; Everest
  must route this through an operator- or workflow-gated recovery path, not an automatic retry,
  since the underlying fault condition may still be physically present.

## Addendum (2026-08-29): MIT-mode calibration monitor port

*(Historical: this addendum introduced the MIT port as a narrow exception; ADR-0002 has
since made it this arm's driver and carries the current scope and qualification rules.)*

The motors on this project's arm are currently provisioned in **MIT** mode (they answer
the makermodslab teleop stack, and `maker_arm.Arm.connect()` reports "online: none"), and
switching them back to the private protocol was not practical for the calibration session
at hand. An MIT-protocol `ArmPort` therefore exists:
`src/everest_robot/robot/robstride_mit_port.py` (`RobstrideMitPort`, over the lerobot
fork's `RobstrideMotorsBus`), selected by `EVEREST_ARM_DRIVER=mit`.

Its qualified scope is the **lease-local calibration teleoperation monitor**
(`just monitor`) only. The gaps this ADR documents remain in force and accepted for that
scope: no motor-side CAN watchdog, no host-side feedback-age watchdog, no fault-hold. The
teleoperation loop mitigates at its level (hold-on-failure, torque-off on disconnect), and
frame reconciliation goes through the parameters file's `lerobot_frame` offsets
(docs/lerobot-frame-reconciliation.md).

Production replay and the durable workflow remain on the private protocol with
`maker-arm-sdk`; this ADR's decision stands. Do not wire `RobstrideMitPort` into replay or
workflow paths — that requires reopening this ADR under its revisit criteria.

## Revisit criteria

Reopen this decision only if one or more of the following becomes true:

- `MakerFollower`/`RobstrideMotorsBus` completes real-hardware validation as a standalone
  production safety boundary, **and** gains a motor-side CAN watchdog and a host-side
  feedback-age watchdog with behavior at least equivalent to `maker_arm`'s
  `motor_can_timeout_ms`/`feedback_timeout`, **and** gains fault-hold behavior equivalent to
  `hold_on_fault` (rather than raising on the first detected fault).
- Everest requires a LeRobot capability that only works through LeRobot's own motor bus stack
  (for example, a LeRobot-maintained calibration/record tool, or a future feature that assumes
  `RobstrideMotorsBus` directly and cannot be adapted), such that the adapter-layer cost of
  staying on the private protocol exceeds the cost of qualifying the MIT-mode driver for
  production.
- `maker-arm-sdk` is deprecated or unmaintained upstream, making the private-protocol path a
  long-term liability regardless of its current feature advantage.

Any reopening must re-derive the ownership table above against the drivers' state at that time,
not against this document's snapshot of them.
