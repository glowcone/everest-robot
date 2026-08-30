# 0002. MIT-protocol motor operation for maker-arm-02

- Status: Accepted
- Date: 2026-08-29
- Supersedes: [0001-production-motor-protocol.md](0001-production-motor-protocol.md)

## Context

ADR-0001 chose RobStride's **private** protocol with `maker-arm-sdk` as the production
motor driver and hardware safety boundary, on the strength of that driver's implemented
protections (motor-side CAN watchdog with read-back, host feedback watchdog, fault-hold,
unconditional velocity limiting). That analysis of the two drivers' code was correct and
is retained by reference. What changed is contact with the machine.

### What the local rollouts established

Everything below was measured on maker-arm-02 over a CANable slcan adapter
(`/dev/tty.usbmodem1101`) and the Star 102 leader (`/dev/tty.usbserial-130`) on macOS,
2026-08-29, during bring-up of the calibration teleoperation monitor.

1. **The motors are provisioned in MIT mode, and switching is not a paper exercise.**
   `maker_arm.Arm.connect()` reports "online: none" against these motors despite verified
   power, wiring and bitrate: a motor answers only its persisted protocol mode. The switch
   back to private mode requires per-motor protocol writes (possibly via MotorStudio) plus
   a power cycle, and invalidates the team's working makermodslab teleop setup. ADR-0001's
   decision therefore described a configuration this arm has never run.

2. **The MIT stack works end to end on this arm.** `RobstrideMitPort` over the lerobot
   fork's `RobstrideMotorsBus` (pinned `af9072a8`) passed handshake, read-only monitoring,
   the enable gate, and full powered leader-following teleoperation, including gripper
   stall-grip. The `lerobot_frame` offsets in `config/maker_arm_v1.yaml` -- derived from
   the two drivers' limit tables -- survived their first powered session: mapped leader
   targets and follower feedback agree in calibrated radians.

3. **The bus stack is not thread-safe, and this is load-bearing.** One thread reading
   feedback while another commands (the monitor TUI plus the teleoperation loop) tears
   slcan lines apart mid-frame; python-can surfaces the shear as
   `ValueError: invalid literal for int() with base 16: '0tFD8057'`. `maker_arm.Arm` hid
   this class of problem behind its internal control loop and feedback cache;
   `RobstrideMotorsBus` does not. The port serializes every bus operation behind one lock.

4. **The CANable's RX FIFO cannot absorb bursts.** A 7-frame `sync_write`/`sync_read`
   burst drops replies ("Packet drop" on every tick). Per-motor request/reply I/O is
   required, as `MakerFollower` itself does. Measured costs: 7 follower command writes
   ≈ 7 ms, 7 leader UART reads + mapping ≈ 18 ms -- a 25 Hz control tick holds its period
   with margin.

5. **Residue on the serial line survives a session.** A SIGINT'd process leaves partial
   slcan lines in the adapter; the next session's first parse desyncs. The port flushes
   the RX queue at connect and again before the enable gate, and treats a torn frame
   during operation as "no feedback this sample" (`nan`), never as a crash.

6. **Full-turn encoder wraps are real.** A RobStride motor can return from a power cycle
   reporting a whole turn off (seen upstream on shoulder_lift). The port applies a single
   ±360° correction at enable when that explains the reading, and refuses to enable
   otherwise -- an arm whose zero cannot be trusted must never be put under torque.

7. **The Star leader's `reliable` flag is a near-zero glitch filter, not a liveness
   signal.** `motorbridge_smart_servo` flags raw readings of ~0° as suspected power-cycle
   glitches; a leader joint genuinely resting at zero reads 4% "reliable" at rest and 100%
   when bent. Gating on that flag produced spurious "leader lost" aborts. Liveness now
   comes from per-servo `read_angle()` (which raises on a true non-response) with the
   native loss threshold disabled; loss policy lives only in the controller's timeout.

8. **Gripping means commanding past the object.** The Star gripper mapping's `base_rad`
   (-0.039 rad) sits exactly on the follower's upper soft limit, so any squeeze maps past
   a limit; and a MakerMod gripper grips by stalling compliantly against a target beyond
   the object (grip force = kp × position error, gripper kp deliberately 20). A hard
   limit-refusal on the gripper mapping is therefore wrong by design; the teleoperation
   layer clamps the gripper and hard-refuses only arm joints, where an out-of-limit
   mapping really does mean a bad mapping or frame.

9. **Perceived speed is governed by the velocity clamp, not the I/O.** With per-motor I/O
   the loop holds 25 Hz; the operator-facing speed is the teleoperation clamp
   (default 0.25 rad/s, `just monitor <poll_hz> <max_velocity>`), applied against the
   measured tick time so bus contention cannot slow the arm below the configured rate,
   capped so a stalled tick cannot become a jump.

### What did not change

`RobstrideMotorsBus` still has no motor-side CAN watchdog, no host-side feedback-age
watchdog, and no fault-hold; a fault surfaces as a raised exception. ADR-0001's code
analysis stands. A hard-killed host process (`kill -9`) leaves the last MIT command in
force at the motors indefinitely.

## Decision

maker-arm-02's motors **remain provisioned in RobStride's MIT protocol**, and
`RobstrideMitPort` (`src/everest_robot/robot/robstride_mit_port.py`, selected by
`EVEREST_ARM_DRIVER=mit`) is this arm's motor driver. Everest owns the compensating
controls the driver lacks, at the port and teleoperation layers (ownership table below).

Qualification is explicit and staged:

- **Qualified now: lease-local calibration teleoperation** (`just monitor` and the
  read-only monitor modes). Powered use is operator-attended by construction: the same
  process renders feedback, holds the lease, and can pause (space) or exit (torque-off on
  disconnect) at any time.
- **Not yet qualified: replay and the durable workflow.** These must not run powered on
  this driver until the checklist below is worked through. Note they cannot silently run
  on the old path either -- the private-protocol driver gets no replies from MIT motors --
  so the failure mode of a misconfiguration is inert, not dangerous.

`maker-arm-sdk` stays in the dependency set: it is the source of the Star leader mapping
and the hardware profile the `lerobot_frame` offsets were derived from, and it remains the
path back if the motors are ever deliberately re-provisioned to the private protocol. Its
driver is no longer described as this arm's production safety boundary, because it has
never operated this arm.

### Qualification checklist for replay/workflow on the MIT driver

1. A host-side feedback-age watchdog in `RobstrideMitPort` (or the fork) with behavior
   equivalent to `maker_arm`'s `feedback_timeout`: sustained stale feedback must stop
   commanding, not merely render as `nan`.
2. A defined crash story replacing the missing motor-side CAN watchdog: either the fork
   gains a `CAN_TIMEOUT`-equivalent write with read-back, or the deployment accepts and
   documents operator-attended powered runs only.
3. Formal `lerobot_frame` reconciliation per `docs/lerobot-frame-reconciliation.md`
   (command a known pose, read it back through both conventions), beyond the incidental
   validation teleoperation provided.
4. Replay preflight and the hardware acceptance procedure in `docs/session-replay.md`
   re-run against this driver.

## Ownership table

| Concern | Owner | Concrete symbol |
|---|---|---|
| MIT frame encoding, per-motor state cache (20 ms TTL) | lerobot fork | `RobstrideMotorsBus` (`robstride.py`, pinned `af9072a8`) |
| deg↔rad / zero-pose frame reconciliation | Everest port + parameters | `JointFrame`, `lerobot_frame` in `config/maker_arm_v1.yaml` |
| Soft limits | Everest port | `MakerFollowerConfigBase.joint_limits` via `RobstrideMitPort.limits()`; gated at `enable()`, clipped in `send_targets()` |
| Full-turn (±360°) wrap correction | Everest port | `RobstrideMitPort.enable()` (mirrors `MakerFollower._detect_full_turn_offsets`) |
| Bus serialization (thread safety) | Everest port | `RobstrideMitPort._bus_lock` |
| Burst avoidance / adapter FIFO | Everest port | per-motor `bus.read` / `bus.write`, never `sync_*` for motion |
| Torn-frame and no-reply tolerance | Everest port | `read_state()` → `nan`; `send_targets()` → refusal |
| RX residue flush | Everest port | `connect()`, `enable()` |
| Velocity limiting | Everest teleoperation | `TeleoperationController` measured-dt clamp (`max_velocity_rad_s`) |
| Leader liveness | Everest teleoperation | `Star102LeaderPort.read_positions()` exception-based; controller loss timeout |
| Gripper stall-grip | Everest teleoperation | `clamp_joints=("gripper",)` in `_mapped_targets()` |
| Torque release / e-stop | Everest port | `estop()`, `disconnect(disable_torque=True)` |
| Motor-side CAN watchdog | **nobody (gap)** | accepted for attended calibration use only |
| Host feedback-age watchdog | **nobody (gap)** | checklist item 1 |
| Fault-hold | **nobody (gap)** | fault → exception → teleop holds last pose, then stops |

## Consequences

**Positive**

- The driver in production is the one that has actually operated this arm, with every
  compensating control unit-tested against a stub bus (including a concurrency test that
  fails on any unserialized bus access) and validated in powered rollouts.
- No protocol-switch procedure, no divergence from the team's other MIT-mode tooling.
- The empirical constraints of this exact hardware chain (CANable FIFO, slcan parser,
  leader glitch filter, gripper convention) are encoded in code and tests rather than
  operator folklore.

**Negative**

- The three watchdog/fault-hold gaps are now Everest's risk to manage, mitigated only by
  attended operation, hold-on-failure, and torque-off on disconnect. This is acceptable
  for a calibration tool and explicitly not yet acceptable for unattended workflow motion.
- Two RobStride drivers remain in the dependency graph. `EVEREST_ARM_DRIVER` defaults to
  `maker-arm`, which on this arm fails closed (no replies); deployments must set `mit`
  knowingly.
- Anything recorded or tuned against `maker_arm`'s calibrated radians continues to depend
  on the `lerobot_frame` offsets being right; checklist item 3 stands between here and
  trusting them for replay.

## Revisit criteria

- The fork gains a motor-side CAN watchdog, host feedback watchdog, or fault-hold →
  shrink the gap rows above and re-evaluate the replay/workflow qualification.
- The motors are re-provisioned to the private protocol for any reason → ADR-0001's
  analysis becomes operative again; do not run two protocols' tooling against one bus.
- The CANable is replaced with an adapter with a deeper FIFO or native CAN → re-measure
  before assuming per-motor I/O is still required; it costs about half the tick budget.
