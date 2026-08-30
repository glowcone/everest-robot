# Wrist-servo calibration

The wrist camera's alternative to `docs/`'s fixed-camera pixel map, for the same
`SEARCH_CV` state. Read this before running `robot-wrist-servo teach`; the model it
produces is documented in `src/everest_robot/robot/wrist_servo.py` and consumed by
`src/everest_robot/robot/wrist_follower.py`.

## Why there are two

The two cameras are different kinds of instrument, and the difference decides what can be
calibrated at all.

The bench camera is bolted down, so one of its pixels names a place in the world. That is
what makes `robot-pixel-map` possible: teach ~30 (pixel, pre-grasp pose) pairs and a fit
answers "which joint vector grasps at this pixel?" forever after. It is open-loop and
arrives in one move, which is the best property any of this has.

The wrist camera moves with the arm. The same pixel means a different place at every pose,
so there is nothing to sample and no map to look a pose up in. What *is* stable is the
derivative -- how the carabiner's image moves when a joint moves -- and that is what this
calibration measures. The servo is then image-based: difference the current image against a
taught goal image, solve for the joint step that removes the difference, repeat.

What that buys is that **nothing is registered to the bench**. No camera has to be placed
and kept still, and the carabiner can be anywhere the wrist camera can see it, including
somewhere nobody demonstrated. What it costs is that "visible" now depends on where the arm
is pointing, and that arrival takes several ticks instead of one.

## The features

Four numbers, taken from one `everest_robot.carabiner_detect` detection:

| feature | source | tolerance |
| --- | --- | --- |
| `u`, `v` | the detector's insertion point, in pixels | 8 mm |
| `scale_px` | `sqrt(mask area)` | 10 mm |
| `spine_deg` | the spine tangent, wrapped into `[-90, 90)` | 6° |

`scale_px` is the square root of the area, not the area, so it moves linearly with the
other three under approach — otherwise the normalized least squares would be weighing a
quadratic against three linears. It is the only proxy a monocular camera has for range.

`spine_deg` is the only angular feature, and it is the reason error is computed inside the
calibration rather than by a caller subtracting two tuples. The detector's spine angle is
undirected: +89° and −89° are two degrees apart, and an unwrapped subtraction would call
that 178 and swing the wrist most of a half turn.

The pixel tolerances are stated in millimetres and converted at **1.44 mm/px**, measured at
the goal pose. Monocular, so that is a ratio at one range, not a property of the camera —
re-measure it if the lens, the resolution or the goal pose changes. It is used for nothing
but making the tolerances checkable against the part (a Petzl Spirit is 100 × 60 mm).

The tolerances do double duty: they are the arrival test *and* the weights the solve
normalizes by. That is deliberate — dividing each row by its tolerance makes the residual
dimensionless and makes the objective and the stopping test agree, so "settled" is exactly
the unit cube. Loosening a tolerance is how you tell the servo to care less about a feature.

## The procedure

```
just wrist-teach <your-name>     # 0. POWERED
just wrist-check                 # 1. no hardware
just wrist-look                  # 2. claims the arm, never enables it
just wrist-centre-dry            # 3a. the centring loop, never energized
just wrist-centre                # 3. POWERED  <- debug here
just wrist-track-dry             # 4a. the whole loop, never energized
just wrist-track                 # 4. POWERED
```

**Before step 0**, put the arm where you want `SEARCH_CV` to hand over to the clip policy:
gripper at pre-grasp over the carabiner, carabiner clearly in the wrist view. `robot-monitor`
under teleoperation or `robot-goto` are the ways to get there. Everything is measured from
that pose, and the image at it becomes the goal — an operator attestation with the same
standing as an approved named position, which is why `--approved-by` is required.

Step 0 then, for each of `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
`wrist_roll`:

1. Return to the start pose and measure the image (median of `--samples` detections; the
   failure mode of a threshold segmentation is not noise around the truth but an occasional
   frame that is wrong by a lot, which a median discards and a mean would carry into the
   derivative).
2. Move the joint by `--delta`, measure again.
3. Record the trial against the **measured** joint displacement, never the commanded one.
   Backlash and tracking error make those different numbers, and the derivative is only as
   good as its denominator.
4. Repeat in the other direction.

The gripper is excluded on purpose: opening and closing it changes the image by occluding
it, not by moving the camera, so its "derivative" would be an artefact of the fingers
entering the frame.

Each column is fitted through the origin over that joint's own trials, so a joint with
backlash or barely any effect on the image shows up as its own residual instead of being
smeared across its neighbours. `wrist-check` prints them.

Finally the arm returns to the start and the image is measured once more. **If it has not
come back to the goal, nothing is saved.** This is the one check that catches the failure
that matters most and is otherwise invisible: the carabiner was nudged partway through, so
every column measured after that point is about a different scene and nothing in the numbers
themselves would say so. The gate is `--return-scale` × tolerance rather than tolerance
itself, because a teach that failed on detector spread would just be re-run until it passed.

## Debugging the loop: `wrist-centre`

`wrist-track` answers a question that is too big to debug. "Did the handover converge?"
mixes translation, range, rotation, the taught goal and the detector together, so a failure
rules none of them out.

`wrist-centre` strips it to the one part a person can check by eye. It servos the **spine
midpoint** — the middle of the bar the gripper closes on, which the detector already marks
on its overlay — to the centre of the wrist frame, and gives `scale_px` and `spine_deg` an
`IGNORED_TOLERANCE`, so the arm does not try to change its range or rotation while doing it.
It runs slower than `track` by default (0.06 rad/s), because it is the command you watch.

Read it like this:

| what you see | what it means |
| --- | --- |
| marker walks to the middle and stays | camera, detector, Jacobian signs and servo all work; any remaining trouble is in the taught goal |
| marker walks the *wrong* way | a Jacobian column has the wrong sign — re-teach, and check nothing moved during the bumps |
| `reason` stays `no detection` | the detector, not the servo: check the overlay, the lighting, and `EVEREST_WRIST_CAMERA_COLOR` |
| `reason` is `servo step outside the taught range` | the carabiner is far outside where the Jacobian was measured; move it closer and re-run |
| `moved` never becomes true | lock-on, limits or feedback — the reason column says which |

Nothing is written. The stored calibration is retargeted in memory, so this is safe to run
at any time without disturbing what the FSM uses. `--point` also accepts `insert` and
`aperture`.

One approximation is being made and is printed when it applies: if `--point` differs from
the point the Jacobian was taught on, only the Jacobian's `u`/`v` rows are used. Those
transfer between points on one rigid object under small camera translations; the `scale_px`
and `spine_deg` rows do not — which is why this mode does not servo either of them.

## Running the FSM on it

```
EVEREST_SEARCH_CV=wrist just attach-fsm-act <checkpoint>
# or
just attach-fsm-act <checkpoint> auto wrist
```

The calibration is loaded and refused before the robot is claimed; it is checked against the
arm's identity and against `EVEREST_CAMERAS` once the session is open, before anything is
energized. Supplying both calibrations is refused rather than ranked — two calibrations
commanding one arm with no arbiter between them is not a configuration to pick a winner
from.

## Limits worth knowing

- **The Jacobian is measured at one pose.** Away from it the calibration gives a direction,
  not a step. That is enough because `VisualTracker` clamps the step anyway, but a solve
  asking for more than `max_delta_rad` on any joint is refused rather than clamped, and the
  arm holds.
- **No range sensing.** `scale_px` is apparent size. A carabiner of a different size, or
  seen against a very different background, changes it.
- **The detector is the ceiling.** `carabiner_detect` is a hysteresis threshold with shape
  validation tuned on a green Petzl Spirit on wood. Nothing here reports a confidence,
  because there is no score behind it to report.
