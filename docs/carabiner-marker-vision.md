# Carabiner marker vision

The overhead-camera detector uses two white tape markers to define the carabiner axis and
the black gate to resolve its direction. These commands perform vision and target planning
only. They do not connect to or move the robot arm.

## Setup

Install the locked environment and fix the camera in its final position before choosing an
ROI or recording calibration points:

```bash
uv sync
```

On macOS, allow the terminal application to use the camera under **System Settings >
Privacy & Security > Camera**. Camera `0` is the current KD-USB camera; try `1` if device
ordering changes.

`--roi X Y WIDTH HEIGHT` is the fixed pickup zone in full-frame pixels. It should cover the
whole area where the carabiner may be placed, not a tight crop around its current position.
The current setup uses `740 430 560 460`.

## Capture one debug frame

```bash
uv run robot-two-white-black-debug \
    --camera 0 \
    --roi 740 430 560 460 \
    --image-out /tmp/two_white_black_pickup_zone_debug.jpg
```

The command discards 15 frames by default so auto-exposure and white balance can settle.
Use `--warmup-frames N` to change that. A successful run prints all three image points and
writes an annotated image.

## Run continuous detection

```bash
uv run robot-two-white-black-live \
    --camera 0 \
    --roi 740 430 560 460
```

Press `q` or Escape to stop. For a terminal-only run that continually updates an annotated
image:

```bash
uv run robot-two-white-black-live \
    --camera 0 \
    --roi 740 430 560 460 \
    --no-window \
    --image-out /tmp/two_white_black_live.jpg
```

`--max-frames N` makes the live command exit after a finite number of frames and is useful
for hardware checks or scripts.

The printed values are:

- `white_a=(u,v)` and `white_b=(u,v)`: centers of the two white tapes in full-frame pixels.
- `black_gate=(u,v)`: center of the black gate in full-frame pixels.

Pixel `u` increases from left to right. Pixel `v` increases from top to bottom.

## Optional pregrasp calculation

Copy the example and replace all calibration and grasp values with measured values from the
actual camera and robot:

```bash
cp pickup_config.example.json pickup_config.json
```

`image_points_px` and `robot_points_m` must list the same table points in the same order.
Use at least four non-collinear points spread across the pickup area. More measured pairs
are supported and give the matrix solver a least-squares fit that is less sensitive to
clicking and measurement noise. `robot_points_m` must be measured in the robot base frame;
they cannot be inferred from camera pixels alone. The example image points are the four
black tape centers captured from the current KD-USB setup in top-left, top-right,
bottom-right, bottom-left order. The example robot coordinates are placeholders and must
not be used to command the arm.

The solver computes a 3x3 homography `H`:

```text
[rx, ry, w]^T = H [u, v, 1]^T
x = rx / w
y = ry / w
```

Check a camera point without opening the camera or commanding the robot:

```bash
just camera-to-xy 1000 650 pickup_config.json
```

The command prints the solved matrix and the resulting robot-base `x` and `y` in meters.
You can also convert several points in one invocation:

```bash
uv run robot-camera-to-xy --config pickup_config.json \
    --point 1000 650 --point 1100 700
```

Once calibrated, pass the file to print the calculated pregrasp target alongside each
detection:

```bash
uv run robot-two-white-black-live \
    --camera 0 \
    --roi 740 430 560 460 \
    --config pickup_config.json
```

This still does not move the robot. It only prints `PREGRASP x=... y=... z=... yaw_deg=...`.

## Troubleshooting

- `Failed to spawn`: run `uv sync` and confirm the command appears under `[project.scripts]`
  in `pyproject.toml`.
- `Could not open camera index`: close other camera processes and verify macOS camera
  permission.
- `MISS Could not find green carabiner body`: verify the carabiner is inside the ROI and
  reduce reflections or camera exposure changes.
- Too many white candidates: remove bright objects from the pickup zone or make the ROI
  smaller while still covering the full allowed zone.
- Unstable black-gate detection: use even table lighting and keep dark cables and tools out
  of the pickup zone.
