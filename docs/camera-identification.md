# Identifying the cameras

`EVEREST_CAMERAS` pairs a **name** the policy uses (`wrist`, `front`) with an **id** the
host uses (`/dev/video3`, `2`). Nothing in the running system can check that pairing. Swap
the two ids and every layer still passes: the shapes match, the checkpoint's feature
mapping matches, `INITIAL`'s readiness gate is satisfied, frames arrive at the right rate.
The policy simply looks through the wrong lens, and the failure surfaces as a model that
"doesn't work on this rig".

`robot-cameras` is how that pairing gets established and re-checked — by eye, by putting
each camera on screen with its id drawn into the picture. It is read-only: it opens
cameras, draws on the frames and commands nothing, and it never claims the robot lease. It
does hold the cameras, though, so quit it before starting a rollout, a recording session or
`pixel-track`.

## 1. See what the host has

```
just camera-scan
```

One window per capture device that actually delivers frames, laid out in a grid, each
labelled with:

```
id /dev/video3                          <- what to configure
stable id  /dev/v4l/by-id/usb-Chi...index0
640x480   30 fps                        <- what the driver actually delivers
```

Wave a hand in front of a lens and read the id off that window. Write it down against the
name the checkpoint uses for that view.

Keys: `q` or `Esc` quits, `s` saves every window to `out/camera-ids/`.

Devices that open but never produce a frame are listed as skipped rather than shown. On
Linux that is normally the metadata node a UVC camera exposes next to its video node — the
reason a host with two cameras shows five `/dev/video*` entries.

Useful flags:

| flag | when |
| --- | --- |
| `--only /dev/video3 2` | probe just these ids |
| `--fourcc MJPG` | several USB cameras cannot stream at once — MJPG fits them in the bus budget |
| `--width 1280 --height 720` | check whether the driver will actually give a size |
| `--display-width 0` | stop downscaling the windows |
| `--no-window` | headless or over ssh: report only |

## 2. Write the configuration

```
just camera-scan-json
```

prints the same report with no windows, plus an `EVEREST_CAMERAS` skeleton for what it
found. The names in it are placeholders (`camera0`, `camera1`) on purpose — which one is
`wrist` is exactly the fact step 1 establishes by eye, and inventing it here would defeat
the point. Rename them and put the result in `.env`:

```
EVEREST_CAMERAS='[{"name":"wrist","kind":"opencv","index_or_path":"/dev/v4l/by-id/usb-Chicony...-video-index0","width":640,"height":480,"fps":30}]'
```

**Prefer the `by-id` path over `/dev/videoN`.** Device numbers are handed out in probe
order, so they move when a camera is unplugged, when another is added, and sometimes across
a plain reboot — silently swapping `wrist` and `front` in a configuration that was correct
yesterday. The by-id link comes from the device's own USB identity and does not move. The
scan prints it as `configure as ...` wherever the host offers one.

The name must match what the checkpoint calls the view. `just policy-check <checkpoint>`
prints the names and shapes a checkpoint needs and compares them against `EVEREST_CAMERAS`.

## 3. Check the configuration you wrote

```
just camera-show
```

opens the cameras `EVEREST_CAMERAS` already configures, by name, through the same
`CameraRuntime` a policy rollout uses. This is the step that catches:

* an id that points at the wrong camera — the window says `wrist` over the front view;
* an id nothing answers on, or a device another process is holding;
* **a size the driver quietly refused.** A camera asked for 1280x720 that hands back
  640x480 looks perfectly correct in the window, and only breaks at the first inference,
  where the checkpoint's feature shape no longer matches. `camera-show` flags it as
  `MISMATCH` and exits non-zero, `--no-window` included, so it can gate a rollout.

`--camera wrist` restricts it to one; `--no-window` reports without opening any.

## Troubleshooting

**"OpenCV cannot open a window"** — the environment has the headless OpenCV build.
`lerobot` depends on `opencv-python-headless`, which ships the same `cv2` module as
`opencv-python` and overwrites it, so `just setup-hardware` can leave a `cv2` with no
`imshow` at all. This affects every window in this repository, `robot-pixel-map collect`
included. Repair it with `uv pip install --reinstall opencv-python`, or use `--no-window`.

**A camera opens but produces no frame** — either it is a metadata node (expected), or
another process is holding it. Stop any running `robot-pixel-map`, `tools/preview.py`,
`robot-two-white-black-live` or rollout. On macOS, also check the terminal's Camera
permission in System Settings › Privacy & Security.

**Several cameras will not open together** — a USB controller's bandwidth, not a bad id.
Try `--fourcc MJPG`, a smaller `--width`/`--height`, or scan them one at a time with
`--only`.

**No devices at all on Linux** — check that `/dev/video*` exists and that you are in the
`video` group.
