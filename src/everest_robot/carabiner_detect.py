"""Classical CV detection of a green Petzl Spirit carabiner lying on a table.

The wrist camera runs auto white balance and the anodised frame is specular, so
absolute colour thresholds tuned on one frame do not survive to the next. Every
threshold here is therefore relative to the frame's own background: the table
fills most of the view, so its median (a*, b*) is a per-frame estimate of "what
neutral looks like right now". The carabiner is the teal side of that estimate.

Output is a GraspTarget in pixel coordinates. Nothing here touches the robot.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GraspTarget:
    """Where to put the fingertip, in image space."""

    aperture: tuple[float, float]  # centroid of the frame's inner hole
    spine: tuple[float, float]  # midpoint of the spine (side opposite the gate)
    spine_angle: float  # local spine tangent, degrees, in [-90, 90)
    insert: tuple[float, float]  # aperture centroid biased toward the spine
    area: float  # frame mask area, px
    mask: np.ndarray
    outer: np.ndarray
    inner: np.ndarray


class NotFound(Exception):
    """No component in the frame looked like the carabiner."""


# Area bounds, px, for a 100x60 mm carabiner in the wrist view. The near/far
# extremes of the descent change apparent size by roughly 4x.
MIN_AREA = 900
MAX_AREA = 60000

#: Hysteresis floors, in background sigmas, tried strictest first. See `detect`:
#: 3.0 is what the near-field frames were tuned on and stays the default answer;
#: the lower rungs exist for the far end of the descent, where the ring thins to a
#: few pixels of stock and its shaded spine falls under 3 sigma. 2.0 is below the
#: ~2.2 sigma at which the worst overexposed wood grain leaks, so a detection that
#: needed it rests on weak evidence -- which is why it is last and never preferred.
LOW_THRESHOLDS = (3.0, 2.5, 2.0)


def teal_score(bgr: np.ndarray) -> np.ndarray:
    """Per-pixel distance, in background sigmas, into the teal quadrant.

    Lab a*/b* are re-centred on the frame median, which absorbs the camera's
    auto white balance: the table is warm wood (both channels positive after
    centring), the carabiner is the opposite quadrant.

    The background is scored with a full 2D Mahalanobis distance rather than
    per-channel thresholds because wood grain is strongly correlated across
    a*/b* (measured r = 0.53-0.67) -- an axis-aligned threshold is the wrong
    shape for that cloud and has to be loosened until overexposed grain leaks
    through. Under the correct metric the anodised frame sits at ~9 sigma and
    the worst leaking pixels at ~2.2, which is the gap the thresholds use.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    a0, b0 = np.median(a), np.median(b)
    da, db = a - a0, b - b0

    # Covariance from a trimmed core so the carabiner itself cannot inflate the
    # background model it is being measured against.
    ma = 1.4826 * np.median(np.abs(da)) + 1e-3
    mb = 1.4826 * np.median(np.abs(db)) + 1e-3
    inlier = (np.abs(da) < 3 * ma) & (np.abs(db) < 3 * mb)
    C = np.cov(np.stack([da[inlier], db[inlier]])) + np.eye(2) * 1e-3
    Ci = np.linalg.inv(C)

    d = np.sqrt(
        np.maximum(
            Ci[0, 0] * da * da + 2 * Ci[0, 1] * da * db + Ci[1, 1] * db * db,
            0,
        )
    )

    # Mahalanobis is unsigned, so keep only the teal side -- otherwise anything
    # merely far from wood (a bright window, a red shirt) scores just as high.
    d = np.where(da + db < 0, d, 0.0)

    # Suppress the fingers. They are NOT neutral: they sit slightly blue of the
    # warm wood median, so they score positive on the teal axis and at 2 sigma
    # they outnumber real carabiner pixels roughly 3:1. Lightness is what
    # separates them -- measured L is ~148 on the anodised frame against ~38-57
    # on the fingers. The gate scales with the frame's own median so it tracks
    # exposure rather than pinning an absolute level.
    return np.where(0.55 * np.median(L) < L, d, 0.0)


def in_roi(score: np.ndarray, roi_xywh: Sequence[int] | None) -> np.ndarray:
    """Zero the score outside a region of interest, keeping full-frame coordinates.

    Masking the score rather than cropping the image is deliberate: every caller
    works in absolute pixels -- the wrist servo's goal, the pixel map, the
    overlay -- and a crop would silently shift all of them by the ROI origin.

    This is the answer to background clutter, and it is not a nicety. The teal
    score is relative to the frame's own median, so a bench in a room with a
    bright screen, plants and people produces teal-scoring blobs that are
    genuinely ring-shaped. Nothing in one frame distinguishes a small carabiner
    far away from a ring-shaped object across the room; what distinguishes them
    is that the robot cannot reach across the room. That is a fact about the
    workspace, so it has to be told, not inferred.
    """
    if roi_xywh is None:
        return score
    x, y, w, h = (int(value) for value in roi_xywh)
    if w <= 0 or h <= 0:
        raise ValueError(f"roi_xywh must have positive width and height, got {w}x{h}")
    kept = np.zeros(score.shape, bool)
    kept[max(0, y) : y + h, max(0, x) : x + w] = True
    return np.where(kept, score, 0.0)


def chroma_mask(
    bgr: np.ndarray,
    hi: float = 5.0,
    lo: float = 3.0,
    score: np.ndarray | None = None,
) -> np.ndarray:
    """Hysteresis threshold on the teal score, in background sigmas.

    A single threshold cannot win here: set it high and the washed-out lower
    arc of the ring drops out, breaking the loop; set it low and wood grain
    leaks in. Hysteresis takes confident seeds at `hi` sigma and grows them
    through contiguous `lo`-sigma pixels, so a marginal arc survives as long as
    some part of the ring is confidently teal.

    `score` lets a caller supply an already-computed (and possibly ROI-masked)
    score, so the ladder in `detect` pays for the Mahalanobis pass once.
    """
    d = teal_score(bgr) if score is None else score
    strong = d > hi
    weak = (d > lo).astype(np.uint8)

    n, lbl = cv2.connectedComponents(weak, 8)
    keep = np.zeros(n, bool)
    keep[np.unique(lbl[strong])] = True
    keep[0] = False  # background label
    m = keep[lbl].astype(np.uint8) * 255

    # Kernel must stay well under the aperture width (~40 px) or the close
    # welds the hole shut; 3x3 is enough to bridge specular pinholes.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return m


def _plausible(mask: np.ndarray) -> list[np.ndarray]:
    """Every component of the right rough size and build, largest first.

    Largest first, and *every* one rather than only the largest: the largest
    teal thing in view is often not the carabiner -- a bright screen or a
    plant scores teal against a wood median and can be an order of magnitude
    bigger. Size and aspect are cheap pre-filters; `_validate` is the arbiter,
    so the caller tries them in turn instead of committing to one.
    """
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    ranked = []
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if max(w, h) / max(1, min(w, h)) > 4.0:  # carabiner is ~1.7:1, not a sliver
            continue
        ranked.append((area, i))
    return [
        (lbl == i).astype(np.uint8) * 255 for _, i in sorted(ranked, key=lambda r: -r[0])
    ]


def detect(bgr: np.ndarray, roi_xywh: Sequence[int] | None = None) -> GraspTarget:
    """Locate the aperture and spine of the carabiner in a wrist-camera frame.

    The hysteresis floor is a ladder walked strictest first, not a constant.
    One floor cannot serve the whole descent: near the table the ring is broad
    and bright and `LOW_THRESHOLDS[0]` is right, while a leak at that range
    fills the aperture and fails validation. Far away the ring is a few pixels
    of stock, the spine falls to ~3 sigma, and the loop breaks into arcs that
    no longer contain a hole. Lowering the floor globally to fix the far case
    breaks the near one, which is measurable: on the tuning frames it costs
    more detections than it wins.

    So the floor descends only when *nothing* validated above it. Every frame
    that resolved at the strictest floor still resolves there, with the same
    answer; only frames that would otherwise have been a miss pay the extra
    passes and accept weaker evidence.
    """
    score = in_roi(teal_score(bgr), roi_xywh)
    refusals: list[str] = []

    for lo in LOW_THRESHOLDS:
        mask_at = chroma_mask(bgr, lo=lo, score=score)
        for mask in _plausible(mask_at):
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            outer = max(cnts, key=cv2.contourArea)
            try:
                inner = _aperture(mask, outer)
                _validate(mask, outer, inner)
            except (NotFound, ValueError) as error:
                refusals.append(f"{lo:g}sigma: {error}")
                continue
            return _target(bgr, mask, outer, inner)

    where = "" if roi_xywh is None else f" inside ROI {tuple(int(v) for v in roi_xywh)}"
    if not refusals:
        raise NotFound(f"no component in the expected size/shape range{where}")
    raise NotFound(f"no carabiner-shaped component{where} ({'; '.join(refusals[:3])})")


def _target(
    bgr: np.ndarray, mask: np.ndarray, outer: np.ndarray, inner: np.ndarray
) -> GraspTarget:
    """Turn one accepted component into the grasp target."""
    mi = cv2.moments(inner)
    ap = (mi["m10"] / mi["m00"], mi["m01"] / mi["m00"])

    spine_pt, ang = _spine(bgr, mask, outer, ap)

    # Bias the insertion point off the aperture centre toward the spine so the
    # fingertip drops through the hole close to the bar it will close on.
    v = np.array([spine_pt[0] - ap[0], spine_pt[1] - ap[1]], np.float32)
    n = np.linalg.norm(v) + 1e-6
    px_per_mm = _px_per_mm(outer)
    ins = (ap[0] + v[0] / n * 8.0 * px_per_mm, ap[1] + v[1] / n * 8.0 * px_per_mm)

    return GraspTarget(ap, spine_pt, ang, ins, float(mask.sum() / 255), mask, outer, inner)


def _validate(mask: np.ndarray, outer: np.ndarray, inner: np.ndarray) -> None:
    """Reject detections whose shape is not carabiner-like.

    Finding *a* component is not the same as finding the part: a mask that has
    leaked into the fingers still yields a contour and a hole, just a wrong
    one. These three ratios are what separate a ring from a puddle, and all are
    scale-free so they hold across the descent.
    """
    hull_area = cv2.contourArea(cv2.convexHull(outer))
    if hull_area <= 0:
        raise NotFound("degenerate hull")

    (_, _), (w, h), _ = cv2.minAreaRect(outer)
    aspect = max(w, h) / max(1e-6, min(w, h))
    if not 1.15 <= aspect <= 2.8:  # 100x60 mm is 1.67 before perspective
        raise NotFound(f"aspect {aspect:.2f} not carabiner-like")

    hole = cv2.contourArea(inner) / hull_area
    if not 0.15 <= hole <= 0.62:  # the aperture is a big fraction of the frame
        raise NotFound(f"aperture/hull {hole:.2f} out of range")

    ring = float(mask.sum() / 255) / hull_area
    if not 0.20 <= ring <= 0.70:  # a ring fills part of its hull, a blob fills it
        raise NotFound(f"ring/hull {ring:.2f} out of range")


def _aperture(mask: np.ndarray, outer: np.ndarray) -> np.ndarray:
    """The frame's inner hole, without requiring the ring to be topologically closed.

    Asking findContours for a child contour fails whenever a washed-out arc
    breaks the loop -- and that is common here. Instead take the convex hull of
    the ring and subtract the ring itself: what is left is the aperture plus
    thin slivers along the concave gate side. The aperture is by far the
    largest of those, and this survives a broken arc because the hull closes
    the gap for us.
    """
    hull = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(hull, [cv2.convexHull(outer)], -1, 255, -1)
    interior = cv2.erode(hull, np.ones((5, 5), np.uint8)) & ~mask

    n, lbl, st, _ = cv2.connectedComponentsWithStats(interior, 8)
    if n < 2:
        raise NotFound("no interior region inside the frame hull")
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < 0.08 * cv2.contourArea(cv2.convexHull(outer)):
        raise NotFound("largest interior region too small to be the aperture")

    c, _ = cv2.findContours(
        (lbl == i).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return max(c, key=cv2.contourArea)


def _px_per_mm(outer: np.ndarray) -> float:
    """Scale from the frame's long axis, which is ~100 mm."""
    (_, _), (w, h), _ = cv2.minAreaRect(outer)
    return max(w, h) / 100.0


def _spine(bgr, mask, outer, ap) -> tuple[tuple[float, float], float]:
    """Spine midpoint and tangent.

    The gate is the black bar closing one side of the frame. Finding it tells
    us which side is the spine: the spine is simply the far side. We look for
    dark, low-chroma pixels inside the frame's convex hull but outside the
    green mask -- that is exactly the gate.
    """
    hull = cv2.convexHull(outer)
    inside = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(inside, [hull], -1, 255, -1)
    inside = cv2.erode(inside, np.ones((7, 7), np.uint8))

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    dark = ((np.percentile(L[inside > 0], 25) > L) & (inside > 0) & (mask == 0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # The gripper fingers are also dark and they overlap the hull on approach,
    # which drags the gate centroid toward them and flips the spine to the
    # wrong side. They always run off the edge of the frame; the gate, being
    # inside the carabiner, never does. So drop border-touching components.
    n, lbl, st, cent = cv2.connectedComponentsWithStats(dark, 8)
    h, w = dark.shape
    best, best_area = None, 0
    for i in range(1, n):
        x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
        cw, ch = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if x <= 0 or y <= 0 or x + cw >= w or y + ch >= h:
            continue
        if st[i, cv2.CC_STAT_AREA] > best_area:
            best, best_area = i, st[i, cv2.CC_STAT_AREA]

    pts = outer.reshape(-1, 2).astype(np.float32)
    # None means no gate was visible (occluded or blown out); the caller then falls back
    # to the aperture's offset from centre, which is a weaker cue for the same thing.
    gate = np.array(cent[best], np.float32) if best is not None and best_area > 30 else None

    # Work in the carabiner's own axes. "Furthest from the gate centroid" picks
    # a diagonal corner, because the gate is a long bar whose centroid is also
    # offset along its own length. The spine is specifically the long side
    # opposite the gate, so decide the side along the SHORT axis and then take
    # the middle band along the long axis.
    (cx, cy), (rw, rh), angd = cv2.minAreaRect(outer)
    centre = np.array([cx, cy], np.float32)
    th = np.deg2rad(angd if rw >= rh else angd + 90)
    long_ax = np.array([np.cos(th), np.sin(th)], np.float32)
    short_ax = np.array([-long_ax[1], long_ax[0]], np.float32)
    long_len = max(rw, rh)

    s = (pts - centre) @ short_ax
    t = (pts - centre) @ long_ax

    if gate is not None:
        side = -np.sign((gate - centre) @ short_ax) or 1.0
    else:
        # No gate visible (occluded or blown out): the aperture centroid sits
        # off-centre toward the gate, so it gives the same cue, more weakly.
        side = -np.sign((np.array(ap, np.float32) - centre) @ short_ax) or 1.0

    on_spine = (s * side) > np.percentile(s * side, 75)
    mid_band = np.abs(t) < 0.35 * long_len
    sel = pts[on_spine & mid_band]
    if len(sel) < 5:
        sel = pts[on_spine]

    mid = sel.mean(axis=0)

    # Tangent from the principal direction of the selected spine points.
    _, _, vt = np.linalg.svd(sel - mid, full_matrices=False)
    ang = float(np.degrees(np.arctan2(vt[0][1], vt[0][0])))
    ang = (ang + 90) % 180 - 90
    return (float(mid[0]), float(mid[1])), ang


def draw(bgr: np.ndarray, t: GraspTarget) -> np.ndarray:
    """Debug overlay. cv2.imshow is unavailable (headless build) -- write a PNG."""
    v = bgr.copy()
    cv2.drawContours(v, [t.outer], -1, (0, 255, 0), 2)
    cv2.drawContours(v, [t.inner], -1, (255, 128, 0), 2)
    cv2.circle(v, tuple(np.int32(t.aperture)), 5, (255, 255, 0), -1)
    cv2.circle(v, tuple(np.int32(t.insert)), 6, (0, 0, 255), 2)
    cv2.circle(v, tuple(np.int32(t.spine)), 5, (255, 0, 255), -1)
    th = np.deg2rad(t.spine_angle)
    d = np.array([np.cos(th), np.sin(th)]) * 40
    p = np.array(t.spine)
    cv2.line(v, tuple(np.int32(p - d)), tuple(np.int32(p + d)), (255, 0, 255), 2)
    return v
