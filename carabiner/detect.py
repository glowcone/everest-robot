"""Classical CV detection of a green Petzl Spirit carabiner lying on a table.

The wrist camera runs auto white balance and the anodised frame is specular, so
absolute colour thresholds tuned on one frame do not survive to the next. Every
threshold here is therefore relative to the frame's own background: the table
fills most of the view, so its median (a*, b*) is a per-frame estimate of "what
neutral looks like right now". The carabiner is the teal side of that estimate.

Output is a GraspTarget in pixel coordinates. Nothing here touches the robot.
"""

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
    return np.where(L > 0.55 * np.median(L), d, 0.0)


def chroma_mask(bgr: np.ndarray, hi: float = 5.0, lo: float = 3.0) -> np.ndarray:
    """Hysteresis threshold on the teal score, in background sigmas.

    A single threshold cannot win here: set it high and the washed-out lower
    arc of the ring drops out, breaking the loop; set it low and wood grain
    leaks in. Hysteresis takes confident seeds at `hi` sigma and grows them
    through contiguous `lo`-sigma pixels, so a marginal arc survives as long as
    some part of the ring is confidently teal.
    """
    d = teal_score(bgr)
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


def _largest_valid(mask: np.ndarray) -> np.ndarray:
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    best, best_area = None, 0
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if max(w, h) / max(1, min(w, h)) > 4.0:  # carabiner is ~1.7:1, not a sliver
            continue
        if area > best_area:
            best, best_area = i, area
    if best is None:
        raise NotFound("no component in the expected size/shape range")
    return (lbl == best).astype(np.uint8) * 255


def detect(bgr: np.ndarray) -> GraspTarget:
    """Locate the aperture and spine of the carabiner in a wrist-camera frame."""
    mask = _largest_valid(chroma_mask(bgr))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outer = max(cnts, key=cv2.contourArea)
    inner = _aperture(mask, outer)
    _validate(mask, outer, inner)

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
    dark = ((L < np.percentile(L[inside > 0], 25)) & (inside > 0) & (mask == 0)).astype(np.uint8)
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
    if best is not None and best_area > 30:
        gate = np.array(cent[best], np.float32)
    else:
        # No gate visible (occluded or blown out): fall back to the frame's
        # long axis and take the side further from the aperture's short side.
        gate = None

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
