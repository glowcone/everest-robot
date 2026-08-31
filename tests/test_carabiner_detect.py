"""How the wrist detector chooses among candidates, and where it is allowed to look.

Synthetic frames, because the two mechanisms under test are about *arbitration* rather than
about segmentation quality, and arbitration is only checkable when the right answer is
known. Each frame is warm wood with teal rings painted on it at controlled strength: the
detector's thresholds are all relative to the frame's own background, so a painted frame
exercises the same code path a photograph does.

Both behaviours here were built from a real failure -- a bench photographed in a room with a
bright screen, plants and people behind it, where the detector reported no detection while
the carabiner sat plainly in the middle of the table.
"""

import numpy as np
import pytest

from everest_robot.carabiner_detect import LOW_THRESHOLDS, NotFound, detect, in_roi

#: The frames are built in Lab and converted, not painted in BGR, because every threshold
#: in the detector is measured in *background sigmas*. Controlling the wood's spread
#: directly is what lets a test say "this ring is 6 sigma" and mean it. The a*/b* noise is
#: correlated, as real wood grain is (the module documents r = 0.53-0.67) -- an uncorrelated
#: background would make the Mahalanobis metric trivially equal to a per-channel one and the
#: tests would stop exercising the thing that is actually there.
WOOD_L, WOOD_A, WOOD_B = 150.0, 140.0, 150.0
GRAIN_SIGMA = 4.0
GRAIN_R = 0.6


def frame(width: int = 640, height: int = 480, seed: int = 0) -> np.ndarray:
    """Warm wood with correlated grain, as BGR."""

    import cv2

    rng = np.random.default_rng(seed)
    cov = GRAIN_SIGMA**2 * np.array([[1.0, GRAIN_R], [GRAIN_R, 1.0]])
    noise = rng.multivariate_normal([0.0, 0.0], cov, size=(height, width))
    lab = np.zeros((height, width, 3), np.float32)
    lab[:, :, 0] = WOOD_L + rng.normal(0, 6, (height, width))
    lab[:, :, 1] = WOOD_A + noise[:, :, 0]
    lab[:, :, 2] = WOOD_B + noise[:, :, 1]
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def paint(canvas: np.ndarray, stencil: np.ndarray, strength: float) -> np.ndarray:
    """Push the stencilled pixels into the teal quadrant, and make them light.

    Teal is the *negative* a*/b* side of the wood median, which is what `teal_score` keeps;
    light because the detector suppresses dark pixels to reject the gripper's fingers.

    ``strength`` is a paint level, deliberately *not* a count of sigmas. The score the
    detector ends up measuring is not a linear function of the shift -- the a*/b* noise is
    correlated, so distance along the teal diagonal is compressed, and the trimmed
    covariance is re-estimated per frame. Naming this "sigmas" would invite reasoning that
    does not hold. Measured, for a 16 px stroke on this background:

        strength  1.6   2.0   2.5   3.0   4.0   5.0   8.0
        score     3.4   4.8   6.8   7.8  10.8  13.7  21.6
    """

    import cv2

    lab = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB).astype(np.float32)
    shift = strength * GRAIN_SIGMA / np.sqrt(2.0)
    where = stencil > 0
    lab[:, :, 0] = np.where(where, 200.0, lab[:, :, 0])
    lab[:, :, 1] = np.where(where, WOOD_A - shift, lab[:, :, 1])
    lab[:, :, 2] = np.where(where, WOOD_B - shift, lab[:, :, 2])
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def ring(
    canvas: np.ndarray,
    centre: tuple[int, int],
    size: tuple[int, int],
    thickness: int,
    strength: float = 8.0,
) -> np.ndarray:
    """Paint one carabiner-proportioned ring at a controlled evidence strength."""

    import cv2

    stencil = np.zeros(canvas.shape[:2], np.uint8)
    cv2.ellipse(stencil, centre, (size[0] // 2, size[1] // 2), 0, 0, 360, 255, thickness)
    return paint(canvas, stencil, strength)


def shaded_ring(
    canvas: np.ndarray,
    centre: tuple[int, int],
    size: tuple[int, int],
    thickness: int,
    bright: float,
    shaded: float,
    arc: tuple[int, int] = (200, 340),
) -> np.ndarray:
    """A ring whose spine side is in shadow, which is how a real one fails far away.

    Uniform faintness is not the interesting case -- the hysteresis seeds anywhere on the
    ring and grows through the whole of it. What breaks the loop is one arc falling under
    the floor while the rest stays bright, leaving arcs that no longer enclose a hole.
    """

    import cv2

    canvas = ring(canvas, centre, size, thickness, strength=bright)
    stencil = np.zeros(canvas.shape[:2], np.uint8)
    cv2.ellipse(
        stencil, centre, (size[0] // 2, size[1] // 2), 0, arc[0], arc[1], 255, thickness
    )
    return paint(canvas, stencil, shaded)


def slab(canvas: np.ndarray, box: tuple[int, int, int, int], strength: float = 8.0):
    """A solid teal rectangle: big, plausible by size and aspect, but with no aperture."""

    x, y, w, h = box
    stencil = np.zeros(canvas.shape[:2], np.uint8)
    stencil[y : y + h, x : x + w] = 255
    return paint(canvas, stencil, strength)


# ── the region of interest ─────────────────────────────────────────────────────────
def test_a_ring_outside_the_region_of_interest_is_not_found():
    """The reason the ROI exists. A ring-shaped teal thing across the room is
    indistinguishable from a small carabiner far away in any single frame; what separates
    them is that the arm cannot reach across the room, which only configuration knows."""

    canvas = ring(frame(), (540, 110), (90, 150), 9)

    assert detect(canvas) is not None
    with pytest.raises(NotFound, match="inside ROI"):
        detect(canvas, roi_xywh=(0, 200, 420, 280))


def test_the_region_of_interest_keeps_full_frame_coordinates():
    """Masking the score rather than cropping the image: every caller -- the wrist servo's
    goal, the overlay -- works in absolute pixels, and a crop would shift all of them."""

    canvas = ring(frame(), (400, 300), (90, 150), 9)

    inside = detect(canvas, roi_xywh=(250, 150, 340, 300))

    assert inside.aperture[0] == pytest.approx(400, abs=12)
    assert inside.aperture[1] == pytest.approx(300, abs=12)


def test_a_degenerate_region_of_interest_is_refused():
    with pytest.raises(ValueError, match="positive width and height"):
        in_roi(np.zeros((10, 10), float), (0, 0, 0, 5))


# ── choosing among candidates ──────────────────────────────────────────────────────
def test_the_largest_teal_thing_is_not_assumed_to_be_the_carabiner():
    """A bright screen or a plant outscores the carabiner and passes the cheap size and
    aspect filters. Committing to the largest component is what produced the original
    failure: validation rejected it and the frame was reported empty, with the carabiner
    plainly in view."""

    # A slab bigger than the ring, in the size range, carabiner-ish aspect, and solid --
    # so it has no aperture and cannot survive validation.
    canvas = ring(slab(frame(), (380, 40, 130, 160)), (200, 300), (90, 150), 9)

    found = detect(canvas)

    assert found.aperture[0] == pytest.approx(200, abs=12)
    assert found.aperture[1] == pytest.approx(300, abs=12)


def test_nothing_carabiner_shaped_says_what_was_rejected():
    """The refusal an operator reads. "No detection" alone does not distinguish an empty
    table from a table whose only candidate was a solid slab."""

    canvas = slab(frame(), (200, 100, 140, 160))

    with pytest.raises(NotFound, match="sigma:"):
        detect(canvas)


def test_an_empty_table_says_so_without_pretending_to_have_judged_a_shape():
    with pytest.raises(NotFound, match="expected size/shape range"):
        detect(frame())


def _validates(mask: np.ndarray) -> bool:
    """Whether any component of this mask survives the detector's shape validation."""

    import cv2

    from everest_robot.carabiner_detect import _aperture, _plausible, _validate

    for component in _plausible(mask):
        cnts, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        outer = max(cnts, key=cv2.contourArea)
        try:
            _validate(component, outer, _aperture(component, outer))
        except (NotFound, ValueError):
            continue
        return True
    return False


# ── the threshold ladder ───────────────────────────────────────────────────────────
def test_the_strictest_threshold_is_preferred_when_it_already_works():
    """Every frame that resolved before the ladder existed must still resolve the same way.
    The lower rungs admit weaker evidence, and a detector that reached for them when it did
    not have to would quietly trade accuracy for recall on the near-field frames the
    thresholds were measured on."""

    canvas = ring(frame(), (320, 240), (110, 180), 11)

    strict = detect(canvas)

    # Same answer as pinning the floor at the strictest rung: the ladder never ran on.
    from everest_robot.carabiner_detect import chroma_mask, teal_score

    mask = chroma_mask(canvas, lo=LOW_THRESHOLDS[0], score=teal_score(canvas))
    assert mask[int(strict.spine[1]), int(strict.spine[0])] == 255


def test_a_faint_ring_is_recovered_by_descending_the_ladder():
    """The far end of the descent: the ring thins to a few pixels of stock and its shaded
    side falls under the strictest floor, breaking the loop so no component contains a
    hole. Lowering the floor globally is not the fix -- it fills the aperture on near-field
    frames -- so the floor descends only when nothing validated above it."""

    from everest_robot.carabiner_detect import chroma_mask, teal_score

    canvas = shaded_ring(frame(), (320, 240), (100, 160), 16, bright=8.0, shaded=1.4)

    # The strictest floor genuinely cannot answer: the shaded arc is below it, so the ring
    # is left as arcs enclosing nothing. Asserted, so this test cannot quietly stop
    # exercising the descent if the frame or the thresholds change.
    score = teal_score(canvas)
    assert not _validates(chroma_mask(canvas, lo=LOW_THRESHOLDS[0], score=score))
    assert _validates(chroma_mask(canvas, lo=LOW_THRESHOLDS[1], score=score))

    found = detect(canvas)
    assert found.aperture[0] == pytest.approx(320, abs=15)
    assert found.aperture[1] == pytest.approx(240, abs=15)


def test_the_ladder_is_ordered_strictest_first():
    """The order is the whole safety property: it is what makes the extra rungs a
    fallback rather than a change of default."""

    assert list(LOW_THRESHOLDS) == sorted(LOW_THRESHOLDS, reverse=True)
