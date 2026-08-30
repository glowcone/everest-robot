"""Compatibility shim. The detector now lives in the installed package.

It moved to :mod:`everest_robot.carabiner_detect` because the attachment FSM's perception
imports it at runtime, and ``robot-attach-fsm`` is an installed console script: ``sys.path``
starts at the script's directory, not the working directory, so a top-level ``carabiner``
package next to the repository root is not importable from it.

This module is kept so existing scratch scripts and notebooks keep working. New code should
import from :mod:`everest_robot.carabiner_detect` directly.
"""

from everest_robot.carabiner_detect import (
    MAX_AREA,
    MIN_AREA,
    GraspTarget,
    NotFound,
    chroma_mask,
    detect,
    draw,
    teal_score,
)

__all__ = [
    "MAX_AREA",
    "MIN_AREA",
    "GraspTarget",
    "NotFound",
    "chroma_mask",
    "detect",
    "draw",
    "teal_score",
]
