"""Recording a measured pose into the robot parameters file.

``robot-monitor`` measures a pose; this writes it down. The two are separate because
writing to ``config/maker_arm_v1.yaml`` is the moment a pose stops being a number on a
terminal and becomes something the workflow will drive to, and that transition deserves its
own validated, reversible code path.

Three constraints shape everything here.

*The file is documentation.* Its comments carry the derivation of the LeRobot frame offsets
and the rules for presets -- content no loader preserves. So this edits the file as text,
splicing one entry into the ``named_positions`` block and leaving every other byte alone.
Round-tripping through a YAML dumper would silently delete the comments.

*A text edit can be wrong in ways a dumper cannot.* So the write is verified: the file is
replaced atomically, re-read through the strict loader, and rolled back to the original
bytes unless the preset that comes back is the one that went in. A failed save leaves the
file exactly as it was.

*Provenance is an attestation, not metadata.* ``approved_by`` and ``captured_at`` say a
person stood there and watched. Nothing here invents them, and nothing here validates the
approach to the pose -- that is docs/named-position-capture.md step 3, which happens after
the preset exists and before it is trusted.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from everest_robot.robot.parameters import ParameterError, RobotParameters

# A preset name is a YAML key and a shell argument to `just goto`. Restricting it to this
# means it never needs quoting in either, and cannot inject structure into the file.
NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
# Enough decimals to be below any tolerance the motion layer works in, and the same
# precision the monitor prints, so the written value matches what the operator read.
JOINT_DECIMALS = 7
# What the re-read value may differ from the captured one by: pure decimal truncation.
ROUNDTRIP_TOLERANCE_RAD = 1e-6

_BLOCK_KEY = "named_positions:"
_ENTRY_INDENT = "  "
_FIELD_INDENT = "    "


class CaptureRefused(RuntimeError):
    """The pose was not written, and the file was not changed."""


@dataclass(frozen=True, slots=True)
class CapturedPose:
    """One measured pose with the provenance the loader requires."""

    name: str
    joints: tuple[float, ...]
    calibration_id: str
    approved_by: str
    captured_at: date
    notes: str | None = None


def validate_name(name: str) -> str:
    """Reject a preset name before it becomes a YAML key."""

    if not NAME_PATTERN.match(name):
        raise CaptureRefused(
            f"{name!r} is not a usable preset name: start with a letter and use only "
            "letters, digits, hyphens and underscores"
        )
    return name


def _quote(value: str) -> str:
    """A double-quoted YAML scalar, which is safe for any operator-typed text."""

    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_preset(pose: CapturedPose) -> str:
    """The YAML block for one preset, indented to sit under ``named_positions``.

    Only schema fields are written. The robot id and config digest the monitor prints
    alongside a capture are identity for the operator to check, not preset data -- the
    loader rejects them, and ``calibration_id`` is the field that carries the same guard.
    """

    joints = ", ".join(f"{value:.{JOINT_DECIMALS}f}" for value in pose.joints)
    lines = [
        f"{_ENTRY_INDENT}{pose.name}:\n",
        f"{_FIELD_INDENT}joints: [{joints}]\n",
        f"{_FIELD_INDENT}calibration_id: {pose.calibration_id}\n",
        f"{_FIELD_INDENT}approved_by: {_quote(pose.approved_by)}\n",
        f"{_FIELD_INDENT}captured_at: {pose.captured_at.isoformat()}\n",
    ]
    if pose.notes:
        lines.append(f"{_FIELD_INDENT}notes: {_quote(pose.notes)}\n")
    return "".join(lines)


def _block_bounds(lines: list[str]) -> tuple[int, int, bool]:
    """Locate the ``named_positions`` block: its header, its end, and whether it is empty.

    The block ends at the first line that carries content in column zero -- the next
    top-level key, or the comment introducing it. Those comments belong to what follows,
    which is why they are not treated as part of this block.
    """

    header = next(
        (index for index, line in enumerate(lines) if line.startswith(_BLOCK_KEY)), None
    )
    if header is None:
        raise CaptureRefused(
            f"no top-level {_BLOCK_KEY!r} in the parameters file; this writer will not "
            "restructure a file it does not recognise"
        )
    inline = lines[header][len(_BLOCK_KEY) :].strip()
    if inline not in ("", "{}"):
        raise CaptureRefused(
            f"{_BLOCK_KEY} has the inline value {inline!r}; rewrite it as a block mapping "
            "before saving presets into it"
        )
    end = len(lines)
    for index in range(header + 1, len(lines)):
        if lines[index].strip() and not lines[index][0].isspace():
            end = index
            break
    return header, end, inline == "{}"


def _entry_bounds(lines: list[str], start: int, end: int, name: str) -> tuple[int, int] | None:
    """The line range of an existing preset inside the block, if it is there."""

    for index in range(start, end):
        match = re.match(r"^ {2}(\S+):", lines[index])
        if match is None or match.group(1).strip("'\"") != name:
            continue
        stop = end
        for following in range(index + 1, end):
            if re.match(r"^ {2}\S", lines[following]):
                stop = following
                break
        # Blank lines between presets separate them; they belong to neither.
        while stop > index and not lines[stop - 1].strip():
            stop -= 1
        return index, stop
    return None


def insert_preset(text: str, name: str, block: str, *, replace: bool = False) -> str:
    """Splice one rendered preset into ``named_positions``, leaving the rest untouched.

    Pure text in, pure text out, so it can be tested exhaustively without a filesystem.
    """

    lines = text.splitlines(keepends=True)
    header, end, empty = _block_bounds(lines)

    if empty:
        if replace:
            raise CaptureRefused(f"no preset named {name!r} to replace; the block is empty")
        lines[header] = f"{_BLOCK_KEY}\n"
        lines.insert(header + 1, block)
        return "".join(lines)

    existing = _entry_bounds(lines, header + 1, end, name)
    if existing is not None:
        if not replace:
            raise CaptureRefused(
                f"{name!r} is already a named position; recapturing one is a re-approval "
                "(docs/named-position-capture.md step 4), so say so explicitly"
            )
        start, stop = existing
        return "".join(lines[:start] + [block] + lines[stop:])
    if replace:
        raise CaptureRefused(f"no preset named {name!r} to replace")

    insert_at = end
    while insert_at > header + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines.insert(insert_at, block)
    return "".join(lines)


def _write_atomically(path: Path, text: str) -> None:
    """Replace the file in one step, so an interrupted save cannot truncate it."""

    temporary = path.with_name(f"{path.name}.robot-capture.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def save_preset(path: str | Path, pose: CapturedPose, *, replace: bool = False) -> Path:
    """Write ``pose`` into the parameters file at ``path``, or leave the file unchanged.

    Everything decidable without touching the file is checked first, and the edit itself
    is only kept if the file still loads and yields back the preset that was written.
    """

    path = Path(path)
    validate_name(pose.name)

    try:
        parameters = RobotParameters.from_yaml(path)
    except (OSError, ParameterError) as error:
        raise CaptureRefused(
            f"{path} does not load as it stands, so it will not be edited: {error}"
        ) from None

    identity = parameters.identity
    if pose.calibration_id != identity.calibration_id:
        raise CaptureRefused(
            f"the pose was measured under calibration {pose.calibration_id!r} but {path} "
            f"describes {identity.calibration_id!r}; they are different physical poses"
        )
    if len(pose.joints) != len(identity.joint_names):
        raise CaptureRefused(
            f"the pose has {len(pose.joints)} joint values but this arm has "
            f"{len(identity.joint_names)}"
        )
    if not all(math.isfinite(value) for value in pose.joints):
        raise CaptureRefused("the pose has a joint with no feedback; nothing was written")

    original = path.read_text()
    updated = insert_preset(original, pose.name, render_preset(pose), replace=replace)
    _write_atomically(path, updated)

    try:
        written = RobotParameters.from_yaml(path).named_positions.get(pose.name)
        if written is None:
            raise ParameterError(f"{pose.name!r} is not in the file after writing it")
        drifted = [
            joint
            for joint, before, after in zip(
                identity.joint_names, pose.joints, written.joints, strict=True
            )
            if not math.isclose(before, after, abs_tol=ROUNDTRIP_TOLERANCE_RAD)
        ]
        if drifted:
            raise ParameterError(f"{', '.join(drifted)} did not survive the round trip")
    except (OSError, ParameterError) as error:
        _write_atomically(path, original)
        raise CaptureRefused(
            f"the edit did not read back as written and {path} was restored unchanged: {error}"
        ) from None

    return path
