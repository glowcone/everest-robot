from everest_robot.adapters import ScaffoldRobot
from everest_robot.domain import RecoveryTarget, RopePose


def test_scaffold_robot_is_deterministic() -> None:
    robot = ScaffoldRobot()
    pickup = robot.pick_up_carabiner("rl-policy")
    rope = robot.locate_rope("deterministic-cv")
    attachment = robot.attach_carabiner(rope)
    verification = robot.verify_attachment(attachment)

    assert pickup.secured
    assert rope == RopePose("robot_base", 0.42, 0.0, 0.18, "deterministic-cv")
    assert attachment.motion_completed
    assert verification.secure


def test_verification_can_request_attachment_recovery() -> None:
    robot = ScaffoldRobot(verification_failures=1)
    rope = robot.locate_rope("vlm")
    attachment = robot.attach_carabiner(rope)

    first = robot.verify_attachment(attachment)
    second = robot.verify_attachment(attachment)

    assert first.recovery_target is RecoveryTarget.ATTACH
    assert not first.secure
    assert second.secure

