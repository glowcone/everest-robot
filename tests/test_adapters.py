from everest_robot.adapters import ScaffoldRobot
from everest_robot.domain import CarabinerPickupResult, RecoveryTarget


def test_scaffold_robot_is_deterministic() -> None:
    robot = ScaffoldRobot()
    pickup = robot.localize_and_pick_up_carabiner("deterministic-cv", "graspnet")
    position = robot.go_to_known_position("clip-attachment-ready")
    attachment = robot.attach_clip("vla")
    verification = robot.verify_attachment(attachment)

    assert pickup == CarabinerPickupResult(
        True,
        "robot_base",
        0.42,
        0.0,
        0.18,
        "deterministic-cv",
        "graspnet",
    )
    assert position.reached
    assert attachment.motion_completed
    assert attachment.controller == "vla"
    assert verification.secure


def test_verification_can_request_attachment_recovery() -> None:
    robot = ScaffoldRobot(verification_failures=1)
    attachment = robot.attach_clip("rl-policy")

    first = robot.verify_attachment(attachment)
    second = robot.verify_attachment(attachment)

    assert first.recovery_target is RecoveryTarget.ATTACH
    assert not first.secure
    assert second.secure
