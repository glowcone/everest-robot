"""The camera viewer's decisions, separated from anything that opens a device.

Everything a window shows -- which ids are worth probing, which id is the stable one, what
the label says, what the skeleton configuration would be -- is a pure function here, so it
is tested without a camera, a window, or an arm.
"""

import pytest

from everest_robot.cameras_cli import (
    CameraView,
    build_parser,
    cameras_env_document,
    candidate_targets,
    elide_middle,
    overlay_lines,
    shape_mismatch,
    stable_ids,
    window_positions,
)


def test_linux_scans_device_nodes_in_numeric_order(tmp_path) -> None:
    for name in ("video0", "video2", "video10", "videoX", "not-a-camera"):
        (tmp_path / name).touch()

    assert candidate_targets(system="Linux", dev_root=tmp_path) == (
        f"{tmp_path}/video0",
        f"{tmp_path}/video2",
        f"{tmp_path}/video10",
        f"{tmp_path}/videoX",
    )


def test_other_platforms_probe_indices() -> None:
    assert candidate_targets(system="Darwin", max_index=3) == ("0", "1", "2")


def test_stable_ids_resolve_by_id_links_and_skip_dangling(tmp_path) -> None:
    dev = tmp_path / "dev"
    by_id = tmp_path / "by-id"
    dev.mkdir()
    by_id.mkdir()
    (dev / "video3").touch()
    (by_id / "usb-Chicony_Camera-video-index0").symlink_to(dev / "video3")
    (by_id / "usb-Gone_Camera-video-index0").symlink_to(dev / "video9")

    assert stable_ids(by_id) == {
        str(dev / "video3"): str(by_id / "usb-Chicony_Camera-video-index0")
    }


def test_no_by_id_directory_is_not_an_error(tmp_path) -> None:
    assert stable_ids(tmp_path / "absent") == {}


def test_a_driver_that_ignored_the_requested_size_is_named() -> None:
    matching = CameraView("0", name="wrist", delivered=(480, 640, 3), configured=(480, 640, 3))
    refused = CameraView("0", name="wrist", delivered=(480, 640, 3), configured=(720, 1280, 3))

    assert shape_mismatch(matching) is None
    assert shape_mismatch(CameraView("0", delivered=(480, 640, 3))) is None
    assert shape_mismatch(refused) == "MISMATCH: configured 1280x720, delivering 640x480"


def test_the_label_carries_the_id_the_operator_has_to_copy() -> None:
    scanned = CameraView(
        "/dev/video3",
        stable_id="/dev/v4l/by-id/usb-Chicony_Camera-video-index0",
        delivered=(480, 640, 3),
    )

    lines = overlay_lines(scanned, measured_fps=29.7)

    assert lines[0] == "id /dev/video3"
    assert lines[1] == "stable id  /dev/v4l/by-id/usb-Chicony_Camera-video-index0"
    assert lines[2] == "640x480   30 fps"


def test_a_configured_camera_is_labelled_by_name_and_flags_its_problems() -> None:
    view = CameraView(
        "/dev/video3",
        name="wrist",
        delivered=(480, 640, 3),
        configured=(720, 1280, 3),
        problem="opened but produced no frame",
    )

    lines = overlay_lines(view)

    assert lines[0] == "wrist   id /dev/video3"
    assert "MISMATCH" in lines[-2]
    assert lines[-1] == "opened but produced no frame"


def test_the_skeleton_recommends_the_stable_id_and_leaves_naming_to_the_operator() -> None:
    views = [
        CameraView(
            "/dev/video3",
            stable_id="/dev/v4l/by-id/usb-Chicony_Camera-video-index0",
            delivered=(480, 640, 3),
            driver_fps=29.97,
        ),
        CameraView("/dev/video5", delivered=(720, 1280, 3)),
    ]

    assert cameras_env_document(views) == [
        {
            "name": "camera0",
            "kind": "opencv",
            "index_or_path": "/dev/v4l/by-id/usb-Chicony_Camera-video-index0",
            "width": 640,
            "height": 480,
            "fps": 30,
        },
        {
            "name": "camera1",
            "kind": "opencv",
            "index_or_path": "/dev/video5",
            "width": 1280,
            "height": 720,
            "fps": 30,
        },
    ]


def test_the_skeleton_is_a_valid_camera_configuration() -> None:
    import json

    from everest_robot.robot.cameras import load_camera_specs

    document = cameras_env_document([CameraView("2", delivered=(480, 640, 3), driver_fps=30.0)])
    (spec,) = load_camera_specs(json.dumps(document))

    assert (spec.index_or_path, spec.frame_shape) == ("2", (480, 640, 3))


def test_windows_are_laid_out_in_a_grid_rather_than_stacked() -> None:
    positions = window_positions(4, width=480, height=360, screen_width=1100, margin=20)

    assert positions == ((0, 0), (500, 0), (0, 412), (500, 412))
    assert len(set(positions)) == 4


def test_one_camera_wider_than_the_screen_still_gets_a_position() -> None:
    assert window_positions(2, width=4000, height=3000, screen_width=1100) == ((0, 0), (0, 3056))


def test_the_cli_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_scan_and_show_reach_their_handlers() -> None:
    scan = build_parser().parse_args(["scan", "--only", "0", "2", "--json", "--no-window"])
    show = build_parser().parse_args(["show", "--camera", "wrist"])

    assert (scan.command, scan.only, scan.json, scan.no_window) == ("scan", ["0", "2"], True, True)
    assert scan.handler.__name__ == "scan"
    assert (show.command, show.camera, show.color) == ("show", ["wrist"], "rgb")
    assert show.handler.__name__ == "show"


def test_a_long_by_id_name_keeps_both_ends() -> None:
    name = "/dev/v4l/by-id/usb-Chicony_Electronics_Integrated_Camera-video-index1"

    elided = elide_middle(name, 40)

    assert len(elided) == 40
    assert elided.startswith("/dev/v4l/by-id/usb-")
    # index1 is what separates a camera's video node from its metadata node.
    assert elided.endswith("index1")
    assert elide_middle(name, len(name)) == name
    assert elide_middle("short", 40) == "short"
