"""Regressions for real board omissions, UI ghosts and competing classes."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from live_perception import (BoardUnitDetector, configure_unit_detection,
                             health_bar_side, merge_board_predictions)


def test_badge_and_border_scenery_are_not_units():
    rows = [[174, 484, 192, 501, .7, 55],  # actual false king-level lava pup
            [354, 122, 368, 164, .8, 1],  # actual right-border decoration
            [0, 149, 15, 200, .8, 2],     # actual left-border decoration
            [84, 291, 125, 329, .85, 53]]
    result = merge_board_predictions(rows, 368, 652)
    assert result.shape == (1, 6)
    assert result[0, 5] == 53


def test_competing_classes_for_identical_box_count_once():
    result = merge_board_predictions([[109,342,151,395,.6,53],
                                     [109,342,151,395,.4,30]], 368, 652)
    assert len(result) == 1 and result[0, 5] == 53


def test_nearby_overlapping_troops_are_preserved():
    result = merge_board_predictions([[74,260,108,313,.7,53],
                                     [84,291,125,329,.85,53]], 368, 652)
    assert len(result) == 2


def test_heavily_overlapping_swarm_units_are_preserved():
    result = merge_board_predictions([[150,300,170,330,.8,84],
                                     [154,302,174,332,.7,84]], 368, 652)
    assert len(result) == 2


def test_refined_building_replaces_weaker_wrong_building_label():
    result = merge_board_predictions([[153,314,197,369,.33,36],
                                     [158,327,192,363,.82,13]], 368, 652, {13,36})
    assert len(result) == 1 and result[0, 5] == 13


@pytest.mark.parametrize("color,side", [((80,190,245), "ally"), ((245,40,80), "enemy")])
def test_health_bar_side_requires_a_horizontal_bar(color, side):
    image = Image.new("RGB", (368,652), (100,100,100))
    draw = ImageDraw.Draw(image)
    draw.rectangle((155,315,190,319), fill=color)
    assert health_bar_side(image, (154,333,196,382)) == side


def test_spell_blob_is_not_a_health_bar():
    image = Image.new("RGB", (368,652), (100,100,100))
    ImageDraw.Draw(image).ellipse((150,305,190,345), fill=(245,40,80))
    assert health_bar_side(image, (154,333,196,382)) is None


def test_wrapping_is_idempotent_and_yolo_still_skips_onnx():
    detector = SimpleNamespace(unit_detector=SimpleNamespace(_infer=lambda x: x))
    configure_unit_detection(detector, None)
    wrapper = detector.unit_detector
    assert isinstance(wrapper, BoardUnitDetector)
    configure_unit_detection(detector, None)
    assert detector.unit_detector is wrapper
    configure_unit_detection(detector, SimpleNamespace(ready=True))
    assert detector.unit_detector.run(None) == ([], [])


@pytest.fixture(scope="module")
def board_detector():
    model = ROOT / "vendor/ClashRoyaleBuildABot/clashroyalebuildabot/models/units_M_480x352.onnx"
    if not model.exists():
        pytest.skip("vendored ONNX model is unavailable")
    pytest.importorskip("onnxruntime")
    import cr_bot as live
    return BoardUnitDetector(live.Detector(live.DECK).unit_detector)


@pytest.mark.parametrize("frame_id", [3,6,10,11])
def test_recorded_blue_cannon_is_recognised_as_ours(board_detector, frame_id):
    with Image.open(ROOT / f"data/validation/board_regression_20261021/{frame_id:02d}.png") as image:
        allies, enemies = board_detector.run(image)
    assert sum(d.unit.name == "cannon" for d in allies) == 1
    assert not any(d.unit.name in {"cannon", "goblin_cage"} for d in enemies)


def test_recorded_empty_arena_stays_empty(board_detector):
    with Image.open(ROOT / "data/validation/board_regression_20261021/02.png") as image:
        assert board_detector.run(image) == ([], [])
