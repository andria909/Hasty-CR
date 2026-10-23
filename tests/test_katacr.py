"""KataCR's team output must never be interpreted as a troop class."""
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from brain.katacr import KataCRDetector, decode_predictions, unit_name
from audit_board_cv import score


def test_comparison_requires_class_side_iou_and_one_to_one_matching():
    expected = [{"name":"knight","side":"ally","bbox_small":[10,10,30,40]},
                {"name":"cannon","side":"ally","bbox_small":[60,60,90,90]}]
    predicted = [{**expected[0],"confidence":.9}, {**expected[0],"confidence":.8},
                 {**expected[1],"side":"enemy","confidence":.9}]
    assert score(predicted, expected) == {"tp":1,"fp":2,"fn":1}


def test_high_team_probability_alone_is_not_a_detection():
    output = torch.tensor([[[100.], [100.], [20.], [30.], [.01], [.02], [.999]]])
    assert len(decode_predictions(output, {0:"knight",1:"cannon",2:"padding_belong"})) == 0


def test_team_probability_is_preserved_separately_from_class():
    output = torch.tensor([[[100.,200.], [100.,200.], [20.,20.], [30.,30.],
                            [.9,.01], [.01,.8], [.03,.99]]])
    rows = decode_predictions(output, {0:"knight",1:"cannon",2:"padding_belong"})
    assert rows.shape == (2,7)
    assert rows[:,5].tolist() == [0.,1.]
    assert rows[:,6].tolist() == pytest.approx([.03,.99])
    assert rows[0,:4].tolist() == [90.,85.,110.,115.]


def test_decoder_rejects_wrong_channel_layout():
    with pytest.raises(ValueError, match="output shape"):
        decode_predictions(torch.zeros(1,6,1), {0:"knight",1:"cannon",2:"padding_belong"})


@pytest.mark.parametrize("raw", ["bar", "padding_belong", "padding_1", "king-tower",
                                  "fireball", "ice-spirit-evolution-symbol", "void",
                                  "goblin-curse", "giant-snowball-hero", "giant-snowball-evolution"])
def test_ui_spells_and_padding_are_excluded(raw):
    assert unit_name(raw) is None


@pytest.mark.parametrize("raw,expected", [("hog-rider","hog_rider"),
    ("mini-pekka","minipekka"),("skeleton-dragon","skeleton_dragon"),
    ("knight-evolution","knight"),("cannon","cannon")])
def test_canonical_unit_names(raw, expected):
    assert unit_name(raw) == expected


def test_recorded_hog_cannon_and_musketeer_with_published_pair():
    weights = [ROOT / f"tmp/katacr_eval_20261021/detector{i}_v0.7.13.pt" for i in (1,2)]
    if not all(path.exists() for path in weights):
        pytest.skip("published KataCR weights are not installed")
    pytest.importorskip("ultralytics")
    torch.set_num_threads(4)
    detector = KataCRDetector(weights, device="cpu")
    with Image.open(ROOT / "data/validation/board_regression_20261021/10.png") as image:
        rows = detector.detect(image)
    assert {(row["name"],row["side"]) for row in rows} == {
        ("hog_rider","ally"),("musketeer","ally"),("cannon","ally")}
