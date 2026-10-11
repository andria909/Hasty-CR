"""The researched playbook and extracted replay corpus stay executable."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.env import DECK_26, NUM_PLANES, tactical_plane_values  # noqa: E402
from sim.replay_bc import ADAPT_CARD  # noqa: E402


def _json(name: str):
    return json.loads((ROOT / "data" / "strategy" / name).read_text())


def test_playbook_targets_the_actual_live_deck_and_replacements():
    book = _json("hog_cycle_playbook_v1.json")
    assert book["live_deck"] == DECK_26
    replacements = {classic: row["live_card"]
                    for classic, row in book["adaptation"].items()}
    assert replacements == ADAPT_CARD
    assert all(row["preserve"] and row["do_not_assume"]
               for row in book["adaptation"].values())


def test_playbook_places_fast_and_slow_cannons_centrally():
    rules = {row["id"]: row for row in _json(
        "hog_cycle_playbook_v1.json")["rules"]}
    fast = rules["central_fast_pull"]["placement"]
    slow = rules["deep_slow_pull"]["placement"]
    assert (fast["left_x"], fast["right_x"], fast["y"]) == (8, 9, 20)
    assert (slow["left_x"], slow["right_x"], slow["y"]) == (8, 9, 21)


def test_expert_corpus_supports_central_cannon_and_bridge_hog():
    summary = _json("hog26_expert_v1_summary.json")
    assert summary["hog_sides"] >= 300
    assert summary["actions"] >= 10_000
    cannon = summary["placement"]["cannon"]
    hog = summary["placement"]["hog_rider"]
    assert 8.0 <= cannon["x"]["median"] <= 10.0
    assert 19.0 <= cannon["y"]["median"] <= 23.0
    assert 16.0 <= hog["y"]["median"] <= 19.0


def test_tactical_observation_planes_have_bounded_normalisation():
    values = tactical_plane_values(
        damage=100_000, hit_speed_ms=1, range_mt=100_000,
        speed_mt_per_sec=100_000, attacks_air=True,
        target_only_buildings=True, splash_radius_mt=100_000)
    assert NUM_PLANES == 20
    assert values == pytest.approx((3.0, 2.0, 2.0, 1.0, 1.0, 2.0))
