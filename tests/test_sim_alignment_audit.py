import json
from pathlib import Path

import pytest

from scripts.gamestate.audit_sim_alignment import (
    geometry, simulate_speed, stats, targeting_cases, unit_profiles)


def test_geometry_compares_recorded_native_anchors_without_warping_them():
    recorded = json.loads(Path('data/validation/parser_memory_alignment_20261022.json').read_text())
    rows = geometry(recorded['cases'][0])
    assert len(rows) == 6
    top_left = next(r for r in rows if r['engine_side'] == 0 and r['lane'] == 'left')
    bottom_left = next(r for r in rows if r['engine_side'] == 1 and r['lane'] == 'left')
    assert top_left['real'] == [3.5, 6.5]
    assert top_left['real_minus_sim_tiles'] == [0, -1]
    assert bottom_left['real_minus_sim_tiles'] == [0, 1]


def test_targeting_report_does_not_present_model_predictions_as_real_labels():
    cases = targeting_cases(unit_profiles())
    assert [x['sim_selected'] for x in cases[:3]] == ['musketeer', 'mega_minion', 'cannon']
    assert all(x['real_comparison'].startswith('unverified') for x in cases)
    assert cases[-1]['sim_switched_on_rescan'] is True


@pytest.mark.parametrize('name,speed', [('knight', 1.2), ('musketeer', 1.2),
    ('barbarian', 1.2), ('mega_minion', 1.2), ('wizard', 1.2),
    ('skeleton', 1.8), ('skeleton_dragon', 1.8), ('electro_spirit', 2.4)])
def test_isolated_speed_measurement_is_not_limited_by_arrival_or_attack(name, speed):
    assert simulate_speed(unit_profiles()[name]) == pytest.approx(speed, abs=.01)


def test_audit_percentiles_handle_small_and_empty_samples():
    assert stats([]) is None
    assert stats([1]) == {'n': 1, 'median': 1, 'p05': 1, 'p95': 1, 'min': 1, 'max': 1}
