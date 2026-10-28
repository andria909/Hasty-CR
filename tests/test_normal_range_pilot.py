from copy import deepcopy
import json

import pytest

from scripts.kaggle.prepare_normal_range_pilot import protocol_from, validate_protocol
from scripts.attack.verify_normal_range_pilot import audit_geometry
from scripts.attack.evaluation_execution import CORRECTED, combat_modes, environment_kwargs
from tests.test_tactical_starts_pilot import inputs


def test_corrected_grid_keeps_rewards_and_start_fractions_fixed():
    old, proof = inputs()
    p = protocol_from(old, proof)
    assert [r['tactical_start_fraction'] for r in p['cases']] == [0, .1, .25, .5]
    assert all(r['tower_support_weight'] == .15 for r in p['cases'])
    assert p['constants']['source_radius_reach'] == 1
    for change in ({'source_radius_reach': 0}, {'target_lock_in_range': 1},
                   {'tower_support_weight': .4}):
        wrong = deepcopy(p)
        wrong['cases'][0].update(change)
        with pytest.raises(ValueError):
            validate_protocol(wrong)


def test_geometry_audit_requires_actual_episode_modes():
    config = environment_kwargs(CORRECTED)
    row = combat_modes(CORRECTED)
    audit_geometry(config, [row])
    for rows in ([], [{}], [dict(row, normal_range_mode='legacy_target_edge_v1')]):
        with pytest.raises(ValueError):
            audit_geometry(config, rows)
    config['continuous_profile']['source_radius_reach'] = 0
    with pytest.raises(ValueError):
        audit_geometry(config, [row])


def test_summary_calls_geometry_verifier(tmp_path, monkeypatch):
    from scripts.attack.summarize_clock_pilots import summarize
    import scripts.attack.verify_normal_range_pilot as verifier
    (tmp_path / 'dataset').mkdir()
    (tmp_path / 'dataset/protocol.json').write_text(json.dumps(
        {'study_variant': 'edge_to_edge_paid_tactical_v1'}))
    def rejected(*args):
        raise ValueError('actual geometry rejected')
    monkeypatch.setattr(verifier, 'verify_normal_range', rejected)
    with pytest.raises(ValueError, match='actual geometry rejected'):
        summarize(tmp_path)
