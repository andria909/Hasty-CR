import copy
import json
from pathlib import Path

import pytest

from scripts.gamestate.replay_recorded_episode import extract, replay

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT/'data/recorded_mechanics_20261028'


def native_row(tick, bodies, world='world-1'):
    return dict(status='ok', own_state=dict(status='ok', tick=tick, world=world),
                frame=dict(battle_tick=tick, entities=bodies))


@pytest.mark.parametrize('corruption', ['clock', 'reuse', 'gap', 'duplicate'])
def test_extraction_rejects_invalid_time_and_identity(tmp_path, corruption):
    original = json.loads((CLIPS/'hog_cannon_right_pull.json').read_text())
    bodies = original['frames'][0]['bodies']
    uid = original['focal_identity'][0]
    rows = [native_row(t, copy.deepcopy(bodies)) for t in (100, 101, 102, 103)]
    if corruption == 'clock':
        rows[1]['own_state']['tick'] = 105
    elif corruption == 'reuse':
        next(b for b in rows[1]['frame']['entities'] if b['entity_id'] == uid)['address'] = '0xnew'
    elif corruption == 'gap':
        rows[1] = native_row(110, copy.deepcopy(bodies))
    else:
        rows[1] = native_row(100, copy.deepcopy(bodies))
        next(b for b in rows[1]['frame']['entities'] if b['entity_id'] == uid)['x'] += 1
    path = tmp_path/'frames.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):
        extract(path, 0, 112 if corruption == 'gap' else 103, uid, 'invalid')


def test_portable_episode_replay_preserves_input_and_reports_conditioning():
    clip = json.loads((CLIPS/'musketeer_cannon_then_tower.json').read_text())
    original = copy.deepcopy(clip)
    result = replay(clip)
    assert clip == original
    assert result['rows'][0]['error_tiles'] == 0
    assert len(result['rows']) == len(clip['frames'])
    assert result['simulated_shots']
    assert not result['assumptions']['native_target_pointer_available']
    assert result['ignored_context_observations']['MusketeerProjectile'] > 0
    assert result == replay(clip)
