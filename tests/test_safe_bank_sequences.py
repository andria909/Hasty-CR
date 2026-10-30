from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts.attack.build_bank_counterfactuals import opening
from scripts.attack.build_safe_bank_sequences import choose, replay
from scripts.attack.search_late_air_defense import rollout
from scripts.kaggle.prepare_bank_cannon_search import prepare


def test_delayed_cannon_replays_wait_and_actual_legal_deployment():
    env = opening(352, 'left', 3000, 15)
    hp = {k: t.hitpoints for k, t in env.match.towers[1].items()}
    plan = rollout(env.match, hp, env.cards, env.spells, 'left',
                   second=dict(card='cannon', cell=[6, 18], offset_ms=3000))
    assert plan['valid'] and plan['clear_without_damage']
    row = dict(seed=352, lane='left', bank=3000, enemy_y=15)
    states = replay(row, plan)
    assert len(states) == 13
    assert all(s['action'] == 0 for s in states[:-1])
    assert states[-1]['action'] > 0
    assert all(np.all(~s['success_masks'] | s['masks']) for s in states)
    corrupt = deepcopy(plan); corrupt['own_tower_damage']['left'] += 1
    with pytest.raises(ValueError, match='outcome did not reproduce'):
        replay(row, corrupt)


def test_teacher_does_not_prefer_costly_late_or_unsafe_defense():
    def candidate(cost, delay, safe=True):
        return dict(valid=True, clear_without_damage=safe, elixir_spent=cost,
                    delay_ms=delay, commands=[{}], candidate=dict(card='cannon', cell=[6, 18]))
    a, b, c, d = candidate(3, 0), candidate(3, 3000), candidate(4, 9000), candidate(3, 12000, False)
    row = dict(cannon_first_audited=True, trials=[a, b, c, d])
    assert choose(row) is b
    row['cannon_first_audited'] = False
    with pytest.raises(ValueError, match='Cannon-first'):
        choose(row)


@pytest.mark.parametrize('timing', [False, True])
def test_remote_probe_bundles_helper_regardless_of_import_spacing(tmp_path, timing):
    prepare('reports/defense_plan/bank_cannon_single_bootstrap_20261029.py',
            tmp_path/'package', 'fixture-search', timing=timing)
    code = (tmp_path/'package/expanded_probe.py').read_text()
    assert 'from scripts.attack.search_late_air_defense import' not in code
    assert 'def rollout(' in code and 'def rank(' in code
    compile(code, 'probe.py', 'exec')
