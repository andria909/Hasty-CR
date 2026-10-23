import copy
import json
from pathlib import Path

import pytest

from sim.ppo_tuning import quality


def data(joint=False):
    rows=[dict(split='validation',task='defense',difficulty=d,no_tower_damage=True,
        own_towers_alive=dict(left=True,right=True,king=True),own_tower_fraction_loss=dict(left=0,right=0,king=0),
        elixir_spent=5,terminal_reason='cleared')for d in ('simple','medium','complex','impossible')]
    if joint:rows += [dict(split='validation',task='attack',goal_kind=goal,goal_success=True,elixir_spent=4,terminal_reason='goal_success')for goal in ('hog_tower_damage','spell_finish')]
    games=[dict(seed=seed,candidate_side=side,outcome='win')for seed in range(970001,970006)for side in (1,-1)]
    return rows,games


def test_ranking_uses_common_quality_not_profile_reward_scale():
    a,games=data();b=copy.deepcopy(a)
    for row in a:row['reward']=1e9
    for row in b:row['reward']=-1e9
    assert quality(a,games,'defense')==quality(b,games,'defense')
    assert quality(a,games,'defense')['score']==1


def test_joint_score_requires_both_attack_goals_and_defense():
    rows,games=data(True);perfect=quality(rows,games,'joint')
    rows[-1]['goal_success']=False
    weaker=quality(rows,games,'joint')
    assert weaker['score']<perfect['score']and weaker['attack_quality']==.5
    with pytest.raises(ValueError,match='stratum'):quality(rows[:-1],games,'joint')


def test_test_or_duplicate_validation_games_cannot_select_coefficients():
    rows,games=data();rows[0]['split']='test'
    with pytest.raises(ValueError,match='forbids'):quality(rows,games,'defense')
    rows[0]['split']='validation';games[-1]=games[0]
    with pytest.raises(ValueError,match='schedule'):quality(rows,games,'defense')


def test_unknown_outcomes_or_invalid_episodes_are_not_ranked_as_success():
    rows,games=data();games[0]['outcome']='unknown'
    assert quality(rows,games,'defense')['eligible']is False
    games[0]['outcome']='win';rows[0]['terminal_reason']='timeout'
    assert quality(rows,games,'defense')['eligible']is False


def test_short_budget_and_materialized_contexts_are_fixed():
    protocol=json.loads(Path('data/ppo_tuning/protocol.json').read_text())
    assert protocol['full_environment_steps']==20*protocol['short_environment_steps']
    assert protocol['short_updates']*protocol['envs']*protocol['rollout']==102400
    contexts=json.loads(Path('data/ppo_tuning/validation_contexts.json').read_text())
    assert contexts['split']=='validation'
    assert all(len(v)==128 and len({(r['scenario_id'],r['seed'])for r in v})==128 for v in contexts['contexts'].values())
