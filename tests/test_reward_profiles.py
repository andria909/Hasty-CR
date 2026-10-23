from dataclasses import asdict
import json
from pathlib import Path

import pytest

from sim.reward_profiles import load_reward_profile,reward_objects
from sim.defense_env import DefenseRewards
from sim.attack_env import AttackRewards
from sim.strategy_env import StrategyEnv


def test_default_profile_exactly_preserves_existing_coefficients():
    profile,digest=load_reward_profile()
    defense,attack=reward_objects(profile)
    assert asdict(defense)==asdict(DefenseRewards())
    assert asdict(attack)==asdict(AttackRewards())and len(digest)==64
    for path in Path('data/ppo_tuning/profiles').glob('*.json'):
        profile,digest=load_reward_profile(path)
        assert len(digest)==64


@pytest.mark.parametrize('value',[True,'-5',float('nan'),float('inf'),5])
def test_invalid_hp_penalty_is_rejected(tmp_path,value):
    path=tmp_path/'profile.json'
    path.write_text(json.dumps(dict(schema='ppo-reward-profile-v1',id='trial',defense={'own_hp_fraction':value})))
    with pytest.raises(ValueError):load_reward_profile(path)


def test_profiles_reach_both_tasks_and_snapshot_cannot_change_reward_contract():
    baseline,_=load_reward_profile();changed,_=load_reward_profile('data/ppo_tuning/profiles/tower_survival.json')
    defense,attack=reward_objects(changed)
    env=StrategyEnv(defense_rewards=defense,attack_rewards=attack)
    assert env._defense.rewards.own_hp_fraction==-30
    assert env._attack.rewards.own_king_death==-20
    scenario=next(s.scenario_id for s in env.scenarios if s.task=='defense')
    env.reset(seed=42,scenario_id=scenario);snapshot=env.snapshot()
    clone=StrategyEnv(defense_rewards=defense,attack_rewards=attack);clone.restore(snapshot)
    assert clone._defense.rewards==defense
    with pytest.raises(ValueError,match='reward profile'):StrategyEnv().restore(snapshot)
