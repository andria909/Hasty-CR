from collections import Counter

import numpy as np
import pytest

from sim.env import NUM_SCALARS
from sim.strategy_env import StrategyEnv,OBSERVATION_CONTRACT
from sim.strategy_sampler import StrategyScenarioCycle


def test_one_policy_observation_contract_and_task_features():
    env=StrategyEnv()
    for task in ('attack','defense'):
        case=next(s for s in env.scenarios if s.task==task)
        obs,info=env.reset(seed=42,scenario_id=case.scenario_id)
        assert len(obs['scalars'])==NUM_SCALARS+3
        assert obs['scalars'][-1]==(task=='attack')
        assert info['metrics']['task']==task
        assert info['metrics']['observation_contract']==OBSERVATION_CONTRACT
        if task=='defense':assert obs['scalars'][-3:-1].tolist()==[0.,0.]
        assert info['action_mask'][0]
        result=env.step(0)
        assert len(result[0]['scalars'])==NUM_SCALARS+3


def test_joint_cycle_balanced_full_catalog_same_hand_seed_and_exact_resume():
    env=StrategyEnv()
    cycle=StrategyScenarioCycle(env.scenarios,seed=42)
    size=2*max(map(len,cycle.groups.values()))
    rows=[cycle.next()for _ in range(size)]
    counts=Counter(r['scenario_id'].split('::')[0]for r in rows)
    assert counts['attack']==counts['defense']
    assert set(r['scenario_id']for r in rows)=={s.scenario_id for s in env.scenarios}
    assert {r['seed']for r in rows}=={42}
    saved=cycle.state_dict()
    restored=StrategyScenarioCycle(env.scenarios,seed=42);restored.load_state_dict(saved)
    for _ in range(20):assert cycle.next()==restored.next()
    assert cycle.cycle==1
    with pytest.raises(ValueError):StrategyScenarioCycle(StrategyEnv(split='validation').scenarios)


def test_joint_snapshot_restores_both_tasks_and_rejects_changed_timers():
    env=StrategyEnv()
    for task in ('attack','defense'):
        case=next(s for s in env.scenarios if s.task==task)
        env.reset(seed=42,scenario_id=case.scenario_id)
        state=env.snapshot();first=env.step(0)
        env.restore(state);second=env.step(0)
        assert np.array_equal(first[0]['scalars'],second[0]['scalars'])
        assert first[1:4]==second[1:4]
    other=StrategyEnv(deadline_scale=2)
    with pytest.raises(ValueError):other.restore(state)


def test_unknown_or_wrong_split_scenario_cannot_enter_training():
    env=StrategyEnv()
    case=StrategyEnv(split='validation').scenarios[0]
    with pytest.raises(ValueError):env.reset(seed=42,scenario_id=case.scenario_id)
