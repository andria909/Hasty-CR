import json

import pytest

from sim.defense_sampler import ScenarioCycle


def test_cycle_covers_every_attack_once_with_one_hand_seed_then_changes():
    ids = tuple(f'attack-{i}' for i in range(7))
    sampler = ScenarioCycle(ids, seed=42)
    cycle = [sampler.next() for _ in ids]
    assert {r['scenario_id'] for r in cycle} == set(ids)
    assert {r['seed'] for r in cycle} == {42}
    following = [sampler.next() for _ in ids]
    assert {r['scenario_id'] for r in following} == set(ids)
    assert {r['seed'] for r in following} == {43}


def test_algorithms_receive_identical_states_and_sampler_has_no_global_rng():
    import random
    ids=('one','two','three')
    ppo, grpo = ScenarioCycle(ids, seed=2), ScenarioCycle(ids, seed=2)
    for _ in range(19):
        random.random()
        assert ppo.next() == grpo.next()


def test_json_state_resume_is_exact_including_cycle_boundary():
    ids=('one','two','three')
    for prefix in (0,1,3,4,18):
        original=ScenarioCycle(ids,seed=7)
        for _ in range(prefix):
            original.next()
        restored=ScenarioCycle(ids,seed=7)
        restored.load_state_dict(json.loads(json.dumps(original.state_dict())))
        assert [original.next() for _ in range(11)] == [restored.next() for _ in range(11)]


def test_rejects_incompatible_or_corrupt_sampler_state():
    sampler=ScenarioCycle(('one','two'))
    state=sampler.state_dict()
    for changed in ({'seed':9}, {'scenario_ids':['two','one']}, {'offset':3}, {'cycle':-1}):
        with pytest.raises(ValueError):
            sampler.load_state_dict({**state, **changed})
    with pytest.raises(ValueError):
        ScenarioCycle(('one','one'))


def test_shared_environment_resets_use_same_real_hand_within_pass():
    from sim.defense_env import DefenseEnv
    env=DefenseEnv(seed=42)
    ids=tuple(s.scenario_id for s in env.scenarios[:3])
    sampler=ScenarioCycle(ids,seed=42)
    hands=[]
    for _ in ids:
        env.reset(**sampler.next())
        hands.append((tuple(env.match.players[1].hand),tuple(env.match.players[1].queue)))
    assert len(set(hands))==1
