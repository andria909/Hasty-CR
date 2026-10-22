from types import SimpleNamespace as NS

import pytest

from sim.attack_sampler import AttackScenarioCycle


def cases():
    return [NS(scenario_id=f'{kind}:{i}',split='train',goal_kind=kind)
            for kind in ('hog_tower_damage','spell_finish')for i in range(4)]


def test_full_cycle_no_repeats_mixed_types_one_seed_then_new_seed():
    source=cases()
    sampler=AttackScenarioCycle(source,seed=42)
    first=[sampler.next()for _ in source]
    second=[sampler.next()for _ in source]
    assert {r['scenario_id']for r in first}=={s.scenario_id for s in source}
    assert len({r['scenario_id']for r in first})==len(source)
    assert {r['seed']for r in first}=={42}
    assert {r['seed']for r in second}=={43}
    # A random pass can coincidentally group types; do not enforce a hidden
    # curriculum or mutate the shared paired shuffle to force alternation.
    orders=[first,second]
    orders += [[sampler.next()for _ in source]for _ in range(3)]
    assert any(sum(a['scenario_id'].split(':')[0]!=b['scenario_id'].split(':')[0]
                   for a,b in zip(order,order[1:]))>1 for order in orders)
    assert [r['scenario_id']for r in first]!=[s.scenario_id for s in source]


def test_paired_sequences_and_exact_cursor_resume():
    a,b=AttackScenarioCycle(cases()),AttackScenarioCycle(cases())
    for _ in range(5):assert a.next()==b.next()
    state=a.state_dict()
    c=AttackScenarioCycle(cases())
    c.load_state_dict(state)
    for _ in range(20):assert a.next()==b.next()==c.next()


def test_train_split_and_hand_mode_and_catalog_contracts_fail_closed():
    source=cases()
    source[0].split='validation'
    with pytest.raises(ValueError,match='train'):AttackScenarioCycle(source)
    state=AttackScenarioCycle(cases()).state_dict()
    other=AttackScenarioCycle(cases(),initial_hand_mode='scenario')
    with pytest.raises(ValueError,match='contract'):other.load_state_dict(state)
    changed=cases();changed[0].goal_kind='changed'
    with pytest.raises(ValueError,match='contract'):
        AttackScenarioCycle(changed).load_state_dict(state)
