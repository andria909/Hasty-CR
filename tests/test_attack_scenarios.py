"""Attack curriculum contracts only; no training or large simulation runs."""
from collections import defaultdict
from dataclasses import replace

import pytest

from sim.attack_scenarios import (AttackScenario, AttackScheduledPlay, TowerHPSetup,
    UnitSetup, build_catalog, sample_episode_context, select_scenarios)
from sim.spells import SpellSpec


@pytest.fixture(scope='module')
def catalog():
    spells={name:SpellSpec(name,damage,1000,1000,pct,0)
            for name,damage,pct in [('fireball',1000,23),('zap',300,17)]}
    return build_catalog(spells=spells)


def test_identity_serialization_and_family_split_isolation(catalog):
    rows,report=catalog
    assert len(rows)==len({s.scenario_id for s in rows})==report['count']
    families=defaultdict(set)
    for s in rows:
        assert AttackScenario.from_dict(s.to_dict())==s
        families[s.family_id].add(s.split)
    assert all(len(splits)==1 for splits in families.values())
    assert {s.lane for s in rows}=={'left','right'}
    assert all({s.deadline_visibility for s in rows if s.paired_case_id==case}=={'hidden','visible'}
               for case in {s.paired_case_id for s in rows})
    assert rows==build_catalog(spells={name:SpellSpec(name,d,1000,1000,p,0)
        for name,d,p in [('fireball',1000,23),('zap',300,17)]})[0]


def test_crown_damage_boundaries_are_not_full_spell_damage(catalog):
    rows,report=catalog
    assert report['crown_damage_level11']=={'fireball':230,'zap':51}
    for s in rows:
        if s.goal_kind!='spell_finish':continue
        hp=next(t.hp for t in s.tower_hp if t.side==-1 and t.lane==s.target_lane)
        delta={'below':-1,'equal':0,'above':1}[s.finisher_hp_relation]
        assert hp==report['crown_damage_level11'][s.finisher_spell]+delta
        assert s.single_cast_sufficient==(delta<=0)
        assert {'fireball','zap'}<=set(s.initial_hand)
        assert s.initial_elixir in (6,10)  # above boundary remains solvable by a combo
        assert len(s.spell_profile_sha256)==64
    invalid=next(s for s in rows if s.goal_kind=='spell_finish')
    with pytest.raises(ValueError,match='labels disagree'):
        replace(invalid,single_cast_sufficient=not invalid.single_cast_sufficient)


def test_hog_setup_target_and_reactive_building_contract(catalog):
    rows,_=catalog
    solo=[s for s in rows if s.family_id=='hog:solo:unopposed']
    assert {s.setup_kind for s in solo}=={'bridge','back','pocket'}
    assert any(s.tower_hp[0].hp==s.tower_hp[1].hp for s in solo)
    assert all(s.target_lane=='auto' for s in solo)
    assert all(any(t.hp==0 for t in s.tower_hp) for s in solo if s.setup_kind=='pocket')
    assert {'musketeer','giant','knight','skeletons','electro_spirit'}<=set(
        u.card for s in rows for u in s.initial_units)
    for building in ('cannon','tesla','bomb_tower'):
        subset=[s for s in rows if s.family_id=='hog:building:'+building]
        assert {p.position_kind for s in subset for p in s.enemy_plays}=={'fixed_guess','reactive_visible'}
        for s in subset:
            p=s.enemy_plays[0]
            assert p.y==9000  # explicit enemy-half placement, not visible Hog's body
            assert (p.trigger_card=='hog_rider')==(p.position_kind=='reactive_visible')


def test_private_timer_draws_pair_variants_and_keep_independent_visibility(catalog):
    rows,_=catalog
    first=rows[0]
    pair=next(s for s in rows if s.paired_case_id==first.paired_case_id and s!=first)
    draws=[]
    for seed in range(20):
        a,b=sample_episode_context(first,seed),sample_episode_context(pair,seed)
        assert a['deadline_ms']==b['deadline_ms']
        assert a['hand_seed']==b['hand_seed']==seed
        assert a['deadline_visible']!=b['deadline_visible']
        assert a==sample_episode_context(first,seed)
        draws.append(a['deadline_ms'])
    assert len(set(draws))>1
    mixed=replace(first,deadline_visibility='mixed')
    assert not sample_episode_context(mixed,1,visible_probability=0)['deadline_visible']
    assert sample_episode_context(mixed,1,visible_probability=1)['deadline_visible']
    assert sample_episode_context(replace(first,hand_seed=81),2)['hand_seed']==81


def test_setup_validation_fail_closed(catalog):
    base=catalog[0][0]
    for changes in ({'deadline_ms_range':(0,1000)},{'deadline_ms_range':(1000,1000)},
                    {'initial_hand':('zap',)*4},{'geometry':'unknown'},
                    {'tower_hp':(TowerHPSetup(-1,'left',1),TowerHPSetup(-1,'left',2))}):
        with pytest.raises(ValueError):replace(base,**changes)
    with pytest.raises(ValueError):TowerHPSetup(0,'left',1)
    with pytest.raises(ValueError):UnitSetup('hog_rider',1,-1,4000)
    with pytest.raises(ValueError):AttackScheduledPlay(0,'cannon',6000,9000,'reactive_visible')
    with pytest.raises(ValueError):select_scenarios('invalid')


def test_coverage_report_is_not_quality_acceptance(catalog):
    rows,report=catalog
    assert not report['calibrated'] and not report['trained'] and not report['quality_accepted']
    assert report['limitations']
    assert any(s.enemy_plays for s in rows if s.goal_kind=='spell_finish')
    kings=[s for s in rows if s.goal_kind=='spell_finish'and s.target_lane=='king']
    assert kings and all(any(t.side==-1 and t.lane in ('left','right')and t.hp==0 for t in s.tower_hp)for s in kings)
    for split,counts in report['goal_split_coverage'].items():
        assert sum(counts.values())==sum(s.split==split for s in rows)
    # Never scatter HP/lane/visibility siblings across splits to manufacture coverage.
    assert all(count>0 for goals in report['goal_split_coverage'].values()for count in goals.values())
    assert report['full_goal_split_coverage'] and not report['missing_goal_split_coverage']


def test_king_finish_and_counterspell_are_substantive_whole_families(catalog):
    rows,_=catalog
    for spell in ('fireball','zap'):
        kings=[s for s in rows if s.family_id==f'finisher:{spell}:king']
        races=[s for s in rows if s.family_id==f'finisher:{spell}:counterspell']
        assert kings and races
        assert len({s.split for s in kings})==len({s.split for s in races})==1
        for s in kings:
            assert s.target_lane=='king'
            assert any(t.side==-1 and t.lane==s.lane and t.hp==0 for t in s.tower_hp)
            assert any(t.side==-1 and t.lane=='king'and t.hp>0 for t in s.tower_hp)
        for s in races:
            assert any(t.side==1 and t.lane==s.lane and t.hp>0 for t in s.tower_hp)
            assert s.enemy_plays[0].position_kind=='counterspell'
            assert s.enemy_plays[0].card=='fireball'and s.enemy_plays[0].at_ms>0
