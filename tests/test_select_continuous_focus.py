import pytest
from scripts.attack.select_continuous_focus import select_initial,select_profiles
from sim.continuous_validation import FAMILIES


def validation(wins):
    rows=[dict(split='validation',family_id=f,outcome='win'if i<n else'loss')
          for f,n in zip(FAMILIES,wins)for i in range(12)]
    for i,r in enumerate(rows):r.update(episode_seed=900100+i,match_seed=1000900100+i)
    return {'records':rows}


def runs():
    return [dict(trial=dict(id=str(i),entropy_coef=.01,focus='uniform'),seed=seed,target_met=True,step=102400,
                 validation=validation([6]*6))for i in range(4)for seed in(62,63)]


def test_warm_start_prefers_broad_skill_and_parent_on_tie():
    refs=[dict(id='parent',validation=validation([6]*6)),dict(id='specialist',validation=validation([0,12,12,0,12,12]))]
    assert select_initial(refs)['selected']=='parent'
    refs.append(dict(id='new',validation=validation([6]*6)))
    assert select_initial(refs)['selected']=='parent'


def test_two_seeds_and_floor_guard_prevent_one_seed_specialist_promotion():
    r=runs();r[0]['validation']=validation([0,12,12,12,12,12])
    selected=select_profiles(r)
    assert len(selected['selected'])==2 and not next(x for x in selected['ranking']if x['trial']['id']=='0')['eligible']
    r.pop()
    with pytest.raises(ValueError):select_profiles(r)


def test_fewer_than_two_eligible_profiles_does_not_start_a_full_run():
    r=runs()
    for row in r:row['validation']=validation([0]*6)
    assert not select_profiles(r)['gate_met'] and not select_profiles(r)['selected']


def test_replayed_task_seeds_are_not_independent_validation_games():
    refs=[dict(id='parent',validation=validation([6]*6))]
    rows=refs[0]['validation']['records'];rows[-1]['match_seed']=rows[0]['match_seed']
    with pytest.raises(ValueError,match='distinct'):select_initial(refs)
