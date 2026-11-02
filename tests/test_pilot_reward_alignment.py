from copy import deepcopy

import pytest

from scripts.attack.audit_pilot_reward_alignment import components,paired
from tests.test_paired_ppo_behavior import records


def config():
    return dict(continuous_profile=dict(enemy_hp_reward=3.,own_hp_penalty=6.,overflow_penalty=.3,
        missed_fireball_penalty=0.,fireball_value_reward=.25,win_reward=20.,loss_penalty=20.),
        tower_support_weight=.15,spell_waste_weight=.1,crown_only_weight=0.,dormant_king_weight=.4)


def measured_records(c):
    rows=records()
    for r in rows:
        r.update(enemy_fraction_damage=0.,illegal_actions=0,missed_fireball_elixir=0.,board_shaping_return=0.,
            reserve_shaping_return=0.,fireball_exchange_value=0.,tower_support_paid_value=0.,
            dormant_at_request_king_only_elixir=0.,nonlethal_crown_only_elixir=0.,
            **{k:c[k]for k in ('tower_support_weight','spell_waste_weight','crown_only_weight','dormant_king_weight')})
        r['episode_return']=-20.
    return rows


def recompute(row,c):
    row['episode_return']=sum(components(row,c,rescore=True).values())


def test_paid_bins_are_disjoint_and_return_includes_all_wrappers():
    c=config();r=measured_records(c)[0]
    r.update(empty_spell_elixir=4.,nonlethal_crown_only_elixir=6.,dormant_at_request_king_only_elixir=4.,
        tower_support_paid_value=10.,own_fraction_loss=.5,enemy_fraction_damage=.2)
    recompute(r,c);terms=components(r,c)
    assert terms['crown_spell']==pytest.approx(-1.6)
    assert terms['empty_spell']==pytest.approx(-.4)
    assert terms['tower']==pytest.approx(1.5)
    assert r['episode_return']==pytest.approx(-22.9)
    r['episode_return']+=.2
    with pytest.raises(ValueError,match='Return does not match'):components(r,c)


def test_rescoring_prevents_false_outcome_inversion_across_reward_coefficients():
    old=config();new=dict(old,spell_waste_weight=1.)
    before=measured_records(old);after=measured_records(new)
    before[0]['empty_spell_elixir']=after[0]['empty_spell_elixir']=50.
    after[0]['outcome']='win';recompute(before[0],old);recompute(after[0],new)
    assert after[0]['episode_return']<before[0]['episode_return']
    result=paired(before,after,new)
    assert result['changed_outcomes']==1 and not result['outcome_inversions']
    before[0]['empty_spell_elixir']=0.;recompute(before[0],old)
    result=paired(before,after,new)
    assert len(result['outcome_inversions'])==1
    assert result['outcome_inversions'][0]['return_delta']==pytest.approx(-10.)


def test_same_outcome_four_metric_flag_retains_duration_and_term_explanation():
    c=config();before=measured_records(c);after=deepcopy(before)
    after[0].update(own_fraction_loss=.1,tower_support_paid_value=10.,elapsed_ms=120000)
    recompute(after[0],c);result=paired(before,after,c)
    assert not result['outcome_inversions']
    assert result['same_outcome_hp_tradeoff_count']==1
    item=result['higher_return_worse_on_four_metrics'][0]
    assert item['higher_return_policy']=='after' and not item['same_duration']
    assert item['component_deltas']['tower']==pytest.approx(1.5)
    assert item['component_deltas']['own_damage']==pytest.approx(-.6)


@pytest.mark.parametrize('corruption',['baseline_return','coefficient','bins','nonfinite'])
def test_invalid_accounting_cannot_enter_pair_comparison(corruption):
    c=config();before=measured_records(c);after=deepcopy(before)
    if corruption=='baseline_return':before[0]['episode_return']+=1
    if corruption=='coefficient':after[0]['spell_waste_weight']=1.
    if corruption=='bins':after[0]['dormant_at_request_king_only_elixir']=1.
    if corruption=='nonfinite':after[0]['fireball_exchange_value']=float('nan')
    with pytest.raises(ValueError):paired(before,after,c)
