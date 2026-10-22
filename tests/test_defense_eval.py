import pytest

from sim.eval_defense import aggregate, describe


def record(**changes):
    result=dict(terminal_reason='cleared',own_tower_hp_loss={'left':0,'right':0,'king':0},
        own_tower_fraction_loss={'left':0.,'right':0.,'king':0.},
        own_towers_alive={'left':True,'right':True,'king':True},
        enemy_tower_hp_damage={'left':100,'right':0,'king':0},
        elixir_spent=5.,spell_quality=1.,spell_misses=0,illegal_actions=0,elapsed_ms=1000,
        scenario_id='s1',family_id='f1',difficulty='simple',split='validation',lane='left',
        no_tower_damage=True,shuffle_seed=1,unavoidable_tower_damage=False,
        coverage_flags=['approximate_simulation'])
    result.update(changes)
    return result


def test_summary_metrics_and_stratification():
    first=record()
    second=record(scenario_id='s2',lane='right',shuffle_seed=2,difficulty='hard',
        own_tower_hp_loss={'left':100,'right':0,'king':0},
        own_towers_alive={'left':False,'right':True,'king':True},
        no_tower_damage=False,terminal_reason='timeout',elixir_spent=9.)
    result=aggregate([first,second],expected_scenarios=['s1','s2'],expected_lanes=['left','right'])
    assert result['overall']['no_damage']['rate']==.5
    assert result['overall']['timeout_rate']==.5
    assert result['overall']['own_tower_hp_loss']['left']['mean']==50
    assert result['overall']['elixir_spent']['mean']==7
    assert result['overall']['tower_survival']['left']['rate']==.5
    assert result['by']['shuffle_seed']['1']['episodes']==1
    assert result['coverage']['all_expected_scenarios_seen'] is True
    assert result['coverage']['all_expected_lanes_seen'] is True
    assert result['coverage']['environment_flags']==['approximate_simulation']
    assert result['gates']['overall']['passed'] is None


def test_missing_labels_and_unavoidable_spells_cannot_claim_simple_pass():
    unlabeled=record()
    del unlabeled['unavoidable_tower_damage']
    assert aggregate([unlabeled])['gates']['simple_no_damage']['passed'] is None
    unavoidable=record(unavoidable_tower_damage=True,no_tower_damage=False)
    assert aggregate([unavoidable])['gates']['simple_no_damage']['passed'] is None
    assert aggregate([record()])['gates']['simple_no_damage']['passed'] is True
    assert aggregate([record(no_tower_damage=False)])['gates']['simple_no_damage']['passed'] is False


def test_coverage_missing_values_are_unknown_not_success():
    result=aggregate([{'difficulty':'stress'}],expected_scenarios=['missing'])
    assert result['coverage']['metrics_complete'] is False
    assert result['coverage']['all_expected_scenarios_seen'] is False
    assert result['coverage']['missing_scenarios']==['missing']
    assert result['overall']['tower_survival']['king']['rate'] is None
    assert result['overall']['no_damage']['rate'] is None
    assert result['gates']['medium_hp_tolerance']['passed'] is None
    assert result['coverage']['snapshot_parity_verified'] is False


def test_empty_stats_and_quantiles_are_serializable():
    import json
    json.dumps(aggregate([]),allow_nan=False)
    values=describe([0,1,2,None])
    assert values['mean']==1
    assert values['p95']==pytest.approx(1.9)
    assert describe([])['mean'] is None


def test_invalid_script_truncation_is_not_a_gameplay_timeout():
    result=aggregate([record(terminal_reason='invalid_script_trigger_timeout',truncated=True)])
    assert result['overall']['timeout_rate']==0
    assert result['overall']['invalid_script_rate']==1
    assert result['overall']['truncated_rate']==1
