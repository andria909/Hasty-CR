import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from sim.defense_protocol import (EvaluationProtocol, V25_PATH, V25_SHA256,
                                 acceptance_gate, load_baseline_allowlist,
                                 paired_schedule, wilson_interval)


def protocol(**kwargs):
    return EvaluationProtocol(test_scenario_ids=('simple:left',), defense_seeds=(10,), **kwargs)


def defense(difficulty='simple', damage=0., unavoidable=False):
    return dict(scenario_id='simple:left', seed=10, split='test', difficulty=difficulty,
                terminal_reason='cleared', own_tower_fraction_loss=dict(left=damage,right=0.,king=0.),
                own_towers_alive=dict(left=damage < 1,right=True,king=True), lane='left',
                unavoidable_tower_damage=unavoidable, elixir_spent=5, coverage_flags=[])


def matches(p, wins=10):
    return [dict(row,result='win' if i < wins else 'loss') for i,row in enumerate(paired_schedule(p))]


def test_frozen_protocol_roundtrip_fingerprint_and_side_balance():
    p = protocol()
    assert EvaluationProtocol.from_dict(json.loads(json.dumps(p.to_dict()))) == p
    assert p.fingerprint == EvaluationProtocol.from_dict(p.to_dict()).fingerprint
    with pytest.raises(FrozenInstanceError):
        p.medium_max_hp_loss = .5
    schedule = paired_schedule(p)
    assert len(schedule) == 10
    for seed in p.paired_seeds:
        assert {r['candidate_side'] for r in schedule if r['seed']==seed} == {-1,1}


def test_hard_simple_avoidable_damage_fails_despite_reward_or_wins():
    p=protocol()
    metric=defense(damage=.001)
    metric['reward']=1e9
    result=acceptance_gate(p,[metric],matches(p),coverage_complete=True)
    assert result['status']=='fail'
    assert 'simple_avoidable_tower_damage' in result['failures']
    assert acceptance_gate(p,[defense(damage=.5,unavoidable=True)],matches(p),coverage_complete=True)['status']=='pass'


def test_medium_threshold_is_explicit_provisional_assumption():
    p=protocol()
    result=acceptance_gate(p,[defense('medium',damage=.01)],matches(p),coverage_complete=True)
    assert result['status']=='undetermined'
    assert 'untuned_medium_threshold' in result['undetermined_reasons']
    p=replace(p,medium_threshold_confirmed=True)
    assert acceptance_gate(p,[defense('medium',damage=.02)],matches(p),coverage_complete=True)['status']=='pass'
    assert acceptance_gate(p,[defense('medium',damage=.021)],matches(p),coverage_complete=True)['status']=='fail'


def test_soft_complex_and_impossible_survival_not_mandatory():
    p=protocol()
    for difficulty in ['complex','impossible']:
        metric=defense(difficulty,damage=1.)
        metric['own_towers_alive']['king']=False
        metric['terminal_reason']='own_king_destroyed'
        assert acceptance_gate(p,[metric],matches(p),coverage_complete=True)['status']=='pass'


def test_missing_coverage_and_incomplete_pairs_cannot_pass():
    p=protocol()
    result=acceptance_gate(p,[],matches(p)[:9])
    assert result['status']=='undetermined'
    assert {'missing_coverage','not_enough_defense_episodes','not_enough_paired_games:v25'} <= set(result['undetermined_reasons'])


def test_target_is_per_prior_not_pooled_and_draws_are_nonwins():
    p=protocol(baseline_ids=('v25','v24'))
    games=matches(p,wins=18)  # v25 10/10, v24 8/10
    result=acceptance_gate(p,[defense()],games,coverage_complete=True)
    assert result['status']=='pass'
    assert result['head_to_head']['v24']['observed_winrate']==.8
    assert result['head_to_head']['v24']['wilson_95'][0] < .8
    assert not result['head_to_head']['v24']['population_80_percent_established']
    games[-3]['result']='draw'
    result=acceptance_gate(p,[defense()],games,coverage_complete=True)
    assert result['status']=='fail'
    assert 'observed_winrate_below_target:v24' in result['failures']


def test_wrong_split_invalid_script_and_duplicate_rows_are_not_quality_evidence():
    p=protocol()
    metric=defense()
    metric['split']='train'
    assert acceptance_gate(p,[metric],matches(p),coverage_complete=True)['status']=='undetermined'
    metric['terminal_reason']='invalid_script_play_timeout'
    report=acceptance_gate(p,[metric,metric],matches(p)+matches(p)[:1],coverage_complete=True)
    assert 'invalid_script_episode' in report['undetermined_reasons']
    assert 'unexpected_duplicate_or_invalid_match' in report['undetermined_reasons']


def test_allowlist_requires_canonical_v25_and_checks_hashes(tmp_path):
    path=tmp_path/V25_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(b'fake weights, not deserialized')
    manifest=tmp_path/'allowlist.json'
    data=dict(schema='defense-baselines-v1',checkpoints=[dict(id='v25',path=V25_PATH,sha256=V25_SHA256)])
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='hash mismatch'):
        load_baseline_allowlist(manifest,project_root=tmp_path)
    assert set(load_baseline_allowlist(manifest,project_root=tmp_path,verify_hashes=False)) == {'v25'}
    data['checkpoints'][0]['path']='../escape.pt'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='outside checkpoints'):
        load_baseline_allowlist(manifest,project_root=tmp_path,verify_hashes=False)


def test_protocol_rejects_too_few_games_or_missing_v25():
    with pytest.raises(ValueError,match='10–50'):
        protocol(paired_seeds=(1,2))
    with pytest.raises(ValueError,match='v25'):
        protocol(baseline_ids=('v24',))
    assert wilson_interval(0,0)==[0.,1.]
