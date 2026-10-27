from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from scripts.attack import evaluate_full_candidates as module


def blueprint():
    return json.loads(Path('data/ppo_tuning/full_candidate_evaluation_20261027/protocol.json').read_text())


def fixture(tmp_path):
    spec=blueprint();models=[];proof_models=[]
    for index in range(2):
        name=f'candidate_{index:02d}';path=tmp_path/(name+'.pt');path.write_bytes(name.encode())
        models.append(dict(id=name,file=path.name,sha256=module.sha(path),role='candidate',full_steps=4096000,
            mode='joint_enemy_presence_v1',scalars=309,full_proof_file='full_verified.json',origin_model_id=f'global{index}_full'))
        proof_models.append(dict(id=f'global{index}_full',checkpoint_sha256=module.sha(path)))
    proof=tmp_path/'full_verified.json';proof.write_text(json.dumps(dict(full_steps_verified=8192000,models=proof_models)))
    for row in models:row['full_proof_sha256']=module.sha(proof)
    for baseline in spec['baselines']:
        name=baseline['id'];path=tmp_path/(name+'.pt');path.write_bytes(name.encode());baseline['checkpoint_sha256']=module.sha(path)
        models.append(dict(id=name,file=path.name,sha256=module.sha(path),role='baseline',mode=baseline['mode'],scalars=baseline['scalars']))
    path=tmp_path/'evaluation_blueprint.json';path.write_text(json.dumps(spec))
    return dict(schema='bound-full-candidate-evaluation-v1',models=models,blueprint_sha256=module.sha(path))


def test_validated_candidate_budget_and_all_frozen_baselines_required(tmp_path):
    protocol=fixture(tmp_path)
    assert module.validate_protocol(protocol,tmp_path)['allowed_candidate_counts']==[2,3,4]
    protocol['models'][0]['full_steps']=102400
    with pytest.raises(ValueError,match='full budget'):module.validate_protocol(protocol,tmp_path)


def test_baseline_cannot_be_replaced_even_if_new_file_digest_is_self_consistent(tmp_path):
    protocol=fixture(tmp_path);entry=next(m for m in protocol['models']if m['id']=='v25')
    path=tmp_path/entry['file'];path.write_bytes(b'replacement');entry['sha256']=module.sha(path)
    with pytest.raises(ValueError,match='Baseline identity'):module.validate_protocol(protocol,tmp_path)


@pytest.mark.parametrize('count',[2,3,4])
def test_schedule_covers_two_separate_tests_and_every_pair_with_both_seats(count):
    spec=blueprint();protocol=dict(models=[dict(id=f'model_{i}')for i in range(count+3)])
    jobs=module.schedule(protocol,spec)
    assert len({j['file']for j in jobs})==len(jobs)
    assert sum(j['games']for j in jobs)==spec['budgets_by_candidate_count'][str(count)]['total_games']
    assert len([j for j in jobs if j['kind']=='test'])==2*(count+3)
    assert all(j['games']==60 for j in jobs if j['kind']=='duel')


def test_lane_metrics_are_in_each_players_own_coordinates():
    match=NS(tower_fractions=lambda side:dict(left=.5 if side==1 else .25,right=.75 if side==1 else 1.,king=1.),
        play_log=[dict(side=1,card='cannon',x=3000),dict(side=-1,card='cannon',x=15000)],
        battle=NS(spell_cast_log=[]))
    result=module.public_duel_metrics(match)
    assert result['1']['accepted_card_lane_counts']['left']['cannon']==1
    assert result['-1']['accepted_card_lane_counts']['left']['cannon']==1
    assert result['-1']['own_tower_fraction_loss']['left']==0
    assert result['-1']['own_tower_fraction_loss']['right']==.75
