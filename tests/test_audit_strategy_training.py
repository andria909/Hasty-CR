import json
from pathlib import Path

import pytest
import torch

from scripts.kaggle import audit_strategy_training as checker


def fixture(tmp_path,algorithm):
    root=tmp_path/'collected';root.mkdir()
    manifest=tmp_path/'manifest.json';manifest.write_text('{"archive_sha256":"archive"}')
    launch=tmp_path/'launch.json'
    target=checker.TARGETS[algorithm]
    launch.write_text(json.dumps(dict(dataset='owner/inputs',algorithm=algorithm,target_steps=target,
        pins={'manifest.json':checker.sha(manifest),'v25.pt':checker.BASELINE_SHA})))
    config=dict(task_mode='joint',observation_contract=checker.CONTRACT,num_scalars=309,
        init_sha256=checker.BASELINE_SHA,initial_hand_mode='shuffle',deadline_scale=3,
        target_environment_steps=target,geometry='legacy_v1',coordinate_mode='physical_live_cells_v2',device='cuda:0')
    blob=dict(config=config,state_dict={'scalar.0.weight':torch.zeros(2,309)},
        optimiser={'state':{'moment':torch.ones(1)}},optimizer={'state':{'moment':torch.ones(1)}},
        step=target,environment_steps=target,exact_resume_available=True,
        checkpoint_phase='boundary',resume_supported=True,
        reference_state_dict={'scalar.0.weight':torch.zeros(2,309)},
        actual_environment_steps=dict(train_attack=10,train_defense=10))
    name='ppo/update_002000.pt'if algorithm=='ppo'else 'grpo/final_u001234_s000003.pt'
    path=root/name;path.parent.mkdir();torch.save(blob,path)
    (root/'strategy_result.json').write_text(json.dumps(dict(schema='joint-strategy-account-v1',
        status='trainer_completed',owner='owner',algorithm=algorithm,target_steps=target,
        source_archive_sha256='archive',gpu_names=['Tesla T4'])))
    if algorithm=='ppo':
        (root/'ppo/training_result.json').write_text(json.dumps(dict(target_met=True,step=target)))
        (root/'ppo/metrics.jsonl').write_text(json.dumps(dict(episodes=[dict(task='attack'),dict(task='defense')]))+'\n')
    def receipt():
        files={str(p.relative_to(root)):dict(sha256=checker.sha(p))for p in root.rglob('*')
               if p.is_file()and p.name!='collection_receipt.json'}
        (root/'collection_receipt.json').write_text(json.dumps(dict(complete=True,remote_status='COMPLETE',
            kernel=f'owner/hastycr-joint-{algorithm}-20261023',files=files)))
    receipt()
    return root,launch,manifest,blob,path,receipt


@pytest.mark.parametrize('algorithm',['ppo','grpo'])
def test_completed_target_is_ready_for_evaluation_but_not_accepted(tmp_path,algorithm):
    root,launch,manifest,*_=fixture(tmp_path,algorithm)
    report=checker.audit(root,launch,manifest)
    assert report['target_met'] and report['ready_for_validation']
    assert not report['quality_accepted'] and not report['evaluation_completed']


@pytest.mark.parametrize('algorithm',['ppo','grpo'])
@pytest.mark.parametrize('fault',['short','nan','single_task','wrong_contract'])
def test_runtime_completion_does_not_bypass_training_checks(tmp_path,algorithm,fault):
    root,launch,manifest,blob,path,receipt=fixture(tmp_path,algorithm)
    if fault=='short':blob['step']=blob['environment_steps']=1
    elif fault=='nan':blob['state_dict']['scalar.0.weight'][0,0]=float('nan')
    elif fault=='wrong_contract':blob['config']['task_mode']='attack'
    elif algorithm=='grpo':blob['actual_environment_steps']['train_defense']=0
    else:(root/'ppo/metrics.jsonl').write_text('{"episodes":[{"task":"attack"}]}\n')
    torch.save(blob,path);receipt()
    with pytest.raises(ValueError):checker.audit(root,launch,manifest)


def test_tampered_download_cannot_be_evaluated(tmp_path):
    root,launch,manifest,blob,path,_=fixture(tmp_path,'grpo')
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='SHA mismatch'):checker.audit(root,launch,manifest)


def test_unfinished_grpo_group_cannot_pass(tmp_path):
    root,launch,manifest,blob,path,receipt=fixture(tmp_path,'grpo')
    blob['checkpoint_phase']='collection';blob['resume_supported']=False
    torch.save(blob,path);receipt()
    with pytest.raises(ValueError,match='unfinished group'):checker.audit(root,launch,manifest)
