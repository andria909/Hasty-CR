import hashlib
import json
from pathlib import Path

import pytest
import torch
from scripts.kaggle import select_completed_ppo_full as module
from sim.reward_profiles import load_reward_profile


def fixture(tmp_path):
    root=tmp_path/'collected';root.mkdir();files={}
    def write(name,value):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
        if name.endswith('.pt'):torch.save(value,p)
        else:p.write_text(json.dumps(value))
        files[name]={'sha256':module.sha(p)}
    coefficient_selection=dict(stage='confirmed',source_sha256='frozen',test_used_for_selection=False,modes=dict(defense=dict(full_training_profiles=['baseline'])))
    selection_sha=hashlib.sha256(json.dumps(coefficient_selection,sort_keys=True).encode()).hexdigest()
    config=dict(mode='defense',profile='baseline',selection=coefficient_selection,pins={'teacher_defense.pt':'teacher'})
    _,reward_sha=load_reward_profile(Path('data/ppo_tuning/profiles/baseline.json'))
    blob=dict(step=2048000,update=2000,config=dict(geometry='legacy_v1',coordinate_mode='physical_live',reward_profile_sha256=reward_sha,init_sha256='teacher'),
              state_dict={'scalar.0.weight':torch.zeros(2,306)},optimiser={'state':{'step':torch.tensor(2000.)}},
              exact_resume_available=True,environment_states=[{} for i in range(8)])
    write('train/update_002000.pt',blob);checkpoint_sha=files['train/update_002000.pt']['sha256']
    validation=dict(checkpoint_sha256=checkpoint_sha,training_steps=2048000,test_used_for_selection=False,mode='defense',profile='baseline')
    write('validation.json',validation)
    result=dict(status='completed',step=2048000,update=2000,mode='defense',profile='baseline',source_sha256='frozen',selection_sha256=selection_sha,
                checkpoint='train/update_002000.pt',checkpoint_sha256=checkpoint_sha,validation_sha256=files['validation.json']['sha256'],test_used_for_selection=False)
    write('result.json',result)
    manifest=dict(archive_sha256='frozen',files={'data/ppo_tuning/profiles/baseline.json':module.sha('data/ppo_tuning/profiles/baseline.json')})
    launch=tmp_path/'launch.json';launch.write_text(json.dumps(config));source=tmp_path/'manifest.json';source.write_text(json.dumps(manifest))
    def receipt(complete=True):(root/'receipt.json').write_text(json.dumps(dict(status='COMPLETE',complete=complete,files=files)))
    receipt();return root,launch,source,result,write,receipt


def test_fixed_final_checkpoint_is_preserved_without_test_selection(tmp_path):
    root,launch,source,result,write,receipt=fixture(tmp_path)
    report=module.select(root,launch,source,tmp_path/'selected')
    assert report['training_steps']==2048000 and report['test_used_for_selection'] is False
    assert module.sha(tmp_path/'selected/policy.pt')==result['checkpoint_sha256']
    assert report['quality_accepted'] is False


def test_incomplete_budget_cannot_be_relabelled_as_final(tmp_path):
    root,launch,source,result,write,receipt=fixture(tmp_path)
    result.update(status='incomplete_budget',step=1433600);write('result.json',result);receipt()
    with pytest.raises(ValueError,match='budget'):module.select(root,launch,source,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_modified_weight_rejected_even_if_transport_receipt_is_updated(tmp_path):
    root,launch,source,result,write,receipt=fixture(tmp_path)
    blob=torch.load(root/'train/update_002000.pt',weights_only=False);blob['step']=102400
    write('train/update_002000.pt',blob);receipt()
    with pytest.raises(ValueError,match='SHA'):module.select(root,launch,source,tmp_path/'out')


def test_test_selection_or_unconfirmed_profile_rejected(tmp_path):
    root,launch,source,result,write,receipt=fixture(tmp_path)
    config=json.loads(launch.read_text());config['selection']['test_used_for_selection']=True;launch.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='provenance'):module.select(root,launch,source,tmp_path/'out')


def test_nonfinite_optimizer_rejected_with_consistent_transport_and_report_hashes(tmp_path):
    root,launch,source,result,write,receipt=fixture(tmp_path)
    blob=torch.load(root/'train/update_002000.pt',weights_only=False);blob['optimiser']['lr']=float('nan')
    write('train/update_002000.pt',blob)
    checkpoint_sha=module.sha(root/'train/update_002000.pt')
    validation=json.loads((root/'validation.json').read_text());validation['checkpoint_sha256']=checkpoint_sha
    write('validation.json',validation)
    result.update(checkpoint_sha256=checkpoint_sha,validation_sha256=module.sha(root/'validation.json'))
    write('result.json',result);receipt()
    with pytest.raises(ValueError,match='Nonfinite'):module.select(root,launch,source,tmp_path/'out')
    assert not (tmp_path/'out').exists()
