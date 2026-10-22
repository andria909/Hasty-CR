import json

import pytest
import torch

from scripts.kaggle.select_strategy_checkpoint import select, sha
from tests.test_audit_strategy_training import fixture


def prepared(tmp_path,algorithm):
    root,launch,manifest,terminal,path,receipt=fixture(tmp_path,algorithm)
    terminal['best_validation']=1.5;torch.save(terminal,path)
    best=dict(terminal)
    best['step']=best['environment_steps']=1000
    name='ppo/best_validation_000100.pt'if algorithm=='ppo'else 'grpo/best_validation_u000100_s000002.pt'
    target=root/name;torch.save(best,target);receipt()
    return root,launch,manifest,terminal,path,best,target,receipt


@pytest.mark.parametrize('algorithm',['ppo','grpo'])
def test_trainer_validation_best_is_frozen_without_using_test(tmp_path,algorithm):
    root,launch,manifest,terminal,path,best,target,receipt=prepared(tmp_path,algorithm)
    output=tmp_path/'selected'
    report=select(root,launch,manifest,output)
    assert report['selected_training_steps']==1000
    assert report['terminal_training_audit']['target_met']
    assert report['selected_sha256']==sha(target)==sha(output/'policy.pt')
    assert not report['test_used_for_selection']and not report['quality_accepted']
    with pytest.raises(ValueError,match='new selected-policy directory'):
        select(root,launch,manifest,output)


@pytest.mark.parametrize('fault',['score','contract','nan','counter','missing'])
def test_invalid_selection_does_not_create_output(tmp_path,fault):
    root,launch,manifest,terminal,path,best,target,receipt=prepared(tmp_path,'grpo')
    if fault=='score':best['best_validation']=100
    elif fault=='contract':best['config']=dict(best['config'],task_mode='attack')
    elif fault=='nan':best['state_dict']['scalar.0.weight'][0,0]=float('nan')
    elif fault=='counter':best['environment_steps']=terminal['environment_steps']+1
    if fault!='missing':torch.save(best,target)
    receipt()
    if fault=='missing':
        r=json.loads((root/'collection_receipt.json').read_text())
        r['files'].pop(str(target.relative_to(root)))
        (root/'collection_receipt.json').write_text(json.dumps(r))
    output=tmp_path/'selected'
    with pytest.raises(ValueError):select(root,launch,manifest,output)
    assert not output.exists()
