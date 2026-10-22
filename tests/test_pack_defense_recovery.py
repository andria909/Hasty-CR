"""Frozen recovery patches and boundary migration, without optimizer updates."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from scripts.kaggle.pack_defense_recovery import (CONTRACT_FILES,environment_hash,patch_trainer,
                                                  replace_once,sha)
from sim.defense_sampler import ScenarioCycle
from sim.train_ppo import load_policy_state

# Minimal syntactically valid frozen trainer contract. No main/rollout is run.
TRAINER='''
RESUME_KEYS=('group_size','device_name','torch_version','init_sha256','environment_code_sha256')
def restore_training_state(blob,network,reference,optimizer,sampler,generator,config):
    if not blob.get('resume_supported',False) or blob.get('checkpoint_phase')!='boundary':
        raise ValueError('Checkpoint is inside an unfinished group/validation; exact resume unsupported. Use a boundary checkpoint.')
    if blob.get('resume_schema')!='defense-grpo-boundary-v1':
        raise ValueError('Checkpoint lacks supported exact-resume metadata')
    differences=[key for key in RESUME_KEYS if blob['config'].get(key)!=config.get(key)]
    if differences:
        raise ValueError(f'Resume configuration mismatch: {differences}')
    load_policy_state(network,blob['state_dict'])
    load_policy_state(reference,blob['reference_state_dict'])
    optimizer.load_state_dict(blob['optimizer'])
    sampler.load_state_dict(blob['sampler'])
    torch.set_rng_state(blob['torch_rng'].cpu())
    if blob['cuda_rng']:
        if not torch.cuda.is_available():
            raise ValueError('CUDA RNG cannot be restored without CUDA')
        torch.cuda.set_rng_state_all([state.cpu() for state in blob['cuda_rng']])
    generator.set_state(blob['action_rng'].cpu())
    np.random.set_state(blob['numpy_rng'])
    random.setstate(blob['python_rng'])
    return dict(completed=blob['updates'],steps=blob['environment_steps'],
                degenerate_count=blob['degenerate_groups'],
                actual_steps=dict(blob['actual_environment_steps']),
                best_validation=blob['best_validation'])
def main(argv=None):
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    contract_files=__CONTRACT__
    config['environment_code_sha256'] = contract.hexdigest()
    def save():
        return dict(
            best_validation=best_validation,
        )
'''.replace('__CONTRACT__',repr(CONTRACT_FILES))


def migration():
    return dict(schema='defense-grpo-environment-migration-v1',not_exact_resume=True,
        reset_best_validation=True,reason='escort engagement correction',
        from_environment_code_sha256='a'*64,to_environment_code_sha256='b'*64)


def patched_functions(pin):
    source=patch_trainer(TRAINER,pin)
    tree=ast.parse(source)
    selected=[node for node in tree.body if isinstance(node,ast.Assign)or
              isinstance(node,ast.FunctionDef)and node.name!='main']
    namespace=dict(torch=torch,np=np,random=random,Path=Path,hashlib=hashlib,json=json,
                   copy=copy,load_policy_state=load_policy_state)
    exec(compile(ast.Module(body=selected,type_ignores=[]),'<patched frozen recovery helpers>','exec'),namespace)
    return namespace,source


def boundary_state():
    network=torch.nn.Linear(2,2);reference=copy.deepcopy(network)
    optimizer=torch.optim.Adam(network.parameters())
    sampler=ScenarioCycle(['a','b'],seed=42);sampler.next()
    generator=torch.Generator().manual_seed(123)
    config=dict(group_size=8,device_name='cpu',torch_version='fixture',init_sha256='v25',environment_code_sha256='a'*64)
    blob=dict(resume_supported=True,checkpoint_phase='boundary',resume_schema='defense-grpo-boundary-v1',
        config=config,state_dict=copy.deepcopy(network.state_dict()),reference_state_dict=copy.deepcopy(reference.state_dict()),
        optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),updates=350,groups=350,
        environment_steps=270000,degenerate_groups=5,actual_environment_steps={'train':270000,'validation':1234},
        torch_rng=torch.get_rng_state(),cuda_rng=[],action_rng=generator.get_state(),
        numpy_rng=np.random.get_state(),python_rng=random.getstate(),best_validation=10.)
    return blob,(network,reference,optimizer,sampler,generator)


def test_default_strict_and_explicit_migration_only_permits_code_hash_difference():
    helpers,_=patched_functions('irrelevant')
    blob,state=boundary_state()
    new=dict(blob['config'],environment_code_sha256='b'*64)
    with pytest.raises(ValueError,match='environment_code_sha256'):
        helpers['restore_training_state'](blob,*state,new)
    new.update(code_migration=migration(),code_migration_applied_this_resume=True,not_exact_resume=True)
    restored=helpers['restore_training_state'](blob,*state,new)
    assert restored['completed']==350 and restored['steps']==270000
    assert restored['best_validation']==-float('inf')
    assert state[3].state_dict()==blob['sampler']
    for key in ('group_size','device_name','torch_version','init_sha256'):
        changed=dict(new);changed[key]='different'
        with pytest.raises(ValueError,match='configuration mismatch'):
            helpers['restore_training_state'](blob,*state,changed)
    # New-code continuation restores fresh validation score without reapplying migration.
    later=copy.deepcopy(blob);later['config']=dict(new);later['best_validation']=3.
    restored=helpers['restore_training_state'](later,*state,dict(new,code_migration_applied_this_resume=False))
    assert restored['best_validation']==3.


def test_invalid_migration_partial_or_nonboundary_state_is_rejected_before_updates():
    helpers,_=patched_functions('irrelevant')
    blob,state=boundary_state()
    config=dict(blob['config'],environment_code_sha256='b'*64,code_migration=migration(),code_migration_applied_this_resume=True)
    for changed in (dict(blob,checkpoint_phase='collection'),dict(blob,resume_supported=False),
                    {k:v for k,v in blob.items()if k!='reference_state_dict'},
                    dict(blob,optimizer={}),dict(blob,groups=349),dict(blob,sampler=dict(blob['sampler'],scenario_ids=['other']))):
        with pytest.raises((ValueError,KeyError)):
            helpers['restore_training_state'](changed,*state,config)
    for changed in (dict(migration(),from_environment_code_sha256='c'*64),
                    dict(migration(),to_environment_code_sha256='c'*64),
                    dict(migration(),reset_best_validation=False)):
        with pytest.raises(ValueError,match='exact old/new source pair'):
            helpers['restore_training_state'](blob,*state,dict(config,code_migration=changed))


def test_migration_file_bytes_are_pinned_not_just_user_supplied_endpoints(tmp_path):
    contents=(json.dumps(migration(),sort_keys=True)+'\n').encode()
    helpers,_=patched_functions(sha(contents))
    path=tmp_path/'migration.json';path.write_bytes(contents)
    assert helpers['load_resume_code_migration'](path)==migration()
    path.write_bytes(contents+b' ')
    with pytest.raises(ValueError,match='SHA'):
        helpers['load_resume_code_migration'](path)


def test_environment_hash_uses_exact_old_twelve_files_and_excludes_trainer():
    members={'sim/'+name:name.encode()for name in CONTRACT_FILES}
    members['sim/train_defense_grpo.py']=TRAINER.encode()
    old=environment_hash(members)
    members['sim/train_defense_grpo.py']=patch_trainer(TRAINER,'a'*64).encode()
    assert environment_hash(members)==old
    members['sim/defense_env.py']+=b'changed escort semantics'
    assert environment_hash(members)!=old
    with pytest.raises(ValueError):replace_once('marker marker','marker','new')
    with pytest.raises(ValueError):replace_once('nothing','marker','new')


def test_migration_restores_reference_optimizer_rng_and_sampler_without_optimization():
    helpers,_=patched_functions('irrelevant')
    blob,(network,reference,optimizer,sampler,generator)=boundary_state()
    expected_torch=torch.rand(3)
    expected_action=torch.rand(3,generator=generator)
    expected_numpy=np.random.rand(3)
    expected_python=random.random()
    expected_context=copy.deepcopy(sampler).next()
    with torch.no_grad():
        network.weight.zero_();reference.weight.zero_()
    optimizer.param_groups[0]['lr']=.123
    config=dict(blob['config'],environment_code_sha256='b'*64,
        code_migration=migration(),code_migration_applied_this_resume=True,not_exact_resume=True)
    helpers['restore_training_state'](blob,network,reference,optimizer,sampler,generator,config)
    torch.testing.assert_close(network.weight,blob['state_dict']['weight'])
    torch.testing.assert_close(reference.weight,blob['reference_state_dict']['weight'])
    assert optimizer.param_groups[0]['lr']==blob['optimizer']['param_groups'][0]['lr']
    torch.testing.assert_close(torch.rand(3),expected_torch,rtol=0,atol=0)
    torch.testing.assert_close(torch.rand(3,generator=generator),expected_action,rtol=0,atol=0)
    np.testing.assert_array_equal(np.random.rand(3),expected_numpy)
    assert random.random()==expected_python and sampler.next()==expected_context


def test_checkpoint_dict_persists_nonexact_migration_identity():
    _,source=patched_functions('approved-sha')
    tree=ast.parse(source)
    save=next(node for node in ast.walk(tree)if isinstance(node,ast.FunctionDef)and node.name=='save')
    expression=next(node.value for node in save.body if isinstance(node,ast.Return))
    config=dict(not_exact_resume=True,code_migration_sha256='approved-sha',code_migration_reason='escort correction')
    payload=eval(compile(ast.Expression(expression),'<patched checkpoint metadata>','eval'),
                 dict(config=config,best_validation=-float('inf')))
    assert payload['not_exact_resume']is True
    assert payload['code_migration_sha256']=='approved-sha'
    assert payload['code_migration_reason']=='escort correction'
