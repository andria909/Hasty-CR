import hashlib
import json
from types import SimpleNamespace

import pytest

from sim.defense_curriculum import curriculum_selection


def scenarios():
    return [SimpleNamespace(scenario_id=s,split='train') for s in ('b','a','c')]


@pytest.mark.parametrize('wrapped',[False,True])
def test_explicit_curriculum_is_sorted_train_only_and_hashes_exact_content(tmp_path,wrapped):
    path=tmp_path/'ids.json'
    payload=['c','a']
    path.write_text(json.dumps({'scenario_ids':payload}if wrapped else payload))
    selected,config=curriculum_selection(scenarios(),path)
    assert [s.scenario_id for s in selected]==['a','c']
    assert config['scenario_ids']==['a','c']
    assert config['scenario_ids_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    _,same=curriculum_selection(scenarios(),path)
    assert same==config
    path.write_text(path.read_text()+'\n')
    _,changed=curriculum_selection(scenarios(),path)
    assert changed['scenario_ids']==config['scenario_ids']
    assert changed['scenario_ids_sha256']!=config['scenario_ids_sha256']


@pytest.mark.parametrize('payload',[[],['a','a'],['validation/attack'],['not-known'],[1],{'wrong':[]}])
def test_bad_curriculum_fails_closed(tmp_path,payload):
    path=tmp_path/'ids.json';path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):curriculum_selection(scenarios(),path)


def test_default_full_selected_difficulty_and_no_split_mutation():
    selected,metadata=curriculum_selection(scenarios())
    assert metadata['scenario_ids']==['a','b','c']
    with pytest.raises(ValueError):
        curriculum_selection([SimpleNamespace(scenario_id='v',split='validation')])


def test_both_trainers_offer_same_curriculum_flag():
    from sim.train_defense_ppo import parser
    args=parser().parse_args(['--init','fake.pt','--output','new','--scenario-ids','curriculum.json'])
    assert args.scenario_ids=='curriculum.json'
    # GRPO helper's immutable keys bind both selected IDs and file bytes on resume.
    from sim.train_defense_grpo import RESUME_KEYS
    assert 'scenario_ids' in RESUME_KEYS and 'scenario_ids_sha256' in RESUME_KEYS


def test_ppo_checkpoint_expression_keeps_curriculum_metadata_and_resume_guards(tmp_path):
    """Exercise the actual serialization expression without rollouts or gradients."""
    import ast
    import copy
    import inspect
    import torch
    import sim.train_defense_ppo as trainer
    path=tmp_path/'ids.json';path.write_text(json.dumps(['a','c']))
    _,curriculum=curriculum_selection(scenarios(),path)
    args=trainer.parser().parse_args(['--init','fake.pt','--output','new','--scenario-ids',str(path)])
    config=dict(vars(args),architecture='fake',exact_resume=True)
    config.update(curriculum)
    fake_state=SimpleNamespace(state_dict=lambda:{'weight':torch.tensor([1.])})
    namespace=dict(network=fake_state,optimizer=fake_state,architecture='fake',step=7,update=3,
        best=2.,config=config,args=args,rng_state=lambda:{},counters=[1],
        sampler=fake_state,snapshot=[{'pending':'state'}],observations=[{}],infos=[{}])
    tree=ast.parse(inspect.getsource(trainer.train))
    assignment=next(node for node in ast.walk(tree) if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name)and target.id=='checkpoint' for target in node.targets))
    checkpoint=eval(compile(ast.Expression(assignment.value),'<production checkpoint>','eval'),namespace)
    target=tmp_path/'ppo.pt';torch.save(checkpoint,target)
    loaded=torch.load(target,weights_only=False)
    assert loaded['config']['scenario_ids']==['a','c']
    assert loaded['config']['scenario_ids_sha256']==curriculum['scenario_ids_sha256']
    # Execute only production resume config guards, stopping before state restoration.
    guard=next(node for node in ast.walk(tree) if isinstance(node,ast.For)
        and isinstance(node.iter,ast.Tuple)
        and [getattr(v,'value',None)for v in node.iter.elts]==['scenario_ids','scenario_ids_sha256'])
    code=compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),'<production resume guard>','exec')
    exec(code,dict(old_config=loaded['config'],curriculum=curriculum))
    for key in ('scenario_ids','scenario_ids_sha256'):
        changed=copy.deepcopy(curriculum)
        changed[key]=['b'] if key=='scenario_ids' else 'bad-sha'
        with pytest.raises(ValueError,match='curriculum configuration mismatch'):
            exec(code,dict(old_config=loaded['config'],curriculum=changed))


def test_grpo_checkpoint_expression_and_resume_roundtrip_real_curriculum(tmp_path):
    import ast
    import copy
    import inspect
    import random
    import numpy as np
    import torch
    import sim.train_defense_grpo as trainer
    from sim.defense_sampler import ScenarioCycle
    path=tmp_path/'ids.json';path.write_text(json.dumps({'scenario_ids':['a','c']}))
    _,curriculum=curriculum_selection(scenarios(),path)
    config={key:None for key in trainer.RESUME_KEYS};config.update(curriculum)
    network=torch.nn.Linear(2,2);reference=copy.deepcopy(network)
    optimizer=torch.optim.Adam(network.parameters())
    sampler=ScenarioCycle(curriculum['scenario_ids'],seed=42)
    generator=torch.Generator().manual_seed(17)
    namespace=dict(network=network,reference=reference,optimizer=optimizer,architecture='fake',config=config,
        sampler=sampler,checkpoint_phase='boundary',best_validation=2.,completed=3,steps=7,
        degenerate_count=1,actual_steps={'train':7,'validation':0},torch=torch,np=np,random=random,
        generator=generator,device=torch.device('cpu'))
    tree=ast.parse(inspect.getsource(trainer.main))
    save_call=next(node for node in ast.walk(tree) if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Attribute)and isinstance(node.func.value,ast.Name)
        and node.func.value.id=='torch'and node.func.attr=='save')
    checkpoint=eval(compile(ast.Expression(save_call.args[0]),'<production GRPO checkpoint>','eval'),namespace)
    target=tmp_path/'grpo.pt';torch.save(checkpoint,target)
    loaded=torch.load(target,weights_only=False)
    restored=trainer.restore_training_state(loaded,network,reference,optimizer,sampler,generator,config)
    assert restored['completed']==3 and restored['steps']==7
    assert loaded['config']['scenario_ids']==['a','c']
    for key in ('scenario_ids','scenario_ids_sha256'):
        changed=copy.deepcopy(config);changed[key]=['b'] if key=='scenario_ids' else 'bad-sha'
        with pytest.raises(ValueError,match='configuration mismatch'):
            trainer.restore_training_state(loaded,network,reference,optimizer,sampler,generator,changed)
