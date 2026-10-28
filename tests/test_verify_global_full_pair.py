import json
from pathlib import Path

import pytest
import torch
from scripts.attack import verify_global_full_pair as module


@pytest.mark.parametrize('corruption',[None,'battle','profile','checkpoint','validation','missing_validation'])
def test_new_training_contract_requires_actual_checkpoint_and_validation_modes(tmp_path,corruption):
    from copy import deepcopy
    from types import SimpleNamespace
    from sim.engine import Battle
    from scripts.attack.evaluation_execution import CORRECTED,combat_modes
    profile=dict(source_radius_reach=1,target_lock_in_range=0,cannon_center_only=1,
                 action_cooldown_ms=0,decision_interval_ms=250)
    config=dict(continuous_profile=profile,geometry=CORRECTED['geometry'],
        coordinate_mode=CORRECTED['coordinate_mode'],action_delay_ms=1350,max_ms=310000)
    state=dict(profile=deepcopy(profile),match=SimpleNamespace(battle=Battle()))
    blob=dict(config=deepcopy(config),environment_states=[state])
    metrics=tmp_path/'metrics.jsonl';records=[combat_modes(CORRECTED)]
    if corruption=='battle':state['match'].battle.normal_range_mode='legacy_target_edge_v1'
    if corruption=='profile':state['profile']['source_radius_reach']=0
    if corruption=='checkpoint':blob['config']['continuous_profile']['source_radius_reach']=0
    if corruption=='validation':records[0]['normal_range_mode']='legacy_target_edge_v1'
    if corruption=='missing_validation':records=[]
    metrics.write_text(json.dumps(dict(validation=dict(records=records)))+'\n')
    if corruption:
        with pytest.raises(ValueError):module.audit_training_execution(config,blob,metrics)
    else:
        execution,count=module.audit_training_execution(config,blob,metrics)
        assert execution==CORRECTED and count==1


def fixture(monkeypatch,tmp_path):
    prepared=tmp_path/'prepared';dataset=prepared/'dataset';dataset.mkdir(parents=True)
    (prepared/'job').mkdir();(prepared/'job/train.py').write_text('driver')
    root=tmp_path/'collected';root.mkdir();selected=[];children=[];proofs={};rows={};calls=[]
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    for i in(0,1):
        origin=tmp_path/f'pilot{i}'
        write(origin/'collected/parent_reference.json',dict(validation=dict(records=[])))
        proof=dict(initial_selection=dict(selected='parent'));proofs[origin]=proof
        trial=dict(id='joint_control');choice=dict(prepared=str(origin),study=f'owner/pilot{i}',trial=trial)
        selected.append(choice);bundle=dataset/f'bundle_{i}'
        write(bundle/'pilot_verified.json',proof);write(bundle/'protocol.json',{})
        (bundle/'train.py').write_text('child')
        c=dict(models=[dict(id='parent',file='parent.pt',sha256='init')],pins={
            name:module.sha(bundle/name)for name in('pilot_verified.json','protocol.json')})
        prefix=f'global_full_child_{i}/continuous_focus';run='full_joint_control_s64';childroot=root/prefix
        weight=childroot/run/'update_003750.pt';weight.parent.mkdir(parents=True)
        torch.save(dict(state_dict={'scalar.0.weight':torch.zeros(2,309),'conv.0.weight':torch.zeros(2,20,1,1)}),weight)
        (weight.parent/'update_004000.pt').write_bytes(b'last')
        row=dict(checkpoint=run+'/update_003750.pt',checkpoint_sha256=module.sha(weight),
                 selected_with_family_guard=True,validation=dict(selection_score=.7))
        rows[childroot]=row
        write(childroot/(run+'_summary.json'),row)
        write(childroot/run/'config.json',dict(device=f'cuda:{i}'))
        write(childroot/run/'training_result.json',{});(childroot/run/'metrics.jsonl').write_text('')
        write(childroot/'result.json',dict(status='complete',pilot_steps=0,full_steps=4096000,
            initial_selection=proof['initial_selection'],protocol_sha256=c['pins']['protocol.json'],full_runs=[row]))
        children.append(dict(index=i,remote_output=prefix,selected=choice,config=c,bootstrap_sha256=module.sha(bundle/'train.py')))
    ranking=dict(studies=[dict(prepared=s['prepared'])for s in selected],gate_met=True,selected=selected)
    write(dataset/'global_selection.json',ranking)
    config=dict(pins={str(p.relative_to(dataset)):module.sha(p)for p in dataset.rglob('*')if p.is_file()})
    write(prepared/'preparation.json',dict(config=config,bootstrap_sha256=module.sha(prepared/'job/train.py'),
        children=children,pair_index=0,kernel='owner/full'))
    write(root/'global_full_result.json',dict(status='complete',children=2,full_steps=8192000,pilot_steps_repeated=0))
    write(root/'collection_receipt.json',dict(kernel='owner/full',remote_status='COMPLETE',complete=True,
        files={str(p.relative_to(root)):dict(sha256=module.sha(p))for p in root.rglob('*')if p.is_file()}))
    monkeypatch.setattr(module,'select',lambda *a,**kw:ranking)
    monkeypatch.setattr(module,'verify_pilot',lambda collected,origin:proofs[origin])
    def audit(childroot,*args,**kwargs):
        calls.append(kwargs);return rows[childroot]
    monkeypatch.setattr(module,'audit_run',audit)
    return root,prepared,calls


def test_each_child_uses_its_own_dataset_and_remote_namespace(monkeypatch,tmp_path):
    root,prepared,calls=fixture(monkeypatch,tmp_path)
    result=module.verify(root,prepared)
    assert result['full_steps_verified']==8192000 and result['pilot_steps_repeated']==0
    assert result['ready_for_independent_evaluation']and not result['quality_accepted']
    for i,call in enumerate(calls):
        assert call['dataset']==prepared/'dataset'/f'bundle_{i}'
        assert call['remote_output']==f'/kaggle/working/global_full_child_{i}/continuous_focus'
    assert len({r['id']for r in result['models']})==2


def test_unlisted_required_artifact_is_rejected_even_if_file_exists(monkeypatch,tmp_path):
    root,prepared,calls=fixture(monkeypatch,tmp_path)
    path=root/'collection_receipt.json';receipt=json.loads(path.read_text())
    del receipt['files']['global_full_child_1/continuous_focus/full_joint_control_s64/metrics.jsonl']
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='missing from receipt'):module.verify(root,prepared)


def test_modified_artifact_is_rejected_before_model_acceptance(monkeypatch,tmp_path):
    root,prepared,calls=fixture(monkeypatch,tmp_path)
    (root/'global_full_result.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):module.verify(root,prepared)
    assert not calls
