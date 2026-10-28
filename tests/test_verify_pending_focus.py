import json
import pytest
from scripts.attack.verify_pending_focus import check_input_config,metric_boundaries,input_spec


@pytest.mark.parametrize('pending',[False,True])
def test_input_contract_requires_flag_dimensions_and_identity(pending):
    trial=dict(pending_inputs=pending,history_frames=1);p,s,c=input_spec(trial)
    config=dict(pending_inputs=pending,num_planes=p,num_scalars=s,observation_contract=c)
    assert check_input_config(config,trial)==(p,s,c)
    for key,value in [('pending_inputs',not pending),('num_planes',99),('num_scalars',999),('observation_contract','wrong')]:
        with pytest.raises(ValueError):check_input_config(dict(config,**{key:value}),trial)


def test_complete_flag_does_not_replace_consecutive_cuda_budget(tmp_path):
    p=tmp_path/'metrics.jsonl'
    rows=[dict(update=i,step=i*1024,target_environment_steps=3072,percent=100*i/3,actual_cuda='Tesla T4')for i in(1,2,3)]
    p.write_text('\n'.join(map(json.dumps,rows)))
    assert metric_boundaries(p,3072)==[]
    p.write_text('\n'.join(map(json.dumps,rows[::2])))
    with pytest.raises(ValueError):metric_boundaries(p,3072)


def test_pending_sequence_protocol_rejects_drift():
    from copy import deepcopy
    from scripts.attack.verify_pending_focus import validate_protocol
    p=dict(pilot_seeds=[252,253],cases=[dict(id=name,warmstart_updates=bc,supervised_replay_weight=weight,
        pending_inputs=True,sequence_wait_fraction=.25,warmstart_kl_weight=.1,action_delay_ms=1350,validation_action_delay_ms=1350)
        for name,bc,weight in [('control',0,0.),('bc800',800,0.),('replay100',800,1.),('replay500',800,5.)]])
    validate_protocol(p,dict(allow_full_training=False),'pending_sequence_retention_v1')
    for key,value in [('pending_inputs',False),('sequence_wait_fraction',.9),('action_delay_ms',225),('supervised_replay_weight',2.)]:
        changed=deepcopy(p);changed['cases'][2][key]=value
        with pytest.raises(ValueError):validate_protocol(changed,dict(allow_full_training=False),'pending_sequence_retention_v1')


def test_pending_diversity_protocol_keeps_replay_off():
    from copy import deepcopy
    from scripts.attack.verify_pending_focus import validate_protocol
    p=dict(pilot_seeds=[262,263],cases=[dict(id=name,warmstart_updates=bc,slot_augmentation=slots,
        pending_inputs=True,supervised_replay_weight=0.,sequence_wait_fraction=.25,warmstart_kl_weight=.1,action_delay_ms=1350,validation_action_delay_ms=1350)
        for name,bc,slots in [('control',0,False),('diverse200',200,False),('slot200',200,True),('slot800',800,True)]])
    validate_protocol(p,dict(allow_full_training=False),'pending_diversity_slots_v1')
    for key,value in [('supervised_replay_weight',1.),('slot_augmentation',False),('action_delay_ms',225)]:
        changed=deepcopy(p);changed['cases'][2][key]=value
        with pytest.raises(ValueError):validate_protocol(changed,dict(allow_full_training=False),'pending_diversity_slots_v1')
