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
