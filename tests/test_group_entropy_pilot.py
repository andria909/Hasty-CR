from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest
import requests

from scripts.kaggle.prepare_group_entropy_pilot import CASES,VARIANT,validate_protocol
from scripts.attack.verify_group_entropy_pilot import audit_entropy
from scripts.kaggle import collect_final_evaluation as collector
from tests.test_calibrated_spell_pilot import protocol as base_protocol


def protocol():
    p=base_protocol();base=dict(p['cases'][0],entropy_coef=.01)
    p.update(study_variant=VARIANT,pilot_seeds=[432,433],
             cases=[dict(base,id=name,spatial_entropy_weight=w)for name,w in CASES])
    return p


def test_grid_rejects_reward_or_decoder_confounding():
    p=protocol();validate_protocol(p)
    for key,value in [('spatial_entropy_weight',.7),('spell_waste_weight',.5),('action_delay_ms',0),('entropy_coef',.1)]:
        changed=deepcopy(p);changed['cases'][1][key]=value
        with pytest.raises(ValueError):validate_protocol(changed)


def test_actual_entropy_objective_is_checked():
    row=dict(spatial_weight=.3,optimizer_batches=4,full=4.,group=1.,objective=1.9)
    audit_entropy(row,.3)
    for key,value in [('spatial_weight',1.),('objective',4.),('full',float('nan')),('optimizer_batches',0)]:
        with pytest.raises(ValueError):audit_entropy(dict(row,**{key:value}),.3)


def test_entropy_collector_keeps_all_selected_latest_but_not_other_seeds(monkeypatch,tmp_path):
    prefix='continuous_focus/';data={prefix+'result.json':b'{}',prefix+'parent_reference.json':b'{}'}
    for name,_ in CASES:
        for seed in (432,433):
            run=f'pilot_{name}_s{seed}'
            data[prefix+run+'_summary.json']=json.dumps(dict(id=run,checkpoint=run+'/update_000050.pt')).encode()
            for artifact in ('config.json','metrics.jsonl','training_result.json','update_000050.pt','update_000100.pt'):
                data[prefix+run+'/'+artifact]=b'{}'
    data[prefix+'pilot_control_s434/update_000100.pt']=b'wrong seed'
    monkeypatch.setattr(collector,'list_outputs',lambda *a:({name:NS(url=name)for name in data},[]))
    monkeypatch.setattr(requests,'get',lambda url,**kw:NS(content=data[url],raise_for_status=lambda:None))
    root=tmp_path/'collected'
    r=collector.collect(NS(kernels_status=lambda k:'COMPLETE'),'owner/job',root,'group-entropy-pilot')
    assert r['complete']and len(list(root.glob('*/update_*.pt')))==16
    assert not(root/'pilot_control_s434').exists()
