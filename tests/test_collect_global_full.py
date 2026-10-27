import json
from types import SimpleNamespace as NS

import pytest
import requests
from scripts.kaggle import collect_global_full as module


def fixture(monkeypatch,tmp_path):
    prepared=tmp_path/'prepared';prepared.mkdir()
    children=[dict(index=i,remote_output=f'global_full_child_{i}/continuous_focus',
                   selected=dict(trial=dict(id='joint_control')))for i in(0,1)]
    (prepared/'preparation.json').write_text(json.dumps(dict(kernel='owner/job',children=children)))
    data={'global_full_result.json':b'{}'}
    run='full_joint_control_s64'
    for child in children:
        prefix=child['remote_output']+'/'
        data[prefix+'result.json']=b'{}'
        data[prefix+run+'_summary.json']=json.dumps(dict(id=run,checkpoint=run+'/update_003750.pt')).encode()
        for name in('config.json','metrics.jsonl','training_result.json','update_003750.pt','update_004000.pt','update_000250.pt'):
            data[prefix+run+'/'+name]=(f"child{child['index']}-"+name).encode()
    monkeypatch.setattr(module,'list_outputs',lambda *args:({k:NS(url=k)for k in data},[]))
    calls=[]
    def get(url,**kwargs):
        calls.append(url)
        return NS(content=data[url],raise_for_status=lambda:None)
    monkeypatch.setattr(requests,'get',get)
    return prepared,data,calls


def test_duplicate_profile_ids_stay_separate_and_unselected_weights_are_omitted(monkeypatch,tmp_path):
    prepared,data,calls=fixture(monkeypatch,tmp_path)
    output=tmp_path/'collected'
    receipt=module.collect(NS(kernels_status=lambda *args:'COMPLETE'),prepared,output)
    assert receipt['complete']
    assert not any('update_000250' in name for name in receipt['files'])
    for i in(0,1):
        path=output/f'global_full_child_{i}/continuous_focus/full_joint_control_s64/update_003750.pt'
        assert path.read_bytes().startswith(f'child{i}-'.encode())
    assert len(calls)==len(receipt['files'])


def test_cross_child_checkpoint_reference_rejected_before_output(monkeypatch,tmp_path):
    prepared,data,calls=fixture(monkeypatch,tmp_path)
    summary='global_full_child_0/continuous_focus/full_joint_control_s64_summary.json'
    data[summary]=json.dumps(dict(id='full_joint_control_s64',checkpoint='../other/update_004000.pt')).encode()
    with pytest.raises(ValueError,match='Cross-child'):
        module.collect(NS(kernels_status=lambda *args:'COMPLETE'),prepared,tmp_path/'collected')
    assert not(tmp_path/'collected').exists()


def test_nonterminal_job_cannot_be_collected(monkeypatch,tmp_path):
    prepared,data,calls=fixture(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='terminal'):
        module.collect(NS(kernels_status=lambda *args:'RUNNING'),prepared,tmp_path/'collected')
    assert not calls


def test_verified_cache_avoids_repeated_weight_transfers(monkeypatch,tmp_path):
    prepared,data,calls=fixture(monkeypatch,tmp_path)
    api=NS(kernels_status=lambda *args:'COMPLETE')
    first=module.collect(api,prepared,tmp_path/'cached')
    calls.clear()
    recovered=module.collect(api,prepared,tmp_path/'recovered',cache=tmp_path/'cached')
    assert not calls and recovered['files']==first['files']
