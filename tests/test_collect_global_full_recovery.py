import json
from types import SimpleNamespace as NS

import pytest
import requests

from scripts.kaggle import collect_global_full_recovery as module


def fixture(monkeypatch,tmp_path):
    prepared=tmp_path/'prepared';prepared.mkdir()
    children=[dict(index=i,remote_output=f'global_full_child_{i}/continuous_focus',
                   selected=dict(trial=dict(id='same_profile')))for i in (0,1)]
    (prepared/'preparation.json').write_text(json.dumps(dict(kernel='owner/job',children=children)))
    data={}
    for child in children:
        prefix=child['remote_output']+'/full_same_profile_s64/'
        rows=[dict(update=i,step=1024*i)for i in range(1,4)]
        rows[1]['validation']=dict(selection_score=.4,worst_archetype_points=.2)
        data[prefix+'metrics.jsonl']=(''.join(json.dumps(r)+'\n'for r in rows)+'{"update":4').encode()
        data[prefix+'config.json']=b'{}'
        for i in range(1,5):data[prefix+f'update_{i:06d}.pt']=f'child{child["index"]}update{i}'.encode()
        data[prefix+'unrelated.bin']=b'not collected'
    monkeypatch.setattr(module,'list_outputs',lambda *a:({k:NS(url=k)for k in data},[]))
    calls=[]
    def get(url,**kwargs):
        calls.append(url);return NS(content=data[url],raise_for_status=lambda:None)
    monkeypatch.setattr(requests,'get',get)
    return prepared,data,calls


def test_error_job_preserves_latest_and_historical_selection_without_claiming_completion(monkeypatch,tmp_path):
    prepared,data,calls=fixture(monkeypatch,tmp_path)
    receipt=module.collect(NS(kernels_status=lambda *a:'ERROR'),prepared,tmp_path/'recovery')
    assert receipt['artifacts_complete'] and not receipt['complete']
    assert not receipt['training_budget_verified'] and not receipt['automatic_resume']
    for child in receipt['children']:
        assert child['latest_update']==3 and child['selected_update']==2
        assert child['recovery_boundary_available']
    assert len(receipt['files'])==8
    assert not any(name.endswith(('000001.pt','000004.pt','unrelated.bin'))for name in receipt['files'])
    assert len(calls)==8


def test_recovery_cache_preserves_both_children_without_redownloading(monkeypatch,tmp_path):
    prepared,data,calls=fixture(monkeypatch,tmp_path);api=NS(kernels_status=lambda *a:'ERROR')
    first=module.collect(api,prepared,tmp_path/'first');calls.clear()
    second=module.collect(api,prepared,tmp_path/'second',cache=tmp_path/'first')
    assert first['files']==second['files'] and calls==[]


def test_unstarted_child_is_reported_instead_of_inventing_a_checkpoint(monkeypatch,tmp_path):
    prepared,data,calls=fixture(monkeypatch,tmp_path)
    for name in list(data):
        if name.startswith('global_full_child_1/'):data.pop(name)
    receipt=module.collect(NS(kernels_status=lambda *a:'ERROR'),prepared,tmp_path/'recovery')
    assert receipt['children'][0]['recovery_boundary_available']
    assert not receipt['children'][1]['recovery_boundary_available']


def test_nonterminal_or_changed_remote_state_cannot_be_collected(monkeypatch,tmp_path):
    prepared,data,calls=fixture(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='terminal'):
        module.collect(NS(kernels_status=lambda *a:'RUNNING'),prepared,tmp_path/'recovery')
    states=iter(['ERROR','RUNNING'])
    with pytest.raises(ValueError,match='changed'):
        module.collect(NS(kernels_status=lambda *a:next(states)),prepared,tmp_path/'recovery')
    assert calls==[] and not(tmp_path/'recovery').exists()


def test_only_an_unterminated_last_metrics_line_may_be_ignored():
    assert len(module.logged_prefix(b'{"update":1}\n{"update":2'))==1
    with pytest.raises(ValueError,match='interior'):module.logged_prefix(b'{"update":1}\nbroken\n')
    with pytest.raises(ValueError,match='Nonconsecutive'):module.logged_prefix(b'{"update":2}\n')


def test_missing_historical_selected_boundary_blocks_recovery_selection():
    rows=[dict(update=1,validation=dict(selection_score=.5,worst_archetype_points=.2)),dict(update=2)]
    with pytest.raises(ValueError,match='Historical'):
        module.choose_boundaries(['run/update_000002.pt'],'run/',rows)


def test_changed_preparation_cannot_reuse_cache(monkeypatch,tmp_path):
    prepared,data,calls=fixture(monkeypatch,tmp_path);api=NS(kernels_status=lambda *a:'ERROR')
    module.collect(api,prepared,tmp_path/'first')
    path=prepared/'preparation.json';path.write_text(path.read_text()+'\n');calls.clear()
    with pytest.raises(ValueError,match='identity'):
        module.collect(api,prepared,tmp_path/'second',cache=tmp_path/'first')
    assert not calls and not(tmp_path/'second').exists()


def test_guarded_selection_and_unlogged_checkpoint_are_preserved():
    rows=[dict(update=1,validation=dict(selection_score=.8,worst_archetype_points=0)),
          dict(update=2,validation=dict(selection_score=.3,worst_archetype_points=.1)),
          dict(update=3),dict(update=4)]
    names=[f'run/update_{i:06d}.pt'for i in(1,2,3,5)]
    result=module.choose_boundaries(names,'run/',rows)
    assert result['latest_update']==3 and result['selected_update']==2
    assert result['selected_with_family_guard'] and result['metrics_rows_after_boundary']==1


def test_partial_download_receipt_is_retryable(monkeypatch,tmp_path):
    prepared,data,calls=fixture(monkeypatch,tmp_path);api=NS(kernels_status=lambda *a:'ERROR')
    original=module.download_preserving
    def fail_checkpoint(url,path,**kwargs):
        if url.endswith('.pt'):raise requests.ConnectionError('interrupted')
        return original(url,path,**kwargs)
    monkeypatch.setattr(module,'download_preserving',fail_checkpoint)
    with pytest.raises(requests.ConnectionError):module.collect(api,prepared,tmp_path/'partial')
    receipt=json.loads((tmp_path/'partial/collection_receipt.json').read_text())
    assert not receipt['complete'] and not receipt['artifacts_complete'] and len(receipt['files'])==2
    monkeypatch.setattr(module,'download_preserving',original);calls.clear()
    receipt=module.collect(api,prepared,tmp_path/'retried',cache=tmp_path/'partial')
    assert receipt['artifacts_complete'] and len(calls)==6
