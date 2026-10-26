import json
from types import SimpleNamespace as NS
import requests
import pytest
from scripts.kaggle import collect_final_evaluation as collector


def setup(monkeypatch,selected,run='pilot_uniform_base_s62'):
    prefix='continuous_focus/'
    data={prefix+'result.json':b'{}',prefix+run+'_summary.json':json.dumps(dict(id=run,checkpoint=selected)).encode(),
          prefix+run+'/update_000050.pt':b'chosen_eligible',prefix+run+'/update_000100.pt':b'latest_resume',
          prefix+run+'/update_000025.pt':b'unused',prefix+run+'/update_999999.pt':b'outside_budget'}
    monkeypatch.setattr(collector,'list_outputs',lambda api,kernel:({n:NS(url=n)for n in data},[]))
    calls=[]
    def get(url,**kw):calls.append(url);return NS(content=data[url],raise_for_status=lambda:None)
    monkeypatch.setattr(requests,'get',get)
    return run,calls


def test_downloads_selected_eligible_boundary_and_latest_not_every_weight(monkeypatch,tmp_path):
    run,calls=setup(monkeypatch,'pilot_uniform_base_s62/update_000050.pt')
    r=collector.collect(NS(kernels_status=lambda k:'COMPLETE'),'owner/job',tmp_path/'out','continuous-focus')
    assert r['complete']
    assert (tmp_path/'out'/run/'update_000050.pt').read_bytes()==b'chosen_eligible'
    assert (tmp_path/'out'/run/'update_000100.pt').read_bytes()==b'latest_resume'
    assert not(tmp_path/'out'/run/'update_000025.pt').exists()
    assert not(tmp_path/'out'/run/'update_999999.pt').exists()
    assert calls.count('continuous_focus/'+run+'_summary.json')==1


def test_cross_run_checkpoint_reference_is_rejected(monkeypatch,tmp_path):
    setup(monkeypatch,'pilot_uniform_explore_s62/update_000050.pt')
    with pytest.raises(ValueError,match='path differs'):
        collector.collect(NS(kernels_status=lambda k:'COMPLETE'),'owner/job',tmp_path/'out','continuous-focus')


def test_clock_pilot_collects_selected_and_latest_boundaries(monkeypatch,tmp_path):
    run='pilot_seconds_preserved_miss_s72'
    setup(monkeypatch,run+'/update_000050.pt',run)
    r=collector.collect(NS(kernels_status=lambda k:'COMPLETE'),'owner/job',tmp_path/'out','clock-fireball-pilot')
    assert r['complete']
    assert (tmp_path/'out'/run/'update_000050.pt').read_bytes()==b'chosen_eligible'
