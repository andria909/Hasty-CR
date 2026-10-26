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


def test_history_pilot_keeps_the_selected_temporal_checkpoint(monkeypatch,tmp_path):
    run='pilot_history2_air_golem_s82';setup(monkeypatch,run+'/update_000050.pt',run)
    r=collector.collect(NS(kernels_status=lambda k:'COMPLETE'),'owner/job',tmp_path/'out','history-pilot')
    assert r['complete']and(tmp_path/'out'/run/'update_000050.pt').exists()


def test_selected_full_collection_omits_unselected_weight_archives(monkeypatch,tmp_path):
    prefix='continuous_ppo/';a='full_balanced_s44';b='full_strong_s44'
    chosen=[a+'/best_validation_001000.pt',b+'/best_validation_002750.pt']
    result=dict(status='complete',full_runs=[dict(id=n,checkpoint=p)for n,p in zip((a,b),chosen)])
    data={prefix+'result.json':json.dumps(result).encode(),**{prefix+p:b'chosen'for p in chosen},
        prefix+a+'/best_validation_000250.pt':b'unselected',prefix+a+'/update_004000.pt':b'archive'}
    monkeypatch.setattr(collector,'list_outputs',lambda api,kernel:({n:NS(url=n)for n in data},[]))
    monkeypatch.setattr(requests,'get',lambda url,**kw:NS(content=data[url],raise_for_status=lambda:None))
    r=collector.collect(NS(kernels_status=lambda k:'COMPLETE'),'owner/job',tmp_path/'out','continuous-ppo-selected')
    assert r['complete']
    assert all((tmp_path/'out'/p).read_bytes()==b'chosen'for p in chosen)
    assert not(tmp_path/'out'/a/'update_004000.pt').exists()


@pytest.mark.parametrize('overflow',['00','15'])
@pytest.mark.parametrize('balance',['00','10'])
@pytest.mark.parametrize('seed',[102,103])
def test_bank_control_preserves_selected_and_latest(monkeypatch,tmp_path,overflow,balance,seed):
    run=f'pilot_bank_overflow{overflow}_balance{balance}_s{seed}'
    setup(monkeypatch,run+'/update_000050.pt',run)
    result=collector.collect(NS(kernels_status=lambda k:'COMPLETE'),'owner/job',tmp_path/'out','bank-conservation-pilot')
    assert result['complete']
    assert (tmp_path/'out'/run/'update_000050.pt').read_bytes()==b'chosen_eligible'
    assert (tmp_path/'out'/run/'update_000100.pt').read_bytes()==b'latest_resume'
    assert not (tmp_path/'out'/run/'update_999999.pt').exists()


@pytest.mark.parametrize('run',['full_board_150_s64','full_bank_overflow00_balance10_s64'])
def test_selected_full_profile_collects_both_study_families(monkeypatch,tmp_path,run):
    setup(monkeypatch,run+'/update_000050.pt',run)
    result=collector.collect(NS(kernels_status=lambda k:'COMPLETE'),'owner/job',tmp_path/'out','selected-continuous-full')
    assert result['complete']
    assert (tmp_path/'out'/run/'update_000050.pt').read_bytes()==b'chosen_eligible'
    assert (tmp_path/'out'/run/'update_000100.pt').read_bytes()==b'latest_resume'


def test_air_warmstart_collection_keeps_pretraining_provenance(monkeypatch,tmp_path):
    run='pilot_bc200_kl000_s122';prefix='continuous_focus/'
    data={prefix+'result.json':b'{}',prefix+run+'_summary.json':json.dumps(dict(id=run,checkpoint=run+'/update_000050.pt')).encode(),
        prefix+run+'/update_000050.pt':b'ppo',prefix+run+'_warmstart/policy.pt':b'warm',
        prefix+run+'_warmstart/result.json':b'{"supervised_updates":200}',prefix+run+'_warmstart/updates.json':b'[]'}
    monkeypatch.setattr(collector,'list_outputs',lambda api,kernel:({n:NS(url=n)for n in data},[]))
    monkeypatch.setattr(requests,'get',lambda url,**kw:NS(content=data[url],raise_for_status=lambda:None))
    result=collector.collect(NS(kernels_status=lambda k:'COMPLETE'),'owner/job',tmp_path/'out','air-warmstart-pilot')
    assert result['complete']
    assert (tmp_path/'out'/(run+'_warmstart')/'policy.pt').read_bytes()==b'warm'
    assert (tmp_path/'out'/(run+'_warmstart')/'result.json').exists()
    assert (tmp_path/'out'/(run+'_warmstart')/'updates.json').exists()
