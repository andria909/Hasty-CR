import hashlib
import json
from types import SimpleNamespace
import pytest
from scripts.kaggle import collect_ppo_selected_full as module


def fixture(monkeypatch,complete=True):
    checkpoint='train/update_002000.pt' if complete else 'train/update_001400.pt'
    payload={checkpoint:b'weights','validation.json':json.dumps(dict(training_steps=2048000,checkpoint_sha256=hashlib.sha256(b'weights').hexdigest())).encode()}
    result=dict(status='completed' if complete else 'incomplete_budget',step=2048000 if complete else 1433600,update=2000 if complete else 1400,
                checkpoint=checkpoint,checkpoint_sha256=hashlib.sha256(b'weights').hexdigest(),test_used_for_selection=False)
    if complete:result['validation_sha256']=hashlib.sha256(payload['validation.json']).hexdigest()
    payload['result.json']=json.dumps(result).encode();payload['project/private']=b'forbidden';payload['train/update_000100.pt']=b'old'
    monkeypatch.setattr(module,'list_outputs',lambda *args:({'selected_full/'+k:SimpleNamespace(url=k) for k in payload},[]))
    api=SimpleNamespace(kernels_status=lambda kernel:'COMPLETE')
    def fetch(url,path):path.write_bytes(payload[url])
    return api,payload,fetch,checkpoint


def test_complete_outputs_require_matching_checkpoint_and_validation(monkeypatch,tmp_path):
    api,payload,fetch,checkpoint=fixture(monkeypatch)
    receipt=module.collect(api,'owner/full',tmp_path/'out',fetch)
    assert receipt['complete'] and checkpoint in receipt['files']
    assert not (tmp_path/'out/project').exists()
    assert 'train/update_000100.pt' not in receipt['files']


def test_incomplete_budget_preserves_last_weights_without_claiming_completion(monkeypatch,tmp_path):
    api,payload,fetch,checkpoint=fixture(monkeypatch,False)
    receipt=module.collect(api,'owner/full',tmp_path/'out',fetch)
    assert not receipt['complete'] and receipt['training_status']=='incomplete_budget'
    assert (tmp_path/'out'/checkpoint).read_bytes()==b'weights'


def test_checkpoint_corruption_keeps_failed_receipt(monkeypatch,tmp_path):
    api,payload,fetch,checkpoint=fixture(monkeypatch);payload[checkpoint]=b'corrupt'
    with pytest.raises(ValueError,match='missing or changed'):module.collect(api,'owner/full',tmp_path/'out',fetch)
    assert not json.loads((tmp_path/'out/receipt.json').read_text())['complete']


def test_active_job_does_not_download_or_create_outputs(monkeypatch,tmp_path):
    api,payload,fetch,checkpoint=fixture(monkeypatch);api.kernels_status=lambda kernel:'RUNNING'
    with pytest.raises(ValueError,match='not confirmed terminal'):module.collect(api,'owner/full',tmp_path/'out',fetch)
    assert not (tmp_path/'out').exists()


def test_full_collector_can_reuse_verified_cache_without_overwriting_it(monkeypatch,tmp_path):
    api,payload,fetch,checkpoint=fixture(monkeypatch)
    cache=tmp_path/'cache';module.collect(api,'owner/full',cache,fetch)
    def no_fetch(*args):raise AssertionError('No repeated download expected')
    result=module.collect(api,'owner/full',tmp_path/'recovered',no_fetch,cache=cache)
    assert result['complete'] and (cache/checkpoint).read_bytes()==b'weights'


def test_delayed_observer_preserves_schedule_with_bounded_waits(monkeypatch):
    now=[100.];sleeps=[]
    monkeypatch.setattr(module.time,'monotonic',lambda:now[0])
    def sleep(seconds):sleeps.append(seconds);now[0]+=seconds
    monkeypatch.setattr(module.time,'sleep',sleep)
    module.initial_wait(149.25)
    assert now[0]==249.25 and all(0<seconds<=60 for seconds in sleeps)
    assert sum(sleeps)==149.25
    with pytest.raises(ValueError):module.initial_wait(float('inf'))
    with pytest.raises(ValueError):module.initial_wait(-1)
