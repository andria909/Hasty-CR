import hashlib
import json
from types import SimpleNamespace

import pytest
from scripts.kaggle import collect_ppo_coefficient_study as module

CONFIG={'workers':[[{'mode':'defense','profile':'baseline'}]]}


def fixture(monkeypatch):
    name='defense_baseline_s42';checkpoint=name+'/train/update_000100.pt';validation=name+'/validation.json'
    payload={checkpoint:b'checkpoint',validation:b'{}'}
    row=dict(trial_id=name,status='completed',training_steps=102400,checkpoint=checkpoint,validation=validation,
             checkpoint_sha256=hashlib.sha256(payload[checkpoint]).hexdigest(),validation_sha256=hashlib.sha256(payload[validation]).hexdigest())
    payload[name+'/trial_result.json']=json.dumps(row).encode()
    payload['study_result.json']=json.dumps({'status':'completed','trials':[row]}).encode()
    payload['project/private_token']=b'should never download'
    files={'coefficient_study/'+p:SimpleNamespace(url=p) for p in payload}
    monkeypatch.setattr(module,'list_outputs',lambda *a:(files,[]))
    api=SimpleNamespace(kernels_status=lambda k:'COMPLETE')
    def fetch(url,path):path.write_bytes(payload[url])
    return api,payload,fetch


def test_exact_allowlist_and_manifest_integrity(monkeypatch,tmp_path):
    api,payload,fetch=fixture(monkeypatch)
    result=module.collect(api,'owner/study',tmp_path/'out',CONFIG,fetch)
    assert result['complete']
    assert not (tmp_path/'out/project').exists()
    assert len(result['files'])==4


def test_corrupt_checkpoint_is_preserved_but_ineligible(monkeypatch,tmp_path):
    api,payload,fetch=fixture(monkeypatch)
    payload['defense_baseline_s42/train/update_000100.pt']=b'changed'
    with pytest.raises(ValueError,match='SHA mismatch'):module.collect(api,'owner/study',tmp_path/'out',CONFIG,fetch)
    assert not json.loads((tmp_path/'out/receipt.json').read_text())['complete']


def test_no_download_before_terminal(monkeypatch,tmp_path):
    api,payload,fetch=fixture(monkeypatch);api.kernels_status=lambda k:'RUNNING'
    with pytest.raises(ValueError,match='not confirmed terminal'):module.collect(api,'owner/study',tmp_path/'out',CONFIG,fetch)
    assert not (tmp_path/'out').exists()


def test_duplicate_and_unknown_trial_rejected():
    with pytest.raises(ValueError,match='Duplicate'):module.trial_ids({'workers':CONFIG['workers']*2})
    with pytest.raises(ValueError,match='Unexpected'):module.trial_ids({'workers':[[{'mode':'defense','profile':'../../secret'}]]})


def test_verified_cache_avoids_repeated_download_and_retains_source(monkeypatch,tmp_path):
    api,payload,fetch=fixture(monkeypatch)
    cache=tmp_path/'cache';module.collect(api,'owner/study',cache,CONFIG,fetch)
    def never_fetch(*args):raise AssertionError('Verified cached files must be reused')
    result=module.collect(api,'owner/study',tmp_path/'recovered',CONFIG,never_fetch,cache=cache)
    assert result['complete'] and (cache/'study_result.json').exists()
    (cache/'defense_baseline_s42/train/update_000100.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Cached file changed'):
        module.collect(api,'owner/study',tmp_path/'corrupt-cache-result',CONFIG,fetch,cache=cache)
