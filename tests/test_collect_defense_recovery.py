import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from scripts.kaggle import collect_defense_recovery as collector


def environment(monkeypatch,*,state='ERROR',selected=None):
    result=dict(schema='defense-semantic-recovery-v1',status='failed',selected_artifacts=selected or {})
    data={'recovery/recovery_result.json':json.dumps(result).encode(),
          'recovery/ppo/update_001150.pt':b'old',
          'recovery/ppo/update_001200.pt':b'latest',
          'recovery/selected/ppo.pt':b'chosen',
          'recovery/ppo/metrics.jsonl':b'{}\n',
          'recovery/project/secret.py':b'must not download'}
    monkeypatch.setattr(collector,'list_outputs',lambda api,kernel:(
        {k:NS(url=k)for k in data},['terminal log']))
    api=NS(kernels_status=lambda kernel:state)
    def fetch(url,path):
        with path.open('xb')as stream:stream.write(data[url])
    return api,fetch


def test_partial_run_preserves_latest_checkpoint_metrics_and_original_failure(monkeypatch,tmp_path):
    api,fetch=environment(monkeypatch)
    out=tmp_path/'collected'
    result=collector.collect(api,'owner/kernel',out,fetch)
    assert result['complete'] and result['result_status']=='failed'
    assert not result['quality_accepted'] and not result['remote_job_changed']
    assert (out/'ppo/update_001200.pt').read_bytes()==b'latest'
    assert not (out/'ppo/update_001150.pt').exists()
    assert not (out/'project').exists() and not (out/'selected').exists()
    assert (out/'ppo/metrics.jsonl').exists()


def test_selected_report_hash_verified(monkeypatch,tmp_path):
    digest=hashlib.sha256(b'chosen').hexdigest()
    api,fetch=environment(monkeypatch,selected={'ppo':{'path':'selected/ppo.pt','sha256':digest}})
    result=collector.collect(api,'owner/kernel',tmp_path/'out',fetch)
    assert result['files']['selected/ppo.pt']['reported_hash_verified']


def test_tampered_selected_weight_keeps_incomplete_receipt(monkeypatch,tmp_path):
    api,fetch=environment(monkeypatch,selected={'ppo':{'path':'selected/ppo.pt','sha256':'a'*64}})
    with pytest.raises(ValueError,match='SHA mismatch'):
        collector.collect(api,'owner/kernel',tmp_path/'out',fetch)
    assert not json.loads((tmp_path/'out/collection_receipt.json').read_text())['complete']


def test_running_job_is_observation_only(monkeypatch,tmp_path):
    api,fetch=environment(monkeypatch,state='RUNNING')
    with pytest.raises(ValueError,match='not terminal'):
        collector.collect(api,'owner/kernel',tmp_path/'out',fetch)
    assert not (tmp_path/'out').exists()


def test_report_path_escape_refused(monkeypatch,tmp_path):
    api,fetch=environment(monkeypatch,selected={'ppo':{'path':'../escape.pt','sha256':'a'*64}})
    with pytest.raises(ValueError,match='Unsafe'):
        collector.collect(api,'owner/kernel',tmp_path/'out',fetch)


def test_separate_account_grpo_result_and_selected_artifact(monkeypatch,tmp_path):
    chosen=b'grpo selected'
    report=dict(schema='defense-grpo-account2-v1',status='grpo_target_completed_validation_selected',
        selected_artifact=dict(path='selected/grpo.pt',sha256=hashlib.sha256(chosen).hexdigest()))
    data={'account2/account2_result.json':json.dumps(report).encode(),
          'account2/selected/grpo.pt':chosen,'account2/grpo/updates.jsonl':b'{}\n'}
    monkeypatch.setattr(collector,'list_outputs',lambda api,kernel:({k:NS(url=k)for k in data},[]))
    def fetch(url,path):
        with path.open('xb')as stream:stream.write(data[url])
    receipt=collector.collect(NS(kernels_status=lambda kernel:'COMPLETE'),'other/kernel',tmp_path/'out',fetch,profile='grpo-account2')
    assert receipt['complete'] and receipt['files']['selected/grpo.pt']['reported_hash_verified']
    assert (tmp_path/'out/account2_result.json').is_file()
    assert (tmp_path/'out/grpo/updates.jsonl').is_file()
