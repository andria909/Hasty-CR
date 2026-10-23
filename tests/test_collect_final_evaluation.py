import json
from types import SimpleNamespace as NS

import pytest
import requests

from scripts.kaggle import collect_final_evaluation as collector


@pytest.mark.parametrize('profile,prefix,result', [('defense','evaluation/','evaluation_result.json'),
                                                  ('strategy','strategy_evaluation/','result.json')])
def test_only_declared_terminal_report_files_are_collected(monkeypatch,tmp_path,profile,prefix,result):
    data={prefix+result:b'{"status":"completed"}',prefix+'project/secret.py':b'never',prefix+'policy.pt':b'never'}
    if profile=='strategy':data[prefix+'test.json']=b'{"episodes":804}'
    else:data[prefix+'ppo_defense.json']=b'{"episodes":708}'
    monkeypatch.setattr(collector,'list_outputs',lambda api,kernel:({name:NS(url=name)for name in data},[]))
    monkeypatch.setattr(requests,'get',lambda url,**kw:NS(content=data[url],raise_for_status=lambda:None))
    out=tmp_path/'collected';r=collector.collect(NS(kernels_status=lambda k:'COMPLETE'),'owner/job',out,profile=profile)
    assert r['complete']and not r['remote_job_changed']
    assert (out/result).is_file()and not (out/'project').exists()and not (out/'policy.pt').exists()


def test_running_job_is_never_downloaded(tmp_path):
    with pytest.raises(ValueError,match='not terminal'):
        collector.collect(NS(kernels_status=lambda k:'RUNNING'),'owner/job',tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_changed_remote_state_during_listing_is_rejected(monkeypatch,tmp_path):
    states=iter(['COMPLETE','RUNNING'])
    monkeypatch.setattr(collector,'list_outputs',lambda api,kernel:({},[]))
    with pytest.raises(ValueError,match='changed during listing'):
        collector.collect(NS(kernels_status=lambda k:next(states)),'owner/job',tmp_path/'out')
    assert not (tmp_path/'out').exists()
