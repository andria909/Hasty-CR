import json
from types import SimpleNamespace as NS

import pytest
import requests

from scripts.kaggle import collect_final_evaluation as collector


@pytest.mark.parametrize('profile,prefix,result', [('defense','evaluation/','evaluation_result.json'),
                                                  ('strategy','strategy_evaluation/','result.json'),
                                                  ('fireball','fireball_refinement/','result.json'),
                                                  ('fireball-selectivity','fireball_selectivity/','result.json'),
                                                  ('fireball-balanced','fireball_balanced/','result.json'),
                                                  ('balanced-match-validation','balanced_match_validation/','result.json'),
                                                  ('native-fireball-evaluation','native_fireball_evaluation/','result.json'),
                                                  ('card-usage','card_usage/','result.json'),
                                                  ('card-balance-study','card_balance_study/','result.json'),
                                                  ('continuous-ppo','continuous_ppo/','result.json'),
                                                  ('fireball-evaluation','fireball_evaluation/','result.json')])
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


def test_terminal_download_retry_preserves_partial_files_and_same_notebook(monkeypatch,tmp_path):
    import sys
    output=tmp_path/'collected';calls=[];sleeps=[]
    monkeypatch.setattr(sys,'argv',['collect','--kernel','owner/same-job','--output',str(output),'--profile','card-usage'])
    monkeypatch.setattr(collector,'configure_http_timeout',lambda:None)
    monkeypatch.setattr(collector,'make_api_client',lambda user:NS(kernels_status=lambda kernel:'COMPLETE'))
    monkeypatch.setattr(collector.time,'sleep',lambda seconds:sleeps.append(seconds))
    def collect(api,kernel,destination,profile):
        calls.append((kernel,destination))
        if len(calls)==1:
            destination.mkdir();(destination/'partial.json').write_text('preserve me')
            response=requests.Response();response.status_code=404
            raise requests.HTTPError(response=response)
        assert destination!=output
        assert (output/'partial.json').read_text()=='preserve me'
        return dict(complete=True)
    monkeypatch.setattr(collector,'collect',collect)
    collector.main()
    assert [kernel for kernel,_ in calls]==['owner/same-job','owner/same-job']
    assert sleeps==[1200]
