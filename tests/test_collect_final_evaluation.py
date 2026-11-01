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
                                                  ('continuous-pool','continuous_pool_evaluation/','result.json'),
                                                  ('continuous-focus','continuous_focus/','result.json'),
                                                  ('sequence-pilot','continuous_focus/','result.json'),
                                                  ('pending-inputs-pilot','continuous_focus/','result.json'),
                                                  ('native-fireball-slot-pilot','continuous_focus/','result.json'),
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


@pytest.mark.parametrize('profile',['bank-correction-pair','bank-fit-depth'])
def test_correction_pair_collects_only_the_eight_declared_weight_sets(monkeypatch,tmp_path,profile):
    prefix='bank_fit_depth/'if profile=='bank-fit-depth'else'bank_correction_pair/'
    names=([f'{kind}{updates}_s{seed}'for kind in ('mixed','isolated')for updates in (800,8000)for seed in (402,403)]
        if profile=='bank-fit-depth'else[f'{kind}_kl{kl}_s{seed}'for kind in ('control','corrections')for kl in ('001','01')for seed in (402,403)])
    data={prefix+'result.json':b'{}'}
    for name in names:
        for artifact in ('result.json','updates.json','policy.pt'):data[prefix+name+'/'+artifact]=b'{}'
    data[prefix+'control_kl001_s404/policy.pt']=b'not this study'
    data[prefix+'private.txt']=b'never'
    monkeypatch.setattr(collector,'list_outputs',lambda api,kernel:({name:NS(url=name)for name in data},[]))
    monkeypatch.setattr(requests,'get',lambda url,**kw:NS(content=data[url],raise_for_status=lambda:None))
    out=tmp_path/'collected'
    result=collector.collect(NS(kernels_status=lambda k:'COMPLETE'),'owner/job',out,profile=profile)
    assert result['complete']
    assert len(list(out.glob('*/policy.pt')))==8
    assert not(out/'control_kl001_s404').exists()and not(out/'private.txt').exists()


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
    def collect(api,kernel,destination,profile,cache=None):
        calls.append((kernel,destination))
        if len(calls)==1:
            destination.mkdir();(destination/'partial.json').write_text('preserve me')
            (destination/'collection_receipt.json').write_text(json.dumps(dict(kernel=kernel,remote_status='COMPLETE',complete=False,files={})))
            response=requests.Response();response.status_code=404
            raise requests.HTTPError(response=response)
        assert destination==output and cache!=output
        assert (cache/'partial.json').read_text()=='preserve me'
        return dict(complete=True)
    monkeypatch.setattr(collector,'collect',collect)
    collector.main()
    assert [kernel for kernel,_ in calls]==['owner/same-job','owner/same-job']
    assert sleeps==[1200]
