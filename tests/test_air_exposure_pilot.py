from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import requests

from scripts.kaggle.prepare_air_exposure_pilot import validate_protocol,VARIANT,CASES
from scripts.kaggle.prepare_spell_interaction_pilot import CASES as SPELL_CASES
from scripts.kaggle.prepare_tower_credit_pilot import CASES as TOWER_CASES
from scripts.kaggle.prepare_credit_assignment_pilot import CASES as CREDIT_CASES
from scripts.kaggle.prepare_policy_choice_pilot import CASES as CHOICE_CASES
from scripts.attack.select_air_exposure import select_profiles
from scripts.kaggle import collect_final_evaluation as collector
from tests.test_select_continuous_focus import validation


def protocol():
    p=json.loads(Path('data/ppo_tuning/air_exposure_design_20261102.json').read_text())
    p['study_variant']=VARIANT
    return p


def test_longer_factorial_rejects_changed_budget_seed_focus_or_control():
    p=protocol();validate_protocol(p)
    for key,value in [('pilot_steps_each',102400),('pilot_seeds',[442,442]),('paired_controls',{})]:
        q=deepcopy(p);q[key]=value
        with pytest.raises(ValueError):validate_protocol(q)
    for key,value in [('spell_waste_weight',.5),('focus','uniform'),('action_delay_ms',225)]:
        q=deepcopy(p);q['cases'][3][key]=value
        with pytest.raises(ValueError):validate_protocol(q)


def test_floor_and_both_seeds_still_required_after_increasing_budget():
    p=protocol();runs=[dict(trial=t,seed=s,step=409600,target_met=True,validation=validation([6]*6))
                      for t in p['cases']for s in p['pilot_seeds']]
    assert select_profiles(runs)['gate_met']
    runs[0]['step']=102400
    with pytest.raises(ValueError):select_profiles(runs)
    runs[0]['step']=409600
    for row in runs[::2]:row['validation']=validation([0,12,12,12,12,12])
    assert not select_profiles(runs)['gate_met']


@pytest.mark.parametrize('profile,cases,seeds,wrong',[
    ('air-exposure-pilot',CASES,(442,443),'pilot_uniform01_s444'),
    ('spell-interaction-pilot',SPELL_CASES,(452,453),'pilot_king01_s454'),
    ('tower-credit-pilot',TOWER_CASES,(462,463),'pilot_tower05_01_s464'),
    ('credit-assignment-pilot',CREDIT_CASES,(472,473),'pilot_longer_s474'),
    ('policy-choice-pilot',CHOICE_CASES,(482,483),'pilot_choice_loss_s484')])
def test_collector_gets_selected150_and_last400_not_wrong_seed_or_overbudget(monkeypatch,tmp_path,profile,cases,seeds,wrong):
    prefix='continuous_focus/';data={prefix+'result.json':b'{}'}
    for name,_,_ in cases:
        for seed in seeds:
            run=f'pilot_{name}_s{seed}'
            data[prefix+run+'_summary.json']=json.dumps(dict(id=run,checkpoint=run+'/update_000150.pt')).encode()
            for update in (100,150,400,450):data[prefix+run+f'/update_{update:06d}.pt']=b'weights'
    data[prefix+wrong+'/update_000400.pt']=b'wrong'
    monkeypatch.setattr(collector,'list_outputs',lambda *a:({k:NS(url=k)for k in data},[]))
    monkeypatch.setattr(requests,'get',lambda url,**kw:NS(content=data[url],raise_for_status=lambda:None))
    root=tmp_path/'collected'
    receipt=collector.collect(NS(kernels_status=lambda k:'COMPLETE'),'owner/job',root,profile)
    assert receipt['complete']and len(list(root.glob('*/update_*.pt')))==16
    assert len(list(root.glob('*/update_000400.pt')))==8
    assert not list(root.glob('*/update_000450.pt'))and not(root/wrong).exists()
