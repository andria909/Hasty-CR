from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest
import requests

from scripts.kaggle.prepare_policy_adaptation_pilot import validate_protocol,IDS,INITIALS
from scripts.kaggle import collect_final_evaluation as collector


def protocol():
    common=dict(action_delay_ms=1350,validation_action_delay_ms=1350,cannon_center_only=1,
        action_cooldown_ms=0,tower_damage_all=1,source_radius_reach=1,hog_sight_clipping=1,
        target_lock_in_range=0,tower_support_weight=.15,spell_waste_weight=.1,crown_only_weight=0.,
        dormant_king_weight=0.,tactical_start_fraction=0.,warmstart_updates=0,supervised_replay_weight=0.,
        history_frames=1,card_balance_weight=.03)
    return dict(study_variant='policy_adaptation_v1',pilot_seeds=[412,413],pilot_steps_each=102400,
        total_pilot_steps=819200,cases=[dict(common,id=n,initial_id=m)for n,m in zip(IDS,INITIALS)])


def test_initializer_comparison_rejects_reward_changes_and_wrong_assignment():
    p=protocol();validate_protocol(p)
    for key,value in [('spell_waste_weight',.4),('initial_id','parent'),('supervised_replay_weight',1.)]:
        changed=deepcopy(p);changed['cases'][3][key]=value
        with pytest.raises(ValueError):validate_protocol(changed)


def test_all_eight_adaptation_runs_keep_selected_latest_and_all_initial_refs(monkeypatch,tmp_path):
    prefix='continuous_focus/';data={prefix+'result.json':b'{}'}
    for initial in INITIALS:data[prefix+initial+'_reference.json']=b'{}'
    for name in IDS:
        for seed in (412,413):
            run=f'pilot_{name}_s{seed}'
            data[prefix+run+'_summary.json']=json.dumps(dict(id=run,checkpoint=run+'/update_000050.pt')).encode()
            for artifact in ('config.json','metrics.jsonl','training_result.json','update_000050.pt','update_000100.pt'):
                data[prefix+run+'/'+artifact]=b'{}'
    data[prefix+'pilot_bank_s414/update_000100.pt']=b'wrong seed'
    data[prefix+'unrelated/private.txt']=b'not for collection'
    monkeypatch.setattr(collector,'list_outputs',lambda *a:({name:NS(url=name)for name in data},[]))
    monkeypatch.setattr(requests,'get',lambda url,**kw:NS(content=data[url],raise_for_status=lambda:None))
    out=tmp_path/'collected';r=collector.collect(NS(kernels_status=lambda k:'COMPLETE'),'owner/job',out,'policy-adaptation-pilot')
    assert r['complete']and len(list(out.glob('*/update_*.pt')))==16
    assert len(list(out.glob('*_reference.json')))==4
    assert not(out/'pilot_bank_s414').exists()and not(out/'unrelated').exists()
