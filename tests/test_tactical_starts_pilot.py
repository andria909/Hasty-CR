from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest
import requests

from scripts.kaggle.prepare_tactical_starts_pilot import protocol_from,validate_protocol
from scripts.attack.verify_tactical_starts_pilot import audit_exposure
from scripts.kaggle import collect_final_evaluation as collector


def inputs():
    trial=dict(id='medium',tower_support_weight=.15,cannon_center_only=1,action_cooldown_ms=0,
        tower_damage_all=1,action_delay_ms=1350,validation_action_delay_ms=1350,history_frames=1,
        warmstart_updates=0,supervised_replay_weight=0.,spell_waste_weight=.1,missed_fireball_penalty=0.)
    return dict(pilot_steps_each=102400,total_pilot_steps=819200),dict(
        central_tower_audit={'verified':True},pilot_steps_verified=819200,full_gate_met=False,
        pilot_selection={'ranking':[dict(trial=dict(trial,id=str(i)),score=.3-i*.01)for i in range(4)]})


def test_only_start_fraction_changes_after_verified_predecessor():
    old,proof=inputs();p=protocol_from(old,proof)
    assert p['pilot_seeds']==[322,323]
    assert [r['tactical_start_fraction']for r in p['cases']]==[0,.1,.25,.5]
    assert all(r['tower_support_weight']==.15 for r in p['cases'])
    p['cases'][2]['tower_support_weight']=.4
    with pytest.raises(ValueError,match='Confounded'):validate_protocol(p)
    p=protocol_from(old,proof)
    for row in p['cases']:row['tower_support_weight']=.4
    with pytest.raises(ValueError,match='predecessor'):validate_protocol(p)


@pytest.mark.parametrize('change', [dict(pilot_steps_verified=817152),dict(full_gate_met=True),dict(central_tower_audit={})])
def test_incomplete_or_successful_predecessor_cannot_launch_replacement_pilot(change):
    old,proof=inputs();proof.update(change)
    with pytest.raises(ValueError):protocol_from(old,proof)


def exposure():
    config=dict(tactical_start_fraction=.25,tower_support_weight=.15,spell_waste_weight=.1,
        continuous_profile=dict(cannon_center_only=1,action_cooldown_ms=0,tower_damage_all=1,missed_fireball_penalty=0.))
    row=dict(tactical_start_fraction=.25,tactical_start=None,split='train',
        cannon_center_only=1,cannon_outside_center=0,action_cooldown_ms=0,tower_damage_reward_mode='all_damage',
        tower_support_weight=.15,tower_support_paid_value=2.,tower_support_reward=.3,
        spell_waste_weight=.1,empty_spell_elixir=6.,empty_spell_penalty=.6,
        empty_spell_elixir_by_card={'zap':2.,'fireball':4.})
    start=dict(schema='paid-tactical-start-v1',synthetic_history=True,own_hand=['a','b','c','d'],
        own_queue=['e','f','g','h'],own_princess_mask=0,family='air',
        enemy_setup=[dict(elixir_before=10000,elixir_after=5000)])
    return config,[row,dict(row,tactical_start=start)],[dict(row,split='validation')]


def test_real_exposure_required_and_validation_must_stay_ordinary():
    c,t,v=exposure();r=audit_exposure(c,t,v,.25,.15)
    assert r['counts']==dict(ordinary=1,tactical=1)
    with pytest.raises(ValueError,match='exposure'):audit_exposure(c,t[:1],v,.25,.15)
    with pytest.raises(ValueError,match='Validation'):audit_exposure(c,t,[dict(v[0],tactical_start=t[1]['tactical_start'])],.25,.15)
    broken=deepcopy(t);broken[1]['tactical_start']['enemy_setup'][0]['elixir_after']=-1000
    with pytest.raises(ValueError,match='budget'):audit_exposure(c,broken,v,.25,.15)


def test_collector_includes_only_new_study_selected_and_final_checkpoints(monkeypatch,tmp_path):
    run='pilot_low_s322';prefix='continuous_focus/'
    data={prefix+'result.json':b'{}',prefix+run+'_summary.json':json.dumps(dict(id=run,checkpoint=run+'/update_000050.pt')).encode(),
          prefix+run+'/update_000050.pt':b'selected',prefix+run+'/update_000100.pt':b'final',
          prefix+'pilot_low_s999/update_000100.pt':b'wrong study',prefix+'project/secret.txt':b'never'}
    monkeypatch.setattr(collector,'list_outputs',lambda api,k:({n:NS(url=n)for n in data},[]))
    monkeypatch.setattr(requests,'get',lambda url,**kw:NS(content=data[url],raise_for_status=lambda:None))
    out=tmp_path/'out'
    receipt=collector.collect(NS(kernels_status=lambda k:'COMPLETE'),'owner/job',out,profile='tactical-starts-pilot')
    assert receipt['complete'] and (out/run/'update_000050.pt').read_bytes()==b'selected'
    assert (out/run/'update_000100.pt').read_bytes()==b'final'
    assert not (out/'pilot_low_s999').exists() and not (out/'project').exists()
