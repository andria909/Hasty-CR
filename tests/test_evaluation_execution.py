from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.attack.evaluation_execution import (CURRENT, LEGACY, execution_for,
    environment_kwargs, validate_execution, check_duel_execution)
from scripts.attack import evaluate_continuous_pool as pool
from scripts.attack import evaluate_full_candidates as full
from scripts.attack.verify_full_candidate_evaluation import validate_part
from tests.test_full_candidate_evaluator import fixture
from tests.test_full_candidate_verifier import part
from sim.continuous_push import ContinuousPushEnv


def v2(tmp_path):
    protocol=fixture(tmp_path)
    p=tmp_path/'evaluation_blueprint.json';spec=json.loads(p.read_text())
    spec.update(schema='precommitted-full-candidate-evaluation-v2',execution=deepcopy(CURRENT))
    spec['evaluation_runtime'].update({k:v for k,v in CURRENT.items()if k!='schema'})
    p.write_text(json.dumps(spec))
    protocol.update(schema='bound-full-candidate-evaluation-v2',execution=deepcopy(CURRENT),blueprint_sha256=pool.sha(p))
    return protocol,spec


def test_legacy_environment_defaults_do_not_drift_to_current_training_defaults():
    env=ContinuousPushEnv(**environment_kwargs(execution_for({})))
    assert env.action_delay_ms==225
    assert env.profile['action_cooldown_ms']==2100
    assert env.profile['cannon_center_only']==0
    new=ContinuousPushEnv(**environment_kwargs(CURRENT))
    assert new.action_delay_ms==1350 and new.profile['action_cooldown_ms']==0
    assert new.profile['cannon_center_only']==1


@pytest.mark.parametrize('key,value', [('action_delay_ms',True),('action_delay_ms',-1),
    ('decision_interval_ms',225),('decision_interval_ms',0),('physics_tick_ms',100),
    ('cannon_placement_mode','maybe'),('action_cooldown_ms',float('nan'))])
def test_bad_execution_rejected(key,value):
    with pytest.raises(ValueError):validate_execution(dict(CURRENT,**{key:value}))


def test_execution_bound_to_blueprint_not_user_mutable_defaults(tmp_path):
    protocol,spec=v2(tmp_path)
    assert full.validate_protocol(protocol,tmp_path)==spec
    protocol['execution']['action_delay_ms']=225
    with pytest.raises(ValueError,match='execution differs'):full.validate_protocol(protocol,tmp_path)
    del protocol['execution']
    with pytest.raises(ValueError,match='requires explicit'):full.validate_protocol(protocol,tmp_path)


def test_new_verifier_rejects_old_rows_and_outside_cannon(tmp_path):
    protocol,spec=v2(tmp_path)
    job=full.schedule(protocol,spec)[0];report=part(job,protocol,spec)
    with pytest.raises(ValueError,match='Per-game execution'):validate_part(report,job,protocol,spec)
    for row in report['rows']:row.update(execution=deepcopy(CURRENT),cannon_outside_center=0)
    validate_part(report,job,protocol,spec)
    report['rows'][0]['cannon_outside_center']=1
    with pytest.raises(ValueError,match='outside'):validate_part(report,job,protocol,spec)


@pytest.mark.parametrize('execution',[CURRENT,LEGACY,dict(CURRENT,action_delay_ms=2025)])
def test_actual_duel_queue_parallel_slots_reservation_and_central_cells(monkeypatch,execution):
    env=ContinuousPushEnv();decisions=[];commands=[]
    def greedy(network,match,side,mode,mask,device,reserved_milli=0):
        legal=np.flatnonzero(mask[1:])+1
        # Try Cannon first, then cheap cards so reuse is exercised repeatedly.
        selected=0
        for card in ('cannon','skeletons','electro_spirit','zap','knight','hog_rider','musketeer','fireball'):
            options=[int(a)for a in legal if match.players[side].hand[(a-1)//576]==card]
            if options:
                selected=options[0];break
        if selected:
            slot,x,y=pool.ClashEnv.decode(selected)
            assert env.cards[match.players[side].hand[slot]].cost*1000<=match.players[side].elixir-reserved_milli
            decisions.append((side,slot,match.elapsed_ms))
        return selected
    monkeypatch.setattr(pool,'choose',greedy)
    row=pool.duel((None,'physical_live'),(None,'physical_live'),'cpu',813901,1,
                 env.cards,env.spells,execution=execution,on_command=commands.append)
    check_duel_execution(row,execution)
    assert commands and all(c['accepted']for c in commands)
    if execution['cannon_placement_mode']=='central_defense_v1':
        cannons=[c for c in commands if c['card']=='cannon']
        assert {c['side']for c in cannons}=={1,-1}
        assert all(6<=c['x']<=11 and 18<=c['y']<=23 for c in cannons)
    by_slot={};reused=[];parallel=False
    for side,slot,tick in decisions:
        previous=by_slot.get((side,slot))
        if previous is not None:
            actual_delay=((execution['action_delay_ms']+49)//50)*50
            gap=tick-previous-actual_delay
            assert gap>=execution['action_cooldown_ms']
            reused.append(gap)
        parallel|=any(s==side and other!=slot and 0<tick-at<execution['action_delay_ms']
                      for (s,other),at in by_slot.items())
        by_slot[side,slot]=tick
    assert reused
    if execution['action_delay_ms']>=1000:assert parallel
    if execution['action_cooldown_ms']==0:assert min(reused)<2100
    broken=deepcopy(row);broken['actual_command_delay_max_ms']+=50
    with pytest.raises(ValueError,match='Actual command delay'):check_duel_execution(broken,execution)


def test_actual_new_blueprint_has_no_stale_runtime_and_new_holdout_seeds():
    path=Path('data/ppo_tuning/central_tower_evaluation_20261028/protocol.json')
    spec=json.loads(path.read_text());assert execution_for(spec)==CURRENT
    old=json.loads(Path('data/ppo_tuning/full_candidate_evaluation_20261027/protocol.json').read_text())
    def seeds(p):return {s for t in p['test_sets']for s in t['seeds']}|set(p['tournament']['seeds'])
    assert not seeds(spec)&seeds(old)
    spec['evaluation_runtime']['action_delay_ms']=225
    with pytest.raises(ValueError,match='contradicts'):execution_for(spec)
