from types import SimpleNamespace
import json
import hashlib

import numpy as np
import pytest
import torch

import sim.eval_defense_duel as duel
from sim import arena


def test_outcomes_handle_swapped_seats_and_unknown_separately():
    assert duel.outcome('bottom',1,True)=='win'
    assert duel.outcome('bottom',-1,True)=='loss'
    assert duel.outcome('draw',-1,True)=='draw'
    assert duel.outcome(None,1,False)=='unknown'
    summary=duel.summarize([dict(outcome=o,quality_flags=[])for o in ['win','loss','draw','unknown']])
    assert summary['win_rate_all_matches']==.25
    assert summary['win_rate_known_matches']==pytest.approx(1/3)
    assert summary['unknown']==1 and summary['quality_passed'] is None


def test_physical_observation_rotates_positions_hides_history_and_swaps_top_lanes(monkeypatch):
    entity=SimpleNamespace(pos=arena.Point(4000,23000))
    match=SimpleNamespace(battle=SimpleNamespace(entities={1:entity}),play_log=[{'secret':1}],
        tower_fractions=lambda side:dict(left=.2,right=.8,king=1))
    monkeypatch.setattr(duel,'observe',lambda m,s:dict(y=m.battle.entities[1].pos.y,
        side=s,history=m.play_log,left=m.tower_fractions(s)['left']))
    physical=duel.policy_observation(match,-1,'physical_live')
    assert physical==dict(y=8000,side=-1,history=[],left=.8)
    legacy=duel.policy_observation(match,-1,'legacy_training')
    assert legacy['y']==23000 and legacy['history']==[{'secret':1}] and legacy['left']==.2
    assert match.battle.entities[1].pos.y==23000
    with pytest.raises(ValueError):duel.policy_observation(match,1,'guess')


def test_manifest_requires_approval_hash_and_explicit_observation_contract(tmp_path):
    weights=tmp_path/'weights.pt';weights.write_bytes(b'fake-checkpoint')
    digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    row=dict(label='v25',path='weights.pt',sha256=digest,approved=True,observation_mode='legacy_training')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'baselines':[row]}))
    assert duel.approved_baselines(manifest)[0]['path']==weights
    for changes in [dict(approved=False),dict(sha256='bad'),dict(observation_mode='latest')]:
        manifest.write_text(json.dumps({'baselines':[dict(row,**changes)]}))
        with pytest.raises(ValueError):duel.approved_baselines(manifest)


def test_fake_match_command_delay_and_seat_swap_are_paired(monkeypatch):
    instances=[]
    class FakeMatch:
        def __init__(self,cards,decks,seed,spells):
            self.elapsed_ms=0;self.finished=False;self.result=None
            self.seed=seed;self.commands=[];self.missing_spawns=set()
            self.battle=SimpleNamespace(unmodelled_projectiles=[])
            self.players={s:SimpleNamespace(hand=['knight']*4)for s in (1,-1)}
            self.towers={s:{lane:SimpleNamespace(hitpoints=100)for lane in ('left','right','king')}for s in (1,-1)}
            instances.append(self)
        def play_card(self,side,card,at):
            self.commands.append((self.elapsed_ms,side,card));return True
        def step(self,ms):
            self.elapsed_ms+=ms
            if self.elapsed_ms>=200:self.finished=True;self.result='bottom'
    class Network:
        def __call__(self,p,s):return torch.tensor([[0.,1.]]),torch.zeros(1)
    monkeypatch.setattr(duel,'mechanical_mask',lambda m,c,sp,s,pending:np.array([True,not pending]))
    monkeypatch.setattr(duel,'policy_observation',lambda *a:dict(planes=np.zeros((1,1,1),np.float32),scalars=np.zeros(1,np.float32)))
    results=[duel.run_match(Network(),Network(),'cpu','physical_live','legacy_training',42,seat,{}, {},
                           delay_ms=100,max_ms=1000,match_factory=FakeMatch)for seat in (1,-1)]
    assert [r['outcome']for r in results]==['win','loss']
    assert [m.seed for m in instances]==[42,42]
    assert all(len(m.commands)==2 and all(c[0]==100 for c in m.commands)for m in instances)
    assert results[0]['quality_flags']==results[1]['quality_flags']
    # Actual run_match rows feed the frozen gate via the explicit adapter.
    from sim.defense_protocol import EvaluationProtocol,acceptance_gate
    protocol=EvaluationProtocol(test_scenario_ids=('s',),paired_seeds=(42,43,44,45,46),defense_seeds=(1,))
    report=dict(baselines=[dict(label='v25',sha256='a'*64,matches=results)])
    rows=duel.protocol_rows(report)
    assert rows[0]['observed_result']=='win'
    assert rows[0]['result']=='unknown' and rows[0]['quality_review_required']
    gate=acceptance_gate(protocol,[],rows,coverage_complete=True)
    assert gate['status']=='undetermined'
    assert 'unexpected_duplicate_or_invalid_match' in gate['undetermined_reasons']
    assert gate['head_to_head']['v25']['games']==0



def test_protocol_manifest_accepts_canonical_ids_and_requires_mode(tmp_path,monkeypatch):
    import sim.defense_protocol as protocol
    path=tmp_path/'model.pt';path.write_bytes(b'fake')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(schema='defense-baselines-v1',checkpoints=[dict(id='v25',path='model.pt',sha256=digest)])))
    calls=[]
    def approved(source,project_root):
        calls.append((source,project_root))
        return {'v25':dict(id='v25',path=str(path),sha256=digest)}
    monkeypatch.setattr(protocol,'load_baseline_allowlist',approved)
    rows=duel.approved_baselines(manifest,project_root=tmp_path,default_mode='physical_live')
    assert rows[0]['id']=='v25' and rows[0]['label']=='v25'
    assert rows[0]['observation_mode']=='physical_live'
    assert calls==[(manifest,tmp_path)]
    with pytest.raises(ValueError):duel.approved_baselines(manifest,project_root=tmp_path)
