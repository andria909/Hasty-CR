from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

import sim.eval_defense_duel as duel


def test_joint_task_rule_uses_only_living_public_bodies():
    entities={1:NS(side=-1,alive=True,is_tower=True),
              2:NS(side=-1,alive=False,is_tower=False),
              3:NS(side=-1,alive=True,object_role='pickup_effect_carrier')}
    match=NS(battle=NS(entities=entities))
    assert duel.joint_attack_task(match,1)
    assert duel.joint_attack_task(match,-1)
    entities[4]=NS(side=-1,alive=True,is_building=True)
    assert not duel.joint_attack_task(match,1)
    assert duel.joint_attack_task(match,-1)


def test_joint_observation_keeps_v2_positions_hides_history_and_appends_absent_timer(monkeypatch):
    match=NS(battle=NS(entities={1:NS(side=-1,alive=True,pos=NS(x=4000,y=23000))}),
             play_log=[dict(secret='unavailable')],
             tower_fractions=lambda side:dict(left=.2,right=.8,king=1))
    seen=[]
    def observe(view,side,coordinate_mode):
        seen.append((view.battle.entities[1].pos.y,view.play_log,view.tower_fractions(side)['left'],coordinate_mode))
        return dict(planes=np.zeros((1,1,1),np.float32),scalars=np.zeros(306,np.float32))
    monkeypatch.setattr(duel,'observe',observe)
    bottom=duel.policy_observation(match,1,duel.JOINT_MODE)
    top=duel.policy_observation(match,-1,duel.JOINT_MODE)
    assert bottom['scalars'].shape==(309,)
    assert bottom['scalars'][-3:].tolist()==[0,0,0]
    assert top['scalars'][-3:].tolist()==[0,0,1]
    assert seen==[(23000,[],.2,'physical_live_cells_v2'),(23000,[],.8,'physical_live_cells_v2')]
    assert match.play_log==[dict(secret='unavailable')]
    assert match.battle.entities[1].pos.y==23000


def test_joint_weights_load_exactly_and_cannot_be_silently_trimmed(tmp_path):
    from sim.train_attack_ppo import initialize_attack_policy
    from sim.strategy_env import OBSERVATION_CONTRACT
    net,architecture,_=initialize_attack_policy(num_scalars=309)
    path=tmp_path/'joint.pt'
    blob=dict(state_dict=net.state_dict(),architecture=architecture,config=dict(
        task_mode='joint',observation_contract=OBSERVATION_CONTRACT,
        coordinate_mode='physical_live_cells_v2',geometry='legacy_v1'))
    torch.save(blob,path)
    loaded=duel.load_network(path,'cpu',duel.JOINT_MODE)
    assert loaded.state_dict()['scalar.0.weight'].shape[1]==309
    for key,value in net.state_dict().items():assert torch.equal(value,loaded.state_dict()[key])
    with pytest.raises(ValueError,match='scalar shape'):duel.load_network(path,'cpu')
    blob['config']['coordinate_mode']='physical_live';torch.save(blob,path)
    with pytest.raises(ValueError,match='contract'):duel.load_network(path,'cpu',duel.JOINT_MODE)


@pytest.mark.parametrize('count',[304,306])
def test_historical_baseline_inputs_keep_supported_zero_append_migration(tmp_path,count):
    from sim.train_attack_ppo import initialize_attack_policy
    net,architecture,_=initialize_attack_policy(num_scalars=count)
    path=tmp_path/'baseline.pt'
    torch.save(dict(state_dict=net.state_dict(),architecture=architecture),path)
    loaded=duel.load_network(path,'cpu')
    actual=loaded.state_dict()['scalar.0.weight']
    assert actual.shape[1]==306
    assert torch.equal(actual[:,:count],net.state_dict()['scalar.0.weight'])
    assert not torch.count_nonzero(actual[:,count:])
