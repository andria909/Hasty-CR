import numpy as np
import pytest
import torch

from scripts.attack.evaluate_continuous_pool import load, choose, sha
from scripts.gamestate.elixir import ElixirEstimate
from scripts.gamestate.model_bridge import MemoryPolicy, NativeModelAdapter, OwnState
from sim.entity_policy import EntityTransformer, CONTRACT, vocabulary_sha
from sim.env import ACTIONS
from sim.eval_defense_duel import JOINT_MODE
from tests.test_memory_model_bridge import fixture


def checkpoint(tmp_path, *, architecture='entity_transformer_d64_l2_relative_v1', **changes):
    torch.set_num_threads(1); torch.manual_seed(123)
    net=EntityTransformer(309,ACTIONS,architecture)
    config=dict(task_mode='joint',coordinate_mode='physical_live_cells_v2',
                observation_contract=CONTRACT,entity_vocabulary_sha256=vocabulary_sha(),
                evaluation_action_mode='grouped_argmax')
    config.update(changes)
    p=tmp_path/'policy.pt'
    torch.save(dict(architecture=architecture,state_dict=net.state_dict(),config=config),p)
    return dict(file=p.name,sha256=sha(p),scalars=309,mode=JOINT_MODE)


@pytest.mark.parametrize('side', [1,-1])
@pytest.mark.parametrize('reserved', [0,1000])
def test_full_evaluator_matches_native_input_and_action_in_both_seats(tmp_path,side,reserved):
    entry=checkpoint(tmp_path)
    net=load(entry,tmp_path,'cpu')
    native=MemoryPolicy(tmp_path/entry['file'])
    match,_,_=fixture()
    player=match.players[side]
    rows=[]
    for e in match.battle.entities.values():
        name=('KingTower'if e.name=='king_tower'else'PrincessTower')if e.is_tower else e.name
        rows.append(dict(entity_id=5000000+e.uid,name=name,side=0 if e.side==side else 1,
                         x=e.pos.x,y=e.pos.y,hp=e.hitpoints,max_hp=e.max_hitpoints))
    own=OwnState(0,tuple(player.hand),player.next_card,
        ElixirEstimate(0,player.elixir/1000,player.elixir/1000,True,reserved=reserved/1000),(),
        public_history_complete=True,battle_active=True)
    view=NativeModelAdapter(own_side=0,coordinate_mode='physical_live_cells_v2',
        geometry=match.geometry,entity_inputs=True).encode(dict(status='ok',units=rows),own)
    captured=[];expected=native.decide(view,capture=lambda *args:captured.append(args))
    inputs=[];forward=net.forward
    def recording(planes,scalars):
        inputs.append((planes.detach().numpy()[0].copy(),scalars.detach().numpy()[0].copy()))
        return forward(planes,scalars)
    net.forward=recording
    actual=choose(net,match,side,JOINT_MODE,view.action_mask,'cpu',reserved)
    assert actual==expected['action'] and view.action_mask[actual]
    np.testing.assert_array_equal(inputs[0][0],captured[0][0])
    np.testing.assert_allclose(inputs[0][1],captured[0][1],rtol=0,atol=1e-7)
    assert net.evaluation_action_mode=='grouped_argmax'


@pytest.mark.parametrize('changes', [dict(entity_vocabulary_sha256='bad'),
    dict(observation_contract='raster'),dict(evaluation_action_mode='guess'),dict(task_mode='attack')])
def test_full_evaluator_rejects_wrong_entity_contract(tmp_path,changes):
    entry=checkpoint(tmp_path,**changes)
    with pytest.raises(ValueError):load(entry,tmp_path,'cpu')


def test_declared_decoding_and_observation_mode_are_binding(tmp_path):
    entry=checkpoint(tmp_path)
    with pytest.raises(ValueError,match='Declared action decoding'):
        load(dict(entry,evaluation_action_mode='flat_argmax'),tmp_path,'cpu')
    with pytest.raises(ValueError,match='Entity evaluation input'):
        load(dict(entry,mode='physical_live'),tmp_path,'cpu')


def test_grouped_action_path_does_not_revert_to_flat_argmax():
    match,_,_=fixture()
    mask=np.zeros(ACTIONS,dtype=bool);mask[:577]=True
    class Fixed:
        evaluation_action_mode='grouped_argmax'
        def __call__(self,*args):
            logits=torch.full((1,ACTIONS),-2.);logits[0,0]=0.;logits[0,37]=-1.
            return logits,torch.zeros(1)
    network=Fixed()
    assert choose(network,match,1,JOINT_MODE,mask,'cpu')==37
    network.evaluation_action_mode='flat_argmax'
    assert choose(network,match,1,JOINT_MODE,mask,'cpu')==0
