import hashlib
import json
import numpy as np
import pytest
import torch
from scripts.gamestate.decision_capture import DecisionCapture
from scripts.gamestate.model_bridge import MemoryPolicy,ModelView
from scripts.gamestate.replay_decisions import replay,load_capture
from sim.env import NUM_PLANES,NUM_SCALARS,ACTIONS
from sim.train_ppo import build_network


@pytest.fixture
def captured(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(123)
    checkpoint=tmp_path/'model.pt'
    net=build_network(NUM_PLANES,NUM_SCALARS+3,ACTIONS)
    torch.save(dict(config={'task_mode':'joint'},state_dict=net.state_dict()),checkpoint)
    policy=MemoryPolicy(checkpoint)
    view=ModelView({'planes':np.zeros((NUM_PLANES,32,18),dtype=np.float32),
                    'scalars':np.linspace(0,1,NUM_SCALARS,dtype=np.float32)},
                   np.ones(ACTIONS,dtype=bool),
                   {'ready_for_inference':True,'enemy_bodies_present':True},False)
    view.action_mask[1:577]=False
    # No ability support in native placement decoding.
    view.action_mask[2305:]=False
    context=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                 coordinate_mode=policy.coordinate_mode,geometry=policy.geometry,match=1,tick=100)
    recorder=DecisionCapture(tmp_path/'decisions',every=1)
    policy.decide(view,capture=lambda *args:recorder.capture(*args,context))
    view.action_mask[1:]=False
    policy.decide(view,capture=lambda *args:recorder.capture(*args,context))
    recorder.close()
    return checkpoint,recorder.output


def test_saved_real_network_inputs_reproduce_actions_and_values(captured):
    report=replay(*captured)
    assert report['samples']==report['same_actions']==2
    assert report['maximum_value_difference']==0
    assert report['rows'][1]['action']==0


def test_different_weights_require_explicit_comparison(captured,tmp_path):
    checkpoint,folder=captured
    blob=torch.load(checkpoint,weights_only=False)
    blob['diagnostic_note']='different artifact'
    other=tmp_path/'other.pt';torch.save(blob,other)
    with pytest.raises(ValueError,match='SHA differs'):replay(other,folder)
    assert replay(other,folder,compare=True)['same_actions']==2


def test_illegal_recorded_action_is_rejected(captured):
    _,folder=captured
    file=sorted(folder.glob('*.npz'))[1]
    with np.load(file,allow_pickle=False)as source:data=dict(source)
    metadata=json.loads(str(data['metadata']));metadata['decision']['action']=1
    data['metadata']=np.asarray(json.dumps(metadata));np.savez(file,**data)
    with pytest.raises(ValueError,match='available'):load_capture(file,joint=True)
