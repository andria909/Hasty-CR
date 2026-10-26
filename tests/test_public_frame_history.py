import numpy as np
import pytest
import torch

from sim.env import NUM_PLANES,NUM_SCALARS,ACTIONS
from sim.public_frame_history import PublicFrameHistory,HISTORY_PLANES
from sim.train_ppo import build_network,load_policy_state,COMPACT_ARCHITECTURE
from sim.train_attack_ppo import make_training_env,observation_contract
from sim.public_frame_history import CONTRACT


def frame(value):
    return dict(planes=np.full((NUM_PLANES,32,18),value,dtype=np.float32),
                scalars=np.zeros(NUM_SCALARS+3,dtype=np.float32))


def test_history_is_public_idempotent_and_does_not_alias_input_buffers():
    h=PublicFrameHistory();a=frame(1);first=h.observe(a,at_ms=0,world='a')
    assert not first['planes'][NUM_PLANES:].any()
    second=h.observe(frame(2),at_ms=250,world='a')
    np.testing.assert_array_equal(second['planes'][NUM_PLANES:2*NUM_PLANES],a['planes'])
    assert (second['planes'][-1]==.25).all()
    repeated=h.observe(frame(2),at_ms=250,world='a')
    np.testing.assert_array_equal(second['planes'],repeated['planes'])
    a['planes'][:]=99
    np.testing.assert_array_equal(second['planes'][NUM_PLANES:2*NUM_PLANES],1)
    state=h.snapshot();expected=h.observe(frame(3),at_ms=500,world='a')
    h.restore(state);actual=h.observe(frame(3),at_ms=500,world='a')
    np.testing.assert_array_equal(actual['planes'],expected['planes'])


@pytest.mark.parametrize('at_ms,world',[(200,'a'),(2000,'a'),(500,'different')])
def test_history_clears_at_world_change_clock_rewind_or_stale_gap(at_ms,world):
    h=PublicFrameHistory();h.observe(frame(1),at_ms=0,world='a');h.observe(frame(2),at_ms=250,world='a')
    out=h.observe(frame(3),at_ms=at_ms,world=world)
    assert not out['planes'][NUM_PLANES:].any()


def test_zero_extended_channels_preserve_old_logits_and_can_learn():
    torch.set_num_threads(1);torch.manual_seed(42)
    old=build_network(NUM_PLANES,NUM_SCALARS+3,ACTIONS,COMPACT_ARCHITECTURE)
    new=build_network(HISTORY_PLANES,NUM_SCALARS+3,ACTIONS,COMPACT_ARCHITECTURE)
    assert load_policy_state(new,old.state_dict())
    planes=torch.rand(2,NUM_PLANES,32,18);scalars=torch.rand(2,NUM_SCALARS+3)
    extended=torch.cat((planes,torch.rand(2,HISTORY_PLANES-NUM_PLANES,32,18)),dim=1)
    a,v=old(planes,scalars);b,w=new(extended,scalars)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5);torch.testing.assert_close(v,w,atol=1e-6,rtol=1e-5)
    (b[:,1].mean()+w.mean()).backward()
    assert new.conv[0].weight.grad[:,NUM_PLANES:].abs().sum()>0


def test_real_training_env_history_replays_and_resets_between_games():
    env=make_training_env(curriculum='continuous_v1',history_frames=2)
    obs,info=env.reset(seed=1)
    assert obs['planes'].shape==(41,32,18)and len(obs['scalars'])==309
    assert info['metrics']['observation_contract']==CONTRACT
    assert observation_contract('joint','continuous_v1',2)==CONTRACT
    env.step(0);state=env.snapshot();first=env.step(0)
    env.restore(state);second=env.step(0)
    np.testing.assert_array_equal(first[0]['planes'],second[0]['planes'])
    obs,_=env.reset(seed=2)
    assert not obs['planes'][NUM_PLANES:].any()
