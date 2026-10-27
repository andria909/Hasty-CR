"""Opt-in prototype: encode our own queued commands, never enemy/future state.

Does not modify the default environment or live checkpoint. A live adapter must
supply its own recorded/estimated delivery times, not future server outcomes.
"""
import numpy as np

CONTRACT='joint_public_presence_continuous_slots_pending_v1'
NUM_PLANES=24
NUM_SCALARS=313
TIME_SCALE_MS=3000.


def append_pending(observation,pending,now_ms,*,reflected=False):
    planes=np.asarray(observation['planes']);scalars=np.asarray(observation['scalars'])
    if planes.shape!=(20,32,18) or scalars.shape!=(309,):
        raise ValueError('Base public20/309 observation required')
    if not np.isfinite(now_ms):raise ValueError('Finite own clock required')
    positions=np.zeros((4,32,18),dtype=planes.dtype)
    times=np.zeros(4,dtype=scalars.dtype);used=set()
    for due,card,slot,x,y in pending:
        if (not isinstance(card,str) or not card or type(slot)is not int or not 0<=slot<4
                or type(x)is not int or type(y)is not int or not 0<=x<18 or not 0<=y<32
                or slot in used or not np.isfinite(due)):
            raise ValueError('One valid own queued command per slot required')
        used.add(slot)
        positions[slot,y,17-x if reflected else x]=1
        times[slot]=max(0.,due-now_ms)/TIME_SCALE_MS
    # Base already reserves elixir and masks busy slots; do not subtract twice.
    return dict(observation,planes=np.concatenate((planes,positions)),
                scalars=np.concatenate((scalars,times)))


def initialize_from_base(blob,device='cpu'):
    """Explicit prototype migration; no changes to the default checkpoint loader."""
    from sim.continuous_push import CONTRACT as BASE_CONTRACT
    from sim.train_ppo import build_network,checkpoint_architecture,load_policy_state
    state=blob['state_dict']
    if (blob.get('config',{}).get('observation_contract')!=BASE_CONTRACT
            or state['conv.0.weight'].shape[1]!=20 or state['scalar.0.weight'].shape[1]!=309):
        raise ValueError('Exact20/309 base contract checkpoint required')
    from sim.env import ACTIONS
    network=build_network(NUM_PLANES,NUM_SCALARS,ACTIONS,checkpoint_architecture(blob)).to(device)
    load_policy_state(network,state)
    return network
