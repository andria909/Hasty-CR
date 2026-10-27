import numpy as np
import torch
from scripts.attack.probe_reflection_ensemble import probabilities
from sim.env import ACTIONS
from sim.lane_reflection import reflect_actions,reflect_observation


class Biased(torch.nn.Module):
    def forward(self,planes,scalars):
        value=torch.arange(ACTIONS,dtype=torch.float32)[None]/100
        return value+planes[:,0,0,0,None]*torch.sin(value)+scalars[:,3,None],torch.zeros(len(planes))


def test_averaged_distribution_is_reflection_equivariant_and_legal():
    rng=np.random.default_rng(91)
    obs=dict(planes=rng.normal(size=(20,32,18)).astype('float32'),scalars=rng.normal(size=309).astype('float32'))
    mask=rng.random(ACTIONS)>.4;mask[0]=True
    p=probabilities(Biased(),obs,mask,True)
    back=reflect_actions(probabilities(Biased(),reflect_observation(obs),reflect_actions(mask),True))
    np.testing.assert_allclose(p,back,atol=2e-7,rtol=2e-6)
    assert np.all(p[~mask]==0)
    np.testing.assert_allclose(p.sum(),1.,atol=1e-6)


def test_plain_mode_preserves_original_masked_policy():
    obs=dict(planes=np.zeros((20,32,18),dtype='float32'),scalars=np.zeros(309,dtype='float32'))
    mask=np.zeros(ACTIONS,dtype=bool);mask[[0,10]]=True
    p=probabilities(Biased(),obs,mask,False)
    assert p.argmax()==10 and np.count_nonzero(p)==2
