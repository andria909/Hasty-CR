import pytest
import torch
from scripts.attack.diagnose_action_decoding import decode


def fixture():
    logits=torch.full((2305,),-20.);mask=torch.zeros(2305,dtype=torch.bool);mask[0]=True
    # One concentrated card action versus three individually weaker cells.
    logits[1]=2.;mask[1]=True
    logits[577:580]=1.5;mask[577:580]=True
    return logits,mask


def test_card_marginal_and_joint_map_are_different_decisions():
    logits,mask=fixture()
    assert decode(logits,mask,'joint_greedy')[0]==1
    action,detail=decode(logits,mask,'card_greedy')
    assert action==577 and detail['group_probabilities'][2]>detail['group_probabilities'][1]


def test_masked_cells_and_forced_wait_are_respected():
    logits,mask=fixture();logits[2304]=1000
    for mode in ('joint_greedy','card_greedy','sample'):
        assert mask[decode(logits,mask,mode,torch.Generator().manual_seed(3))[0]]
        only_wait=torch.zeros_like(mask);only_wait[0]=True
        assert decode(logits,only_wait,mode,torch.Generator().manual_seed(3))[0]==0


def test_stochastic_probe_is_reproducible_with_its_own_rng():
    logits,mask=fixture();a=torch.Generator().manual_seed(42);b=torch.Generator().manual_seed(42)
    first=[decode(logits,mask,'sample',a)[0]for _ in range(30)]
    torch.manual_seed(999)
    second=[decode(logits,mask,'sample',b)[0]for _ in range(30)]
    assert first==second and len(set(first))>1
    with pytest.raises(ValueError,match='generator'):decode(logits,mask,'sample')


def test_actual_ability_head_is_preserved_as_singleton_groups():
    logits,mask=fixture();logits=torch.cat((logits,torch.zeros(16)));mask=torch.cat((mask,torch.zeros(16,dtype=torch.bool)))
    logits[-1]=10;mask[-1]=True
    assert decode(logits,mask,'joint_greedy')[0]==2320
    assert decode(logits,mask,'card_greedy')[0]==2320
