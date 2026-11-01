import pytest
import torch

from scripts.attack.audit_sampling_greedy_gap import group_metrics


def test_one_wait_can_win_argmax_despite_overwhelming_play_probability():
    mask=torch.zeros((1,2306),dtype=torch.bool);mask[:,:577]=True
    probabilities=torch.zeros((1,2306));probabilities[:,0]=.01;probabilities[:,1:577]=.99/576
    result=group_metrics(probabilities,mask)
    assert result['greedy_group'].item()==0
    assert result['largest_group'].item()==1
    assert result['group_mass'][0,1].item()==pytest.approx(.99)
    assert result['conditional_cell_entropy'].item()==pytest.approx(.99*torch.log(torch.tensor(576.)).item())


def test_legal_cell_counts_and_illegal_mass_are_not_silently_ignored():
    mask=torch.zeros((1,2306),dtype=torch.bool);mask[:,0]=True;mask[:,1:37]=True
    p=mask.float()/mask.sum(-1,keepdim=True)
    assert group_metrics(p,mask)['legal_cells'].tolist()==[[36,0,0,0]]
    p[:,100]=.01;p[:,0]-=.01
    with pytest.raises(ValueError,match='Illegal'):group_metrics(p,mask)
