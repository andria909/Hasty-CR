"""Regression checks for the family vocabulary and pretrained head transfer."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_field_finetune import family_name


@pytest.mark.parametrize(('raw','expected'),[
    ('knight-evolution','knight'),('valkyrie-hero','valkyrie'),
    ('skeleton-army-evolution','skeleton'),('phoenix-small','phoenix'),
    ('archer-queen','archer-queen'),('boss-bandit','boss-bandit'),
    ('little-prince','little-prince'),('padding_belong',None),('void',None)])
def test_family_vocabulary(raw,expected):
    assert family_name(raw)==expected


def test_head_transfer_merges_variants_and_drops_team_channel():
    torch=pytest.importorskip('torch')
    from torch import nn
    from finetune_field_detector import adapt_class_head
    branches=nn.ModuleList([nn.Sequential(nn.Conv2d(2,4,1)) for _ in range(3)])
    for branch in branches:
        with torch.no_grad():
            for i in range(4):
                branch[-1].weight[i].fill_(i+1)
                branch[-1].bias[i]=i+1
    head=SimpleNamespace(cv3=branches,stride=[8,16,32],reg_max=16)
    model=SimpleNamespace(model=[head],names={0:'knight',1:'knight-evolution',2:'monk',3:'padding_belong'},
                          yaml={'nc':4},criterion=object())
    source=adapt_class_head(model,['knight','monk','boss-bandit'])
    assert source['knight']==['knight','knight-evolution']
    assert source['boss-bandit']==[]
    for branch in branches:
        assert branch[-1].weight.shape==(3,2,1,1)
        assert torch.allclose(branch[-1].weight[0],torch.full((2,1,1),1.5))
        assert branch[-1].bias[1].item()==3
    assert head.nc==3 and head.no==67
    assert model.criterion is None
    assert 'padding_belong' not in model.names.values()
