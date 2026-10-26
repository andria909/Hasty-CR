import pytest
from scripts.attack.continuous_pool_uncertainty import cluster_interval


def test_same_seed_seats_are_one_resampling_unit():
    # Every seed contributes one win and one loss: block means are exactly.5.
    result=cluster_interval({s:[1,0]for s in range(10)},1000)
    assert result['games']==20 and result['independent_seed_blocks']==10
    assert result['percentile_interval_95']==[.5,.5]


def test_bootstrap_reproducible_and_cannot_claim_more_than_observed_range():
    groups={0:[0,0],1:[1,1],2:[.5,.5]}
    a=cluster_interval(groups,1000);b=cluster_interval(groups,1000)
    assert a==b and a['points']==.5
    assert 0<=a['percentile_interval_95'][0]<=.5<=a['percentile_interval_95'][1]<=1
    with pytest.raises(ValueError):cluster_interval({0:[1]},1000)
    with pytest.raises(ValueError):cluster_interval({0:[1],1:[0,0]},1000)
