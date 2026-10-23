from scripts.kaggle.run_five_round_robin import schedule, standings


def test_ten_unique_pairs_and_twenty_paired_seats():
    from collections import Counter
    rows=schedule(['p1','g2','p3','g4','v25'])
    assert len(rows)==200
    assert len({(tuple(r['pair']),r['seed'],r['candidate_side'])for r in rows})==200
    assert set(Counter(tuple(r['pair'])for r in rows).values())=={20}
    for side in (-1,1):assert sum(r['candidate_side']==side for r in rows)==100


def test_seat_flip_result_mapping_draw_and_unknown_not_conflated():
    names=['a','b','c','d','e']
    rows=[dict(pair=['a','b'],candidate_side=-1,outcome=o)for o in ('win','loss','draw','unknown')]
    result=standings(rows,names)
    assert result['totals']['a']['wins']==result['totals']['b']['wins']==1
    assert result['totals']['a']['draws']==result['totals']['a']['unknown']==1
    assert result['totals']['a']['win_rate_all_matches']==.25
    assert result['leaders']==['a','b']


def test_all_participants_have_eighty_games():
    names=['a','b','c','d','e']
    result=standings([dict(**r,outcome='draw')for r in schedule(names)],names)
    assert all(v['matches']==80 and v['draws']==80 for v in result['totals'].values())
    assert result['leaders']==names


def test_skill_adapters_keep_old306_and_joint309_explicit_modes():
    import torch
    from scripts.kaggle.run_five_round_robin import skill_policy,skill_environment
    class Capture(torch.nn.Module):
        def forward(self,planes,scalars):return scalars
    network=Capture();scalars=torch.arange(309)[None]
    assert skill_policy(network,False,torch)(None,scalars).shape==(1,306)
    assert skill_policy(network,True,torch)(None,scalars).shape==(1,309)
    options=dict(coordinate_mode='physical_live_cells_v2',geometry='legacy_v1')
    assert skill_environment(lambda **x:x,False,**options)['coordinate_mode']=='physical_live'
    assert skill_environment(lambda **x:x,True,**options)['coordinate_mode']=='physical_live_cells_v2'
    assert options['coordinate_mode']=='physical_live_cells_v2'
