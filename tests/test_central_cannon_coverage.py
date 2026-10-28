from collections import Counter
import pytest
from scripts.attack.audit_central_cannon_coverage import (contexts,context_key,mirror_key,
    summarize_group,CELLS)
from sim.cannon_placement import CENTRAL_CELLS


def test_schedule_covers_tower_states_routes_latency_and_mirrors_without_illegal_pockets():
    rows=contexts();keys={context_key(r)for r in rows}
    assert len(rows)==len(keys)==224
    assert {mirror_key(r)for r in rows}==keys
    assert len(CELLS)==36 and set(CELLS.values())==set(CENTRAL_CELLS)
    for r in rows:
        if r['route']=='pocket':assert not r['own_mask']&(1 if r['lane']=='left'else 2)
    normal=[r for r in rows if r['route']=='bridge'and r['decision_ms']==1500]
    assert set(Counter((r['own_mask'],r['enemy_mask'])for r in normal).values())=={4}


def test_best_cell_is_paid_counterfactual_and_does_not_hide_worsening_cells():
    rows=[dict(choice='none',total_damage=100)]
    for i,(name,cell)in enumerate(CELLS.items()):
        rows.append(dict(choice=name,cell=cell,paid_cannon=True,pulled=i%2==0,total_damage=0 if i==0 else 200))
    result=summarize_group({},rows)
    assert result['best_prevented']==100 and result['best_paid_damage']==0
    assert result['zero_damage_cells']==result['improved_cells']==1
    assert result['worsening_cells']==35
    assert result['pull_cells']==18
    rows[1]['paid_cannon']=False
    with pytest.raises(ValueError,match='coverage'):summarize_group({},rows)


def test_duplicate_cell_cannot_substitute_missing_central_cell():
    rows=[dict(choice='none',total_damage=100)]
    rows += [dict(choice=name,cell=cell,paid_cannon=True,pulled=True,total_damage=0)for name,cell in CELLS.items()]
    rows[1]['cell']=rows[2]['cell']
    with pytest.raises(ValueError,match='coverage'):summarize_group({},rows)
