import json

import pytest

from scripts.defense.summarize_selected_ppo_train import train_contexts,validate_records


def test_train_schedule_excludes_heldout_catalog_and_fixes_three_seeds():
    members={name:json.dumps({'scenarios':[{'scenario_id':'train','split':'train'},{'scenario_id':'heldout','split':'test'}]})
             for name in ('data/defense/scenarios.json','data/attack/scenarios.json')}
    rows=train_contexts(members)
    assert len(rows)==6 and {r['seed']for r in rows}=={42,43,44}
    assert all('heldout'not in r['scenario_id']for r in rows)


@pytest.mark.parametrize('tamper',['duplicate','split','coordinate','task'])
def test_train_records_cannot_hide_duplicate_or_different_input_contract(tamper):
    contexts=[dict(scenario_id='defense::a',seed=42),dict(scenario_id='attack::b',seed=42)]
    rows=[dict(strategy_scenario_id=r['scenario_id'],seed=42,task=r['scenario_id'].split('::')[0],split='train',
               geometry='legacy_v1',coordinate_mode='physical_live')for r in contexts]
    validate_records(rows,contexts,'defense')
    if tamper=='duplicate':rows[0]=rows[1]
    elif tamper=='split':rows[0]['split']='test'
    elif tamper=='coordinate':rows[0]['coordinate_mode']='physical_live_cells_v2'
    else:rows[0]['task']='attack'
    with pytest.raises(ValueError):validate_records(rows,contexts,'defense')
