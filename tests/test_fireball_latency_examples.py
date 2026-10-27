import json
from pathlib import Path
import numpy as np
import pytest
from scripts.attack.build_fireball_latency_examples import build

SOURCE=Path('data/air_reserve_examples_20261027')
AUDIT=Path('reports/defense_plan/fireball_label_latency_audit_20261027.json')


def test_actual_relabel_preserves_inputs_nonfireball_labels_and_seed_partition(tmp_path):
    result=build(SOURCE,AUDIT,tmp_path/'new')
    old=json.loads((SOURCE/'manifest.json').read_text())
    assert result['examples']==210 and result['fireball_examples']==42 and result['removed_positive_actions']==207
    assert [(r.get('array_index'),r['seed'],r['accepted'])for r in result['records']]==[(r.get('array_index'),r['seed'],r['accepted'])for r in old['records']]
    with np.load(SOURCE/'examples.npz')as a,np.load(tmp_path/'new/examples.npz')as b:
        for key in ('planes','scalars','masks'):assert np.array_equal(a[key],b[key])
        assert np.all(b['success_masks']<=a['success_masks'])
        assert np.all(b['success_masks'].any(axis=1))
        fire={r['array_index']for r in result['records']if r.get('skill')=='fireball_group' and r['accepted']}
        for i in set(range(210))-fire:assert np.array_equal(a['success_masks'][i],b['success_masks'][i])


def test_no_empty_target_is_silently_dropped_or_added_to_train(tmp_path):
    audit=json.loads(AUDIT.read_text());audit['rows'][0]['by_delay']['1350']['successful_action_ids']=[]
    path=tmp_path/'audit.json';path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match='nonempty'):build(SOURCE,path,tmp_path/'new')
    assert not(tmp_path/'new').exists()


def test_changed_audit_original_mask_is_rejected(tmp_path):
    audit=json.loads(AUDIT.read_text());audit['rows'][0]['by_delay']['225']['successful_action_ids']=[]
    path=tmp_path/'audit.json';path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match='success mask'):build(SOURCE,path,tmp_path/'new')
