import json
import numpy as np
import pytest

from scripts.attack.build_correction_only_dataset import build
from scripts.attack.train_bank_correction_pair import depth_protocol,BASE_SKILLS
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from scripts.kaggle.stream_training import record_fit_depth_event


def fixture(tmp_path):
    source=tmp_path/'mixed';source.mkdir();skills=sorted(BASE_SKILLS|{'onpolicy_defense_response'})
    rows=[dict(seed=seed,skill=skill,accepted=True,array_index=i,action=1,context=str(i))
        for i,(seed,skill)in enumerate((s,k)for s in (352,353,354)for k in skills)]
    # Tiny tensor fixtures test subset integrity, not neural execution.
    np.savez_compressed(source/'examples.npz',**{k:np.arange(30).reshape(15,2)for k in ('planes','scalars','masks','success_masks')})
    m=dict(schema='mixed-bank-public-sequences-v1',split='train',seeds=[352,353,354],records=rows,examples=15,
        physics_pins=physics_pins(),command_delay_ms=1350,slot_augmentation_contract='public309_no_pending_busy_slots_v1',
        skills=skills,data_sha256=file_sha(source/'examples.npz'),correction_label_mode='outcome_first_pareto_v1')
    (source/'manifest.json').write_text(json.dumps(m));parent=tmp_path/'parent.pt';parent.write_bytes(b'fixture, not weights')
    subset=tmp_path/'isolated';build(source,subset);return source,subset,parent


def test_same_grouped_subset_and_different_budgets_are_explicit(tmp_path):
    whole,subset,parent=fixture(tmp_path);p=depth_protocol(whole,subset,parent)
    assert sum(c['updates']for c in p['cases'])*len(p['seeds'])==35200
    assert p['datasets']['corrections']['skills']==['onpolicy_defense_response']
    m=json.loads((subset/'manifest.json').read_text());assert {r['seed']for r in m['records']}=={352,353,354}
    assert p['full_training_accepted']is False


def test_self_consistent_but_changed_subset_tensor_is_rejected(tmp_path):
    whole,subset,parent=fixture(tmp_path)
    with np.load(subset/'examples.npz')as z:arrays={k:z[k]for k in z.files}
    arrays['planes'][0,0]+=1;np.savez_compressed(subset/'examples.npz',**arrays)
    m=json.loads((subset/'manifest.json').read_text());m['data_sha256']=file_sha(subset/'examples.npz');(subset/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='tensors differ'):depth_protocol(whole,subset,parent)


def test_depth_progress_uses_each_budget_and_rejects_cpu_and_wrong_target():
    p={};r=dict(event='FIT_DEPTH_PROGRESS',id='mixed8000_s402',updates=800,target_updates=8000,actual_cuda='Tesla T4')
    assert record_fit_depth_event(p,json.dumps(r),1)
    assert p['fit_depth'][r['id']]['percent']==10
    assert 'corrections'not in p and 'trials'not in p
    for changed in (dict(actual_cuda=None),dict(target_updates=800),dict(event='FIT_DEPTH_RUN_COMPLETE'),dict(id='mixed8000_s404')):
        assert not record_fit_depth_event(p,json.dumps(dict(r,**changed)),2)
