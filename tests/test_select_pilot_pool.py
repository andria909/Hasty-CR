from pathlib import Path
import json
import pytest
from scripts.attack import select_pilot_pool as module


def test_validation_reflection_and_nonzero_defaults_cannot_be_ignored():
    source=Path('sim/continuous_push.py').read_text()
    with pytest.raises(ValueError,match='reflection'):
        module.normalized_wrapper(source.replace("self.reflected=bool(self.split=='train'","self.reflected=bool(self.split=='validation'"))
    with pytest.raises(ValueError,match='default'):
        module.normalized_wrapper(source.replace('reserve_potential_weight=0.','reserve_potential_weight=.1'))
    assert module.normalized_wrapper(source)!=module.normalized_wrapper(source.replace('available=max(0,','available=max(1000,'))


def fixture(tmp_path,name,eligible=True):
    root=tmp_path/name;(root/'dataset').mkdir(parents=True);(root/'collected').mkdir()
    (root/'dataset/protocol.json').write_text(json.dumps(dict(pilot_seeds=[1,2])))
    rows=[dict(scenario_id='example',episode_seed=i,match_seed=i)for i in range(72)]
    (root/'collected/parent_reference.json').write_text(json.dumps(dict(validation=dict(records=rows))))
    return root,dict(kernel=name,pilot_steps_verified=819200,pilot_models=[],pilot_selection=dict(floor=.1,ranking=[
        dict(trial=dict(id=name,entropy_coef=.01,focus='uniform'),score=.3,minimum_family_points=.15 if eligible else 0.,eligible=eligible)]))


def test_one_eligible_profile_per_cohort_can_form_global_pair_without_weakening_floor(monkeypatch,tmp_path):
    a,pa=fixture(tmp_path,'a');b,pb=fixture(tmp_path,'b')
    monkeypatch.setattr(module,'verify',lambda collected,root:{a:pa,b:pb}[root])
    monkeypatch.setattr(module,'contract',lambda root:dict(core={'physics':'same'}))
    result=module.select([a,b]);assert result['gate_met']and len(result['selected'])==2
    assert not module.select([a])['gate_met']
    pb['pilot_selection']['ranking'][0].update(eligible=False,minimum_family_points=0.)
    assert not module.select([a,b])['gate_met']
    with pytest.raises(ValueError,match='Duplicate'):module.select([a,a])


def test_global_pair_rejects_incomparable_physics_and_lower_floor(monkeypatch,tmp_path):
    a,pa=fixture(tmp_path,'a');b,pb=fixture(tmp_path,'b')
    monkeypatch.setattr(module,'verify',lambda collected,root:{a:pa,b:pb}[root])
    monkeypatch.setattr(module,'contract',lambda root:dict(core={'physics':str(root)}))
    with pytest.raises(ValueError,match='physics'):module.select([a,b])
    monkeypatch.setattr(module,'contract',lambda root:dict(core={'physics':'same'}))
    pb['pilot_selection']['floor']=0.
    with pytest.raises(ValueError,match='weaken'):module.select([a,b])
