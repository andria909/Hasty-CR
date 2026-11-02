from copy import deepcopy

import pytest

from scripts.attack.select_pilot_pool import economy_identity, behavior_summary, pool_economy_replicates
from scripts.attack.compare_ppo_pilot_behavior import compare_records
from tests.test_paired_ppo_behavior import records


def comparison(seed, delta=0, defense_delta=0):
    before=records();after=deepcopy(before)
    # Different task changes deliberately test an attack gain hiding defense loss.
    if defense_delta<0:
        before[6]['outcome']='win'
    elif defense_delta>0:
        after[6]['outcome']='win'
    for i in range(delta-defense_delta):after[i]['outcome']='win'
    return dict(seed=seed,reference='control',comparison=compare_records(before,after))


def item(study,seeds,score=.2,initial='weights-a'):
    return dict(study=study,trial=dict(id='control'),seed_pair=seeds,score=score,minimum_family_points=.2,
                floor=.1,family_eligible=True,eligible=True,
                training_identity=dict(initial_sha256=initial,profile=dict(spatial_entropy_weight=1.)),
                quality_diagnostics=behavior_summary([comparison(s)for s in seeds]))


def test_more_total_wins_cannot_hide_defensive_regression():
    stable=comparison(1,delta=2,defense_delta=1)
    regressed=comparison(2,delta=2,defense_delta=-1)
    assert stable['comparison']['all']['wins_delta']==regressed['comparison']['all']['wins_delta']==2
    assert not behavior_summary([stable,regressed])['nonregression_gate']
    assert behavior_summary([stable,comparison(2)])['nonregression_gate']


def test_profile_identity_uses_initial_weight_not_alias_and_explicit_default():
    a=dict(config=dict(models=[dict(id='bc800',sha256='abc')]))
    b=dict(config=dict(models=[dict(id='parent',sha256='abc')]))
    first=economy_identity(a,dict(id='bc800',initial_id='bc800',spell_waste_weight=.1))
    assert first==economy_identity(b,dict(id='control',initial_id='parent',spell_waste_weight=.1,spatial_entropy_weight=1.))
    assert first!=economy_identity(b,dict(id='control',initial_id='parent',spell_waste_weight=.5))
    b['config']['models'][0]['sha256']='different'
    assert first!=economy_identity(b,dict(id='control',initial_id='parent',spell_waste_weight=.1))


def test_duplicate_control_is_one_six_seed_profile_not_three_lucky_candidates():
    rows=[item('z',[1,2],.8),item('a',[3,4],.1),item('m',[5,6],.3)]
    pooled=pool_economy_replicates(rows)
    assert len(pooled)==1
    assert pooled[0]['study']=='a'  # Deterministic source, not highest score.
    assert pooled[0]['score']==pytest.approx(.4)
    assert pooled[0]['independent_training_seeds']==6
    assert len(pooled[0]['replicated_controls'])==2
    assert pool_economy_replicates(list(reversed(rows)))==pooled
    rows[0]['family_eligible']=False;rows[0]['minimum_family_points']=0
    assert not pool_economy_replicates(rows)[0]['eligible']


def test_repeated_seed_rejected_and_different_initial_preserved():
    with pytest.raises(ValueError,match='Repeated training seed'):
        pool_economy_replicates([item('a',[1,2]),item('b',[2,3])])
    assert len(pool_economy_replicates([item('a',[1,2]),item('b',[1,2],initial='different')]))==2


def test_regression_in_any_replica_survives_pooling():
    a,b=item('a',[1,2]),item('b',[3,4])
    b['quality_diagnostics']=behavior_summary([comparison(3,delta=1,defense_delta=-1),comparison(4)])
    pooled=pool_economy_replicates([a,b])[0]
    assert pooled['family_eligible'] and not pooled['eligible']


@pytest.mark.parametrize('variant,module_name,function',[
    ('policy_adaptation_v1','verify_policy_adaptation_pilot','verify_adaptation'),
    ('calibrated_spell_cost_v1','verify_calibrated_spell_pilot','verify_calibrated'),
    ('group_entropy_v1','verify_group_entropy_pilot','verify_entropy'),
    ('air_exposure_v1','verify_air_exposure_pilot','verify_exposure'),
    ('spell_interaction_v1','verify_air_exposure_pilot','verify_interaction'),
])
def test_global_selection_requires_variant_specific_audit(tmp_path,monkeypatch,variant,module_name,function):
    import importlib,json
    from scripts.attack.select_pilot_pool import verify_study
    (tmp_path/'dataset').mkdir()
    (tmp_path/'dataset/protocol.json').write_text(json.dumps(dict(study_variant=variant)))
    module=importlib.import_module('scripts.attack.'+module_name)
    calls=[]
    monkeypatch.setattr(module,function,lambda collected,prepared:calls.append((collected,prepared))or{'audited':variant})
    def invalid_fallback(*args):raise AssertionError('Variant audit was bypassed')
    assert verify_study(tmp_path,invalid_fallback)=={'audited':variant}
    assert calls==[(tmp_path/'collected',tmp_path)]
