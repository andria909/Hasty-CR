import pytest
from sim.continuous_validation import score_families,FAMILIES


def cases(rates):
    return [dict(split='validation',family_id=name,outcome='win' if i<n else 'loss')
            for name,n in zip(FAMILIES,rates)for i in range(10)]


def test_balanced_policy_beats_more_specialized_higher_macro_policy():
    specialized=score_families(cases([0,10,10,0,10,10]))
    balanced=score_families(cases([5,5,5,5,5,5]))
    assert specialized['macro_archetype_points']>balanced['macro_archetype_points']
    assert balanced['selection_score']>specialized['selection_score']


def test_selector_refuses_test_and_missing_archetype():
    rows=cases([5]*6);rows[0]['split']='test'
    with pytest.raises(ValueError):score_families(rows)
    with pytest.raises(ValueError):score_families(cases([5]*5))
