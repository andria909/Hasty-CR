import pytest
from scripts.kaggle.prepare_full_evaluation_recovery import corrected_checker,OLD,NEW


def test_only_known_checker_change_is_permitted():
    old='before\n'+OLD+'\nafter\n';new='before\n'+NEW+'\nafter\n'
    assert corrected_checker(old,new)==new
    for changed in (new+'unrelated',new.replace('before','modified'),old):
        with pytest.raises(ValueError):corrected_checker(old,changed)
    with pytest.raises(ValueError):corrected_checker(old+OLD,new)
