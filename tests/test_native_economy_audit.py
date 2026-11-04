import pytest
from scripts.attack.audit_native_economy import aggregate


def row(n,native_below,available_below,native_sum,available_sum):
    return dict(decision_economy=dict(decisions=n,native_below4=native_below,available_below4=available_below,
        native_elixir_sum=native_sum,available_elixir_sum=available_sum))


def test_reservations_are_separated_and_weighted_by_decisions():
    result=aggregate([row(10,2,5,60,45),row(90,88,89,180,150)])
    assert result['native_below4_fraction']==.9
    assert result['available_below4_fraction']==.94
    assert result['reservation_only_below4_fraction']==.04
    assert result['mean_native_elixir']==2.4
    assert result['mean_reserved_gap']==.45


@pytest.mark.parametrize('bad',[row(10,6,5,40,30),row(10,0,0,101,30),row(10,0,0,30,40),row(0,0,0,0,0)])
def test_impossible_native_reserved_accounting_is_rejected(bad):
    with pytest.raises(ValueError):aggregate([bad])
