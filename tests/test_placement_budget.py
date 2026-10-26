import pytest
from scripts.gamestate.placement_budget import PlacementBudget


def test_other_commands_can_spend_only_the_unreserved_balance():
    b=PlacementBudget();b.observe(100,10);b.spend(4)
    assert b.observe(100,10)==6
    b.spend(3);assert b.observe(100,10)==3
    with pytest.raises(ValueError):b.spend(4)


def test_fresh_native_deduction_is_not_subtracted_twice():
    b=PlacementBudget();b.observe(100,10);b.spend(4)
    assert b.observe(102,6)==6
    b.reconcile(6,all_confirmed=True)
    b.spend(3);assert b.value==3


def test_confirmation_reconciles_budget_without_extra_global_delay():
    b=PlacementBudget();b.observe(100,8);b.spend(3)
    b.reconcile(5.1,all_confirmed=False);assert b.value==5
    b.reconcile(5.1,all_confirmed=True);assert b.value==5.1
