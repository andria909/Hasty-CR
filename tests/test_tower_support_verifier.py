import pytest
from scripts.attack.verify_tower_support_pilot import audit_reward


def test_actual_accounting_and_missing_or_wrong_coefficients():
    config=dict(tower_support_weight=.15)
    row=dict(tower_support_weight=.15,tower_support_paid_value=2.,tower_support_reward=.3)
    audit_reward(config,[row],.15)
    for bad in ({},dict(row,tower_support_reward=.4),dict(row,tower_support_weight=0),
                dict(row,tower_support_paid_value=float('nan'))):
        with pytest.raises(ValueError):audit_reward(config,[bad],.15)
    with pytest.raises(ValueError):audit_reward({},[row],.15)
    with pytest.raises(ValueError):audit_reward(config,[],.15)
