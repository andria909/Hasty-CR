from copy import deepcopy
import pytest
from scripts.attack.sequence_physics import physics_pins,require_current_physics


def test_missing_or_changed_simulator_pin_rejects_stale_demonstrations():
    good=dict(physics_pins=physics_pins());require_current_physics(good)
    with pytest.raises(ValueError,match='regenerate'):require_current_physics({})
    changed=deepcopy(good);changed['physics_pins']['sim/match.py']='old-shifted-slots'
    with pytest.raises(ValueError,match='regenerate'):require_current_physics(changed)


def test_execution_contract_is_pinned_and_old_extra_slot_lock_is_rejected():
    from scripts.attack.sequence_execution import EXECUTION,require_execution
    assert 'scripts/attack/sequence_execution.py' in physics_pins()
    require_execution(dict(execution=EXECUTION))
    with pytest.raises(ValueError,match='regenerate'):require_execution({})
    with pytest.raises(ValueError,match='regenerate'):
        require_execution(dict(execution=dict(EXECUTION,action_cooldown_ms=2100)))
