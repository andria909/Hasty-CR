import pytest
from scripts.attack.audit_pending_observation import audit,pending_counterexample


def test_current_inputs_omit_position_of_paid_pending_command():
    r=pending_counterexample()
    assert all(r['input_arrays_equal'].values())
    outcomes=r['outcomes']
    assert all(x['valid'] and x['elixir_spent']==5 for x in outcomes)
    assert outcomes[0]['total_damage']!=outcomes[1]['total_damage']


def test_old_demonstrations_are_rejected_after_physics_changes():
    with pytest.raises(ValueError,match='regenerate'):
        audit('data/late_air_stable_diverse_sequences_20261027')
