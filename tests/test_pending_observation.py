from scripts.attack.audit_pending_observation import audit


def test_current_inputs_omit_position_of_paid_pending_command():
    r=audit('data/late_air_stable_diverse_sequences_20261027')
    assert not r['train_conflict_groups'] and r['cross_split_identical_inputs']==0
    assert all(r['counterexample']['input_arrays_equal'].values())
    outcomes=r['counterexample']['outcomes']
    assert all(x['valid'] and x['elixir_spent']==5 for x in outcomes)
    assert outcomes[0]['total_damage']!=outcomes[1]['total_damage']
