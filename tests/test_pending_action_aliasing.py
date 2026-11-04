from scripts.attack.diagnose_pending_action_aliasing import compare


def outcome(damage, spent):
    return dict(own_king_dead=False, remaining_enemies=0,
                total_damage=damage, elixir_spent=spent)


def test_equal_damage_does_not_imply_shared_economical_optimum():
    a = {0: outcome(0, 4), 1: outcome(0, 5)}
    b = {0: outcome(640, 4), 1: outcome(0, 5)}
    result = compare([a, b])
    assert result['common_optimal_actions'] == []
    assert result['common_minimum_damage_actions'] == [1]
    assert result['best_shared_worst_case_hp_regret'] == 0


def test_regret_retains_ties_and_detects_conflicting_damage_optima():
    a = {0: outcome(0, 5), 1: outcome(0, 5), 2: outcome(640, 5)}
    b = {0: outcome(100, 5), 1: outcome(100, 5), 2: outcome(0, 5)}
    result = compare([a, b])
    assert result['optimal_actions'] == [[0, 1], [2]]
    assert result['common_minimum_damage_actions'] == []
    assert result['shared_minimax_hp_actions'] == [0, 1]
    assert result['best_shared_worst_case_hp_regret'] == 100
