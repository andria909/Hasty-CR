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


def test_probability_audit_separates_card_mass_from_spatial_accuracy():
    import math
    import pytest
    import torch
    from scripts.attack.score_pending_aliasing_policies import probability_diagnostic
    logits = torch.full((2321,), 10000.)  # Illegal logits must not affect the sums.
    mask = torch.zeros(2321, dtype=torch.bool)
    for action, probability in ((0, .5), (577, .3), (578, .2)):
        mask[action] = True
        logits[action] = math.log(probability)
    worlds = [{a: {'outcome': {'total_damage': d}} for a, d in damages.items()}
              for damages in ({0: 100, 577: 0, 578: 50}, {0: 40, 577: 0, 578: 10})]
    result = probability_diagnostic(logits, mask, ['musketeer', 'fireball', 'cannon', 'electro_spirit'],
                                    {'common_minimum_damage_actions': [577], 'minimum_damage': [0, 0]}, worlds)
    assert result['card_probabilities']['fireball'] == pytest.approx(.5)
    assert result['shared_minimum_hp_probability'] == pytest.approx(.3)
    assert result['conditional_spatial']['fireball']['conditional_good_probability'] == pytest.approx(.6)
    assert result['best_shared_hp_action_rank'] == 2
    assert result['best_shared_hp_action'] == 577
    assert result['expected_one_command_hp_regret'] == pytest.approx([60, 22])
