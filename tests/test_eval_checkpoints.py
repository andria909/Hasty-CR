from sim.eval_checkpoints import compare_games, paired_sign_test, wilson_interval


def _game(seed, result, ours=0, theirs=0):
    return {"seed": seed, "result": result,
            "crowns_for": ours, "crowns_against": theirs}


def test_exact_sign_test_detects_fifteen_clean_improvements():
    result = paired_sign_test(15, 0)
    assert result["p_one_sided"] == 1 / 2 ** 15
    assert result["p_two_sided"] == 2 / 2 ** 15


def test_pairing_uses_seed_not_input_order():
    baseline = [_game(1, "top", 0, 3), _game(2, "bottom", 2, 1)]
    candidate = [_game(2, "bottom", 3, 0), _game(1, "bottom", 1, 0)]
    result = compare_games(baseline, candidate)
    assert (result["better"], result["worse"], result["tied"]) == (1, 0, 1)
    assert result["mean_crown_differential_gain"] == 3.0


def test_pairing_rejects_different_seed_sets():
    try:
        compare_games([_game(1, "top")], [_game(2, "bottom")])
    except ValueError as error:
        assert "seeds differ" in str(error)
    else:
        raise AssertionError("unpaired runs were accepted")


def test_wilson_interval_handles_zero_wins_without_zero_uncertainty():
    low, high = wilson_interval(0, 60)
    assert low == 0.0
    assert 0.05 < high < 0.07
