from scripts.attack.watch_air_reserve_diagnostics import summary


def game(seed, frames, below, legal, outcome, musk, loss):
    return dict(seed=seed, unprotected_frames=frames, unprotected_bank_below_four=below,
        unprotected_musketeer_legal=legal,
        metrics=dict(outcome=outcome, own_fraction_loss=loss, accepted_card_counts={'musketeer':musk}))


def test_diagnostic_uses_pooled_frames_and_per_game_card_counts():
    result=summary(dict(checkpoint_sha256='weight', games=[
        game(1, 10, 8, 1, 'win', 4, .2), game(2, 90, 0, 70, 'loss', 2, 1.)]))
    assert result['bank_below_four_fraction']==.08
    assert result['mean_card_plays']['musketeer']==3
    assert result['mean_card_plays']['fireball']==0
    assert result['mean_own_tower_fraction_loss']==.6
    assert result['wins']==1 and result['games']==2 and result['seeds']==[1,2]


def test_no_unprotected_frames_is_absence_of_condition_not_perfect_bank():
    result=summary(dict(checkpoint_sha256='weight',games=[game(1,0,0,0,'win',2,0)]))
    assert result['bank_below_four_fraction'] is None
