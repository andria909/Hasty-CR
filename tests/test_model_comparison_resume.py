import pytest
from scripts.gamestate.run_model_comparison import remaining_matches


def state():
    return dict(requested=10, finished=4, wins=1, losses=3, unknown=0,
                runs=['old'], matches=[dict(run='old') for _ in range(4)])


def test_resume_only_remaining_games_across_multiple_runs():
    saved=state()
    assert remaining_matches(saved,dict(started=4,finished=4))==6
    saved['runs'].append('new')
    assert remaining_matches(saved,dict(started=0,finished=0))==6


@pytest.mark.parametrize('old', [dict(started=5,finished=4),dict(started=5,finished=5)])
def test_unfinished_or_uncounted_match_cannot_resume(old):
    with pytest.raises(ValueError):remaining_matches(state(),old)


def test_corrupt_count_cannot_reset_series():
    saved=state();saved['finished']=0
    with pytest.raises(ValueError):remaining_matches(saved,dict(started=4,finished=4))
