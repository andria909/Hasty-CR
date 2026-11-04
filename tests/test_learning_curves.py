from copy import deepcopy

import pytest

from scripts.attack.audit_learning_curves import summarize
from tests.test_paired_ppo_behavior import records


def test_learning_curve_retains_task_denominators_and_actual_card_plays():
    before=records();after=deepcopy(before)
    after[0]['outcome']='win';after[6]['outcome']='win'
    after[6]['accepted_card_counts']={'musketeer':3}
    result=summarize(list(reversed(after)),before)
    assert result['games']==72 and result['wins']==2
    assert result['defense_games']==36 and result['defense_wins']==1
    assert result['card_plays_per_game']['musketeer']==pytest.approx(3/72)
    assert result['worst_family_points']==0


def test_later_checkpoint_cannot_change_validation_contexts():
    before=records();after=deepcopy(before);after[0]['episode_seed']+=9000
    with pytest.raises(ValueError,match='changed validation contexts'):summarize(after,before)
