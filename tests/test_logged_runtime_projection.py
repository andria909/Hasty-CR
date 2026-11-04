import pytest
from scripts.attack.project_logged_runtime import project


def test_evaluation_frequency_is_not_multiplied_like_training_steps():
    result=project(0,{1:5,2:25,3:27,4:47},pilot_updates=4,pilot_eval_every=2,full_updates=8,full_eval_every=4)
    assert result['observed_pilot_seconds']==47
    assert result['full_evaluations']==2
    assert result['estimated_full_seconds']==55
    assert result['slowest_observed_blocks_full_seconds']==55


def test_incomplete_or_regressed_remote_timing_cannot_predict_completion():
    with pytest.raises(ValueError,match='consecutive'):project(0,{1:1},pilot_updates=4)
    with pytest.raises(ValueError,match='regressed'):
        project(0,{1:5,2:4,3:6,4:8},pilot_updates=4,pilot_eval_every=2)
    with pytest.raises(ValueError,match='Finite'):
        project(0,{1:5,2:25,3:float('nan'),4:47},pilot_updates=4,pilot_eval_every=2)
