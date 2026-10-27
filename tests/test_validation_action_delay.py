import pytest
from sim.train_attack_ppo import validation_options, validate_resume


def test_training_delay_does_not_change_when_validation_is_held_constant():
    training=dict(action_delay_ms=2025,geometry='native_anchors_candidate_v1')
    assert validation_options(training,1350)['action_delay_ms']==1350
    assert training['action_delay_ms']==2025
    assert validation_options(training)==training


@pytest.mark.parametrize('delay',[-1,float('nan'),float('inf'),10001])
def test_invalid_validation_delay_is_rejected(delay):
    with pytest.raises(ValueError,match='validation action delay'):
        validation_options({},delay)


def test_resume_cannot_silently_change_validation_environment():
    blob=dict(schema='attack-ppo-boundary-v1',exact_resume_available=True,
              config=dict(validation_action_delay_ms=1350))
    with pytest.raises(ValueError,match='validation_action_delay_ms'):
        validate_resume(blob,dict(validation_action_delay_ms=225))
