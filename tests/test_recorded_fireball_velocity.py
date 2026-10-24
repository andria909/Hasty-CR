import pytest

from scripts.gamestate.audit_recorded_fireball import fit


def test_observed_xy_velocity_is_fitted_without_assuming_600_per_tick():
    for speed in (500,600):
        result=fit([(t,9000,29000-speed*t)for t in (0,2,5,8,12)])
        assert result['speed_millitiles_per_tick']==pytest.approx(speed)
        assert result['max_fit_residual_millitiles']==pytest.approx(0)


def test_nonatomic_tick_outlier_or_acceleration_rejected():
    assert fit([(0,0,0),(3,0,1800),(6,0,4200),(9,0,5400)])is None


def test_insufficient_flight_span_not_extrapolated():
    assert fit([(0,0,0),(1,0,600),(2,0,1200)])is None
