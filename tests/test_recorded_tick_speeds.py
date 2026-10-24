import pytest

from scripts.gamestate.audit_recorded_tick_speeds import motion_windows,quantiles


def straight():return [(t,3500,3000+60*t,100)for t in range(0,41,2)]


def test_known_native_displacement_uses_ticks_instead_of_host_latency():
    assert motion_windows(straight())==[60.]


@pytest.mark.parametrize('change',['hp','gap','pause','turn'])
def test_nonconstant_or_undersampled_window_rejected(change):
    rows=straight()[:11]
    if change=='hp':rows[3]=(6,3500,3360,99)
    elif change=='gap':rows=rows[:2]+rows[6:]
    elif change=='pause':rows[3]=(6,3500,rows[2][2],100)
    else:rows=[(t,3500+60*min(t,10),3000+60*max(0,t-10),100)for t in range(0,21,2)]
    assert not motion_windows(rows)


def test_quantiles_handle_empty_and_singleton_without_making_up_samples():
    assert quantiles([])['median']is None
    assert quantiles([20.])==dict(n=1,p10=20.,median=20.,p90=20.)
