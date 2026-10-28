"""Native trace filters and the actual damage scheduler share measured units."""
import pytest

from scripts.gamestate.audit_projectile_motion_clips import fitted_clip
from sim.arena import Point
from sim.engine import Battle
from sim.entities import make_unit
from sim.gamedata import load_gamedata
from sim.projectile_calibration import projectile_speed
from sim.public_gamedata import load_public_gamedata


def test_contiguous_trace_measures_speed_and_rejects_clock_corruption():
    # Actual native Cannon XY points, strong run frames7442–7446, tick1656–1661.
    points = [(1656, 10141, 14266), (1657, 10782, 15032),
              (1660, 12707, 17330), (1661, 13349, 18097)]
    result = fitted_clip(points)
    assert result['speed_millitiles_per_tick'] == pytest.approx(999.327, abs=.01)
    assert fitted_clip(points + [(1661, 13349, 18097)]) == result
    assert fitted_clip(points + [(1661, 13350, 18097)]) is None
    assert fitted_clip([points[0], points[1], (1670, 19115, 24995)]) is None
    assert fitted_clip(points[::-1]) is None
    assert fitted_clip([(1, 0, 0), (4, 3000, 0), (6, 3000, 2000)]) is None


@pytest.mark.parametrize('card,expected', [('cannon', 20000), ('archers', 12000)])
def test_native_and_portable_loaders_have_same_measured_speed(tmp_path, card, expected):
    current = load_gamedata(11)[card].unit
    portable = load_gamedata(11, root=tmp_path/'absent')[card].unit
    public, _ = load_public_gamedata(11)
    assert current.projectile_speed_mt_per_sec == expected
    assert portable.projectile_speed_mt_per_sec == expected
    assert public[card].unit.projectile_speed_mt_per_sec == expected


@pytest.mark.parametrize('card,gap,before_speed,old_ms,new_ms', [
    ('cannon', 5000, 16666, 300, 250),
    ('archers', 6000, 10000, 600, 500),
])
def test_measured_speed_changes_actual_first_impact(card, gap, before_speed, old_ms, new_ms):
    cards = load_gamedata(11)

    def run(speed=None):
        battle = Battle()
        shooter = battle.add(make_unit(1, cards[card].unit, 1, Point(9000, 23000)))
        target = battle.add(make_unit(2, cards['knight'].unit, -1, Point(9000, 23000-gap)))
        for body in (shooter, target):
            body.deploy_remaining_ms = 0
            body.speed_mt_per_sec = 0
        target.damage = 0
        shooter.target_uid = target.uid
        shooter.attack_cooldown_ms = shooter.windup_remaining_ms = 0
        if speed is not None:
            shooter.projectile_speed_mt_per_sec = speed
        assert battle._attack(shooter, 50)
        before = target.hitpoints
        for _ in range(25):
            battle.step(50)
            if target.hitpoints < before:
                return battle.now_ms
        pytest.fail('shot never caused damage')

    # Test the scheduler, not just distance/speed arithmetic. Both phases are
    # explicitly ready at t=0; this is not a claim about native first windup.
    assert run(before_speed) == old_ms
    assert run() == new_ms


def test_calibration_does_not_change_unmeasured_or_changed_raw_speed():
    assert projectile_speed('UnknownProjectile', 600) == 10000
    assert projectile_speed('TowerCannonball', 999) == 16650
