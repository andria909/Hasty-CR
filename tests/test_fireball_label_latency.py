import pytest
from scripts.attack.audit_fireball_label_latency import delayed_cast
from scripts.attack.evaluate_fireball_clusters import fixture_cases,cast_outcome,reference_candidates
from sim.eval_defense_duel import mechanical_mask


def test_parameterized_cast_reproduces_original225ms_exactly():
    _,match,cards,spells=next(fixture_cases(seeds=[3800009,3800057,3800085]))
    action=reference_candidates(match,mechanical_mask(match,cards,spells,1))[0]
    original_time=match.elapsed_ms
    assert delayed_cast(match,action,225)==cast_outcome(match,action)
    assert delayed_cast(match,action,1350)['cast_ms']==original_time+1350
    assert match.elapsed_ms==original_time and not match.battle.spell_cast_log


@pytest.mark.parametrize('delay',[-1,10001,float('nan')])
def test_invalid_delays_are_rejected(delay):
    with pytest.raises(ValueError,match='delay'):delayed_cast(None,0,delay)
