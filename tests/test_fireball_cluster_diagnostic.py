from copy import deepcopy
from scripts.attack.evaluate_fireball_clusters import fixture_cases,cast_outcome,reference_candidates
from sim.eval_defense_duel import mechanical_mask


def test_paid_fixture_and_candidate_cast_preserve_source_state():
    label,m,cards,spells=next(fixture_cases())
    assert label['enemy_elixir_paid']==5
    mask=mechanical_mask(m,cards,spells,1)
    candidates=reference_candidates(m,mask)
    before=m.elapsed_ms;bank=m.players[1].elixir
    results=[cast_outcome(m,a)for a in candidates]
    assert any(r['damaged_units']>=3 and r['fireball_exchange_value']>0 for r in results)
    assert all(r['cast_ms']-r['request_ms']>=225 and r['impact_ms']>=r['cast_ms']for r in results)
    assert m.elapsed_ms==before and m.players[1].elixir==bank
    assert not m.battle.spell_cast_log
