from sim.reward_events import score_kite, score_spell_cast


def _event(*hits, cast=1000, side=1, resolved=True):
    return {"caster_side": 1, "side": side, "cast_ms": cast,
            "impact_ms": 2000, "resolved": resolved, "hits": list(hits)}


def _hit(*, damage=500, hp=1000, killed=False, elixir=0,
         spawned=0, at=2000, tower=False):
    return {"target_side": -1, "damage": damage, "max_hp": hp,
            "killed": killed, "elixir_milli": elixir,
            "spawned_ms": spawned, "hit_ms": at, "is_tower": tower}


def test_spell_quality_uses_effective_hp_and_kill_value():
    score = score_spell_cast(_event(
        _hit(damage=500, hp=1000),
        _hit(damage=200, hp=800, killed=True, elixir=4000)))
    assert score.quality == 1.75
    assert not score.missed


def test_tower_chip_is_not_counted_twice():
    score = score_spell_cast(_event(_hit(tower=True)))
    assert score.quality == 0
    assert score.missed


def test_prediction_requires_cast_before_deploy_and_hit_after_it():
    predicted = score_spell_cast(_event(_hit(spawned=1500, at=1750)))
    already_visible = score_spell_cast(_event(_hit(spawned=500, at=1750)))
    assert predicted.prediction > 0
    assert already_visible.prediction == 0


def test_reflected_spell_does_not_reward_original_caster():
    score = score_spell_cast(_event(_hit(), side=-1))
    assert score.quality == score.prediction == 0
    assert score.missed


def test_unresolved_cast_is_not_scored_early():
    assert score_spell_cast(_event(_hit(), resolved=False)).quality == 0


def test_merely_playing_a_cheap_card_is_not_a_kite():
    assert score_kite(retargeted=0, threats=2, displaced_mt=4000,
                      absorbed_hp=500, cheap_max_hp=500,
                      tower_fraction_lost=0) == 0


def test_successful_kite_rewards_retarget_displacement_and_absorption():
    weak = score_kite(retargeted=1, threats=2, displaced_mt=0,
                      absorbed_hp=0, cheap_max_hp=500,
                      tower_fraction_lost=0)
    strong = score_kite(retargeted=2, threats=2, displaced_mt=4000,
                        absorbed_hp=500, cheap_max_hp=500,
                        tower_fraction_lost=0)
    assert 0 < weak < strong <= 1


def test_kite_credit_falls_when_the_tower_still_takes_damage():
    safe = score_kite(retargeted=1, threats=1, displaced_mt=2000,
                      absorbed_hp=300, cheap_max_hp=500,
                      tower_fraction_lost=0)
    leaked = score_kite(retargeted=1, threats=1, displaced_mt=2000,
                        absorbed_hp=300, cheap_max_hp=500,
                        tower_fraction_lost=0.1)
    assert leaked < safe
