"""Event-based rewards for mechanics that tower damage cannot see."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpellReward:
    quality: float = 0.0
    prediction: float = 0.0
    missed: bool = False


def score_kite(*, retargeted: int, threats: int, displaced_mt: int,
               absorbed_hp: int, cheap_max_hp: int,
               tower_fraction_lost: float, center_time_ms: int = 0,
               defender_survival: float = 0.0) -> float:
    """Bounded outcome score for a cheap defensive kite.

    Merely playing a one-elixir card earns nothing. At least one dangerous
    enemy must actually retarget to it; displacement and damage absorbed then
    add credit, while tower damage during the window removes it.
    """
    if retargeted <= 0 or threats <= 0:
        return 0.0
    retarget = min(1.0, retargeted / threats)
    displacement = min(1.0, max(0, displaced_mt) / 4000.0)
    absorbed = min(1.0, max(0, absorbed_hp) / max(1, cheap_max_hp))
    crossfire = min(1.0, max(0, center_time_ms) / 3000.0)
    survival = min(1.0, max(0.0, defender_survival))
    protected = max(0.0, 1.0 - max(0.0, tower_fraction_lost) * 5.0)
    return (0.35 * retarget + 0.25 * displacement + 0.15 * absorbed
            + 0.15 * crossfire + 0.10 * survival) * protected


def score_spell_cast(event: dict, *, our_side: int = 1) -> SpellReward:
    """Score actual spell hits without rewarding overkill or tower chip twice.

    Quality is effective unit HP removed as a fraction of each target's
    maximum HP, plus a small kill-value bonus measured in elixir. Tower hits
    are excluded because the main reward already scores them.

    Prediction only counts an enemy deployed after the cast but hit by that
    already-airborne spell. A 1.5 second falloff favours a deliberate read
    over an unrelated unit wandering into an old area effect.
    """
    if not event.get("resolved") or event.get("caster_side") != our_side:
        return SpellReward()
    if event.get("side") != our_side:  # reflected projectile
        return SpellReward(missed=True)

    quality = prediction = 0.0
    enemy_hits = 0
    cast_ms = int(event.get("cast_ms", 0))
    for hit in event.get("hits", ()):
        if hit.get("target_side") == our_side:
            continue
        damage = max(0, int(hit.get("damage", 0)))
        if damage <= 0:
            continue
        enemy_hits += 1
        # Tower damage is already paid by the main symmetric reward. It still
        # proves the cast was not a miss (important for legitimate finishers),
        # but receives no second quality credit here.
        if hit.get("is_tower"):
            continue
        maximum = max(1, int(hit.get("max_hp", damage)))
        value = min(1.0, damage / maximum)
        if hit.get("killed"):
            value += 0.25 * max(0, int(hit.get("elixir_milli", 0))) / 1000.0
        quality += value

        spawned_ms = int(hit.get("spawned_ms", -1))
        hit_ms = int(hit.get("hit_ms", event.get("impact_ms", cast_ms)))
        if cast_ms < spawned_ms <= hit_ms:
            timing = max(0.0, 1.0 - (hit_ms - spawned_ms) / 1500.0)
            prediction += value * timing
    return SpellReward(quality=quality, prediction=prediction,
                       missed=(enemy_hits == 0))
