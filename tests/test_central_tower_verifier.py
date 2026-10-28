import pytest
from scripts.attack.verify_central_tower_pilot import audit_priority


def test_validation_cannot_silently_use_old_mask_cooldown_or_reward():
    c=dict(tower_support_weight=.4,spell_waste_weight=.1,continuous_profile=dict(
        cannon_center_only=1,action_cooldown_ms=0,tower_damage_all=1,missed_fireball_penalty=0.))
    r=dict(cannon_center_only=1,cannon_outside_center=0,action_cooldown_ms=0,tower_damage_reward_mode='all_damage',
        tower_support_weight=.4,tower_support_paid_value=2.,tower_support_reward=.8,
        spell_waste_weight=.1,empty_spell_elixir=6.,empty_spell_penalty=.6,
        empty_spell_elixir_by_card={'zap':2.,'fireball':4.})
    audit_priority(c,[r],.4)
    for changes in [dict(cannon_center_only=0),dict(cannon_outside_center=1),dict(action_cooldown_ms=2100),
                    dict(tower_damage_reward_mode='assisted'),dict(tower_support_reward=1.)]:
        with pytest.raises(ValueError):audit_priority(c,[dict(r,**changes)],.4)
    with pytest.raises(ValueError):audit_priority(dict(c,continuous_profile=dict(c['continuous_profile'],action_cooldown_ms=2100)),[r],.4)
