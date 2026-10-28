import pytest
from scripts.attack.verify_spell_waste_pilot import audit_reward


def test_emptyspell_accounting_and_double_penalty_rejected():
    config=dict(spell_waste_weight=.5,continuous_profile={'missed_fireball_penalty':0.})
    row=dict(spell_waste_weight=.5,empty_spell_elixir=6.,empty_spell_penalty=3.,
             empty_spell_elixir_by_card={'zap':2.,'fireball':4.})
    audit_reward(config,[row],.5)
    for bad in ({},dict(row,empty_spell_penalty=4.),dict(row,empty_spell_elixir_by_card={'zap':2}),
                dict(row,empty_spell_elixir=float('nan'))):
        with pytest.raises(ValueError):audit_reward(config,[bad],.5)
    config['continuous_profile']['missed_fireball_penalty']=.1
    with pytest.raises(ValueError):audit_reward(config,[row],.5)


def test_legacy_control_requires_original_penalty():
    audit_reward(dict(spell_waste_weight=None,continuous_profile={'missed_fireball_penalty':.1}),[{}],None)
    with pytest.raises(ValueError):
        audit_reward(dict(spell_waste_weight=None,continuous_profile={'missed_fireball_penalty':0.}),[{}],None)
