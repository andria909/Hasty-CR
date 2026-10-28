"""Manual reporting must enforce the same study audit as background reporting."""
import json
import pytest
from scripts.attack.summarize_clock_pilots import summarize


@pytest.mark.parametrize('variant,module,function',[
    ('own_pending_inputs_v1','scripts.attack.verify_frozen_pending','verify'),
    ('pending_diversity_slots_v1','scripts.attack.verify_frozen_pending','verify'),
    ('pending_sequence_retention_v1','scripts.attack.verify_frozen_pending','verify'),
    ('late_air_sequential_teacher_v1','scripts.attack.verify_sequence_pilot','verify_sequence'),
    ('native_fireball_slot_v1','scripts.attack.verify_native_fireball_slot','verify_native_fireball'),
    ('tower_support_reward_v1','scripts.attack.verify_tower_support_pilot','verify_tower_support'),
    ('spell_waste_reward_v1','scripts.attack.verify_spell_waste_pilot','verify_spell_waste'),
    ('central_tower_damage_v1','scripts.attack.verify_central_tower_pilot','verify_central'),
    ('combined_tower_spell_reward_v1','scripts.attack.verify_combined_reward_pilot','verify_combined'),
    ('paid_tactical_fullgame_starts_v1','scripts.attack.verify_tactical_starts_pilot','verify_tactical'),
    ('legacy','scripts.attack.summarize_clock_pilots','verify'),
])
def test_manual_summary_cannot_bypass_study_verifier(tmp_path,monkeypatch,variant,module,function):
    import importlib
    (tmp_path/'dataset').mkdir()
    (tmp_path/'dataset/protocol.json').write_text(json.dumps({'study_variant':variant}))
    calls=[]
    def reject(collected,prepared):
        calls.append((collected,prepared))
        raise ValueError('study audit rejected')
    monkeypatch.setattr(importlib.import_module(module),function,reject)
    with pytest.raises(ValueError,match='study audit rejected'):
        summarize(tmp_path)
    assert calls==[(tmp_path/'collected',tmp_path)]
