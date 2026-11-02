from copy import deepcopy
import json
from pathlib import Path
import pytest
from scripts.kaggle.prepare_spell_interaction_pilot import validate_protocol
from scripts.kaggle.prepare_air_exposure_pilot import validate_protocol as validate_air


def test_interaction_grid_preserves_single_initializer_and_factorial_controls():
    p=json.loads(Path('data/ppo_tuning/spell_interaction_design_20261102.json').read_text())
    validate_protocol(p)
    for key,value in [('focus','air_golem'),('crown_only_weight',.1),('dormant_king_weight',.3),
                      ('spell_waste_weight',.5),('initial_id','bank'),('action_delay_ms',225)]:
        q=deepcopy(p);q['cases'][3][key]=value
        with pytest.raises(ValueError):validate_protocol(q)
    q=deepcopy(p);q['paired_controls']['king03']='control01'
    with pytest.raises(ValueError):validate_protocol(q)
    # The existing active study must not silently accept the new treatment.
    with pytest.raises(ValueError):validate_air(p)


def test_increasing_king_cost_is_not_allowed_in_existing_air_study():
    p=json.loads(Path('data/ppo_tuning/air_exposure_design_20261102.json').read_text())
    p['study_variant']='air_exposure_v1';validate_air(p)
    for r in p['cases']:r['dormant_king_weight']=.4
    with pytest.raises(ValueError):validate_air(p)
