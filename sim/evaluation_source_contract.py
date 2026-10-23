"""Validate the one declared scripted-target migration during evaluation only."""
import hashlib
import json
from pathlib import Path


JOINT_FILES=('sim/attack_env.py','sim/attack_scenarios.py','sim/attack_sampler.py',
    'sim/train_attack_ppo.py','sim/engine.py','sim/match.py','sim/env.py',
    'sim/defense_env.py','sim/defense_assets.py','sim/defense_scenarios.py',
    'sim/public_gamedata.py','sim/public_scenario_spells.py','sim/spells.py','sim/arena.py',
    'sim/reward_profiles.py','sim/strategy_env.py','sim/strategy_sampler.py')


def verify_joint_source(actual,expected,original_files=None,root=None):
    if actual==expected:return
    if original_files is None:raise ValueError('Evaluation/training source semantics differ')
    root=Path(root)if root is not None else Path(__file__).resolve().parents[1]
    current={name:hashlib.sha256((root/name).read_bytes()).hexdigest()for name in JOINT_FILES}
    original={name:original_files[name]for name in JOINT_FILES}
    fingerprint=lambda files:hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    if fingerprint(current)!=expected or fingerprint(original)!=actual:
        raise ValueError('Declared evaluation/training fingerprints differ')
    if {name for name in current if current[name]!=original[name]}!={'sim/defense_env.py'}:
        raise ValueError('Only the declared scripted-target migration is supported')
    text=(root/'sim/defense_env.py').read_text()
    expression='return arena.clamp_to_arena(chosen.pos)'
    if text.count(expression)!=1 or hashlib.sha256(text.replace(expression,'return chosen.pos').encode()).hexdigest()!=original['sim/defense_env.py']:
        raise ValueError('Scripted-target migration changes unrelated environment semantics')
