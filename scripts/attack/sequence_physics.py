"""Bind sequential demonstration states to the simulator implementation."""
from scripts.kaggle.local_artifact_transfer import file_sha

FILES=('sim/match.py','sim/continuous_push.py','sim/eval_defense_duel.py',
       'sim/engine.py','sim/entities.py','sim/spells.py','sim/adapter.py','sim/cannon_placement.py','scripts/attack/sequence_execution.py')


def physics_pins():
    return {name:file_sha(name) for name in FILES}


def require_current_physics(record):
    if record.get('physics_pins')!=physics_pins():
        raise ValueError('Sequential data physics differs or is unpinned; regenerate states')
