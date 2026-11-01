import json
from pathlib import Path

import pytest

from scripts.attack.evaluation_execution import HOG_CLIPPED,execution_for,combat_modes
from scripts.attack.verify_full_candidate_evaluation import validate_part
from scripts.kaggle.queue_full_candidate_evaluation import check_config,file_sha
from tests.test_full_candidate_verifier import setup,part
from tests.test_queue_full_candidate_evaluation import config


def test_new_blueprint_preserves_baselines_and_uses_fresh_test_blocks():
    p=json.loads(Path('data/ppo_tuning/hog_clip_evaluation_20261102/protocol.json').read_text())
    old=json.loads(Path(p['predecessor_blueprint']).read_text())
    assert p['baselines']==old['baselines'] and execution_for(p)==HOG_CLIPPED
    def seeds(x):return {s for t in x['test_sets']for s in t['seeds']}|set(x['tournament']['seeds'])
    assert len(seeds(p))==90 and not seeds(p)&seeds(old)
    assert p['required_candidate_training_execution']==HOG_CLIPPED


@pytest.mark.parametrize('kind',['test','duel'])
def test_version4_checks_actual_hog_geometry_and_delay_in_every_game(tmp_path,kind):
    protocol,blueprint,jobs=setup(tmp_path)
    protocol.update(schema='bound-full-candidate-evaluation-v4',execution=HOG_CLIPPED)
    job=next(j for j in jobs if j['kind']==kind);report=part(job,protocol,blueprint)
    for row in report['rows']:
        row.update(execution=HOG_CLIPPED,cannon_outside_center=0,**combat_modes(HOG_CLIPPED))
        if kind=='duel':row.update(executed_commands=2,actual_command_delay_min_ms=1350,actual_command_delay_max_ms=1350)
    validate_part(report,job,protocol,blueprint)
    report['rows'][0]['hog_sight_mode']='legacy_radial_v1'
    with pytest.raises(ValueError,match='hog_sight_mode'):validate_part(report,job,protocol,blueprint)


def test_queue_cannot_replace_pinned_current_blueprint(tmp_path):
    c=config(tmp_path);p=tmp_path/'blueprint.json'
    p.write_text(Path('data/ppo_tuning/hog_clip_evaluation_20261102/protocol.json').read_text())
    c.update(schema='queued-full-candidate-evaluation-v2',blueprint=str(p),blueprint_sha256=file_sha(p))
    check_config(c)
    p.write_text('{}')
    with pytest.raises(ValueError,match='blueprint changed'):check_config(c)
