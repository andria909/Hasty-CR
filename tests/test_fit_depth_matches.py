import json
from pathlib import Path

import pytest

from scripts.attack.select_continuous_focus import validation_score
from scripts.attack.sequence_physics import physics_pins
from scripts.attack.summarize_correction_full_games import summarize
from scripts.kaggle import watch_fit_depth_matches as watcher
from scripts.kaggle.local_artifact_transfer import file_sha


def fixture(tmp_path):
    """Synthetic scoring/identity fixtures, not models or actual games."""
    (tmp_path/'dataset').mkdir();parent=tmp_path/'dataset/parent.pt';parent.write_bytes(b'parent fixture')
    runs=[]
    for mode in ('isolated','mixed'):
        for budget in (800,8000):
            for seed in (402,403):
                name=f'{mode}{budget}_s{seed}';path=tmp_path/(name+'.pt');path.write_bytes(name.encode())
                runs.append(dict(id=name,checkpoint=str(path),checkpoint_sha256=file_sha(path),seed=seed,
                    case=dict(id=f'{mode}{budget}',dataset='control'if mode=='mixed'else'corrections',updates=budget,kl_weight=.01)))
    proof=dict(schema='verified-bank-fit-depth-v1',actual_cuda_verified=True,supervised_updates_verified=35200,
        initial_sha256=file_sha(parent),runs=runs)
    return proof


def test_only_all_four_mixed_candidates_are_compared(tmp_path):
    proof=fixture(tmp_path);weights=watcher.weights_for_comparison(tmp_path,proof)
    assert len(weights)==5 and weights[0].endswith('parent.pt')
    assert all('mixed' in Path(w).name for w in weights[1:])
    Path(weights[1]).write_bytes(b'changed')
    with pytest.raises(ValueError,match='Candidate changed'):watcher.weights_for_comparison(tmp_path,proof)


def test_incomplete_gpu_evidence_does_not_launch_validation(tmp_path,monkeypatch):
    proof=fixture(tmp_path);proof['actual_cuda_verified']=False
    verified=tmp_path/'verified.json';verified.write_text(json.dumps(proof));state=tmp_path/'state.json'
    monkeypatch.setattr(watcher,'full_validation',lambda *a:pytest.fail('Invalid evidence must not launch matches'))
    with pytest.raises(ValueError,match='eight independently'):
        watcher.watch(tmp_path,verified,state,'config','summary','md','behavior','behavior_md')
    assert json.loads(state.read_text())['status']=='inspection_required'


def test_paired_budget_summary_uses_360_games_and_no_isolated_policy(tmp_path):
    proof=fixture(tmp_path);weights=watcher.weights_for_comparison(tmp_path,proof)
    models=[dict(checkpoint=w,sha256=file_sha(w))for w in weights];reports=[]
    for index,model in enumerate(models):
        records=[dict(episode_seed=i,match_seed=1000+i,task='attack'if i<36 else'defense',
            family_id=('air','bridge','giant','golem','hog','swarm')[i%6],split='validation',
            outcome='win'if i<index else'loss',empty_spell_elixir=0,elixir_overflow=0)for i in range(72)]
        reports.append(dict(model=model,records=records,score=validation_score(dict(records=records)),wins=index,games=72))
    driver=tmp_path/'driver.py';driver.write_text('# fixture')
    raw=dict(protocol=dict(models=models,physics_pins=physics_pins(),source_sha256=file_sha(driver),
        split='validation',independent_test=False,live_games=False),reports=reports)
    a,b=tmp_path/'verified.json',tmp_path/'validation.json';a.write_text(json.dumps(proof));b.write_text(json.dumps(raw))
    result=summarize(a,b,tmp_path/'result.json',tmp_path/'result.md')
    assert result['schema']=='verified-fit-depth-full-games-v1'and result['games_verified']==360
    assert len(result['groups'])==2 and all(r['wins_delta']==2 for r in result['paired_deltas'])
    assert result['full_training_accepted']is False
    raw['reports'][1]['records'][0]['match_seed']+=10000;b.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='Unpaired'):summarize(a,b,tmp_path/'bad.json',tmp_path/'bad.md')
    raw['reports'][1]['records'][0]['match_seed']-=10000
    raw['reports'][1]['records'][0]['task']='defense';b.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='six defense'):summarize(a,b,tmp_path/'bad_task.json',tmp_path/'bad_task.md')
