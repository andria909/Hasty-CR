import json

import pytest

from scripts.attack.summarize_correction_full_games import summarize
from scripts.attack.select_continuous_focus import validation_score
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha


def fixture(tmp_path):
    """Synthetic metadata fixtures only; these are not trained policies."""
    driver=tmp_path/'driver.py';driver.write_text('# synthetic test fixture')
    parent=tmp_path/'parent.pt';parent.write_bytes(b'fixture parent')
    models=[dict(checkpoint=str(parent),sha256=file_sha(parent))];runs=[]
    for dataset in ('control','corrections'):
        for tag,kl in (('001',.01),('01',.1)):
            for seed in (402,403):
                name=f'{dataset}_kl{tag}_s{seed}';path=tmp_path/(name+'.pt');path.write_bytes(name.encode())
                models.append(dict(checkpoint=str(path),sha256=file_sha(path)))
                runs.append(dict(id=name,case=dict(id=f'{dataset}_kl{tag}',dataset=dataset,kl_weight=kl),
                    seed=seed,checkpoint_sha256=file_sha(path)))
    reports=[]
    for index,model in enumerate(models):
        records=[dict(episode_seed=i,match_seed=1000+i,task='attack'if i<36 else'defense',
            family_id=('air','bridge','giant','golem','hog','swarm')[i%6],split='validation',
            outcome='win'if i<index else'loss',empty_spell_elixir=0,elixir_overflow=0)for i in range(72)]
        reports.append(dict(model=model,records=records,score=validation_score(dict(records=records)),wins=index,games=72))
    proof=dict(schema='verified-bank-correction-pair-v1',actual_cuda_verified=True,
        supervised_updates_verified=6400,initial_sha256=file_sha(parent),runs=runs)
    raw=dict(protocol=dict(models=models,physics_pins=physics_pins(),source_sha256=file_sha(driver),
        split='validation',independent_test=False,live_games=False),reports=reports)
    a,b=tmp_path/'verified.json',tmp_path/'validation.json'
    a.write_text(json.dumps(proof));b.write_text(json.dumps(raw));return a,b,raw


def test_all_nine_policies_are_paired_without_automatic_promotion(tmp_path):
    a,b,_=fixture(tmp_path);r=summarize(a,b,tmp_path/'result.json',tmp_path/'result.md')
    assert r['games_verified']==648 and r['unique_match_contexts']==72
    assert len(r['groups'])==4 and all(x['wins_delta']==4 for x in r['paired_deltas'])
    assert r['full_training_accepted']is False


@pytest.mark.parametrize('corruption',['wrong_parent','different_match','missing_policy'])
def test_unpaired_or_incomplete_comparisons_are_rejected(tmp_path,corruption):
    a,b,raw=fixture(tmp_path)
    if corruption=='wrong_parent':
        p=json.loads(a.read_text());p['initial_sha256']='incorrect';a.write_text(json.dumps(p))
    elif corruption=='different_match':raw['reports'][1]['records'][0]['match_seed']+=10000
    else:raw['reports'].pop()
    b.write_text(json.dumps(raw))
    with pytest.raises(ValueError):summarize(a,b,tmp_path/'bad.json',tmp_path/'bad.md')
    assert not(tmp_path/'bad.json').exists()
