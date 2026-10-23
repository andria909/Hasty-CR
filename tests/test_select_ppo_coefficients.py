import json
from pathlib import Path

import pytest
from scripts.defense import select_ppo_coefficients as module


def metric(score,king=1,hp=.1,elixir=5):
    return dict(eligible=True,score=score,king_survival=king,mean_hp_fraction_loss=hp,mean_elixir_spent=elixir,raw_reward_used_for_selection=False)


def setup(monkeypatch,tmp_path):
    protocol=json.loads(Path('data/ppo_tuning/protocol.json').read_text());profiles=protocol['profiles']
    source=tmp_path/'manifest.json';source.write_text(json.dumps({'archive_sha256':'source','files':{'data/ppo_tuning/protocol.json':module.sha('data/ppo_tuning/protocol.json')}}))
    trials={(mode,p,42):metric(.8-i*.05) for mode in protocol['modes'] for i,p in enumerate(profiles)}
    monkeypatch.setattr(module,'read_trials',lambda *args:(trials,[]))
    def select(stage):return module.select([],Path('data/ppo_tuning/protocol.json'),Path('data/ppo_tuning/validation_contexts.json'),source,Path('data/ppo_tuning/profiles'),stage)
    return protocol,trials,select


def test_confirmed_ranks_equal_mean_of_seeds_not_best_single_seed(monkeypatch,tmp_path):
    protocol,trials,select=setup(monkeypatch,tmp_path)
    for mode in protocol['modes']:
        # baseline wins first seed, but loses the second seed substantially.
        for profile,score in zip(protocol['profiles'][:3],(.1,.8,.8)):trials[mode,profile,43]=metric(score)
    report=select('confirmed')
    for mode in protocol['modes']:
        assert report['modes'][mode]['full_training_profiles']==['tower_survival','economy_soft']
    assert report['test_used_for_selection'] is False


def test_refuses_partial_screen_and_unplanned_confirmation(monkeypatch,tmp_path):
    protocol,trials,select=setup(monkeypatch,tmp_path)
    missing=trials.pop(('joint','opportunity',42))
    with pytest.raises(ValueError,match='All16'):select('screen')
    trials['joint','opportunity',42]=missing
    trials['joint','opportunity',43]=metric(1)
    with pytest.raises(ValueError,match='Exactly the6'):select('confirmed')


def test_ineligible_screen_profile_is_skipped_not_ranked_as_zero(monkeypatch,tmp_path):
    protocol,trials,select=setup(monkeypatch,tmp_path)
    trials['defense','baseline',42]={'eligible':False,'score':None}
    result=select('screen')['modes']['defense']
    assert result['confirmation_profiles']==['tower_survival','economy_soft','economy_strong']
    assert result['ineligible_screen_profiles']==['baseline']


def test_nonfinite_and_reward_based_ranking_forbidden():
    with pytest.raises(ValueError,match='Nonfinite'):module.ranking([metric(float('nan'))])
    row=metric(.5);row['raw_reward_used_for_selection']=True
    with pytest.raises(ValueError,match='Ineligible'):module.ranking([row])
    rows=[dict(profile='b',**metric(.5)),dict(profile='a',**metric(.5)),dict(profile='c',**metric(.5,hp=.05))]
    assert [r['profile'] for r in sorted(rows,key=module.order)]==['c','a','b']


def test_read_trials_checks_raw_contexts_receipt_and_teacher(tmp_path):
    protocol=json.loads(Path('data/ppo_tuning/protocol.json').read_text())
    contexts=json.loads(Path('data/ppo_tuning/validation_contexts.json').read_text())
    profile='baseline';mode='defense';trial='defense_baseline_s42'
    root=tmp_path/'study';root.mkdir();files={}
    def write(name,data,raw=False):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data if raw else json.dumps(data).encode());files[name]={'sha256':module.sha(path)}
    checkpoint=trial+'/train/update_000100.pt';write(checkpoint,b'checkpoint',True)
    records=[]
    # Context IDs are the actual frozen validation curriculum, including seed.
    # Classification is irrelevant to context matching; each stratum remains represented.
    for i,context in enumerate(contexts['contexts'][mode]):
        records.append(dict(scenario_id=context['scenario_id'],seed=context['seed'],task='defense',split='validation',
            difficulty=('simple','medium','complex','impossible')[i%4],no_tower_damage=True,
            own_towers_alive=dict(left=True,right=True,king=True),own_tower_fraction_loss=dict(left=0,right=0,king=0),
            elixir_spent=3,terminal_reason='cleared'))
    games=[dict(seed=seed,candidate_side=side,outcome='win') for seed in range(970001,970006) for side in (1,-1)]
    measured=module.quality(records,games,mode)
    _,reward_sha=module.load_reward_profile(Path('data/ppo_tuning/profiles/baseline.json'))
    result=dict(mode=mode,profile=profile,training_steps=102400,checkpoint_sha256=files[checkpoint]['sha256'],
                reward_profile_sha256=reward_sha,test_used_for_selection=False,records=records,duels=games,quality=measured)
    validation=trial+'/validation.json';write(validation,result)
    config={k:v for k,v in protocol['fixed_training'].items() if not k.startswith('joint_')}
    config.update(seed=42,envs=8,rollout=128,updates=100,init_sha256=protocol['initial_policies'][mode]['sha256'],reward_profile_sha256=reward_sha)
    write(trial+'/train/config.json',config)
    row=dict(mode=mode,profile=profile,seed=42,trial_id=trial,status='completed',training_steps=102400,
             checkpoint=checkpoint,validation=validation,checkpoint_sha256=files[checkpoint]['sha256'],validation_sha256=files[validation]['sha256'],quality=measured)
    write(trial+'/trial_result.json',row)
    write('study_result.json',dict(status='completed',source_sha256='source',test_used_for_search=False,training_steps_per_trial=102400,trials=[row]))
    def receipt():
        (root/'receipt.json').write_text(json.dumps(dict(status='COMPLETE',complete=True,trial_ids=[trial],files=files)))
    receipt();manifest=dict(archive_sha256='source',files={'data/ppo_tuning/profiles/baseline.json':module.sha('data/ppo_tuning/profiles/baseline.json')})
    def read():return module.read_trials([root],protocol,contexts,manifest,Path('data/ppo_tuning/profiles'))
    assert read()[0][mode,profile,42]['score']==1
    config['init_sha256']='different_teacher';write(trial+'/train/config.json',config);receipt()
    with pytest.raises(ValueError,match='Teacher'):read()
    config['init_sha256']=protocol['initial_policies'][mode]['sha256'];write(trial+'/train/config.json',config);receipt()
    # Updating the transport receipt alone cannot disguise changed validation contents.
    result['records'][0]['seed']+=1;write(validation,result);receipt()
    with pytest.raises(ValueError,match='provenance'):read()
