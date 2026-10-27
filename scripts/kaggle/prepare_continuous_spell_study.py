"""Self-contained parallel short study; original parent, no automatic full run."""
import argparse
import json
from pathlib import Path
import shutil

from scripts.kaggle.prepare_continuous_focus import freeze_template
from scripts.kaggle.run_five_round_robin import sha


def prepare(output,owner='andriashadrin',study='clock'):
    if study not in ('clock','history','board','bank','warmstart','temperature','retention','tactical'):raise ValueError('Unknown short study')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    extra_files=(('scripts/attack/train_air_position_warmstart.py','sim/supervised_replay.py')if study=='retention'else
        ('scripts/attack/train_air_position_warmstart.py',)if study=='warmstart'else())
    template=root/'template';manifest=freeze_template(template,extra_files=extra_files)
    dataset=root/'dataset';dataset.mkdir();job=root/'job';job.mkdir()
    protocol=json.loads(Path('data/ppo_tuning/continuous_spell_efficiency/protocol.json').read_text())
    # This independent pilot uses a fixed parent and fixed uniform/.01 setup.
    # It does not pretend to inherit the not-yet-finished hard-family winner.
    protocol.update(status='frozen_parallel_parent_pilot',initialization='Original9d6 parent only; fresh optimizer each run',
        after='Compare these pilot results after verification; no full training submitted automatically',
        full='Disabled for this parallel pilot; final promotion/full study requires validation audit')
    for trial in protocol['cases']:
        trial.update(focus='uniform',entropy_coef=.01,overflow_penalty=.15,card_balance_weight=.1,card_balance_target='units')
    if study=='history':
        protocol.update(schema='continuous-public-history-pilot-v1',pilot_seeds=[82,83],
            rationale_source='reports/observation_history_gap_20261026.md',
            cases=[dict(id=f'history{frames}_{focus}',history_frames=frames,focus=focus,
                entropy_coef=.01,gamma=.999,gae_lambda=.95,overflow_penalty=.15,
                card_balance_weight=.1,card_balance_target='units',missed_fireball_penalty=0.)
                for frames in (1,2)for focus in ('uniform','air_golem')])
    if study=='board':
        protocol.update(schema='continuous-board-potential-pilot-v1',pilot_seeds=[92,93],
            rationale_source='reports/defense_plan/history_pilot_summary_20261026.json',
            rationale_split='validation_and_reward_mechanics',
            auxiliary_reward='w*(gamma*Phi(next)-Phi(current)); terminal Phi=0, bounded public bank/board estimate',
            penalty='Missed Fireball penalty disabled in every arm',
            cases=[dict(id=f'board_{int(weight*1000):03d}',board_potential_weight=weight,
                history_frames=1,focus='air_golem',entropy_coef=.01,gamma=.999,gae_lambda=.95,
                overflow_penalty=.15,card_balance_weight=.1,card_balance_target='units',missed_fireball_penalty=0.)
                for weight in (0.,.05,.15,.4)])
    if study=='bank':
        proposal=json.loads(Path('data/ppo_tuning/bank_conservation/protocol.json').read_text())
        protocol.update(schema='continuous-bank-conservation-pilot-v1',pilot_seeds=proposal['pilot_seeds'],
            rationale_source=proposal['rationale_source'],rationale_split='train',cases=proposal['cases'])
    if study in('warmstart','retention'):
        protocol.update(schema='continuous-air-warmstart-pilot-v1',pilot_seeds=[122,123],
            rationale_source='reports/air_warmstart_preflight_20261027.md',rationale_split='train_fixture',
            cases=[dict(id=name,warmstart_updates=updates,warmstart_kl_weight=kl,history_frames=1,
                focus='air_golem',entropy_coef=.01,gamma=.999,gae_lambda=.95,overflow_penalty=.15,
                card_balance_weight=.1,card_balance_target='units',missed_fireball_penalty=0.)
                for name,updates,kl in [('bc000_control',0,0.),('bc200_kl000',200,0.),('bc800_kl000',800,0.),('bc800_kl100',800,.1)]])
        if study=='retention':
            protocol.update(schema='continuous-air-retention-pilot-v1',pilot_seeds=[142,143],
                rationale_source='reports/defense_plan/air_warmstart_pilot_summary_20261027.json',
                cases=[dict(id=f'replay{int(weight*100):03d}',warmstart_updates=800,warmstart_kl_weight=0.,
                    supervised_replay_weight=weight,history_frames=1,focus='air_golem',entropy_coef=.01,
                    gamma=.999,gae_lambda=.95,overflow_penalty=.15,card_balance_weight=.1,
                    card_balance_target='units',missed_fireball_penalty=0.)for weight in(0.,.1,1.,5.)])
        examples=Path('data/air_position_examples_20261027')
        metadata=json.loads((examples/'manifest.json').read_text())
        if metadata['split']!='train'or sha(examples/'examples.npz')!=metadata['data_sha256']:
            raise ValueError('Verified TRAIN demonstrations required')
        shutil.copyfile(examples/'examples.npz',dataset/'air_examples.npz')
        shutil.copyfile(examples/'manifest.json',dataset/'air_manifest.json')
    if study=='temperature':
        protocol.update(schema='continuous-temperature-pilot-v1',pilot_seeds=[132,133],
            rationale_source='reports/defense_plan/air_temperature_probe_20261027.json',rationale_split='train_fixture',
            cases=[dict(id=f'temp{int(t):03d}',policy_temperature=t,history_frames=1,focus='air_golem',
                entropy_coef=.01,gamma=.999,gae_lambda=.95,overflow_penalty=.15,card_balance_weight=.1,
                card_balance_target='units',missed_fireball_penalty=0.)for t in (1.,10.,30.,50.)])
    if study=='tactical':
        protocol.update(schema='continuous-live-tactical-reward-pilot-v1',pilot_seeds=[152,153],
            rationale_source='reports/defense_plan/live_balanced_20261027.json',rationale_split='live_training_feedback',
            initialization='Fixed balanced40d028 checkpoint; fresh optimizer, no test-based initializer search',
            cases=[dict(id=name,history_frames=1,focus='uniform',entropy_coef=.01,gamma=.999,gae_lambda=.95,
                overflow_penalty=.3 if economy else .15,card_balance_weight=.1,card_balance_target='units',
                missed_fireball_penalty=.1 if precision else 0.,fireball_value_reward=.25 if precision else 0.,
                reserve_potential_weight=.1 if economy else 0.,own_hp_penalty=6. if economy else 4.,barbarian_train_fraction=.35,lane_reflection_fraction=.5)
                for name,precision,economy in [('live_control',False,False),('live_precision',True,False),
                    ('live_economy',False,True),('live_combined',True,True)]])
    (dataset/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for name in ('extensions.tar.gz',):shutil.copyfile(template/name,dataset/name)
    source=Path('tmp/kaggle/fireball_balanced_20261025/dataset')
    if sha(source/'manifest.json')!=manifest['source_manifest_sha256']:raise ValueError('Frozen base source changed')
    for name in ('manifest.json','source.tar.gz'):shutil.copyfile(source/name,dataset/name)
    parent=Path('checkpoints/ppo_joint_economy_soft_20261023/policy.pt')
    expected='9d6ac7d1d41ff00df6a9c74354fa8447793e0be8d69f723b27ff43b23c4c73e8'
    if study=='tactical':
        parent=Path('tmp/kaggle/continuous_slots_20261026/collected_selected/full_balanced_s44/best_validation_001000.pt')
        expected='40d028fae25b30a23272d97bcc16d04c45f6e297c4f3b244a501d3bbd181aed9'
    digest=sha(parent)
    if digest!=expected:raise ValueError('Parent changed')
    shutil.copyfile(parent,dataset/'parent.pt')
    stem=('hastycr-history-pilot'if study=='history'else
        'hastycr-board-potential-pilot'if study=='board'else
        'hastycr-bank-conservation-pilot'if study=='bank'else
        'hastycr-air-warmstart-pilot'if study=='warmstart'else
        'hastycr-temperature-pilot'if study=='temperature'else'hastycr-clock-fireball-pilot')
    if study=='retention':stem='hastycr-air-retention-pilot'
    if study=='tactical':stem='hastycr-live-tactical-reward-pilot'
    date='20261027'if study in('bank','warmstart','temperature','retention','tactical')else'20261026'
    slug=stem+'-inputs-'+date;kernel=owner+'/'+stem+'-'+date
    (dataset/'dataset-metadata.json').write_text(json.dumps(dict(id=owner+'/'+slug,title=slug,
        licenses=[dict(name='other')],isPrivate=True),indent=2)+'\n')
    config=dict(owner=owner,slug=slug,source_slug=slug,allow_full_training=False,
        models=[dict(id='parent',file='parent.pt',sha256=digest)],
        source_manifest_sha256=manifest['source_manifest_sha256'],extensions=manifest['extensions'],replaced=manifest['replaced'],
        pins={n:sha(dataset/n)for n in ('manifest.json','extensions.tar.gz','protocol.json','parent.pt')})
    if study in('warmstart','retention'):config['pins'].update({n:sha(dataset/n)for n in('air_examples.npz','air_manifest.json')})
    code=(template/'bootstrap.txt').read_text()+'\nexecute('+repr(config)+')\n'
    compile(code,'train.py','exec');(job/'train.py').write_text(code)
    (job/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,title=kernel.split('/')[-1],code_file='train.py',
        language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,
        dataset_sources=[owner+'/'+slug],competition_sources=[],kernel_sources=[]),indent=2)+'\n')
    result=dict(kernel=kernel,config=config,bootstrap_sha256=sha(job/'train.py'),submitted=False,
        pilot_steps=819200,automatic_full_training=False,quality_accepted=False)
    (root/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    return dict(kernel=kernel,submitted=False,pilot_steps=819200)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--owner',default='andriashadrin')
    p.add_argument('--study',choices=('clock','history','board','bank','warmstart','temperature','retention','tactical'),default='clock')
    a=p.parse_args();print(json.dumps(prepare(a.output,a.owner,a.study)))
