"""Self-contained parallel short study; original parent, no automatic full run."""
import argparse
import json
from pathlib import Path
import shutil

from scripts.kaggle.prepare_continuous_focus import freeze_template
from scripts.kaggle.run_five_round_robin import sha


def prepare(output,owner='andriashadrin'):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    template=root/'template';manifest=freeze_template(template)
    dataset=root/'dataset';dataset.mkdir();job=root/'job';job.mkdir()
    protocol=json.loads(Path('data/ppo_tuning/continuous_spell_efficiency/protocol.json').read_text())
    # This independent pilot uses a fixed parent and fixed uniform/.01 setup.
    # It does not pretend to inherit the not-yet-finished hard-family winner.
    protocol.update(status='frozen_parallel_parent_pilot',initialization='Original9d6 parent only; fresh optimizer each run',
        after='Compare these pilot results after verification; no full training submitted automatically',
        full='Disabled for this parallel pilot; final promotion/full study requires validation audit')
    for trial in protocol['cases']:
        trial.update(focus='uniform',entropy_coef=.01,overflow_penalty=.15,card_balance_weight=.1,card_balance_target='units')
    (dataset/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for name in ('extensions.tar.gz',):shutil.copyfile(template/name,dataset/name)
    source=Path('tmp/kaggle/fireball_balanced_20261025/dataset')
    if sha(source/'manifest.json')!=manifest['source_manifest_sha256']:raise ValueError('Frozen base source changed')
    for name in ('manifest.json','source.tar.gz'):shutil.copyfile(source/name,dataset/name)
    parent=Path('checkpoints/ppo_joint_economy_soft_20261023/policy.pt')
    digest=sha(parent)
    if digest!='9d6ac7d1d41ff00df6a9c74354fa8447793e0be8d69f723b27ff43b23c4c73e8':raise ValueError('Parent changed')
    shutil.copyfile(parent,dataset/'parent.pt')
    slug='hastycr-clock-fireball-pilot-inputs-20261026';kernel=owner+'/hastycr-clock-fireball-pilot-20261026'
    (dataset/'dataset-metadata.json').write_text(json.dumps(dict(id=owner+'/'+slug,title=slug,
        licenses=[dict(name='other')],isPrivate=True),indent=2)+'\n')
    config=dict(owner=owner,slug=slug,source_slug=slug,allow_full_training=False,
        models=[dict(id='parent',file='parent.pt',sha256=digest)],
        source_manifest_sha256=manifest['source_manifest_sha256'],extensions=manifest['extensions'],replaced=manifest['replaced'],
        pins={n:sha(dataset/n)for n in ('manifest.json','extensions.tar.gz','protocol.json','parent.pt')})
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
    a=p.parse_args();print(json.dumps(prepare(a.output,a.owner)))
