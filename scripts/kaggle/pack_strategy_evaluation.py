"""Freeze joint evaluation code plus all15 SHA-pinned historical opponents.

Only explicitly allowlisted evaluation modules may differ from training source. No
training, uploads, weight selection or evaluation happens during packaging.
"""
import argparse
import gzip
import io
import json
from pathlib import Path
import tarfile

from scripts.defense.pack_evaluation import verified_frozen_package, digest, add_member
from sim.defense_protocol import load_baseline_allowlist, V25_SHA256


EVALUATORS=('sim/eval_strategy.py','sim/eval_defense_duel.py','sim/evaluation_terminal_damage.py')
EXTRA=('data/strategy/evaluation_protocol.json','data/defense/baselines.json',
       'data/defense/evaluation_protocol.json')



def supported_training_modes(training,members):
    if training.get('observation_contract')=='joint_attack_defense_timer_task_v1' and training.get('num_scalars')==309:
        return ['joint']
    if training.get('schema')!='ppo-coefficient-source-v1':
        raise ValueError('Wrong training source contract')
    name='data/ppo_tuning/protocol.json'
    if name not in members or digest(members[name])!=training.get('protocol_sha256'):
        raise ValueError('Coefficient training protocol SHA mismatch')
    protocol=json.loads(members[name])
    if (protocol.get('schema')!='ppo-coefficient-study-v1' or protocol.get('algorithm')!='ppo' or
        set(protocol.get('modes',[]))!={'defense','joint'} or protocol.get('test_used_for_search') is not False or
        training.get('test_used_for_search') is not False or
        protocol.get('full_environment_steps')!=2048000 or protocol.get('short_environment_steps')!=102400):
        raise ValueError('Wrong dual-regime PPO source contract')
    if any(name not in members for name in ('sim/strategy_env.py','sim/train_defense_ppo.py','sim/train_attack_ppo.py','sim/reward_profiles.py')):
        raise ValueError('Dual-regime training implementation missing')
    return ['defense','joint']


def build(source,output,*,root,owner,slug):
    import re
    if not re.fullmatch('[A-Za-z0-9_-]+',owner)or not re.fullmatch('[a-z0-9-]+',slug):
        raise ValueError('Invalid dataset owner/slug')
    root=Path(root);output=Path(output)
    if output.exists():raise ValueError('Use a new evaluation directory')
    members,training,manifest_bytes,_=verified_frozen_package(source)
    modes=supported_training_modes(training,members)
    changes={}
    for name in EVALUATORS:
        raw=(root/name).read_bytes()
        changes[name]=dict(training_sha256=digest(members[name])if name in members else None,
                           evaluation_sha256=digest(raw))
        members.pop(name,None);add_member(members,name,raw)
    for name in EXTRA:add_member(members,name,(root/name).read_bytes())
    protocol=json.loads(members[EXTRA[0]])
    if protocol['duel_protocol_sha256']!=digest(members['data/defense/evaluation_protocol.json']):
        raise ValueError('Frozen duel protocol changed')
    allowed=load_baseline_allowlist(root/'data/defense/baselines.json',project_root=root)
    old=json.loads(members['data/defense/evaluation_protocol.json'])
    if set(allowed)!=set(old['baseline_ids'])or len(allowed)!=15:
        raise ValueError('Expected all15 prior baselines')
    for row in allowed.values():
        path=Path(row['path']);raw=path.read_bytes()
        if digest(raw)!=row['sha256']:raise ValueError('Historical baseline changed during packaging')
        add_member(members,path.relative_to(root.resolve()).as_posix(),raw)
    if allowed['v25']['sha256']!=V25_SHA256:raise ValueError('Canonical v25 changed')
    output.mkdir(parents=True,exist_ok=False)
    archive=output/'source.tar.gz'
    with archive.open('xb')as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='')as compressed:
            with tarfile.open(fileobj=compressed,mode='w')as stream:
                for name,contents in sorted(members.items()):
                    info=tarfile.TarInfo(name);info.size=len(contents);info.mode=0o644;info.mtime=0
                    stream.addfile(info,io.BytesIO(contents))
    report=dict(schema='joint-strategy-evaluation-package-v1',archive_sha256=digest(archive.read_bytes()),
        files={name:digest(raw)for name,raw in sorted(members.items())},
        training_source_archive_sha256=training['archive_sha256'],
        training_source_manifest_sha256=digest(manifest_bytes),evaluation_changes=changes,
        protocol_sha256=digest(members[EXTRA[0]]),baseline_ids=sorted(allowed),
        baseline_sha256={key:row['sha256']for key,row in allowed.items()},
        baseline_v25_sha256=V25_SHA256,supported_training_modes=modes,quality_accepted=False,private_dataset=True)
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'dataset-metadata.json').write_text(json.dumps(dict(id=owner+'/'+slug,
        title=slug.replace('-',' '),licenses=[dict(name='other')],isPrivate=True),indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--owner',required=True)
    parser.add_argument('--slug',required=True)
    args=parser.parse_args()
    report=build(args.source,args.output,root=args.root,owner=args.owner,slug=args.slug)
    print(json.dumps(dict(files=len(report['files']),baselines=len(report['baseline_ids']),archive_sha256=report['archive_sha256'])))


if __name__=='__main__':main()
