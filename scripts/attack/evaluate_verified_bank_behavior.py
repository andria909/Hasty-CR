"""Run both local behavior probes for every candidate in one verified cohort.

No shortlisting by imitation accuracy: all eight receive the same scenarios.
This produces development diagnostics, not a full-training promotion decision.
"""
import argparse
import json
from pathlib import Path

from scripts.attack.evaluate_support_closed_loop import evaluate
from scripts.kaggle.local_artifact_transfer import file_sha


def evaluate_cohort(verified,support,output):
    verified,output=Path(verified),Path(output)
    p=json.loads(verified.read_text())
    expected={f'{case}_s{seed}'for case in ('retain01','retain001','retain0','zero01')for seed in (392,393)}
    if (p.get('schema')!='verified-bank-initialization-study-v1'or
            p.get('supervised_updates_verified')!=6400 or p.get('environment_steps')!=0 or
            len(p.get('runs',[]))!=8 or {r['id']for r in p['runs']}!=expected):
        raise ValueError('Eight independently verified BC runs required')
    weights=[]
    for run in p['runs']:
        path=Path(run['checkpoint'])
        if file_sha(path)!=run['score']['checkpoint_sha256']:
            raise ValueError('Verified checkpoint changed')
        weights.append(str(path))
    output.mkdir(parents=True,exist_ok=False)
    reports={}
    for wave in ('none','opposite'):
        target=output/(wave+'.json')
        evaluate(support,weights,target,wave)
        reports[wave]=dict(path=str(target),sha256=file_sha(target))
    result=dict(schema='verified-bank-behavior-development-v1',kernel=p['kernel'],
        verified_source_sha256=file_sha(verified),models=8,episodes=384,
        reports=reports,full_training_accepted=False,
        scope='Eight policies x twelve reused local contexts x two delays x two wave modes. Independent full-match validation still required.')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('verified','support','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();evaluate_cohort(args.verified,args.support,args.output)
