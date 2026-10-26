"""Descriptive seed-cluster bootstrap for an already verified frozen pool.

Seats and opponent families sharing one seed stay together. This does not
turn a small fixed simulator pool into evidence about live matchmaking.
"""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import random
import statistics

from scripts.attack.verify_continuous_pool import verify


def points(outcome):
    return {'win':1.,'draw':.5,'loss':0.}[outcome]


def cluster_interval(groups,replicates=20000,seed=20261026):
    if len(groups)<2 or any(not v for v in groups.values()) or replicates<1000:
        raise ValueError('At least two nonempty seed clusters and1000 replicates required')
    rows=[tuple(groups[k])for k in sorted(groups)]
    if len({len(r)for r in rows})!=1:
        raise ValueError('Balanced seed cluster sizes required')
    means=[statistics.mean(r)for r in rows];rng=random.Random(seed)
    samples=sorted(statistics.mean(rng.choices(means,k=len(means)))for _ in range(replicates))
    def percentile(p):
        index=p*(len(samples)-1);low=int(index);fraction=index-low
        return samples[low]+fraction*(samples[min(low+1,len(samples)-1)]-samples[low])
    return dict(points=statistics.mean(means),games=sum(map(len,rows)),independent_seed_blocks=len(rows),
        percentile_interval_95=[percentile(.025),percentile(.975)],replicates=replicates,
        bootstrap_seed=seed,cluster_unit='seed; all seats/families retained together')


def analyze(collected,prepared,replicates=20000):
    verified=verify(collected,prepared);root=Path(collected)
    ids=[m['id']for m in verified['models']];tests={};duels={};tournament={n:defaultdict(list)for n in ids}
    for name in ids:
        groups=defaultdict(list)
        for row in json.loads((root/(name+'_test.json')).read_text())['rows']:
            groups[row['seed']].append(points(row['outcome']))
        tests[name]=cluster_interval(groups,replicates)
    for a,b in itertools.combinations(ids,2):
        groups=defaultdict(list)
        for row in json.loads((root/(a+'__'+b+'_duels.json')).read_text())['rows']:
            value=points(row['outcome']);s=row['seed'];groups[s].append(value)
            tournament[a][s].append(value);tournament[b][s].append(1-value)
        duels[a+'__'+b]=dict(candidate=a,opponent=b,**cluster_interval(groups,replicates))
    return dict(schema='continuous-frozen-pool-seed-bootstrap-v1',kernel=verified['kernel'],
        tests=tests,duels=duels,tournament={n:cluster_interval(g,replicates)for n,g in tournament.items()},
        test_used_for_selection=False,quality_accepted=False,
        limits=['Percentile bootstrap conditional on fixed models, opponents and simulator',
            'Only10 seed clusters; intervals may be unstable and do not prove live quality',
            'Not simultaneous confidence intervals; no multiple-comparison correction',
            'Model/checkpoint selection still uses validation only'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collected',required=True)
    p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=analyze(a.collected,a.prepared)
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r['duels']))
