"""Verify complete paired counterfactual evidence from immutable search shards."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha


def verify(shards,output):
    contexts={};protocol=None;receipts=[]
    for shard in map(Path,shards):
        result=shard/'result.json';r=json.loads(result.read_text());p=r['protocol']
        if (r.get('complete')is not True or p.get('schema')!='bank-match-action-search-v1'or
                file_sha(shard/'driver.py')!=p['source_sha256']or p['physics_pins']!=physics_pins()or
                p['delays_ms']!=[1350,2025]or p['screen_ms']!=15000):
            raise ValueError('Complete pinned two-delay search required')
        common={k:v for k,v in p.items()if k!='indices'}
        if protocol is None:protocol=common
        if protocol!=common:raise ValueError('Search shard protocols differ')
        if len(r['contexts'])!=len(p['indices'])or {c['index']for c in r['contexts']}!=set(p['indices']):
            raise ValueError('Shard contexts incomplete')
        for row in r['contexts']:
            if row['index']in contexts:raise ValueError('Duplicate context across shards')
            if file_sha(row['path'])!=row['sha256']:raise ValueError('Context changed')
            contexts[row['index']]=row
        receipts.append(dict(root=str(shard),result_sha256=file_sha(result)))
    root=Path(protocol['root']);manifest=root/'manifest.json'
    if file_sha(manifest)!=protocol['manifest_sha256']:raise ValueError('Collection changed')
    collection=json.loads(manifest.read_text())
    if set(contexts)!=set(range(len(collection['records']))):raise ValueError('All saved contexts required')
    if (collection['split']!='train'or collection['weights_sha256']!=protocol['weights_sha256']or
            file_sha(collection['weights'])!=protocol['weights_sha256']):raise ValueError('TRAIN actor differs')
    accepted=[];fold_counts=Counter();all_folds=Counter();branches=0;screens=0
    for index,ref in sorted(contexts.items()):
        row=json.loads(Path(ref['path']).read_text());state=collection['records'][index]
        if row['state']!=state or row['index']!=index or file_sha(root/state['snapshot'])!=state['sha256']:
            raise ValueError('State identity differs')
        if state['pending_count']!=0 or state['split']!='train':raise ValueError('No-pending TRAIN contract required')
        all_folds[str(state['fold'])]+=1
        finals=row['finals'];table={(b['action'],b['delay_ms']):b for b in finals}
        if len(table)!=len(finals):raise ValueError('Duplicate final branch')
        actions={b['action']for b in finals}
        if 0 not in actions or set(table)!={(a,d)for a in actions for d in (1350,2025)}:
            raise ValueError('Both delays for every selected action required')
        if len(finals)!=ref['full_branches']or len(row['screen'])!=ref['screen_branches']:
            raise ValueError('Branch counts differ')
        original=next(g['metrics']for g in collection['games']if g['episode_seed']==state['episode_seed'])
        if table[0,1350]['metrics']!=original:raise ValueError('Original game was not reproduced exactly')
        scored={}
        for key,b in table.items():
            if not b['terminal']or b['metrics']['outcome']not in ('win','draw','loss'):
                raise ValueError('Actual full-game terminal outcome required')
            if b['end_ms']!=b['metrics']['elapsed_ms']or b['start_ms']!=state['at_ms']:
                raise ValueError('Match clocks differ')
            scored[key]=({'win':2,'draw':1,'loss':0}[b['metrics']['outcome']],-b['own_damage'])
        improved=[a for a in sorted(actions-{0})if all(table[a,d]['first_command_accepted'] and
            scored[a,d]>scored[0,d]for d in (1350,2025))]
        if improved!=row['successful_actions']or improved!=ref['successful_actions']:
            raise ValueError('Reported improvement differs from paired results')
        if improved:
            fold_counts[str(state['fold'])]+=1
            accepted.append(dict(index=index,fold=state['fold'],scenario=state['scenario'],
                context_path=ref['path'],context_sha256=ref['sha256'],successful_actions=improved))
        branches+=len(finals);screens+=len(row['screen'])
    proof=dict(schema='verified-bank-match-search-v1',root=str(root),shards=receipts,
        collection_sha256=protocol['manifest_sha256'],source_sha256=protocol['source_sha256'],
        contexts_verified=len(contexts),full_branches=branches,screen_branches=screens,
        counts_by_fold=dict(all_folds),improved_by_fold=dict(fold_counts),accepted=accepted,
        weights_sha256=protocol['weights_sha256'],physics_pins=protocol['physics_pins'],
        labels_ready=False,full_training_accepted=False,
        next='Replay accepted actions independently before teacher assembly; keep fold354 outside training',
        limitations=['Finite screened candidate set; one changed action then original policy',
            'Two fixed delays, not a measured latency distribution',
            'Synthetic TRAIN games, not real matches or independent model validation'])
    Path(output).write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shards',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=verify(a.shards,a.output);print(json.dumps(dict(contexts=r['contexts_verified'],improved=len(r['accepted']))))
