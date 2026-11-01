"""Keep outcome-first nondominated action labels among replayed improvements.

No latency probabilities are assumed: an action is discarded only if another
is no worse at either tested delay and better at one. Tower HP breaks ties only
when the two complete outcome vectors are identical.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from scripts.attack.sequence_physics import require_current_physics
from scripts.kaggle.local_artifact_transfer import file_sha

RANK={'loss':0,'draw':1,'win':2}
DELAYS=(1350,2025)


def frontier(context):
    actions=context['successful_actions']
    if not actions or len(set(actions))!=len(actions):raise ValueError('Unique nonempty improved action set required')
    table={(r['action'],r['delay_ms']):r for r in context['finals']}
    if len(table)!=len(context['finals']):raise ValueError('Duplicate action/delay trial')
    outcomes={};damage={}
    for action in actions:
        pair=[table.get((action,d))for d in DELAYS]
        if any(r is None or not r['terminal']or not r['first_command_accepted']for r in pair):
            raise ValueError('Executed complete paired continuation required')
        outcomes[action]=tuple(RANK[r['metrics']['outcome']]for r in pair)
        damage[action]=tuple(r['own_damage']for r in pair)
    def dominates(a,b):
        if outcomes[a]!=outcomes[b]:
            return all(x>=y for x,y in zip(outcomes[a],outcomes[b]))and any(x>y for x,y in zip(outcomes[a],outcomes[b]))
        return all(x<=y for x,y in zip(damage[a],damage[b]))and any(x<y for x,y in zip(damage[a],damage[b]))
    retained=[a for a in sorted(actions)if not any(dominates(b,a)for b in actions if b!=a)]
    if not retained:raise ValueError('Empty nondominated frontier')
    return retained,dict(outcomes=outcomes,damage=damage,
        discarded={a:[b for b in retained if dominates(b,a)]for a in actions if a not in retained})


def build(data,verified,output):
    data,verified,output=map(Path,(data,verified,output));m=json.loads((data/'manifest.json').read_text())
    proof=json.loads(verified.read_text());require_current_physics(m);require_current_physics(proof)
    if (m.get('correction_search_sha256')!=file_sha(verified)or
            file_sha(data/'examples.npz')!=m['data_sha256']or
            proof.get('schema')!='verified-bank-match-search-v1'):
        raise ValueError('Fully verified assembled corrections required')
    sources={r['context_sha256']:r for r in proof['accepted']}
    with np.load(data/'examples.npz',allow_pickle=False)as archive:arrays={k:archive[k]for k in archive.files}
    result=deepcopy(m);audit=[]
    for record in result['records']:
        if record['skill']!='onpolicy_defense_response':continue
        ref=sources[record['source_context_sha256']]
        if file_sha(ref['context_path'])!=ref['context_sha256']:raise ValueError('Counterfactual source changed')
        row=json.loads(Path(ref['context_path']).read_text());index=record['array_index']
        if set(np.flatnonzero(arrays['success_masks'][index]))!=set(row['successful_actions']):
            raise ValueError('Input labels differ from replayed action set')
        chosen,details=frontier(row)
        arrays['success_masks'][index]=False;arrays['success_masks'][index,chosen]=True
        if (arrays['success_masks'][index]&~arrays['masks'][index]).any():raise ValueError('Illegal frontier action')
        record.update(action=min(chosen),improved_action_count=len(chosen),label_mode='outcome_first_pareto_v1')
        audit.append(dict(index=ref['index'],fold=record['seed'],before=row['successful_actions'],after=chosen,details=details))
    if len(audit)!=m['correction_examples']:raise ValueError('All correction examples must be filtered')
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'examples.npz',**arrays)
    result.update(data_sha256=file_sha(output/'examples.npz'),builder_sha256=file_sha(__file__),
        label_parent_data=str(data),label_parent_manifest_sha256=file_sha(data/'manifest.json'),
        correction_label_mode='outcome_first_pareto_v1',frontier_audit=audit,
        full_game_training_ready=False,quality_accepted=False)
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('data','verified','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();m=build(a.data,a.verified,a.output);print(json.dumps(dict(examples=m['examples'],corrections=m['correction_examples'])))
