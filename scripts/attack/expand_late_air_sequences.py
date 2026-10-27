"""Transfer fixed teacher plans to predeclared new TRAIN hands, retaining failures."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from scripts.attack.audit_air_label_latency import opening
from scripts.attack.search_late_air_defense import rollout, rank
from scripts.attack.build_late_air_sequences import replay
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.runner import resolve_deck


def expand(source, search, output):
    source, search, output = map(Path,(source,search,output))
    manifest=json.loads((source/'manifest.json').read_text())
    teachers=json.loads(search.read_text())
    if file_sha(source/'examples.npz')!=manifest['data_sha256'] or file_sha(search)!=manifest['search_sha256']:
        raise ValueError('Pinned sequence source and teacher search required')
    # All below unchanged maximum held-out seed3800085, disjoint from original.
    new_seeds=list(range(3800010,3800034))
    if set(new_seeds)&set(manifest['seeds']) or max(new_seeds)>=max(manifest['seeds']):
        raise ValueError('New seeds must not alter existing holdout')
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    with np.load(source/'examples.npz',allow_pickle=False) as data:
        arrays={k:list(data[k]) for k in data.files}
    records=deepcopy(manifest['records']);trials=[];accepted=[]
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(seeds=new_seeds,source_sha256=file_sha(__file__),teacher_search_sha256=file_sha(search),
        teacher_source_sha256=file_sha('scripts/attack/search_late_air_defense.py'),
        scope='Fixed transferred plans; no new position search; existing TRAIN holdout unchanged')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for seed in new_seeds:
        for lane in ('left','right'):
            for support in (False,True):
                for bank in (5.,10.):
                    context=24+len(trials)
                    record=dict(seed=seed,lane=lane,support=support,initial_own_elixir=bank,
                        reaction_ms=1500,decision_ms=2000,accepted=True,skill='air_defense')
                    try:
                        match,hp=opening(record,cards,spells)
                    except ValueError as exc:
                        trials.append(dict(context=context,opening=record,accepted=False,reason=str(exc)));continue
                    options={}
                    for teacher in teachers['contexts']:
                        if teacher['opening']['seed']==max(manifest['seeds']):continue
                        if teacher['opening']['lane']!=lane:continue
                        best=teacher['best']
                        key=json.dumps([best['first'],best['second']],sort_keys=True)
                        options[key]=(best['first'],best['second'])
                    outcomes=[]
                    for first,second in options.values():
                        row=rollout(match,hp,cards,spells,lane,first,second)
                        row.update(first=first,second=second)
                        outcomes.append(row)
                    successes=[r for r in outcomes if r['valid'] and r['clear_without_damage']]
                    trial=dict(context=context,opening=record,initial_hand=list(match.players[1].hand),
                        accepted=bool(successes),outcomes=outcomes)
                    trials.append(trial)
                    if not successes:continue
                    chosen=dict(context=context,opening=record,best=min(successes,key=rank))
                    rows=replay(chosen,cards,spells)
                    accepted.append(chosen)
                    for row in rows:
                        index=len(records)
                        for k in arrays:arrays[k].append(row[k])
                        records.append(dict(seed=seed,accepted=True,array_index=index,skill='late_air_sequence',
                            context=context,lane=lane,**{k:row[k]for k in ('decision_ms','action','pending_count','reserved_elixir')}))
        print(json.dumps(dict(seed=seed,attempted_contexts=len(trials),accepted=len(accepted))),flush=True)
    np.savez_compressed(output/'examples.npz',**{k:np.stack(v)for k,v in arrays.items()})
    result=deepcopy(manifest)
    result.update(schema='late-air-diverse-hand-sequences-v1',examples=len(records),records=records,
        seeds=sorted(set(manifest['seeds'])|{r['seed']for r in records}),
        sequences=manifest['sequences']+len(accepted),data_sha256=file_sha(output/'examples.npz'),
        source_data_sha256=manifest['data_sha256'],source_dataset=str(source),expansion_protocol=protocol,
        new_contexts_attempted=len(trials),new_contexts_accepted=len(accepted),
        unchanged_holdout_seed=max(manifest['seeds']),quality_accepted=False)
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'transfer_audit.json').write_text(json.dumps(dict(protocol=protocol,trials=trials,accepted=accepted),indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('source','search','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();r=expand(a.source,a.search,a.output)
    print(json.dumps({k:r[k]for k in ('examples','sequences','new_contexts_attempted','new_contexts_accepted')}))
