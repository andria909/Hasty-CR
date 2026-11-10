"""Verify completed paired architecture pilots and recompute their outcomes."""
import argparse
import json
from pathlib import Path
import statistics

from scripts.kaggle.entity_experiment import ARCHS, SEEDS, STEPS, PROFILE, sha


def summarize(root,study='architecture'):
    architectures,seeds=ARCHS,SEEDS
    if study=='decoder':
        from scripts.kaggle.entity_decoder_experiment import ARCHS as architectures, SEEDS as seeds
    elif study!='architecture':raise ValueError('Unknown entity study')
    root=Path(root);results={};contexts=None;proof=[]
    expected_source=json.loads((root/f'preflight/{next(iter(architectures))}_whole/config.json').read_text())['source_code_sha256']
    for case,arch in architectures.items():
        folder=root/case/'collected';receipt=json.loads((folder/'collection_receipt.json').read_text())
        if receipt['steps']!=2*STEPS:raise ValueError('Wrong collected budget')
        for name,digest in receipt['files'].items():
            if sha(folder/name)!=digest:raise ValueError('Collected file changed: '+name)
        selected_records=[];final_records=[];seed_results=[]
        for seed in seeds:
            identity=f'{case}_s{seed}';directory=folder/identity
            cfg=json.loads((directory/'config.json').read_text())
            boundary=json.loads((directory/'training_result.json').read_text())
            selected=json.loads((folder/(identity+'_summary.json')).read_text())
            metrics=[json.loads(l)for l in (directory/'metrics.jsonl').read_text().splitlines()]
            expected=dict(architecture=arch,seed=seed,init=None,resume=None,value_coef=.25,
                action_delay_ms=1350,card_balance_weight=.03,tower_support_weight=.15,spell_waste_weight=.3,
                target_environment_steps=STEPS,source_code_sha256=expected_source)
            if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('Training config differs: '+identity)
            if study=='decoder' and (cfg.get('evaluation_action_mode')!='grouped_argmax' or
                    cfg.get('spatial_entropy_weight')!=(.1 if case=='sharp64'else 1.)):
                raise ValueError('Decoder/entropy experiment config differs')
            if any(cfg['continuous_profile'].get(k)!=v for k,v in PROFILE.items()):raise ValueError('Physics/rewards differ')
            if not boundary['target_met']or boundary['step']!=STEPS:raise ValueError('Incomplete budget')
            if len(metrics)!=400 or any((r['update'],r['step'],r.get('actual_cuda'))!=(i,i*1024,'Tesla T4')for i,r in enumerate(metrics,1)):
                raise ValueError('CUDA update sequence differs')
            vals=[r for r in metrics if 'validation'in r]
            guarded=[r for r in vals if r['validation']['worst_archetype_points']>=.1]
            winner=max(guarded or vals,key=lambda r:r['validation']['selection_score'])
            if winner['update']!=selected['selected_update']or winner['validation']!=selected['validation']:
                raise ValueError('Checkpoint selection differs')
            checkpoint=folder/selected['checkpoint']
            if sha(checkpoint)!=selected['checkpoint_sha256']:raise ValueError('Weight digest differs')
            rows=selected['validation']['records'];last=metrics[-1]['validation']['records']
            keys=sorted((r['scenario_id'],r['episode_seed'])for r in rows)
            if contexts is None:contexts=keys
            if keys!=contexts or len(keys)!=72 or len(set(keys))!=72:raise ValueError('Validation contexts differ')
            if sorted((r['scenario_id'],r['episode_seed'])for r in last)!=contexts:
                raise ValueError('Final validation contexts differ')
            if study=='decoder' and any(r['validation'].get('action_mode')!='grouped_argmax'for r in vals):
                raise ValueError('Actual validation decoding differs')
            wins=sum(r['outcome']=='win'for r in rows)
            if wins!=selected['validation']['wins']:raise ValueError('Win count differs')
            selected_records+=rows;final_records+=last
            seed_results.append(dict(seed=seed,selected_update=winner['update'],wins=wins,games=len(rows),
                last_update_wins=sum(r['outcome']=='win'for r in last),family_guard_met=bool(guarded),
                checkpoint=str(checkpoint),sha256=sha(checkpoint),training_to_last_update_seconds=metrics[-1]['elapsed_seconds']))
            proof.append(dict(id=identity,metrics_sha256=sha(directory/'metrics.jsonl'),
                              config_sha256=sha(directory/'config.json'),checkpoint_sha256=sha(checkpoint),steps=STEPS))
        def aggregate(rows):
            cards=sorted({c for r in rows for c in r['accepted_card_counts']})
            return dict(wins=sum(r['outcome']=='win'for r in rows),games=len(rows),
                by_task={t:dict(wins=sum(r['outcome']=='win'for r in rows if r['task']==t),
                                  games=sum(r['task']==t for r in rows))for t in ('attack','defense')},
                mean_card_plays={c:statistics.mean(r['accepted_card_counts'].get(c,0)for r in rows)for c in cards},
                empty_spell_elixir_per_game=statistics.mean(r['empty_spell_elixir']for r in rows),
                mean_overflow=statistics.mean(r['elixir_overflow']for r in rows))
        results[case]=dict(seeds=seed_results,selected=aggregate(selected_records),final=aggregate(final_records))
    return dict(schema='entity-'+study+'-results-v1',root=str(root),models=results,proof=proof,
        total_cuda_steps=6*STEPS,unique_validation_contexts=72,seed_repetitions=2,
        test_used=False,live_promoted=False,all_fresh_initialization=True,
        family_guard_passes=sum(s['family_guard_met']for r in results.values()for s in r['seeds']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--study',choices=('architecture','decoder'),default='architecture');a=p.parse_args()
    result=summarize(a.root,a.study);Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    for name,row in result['models'].items():print(name,row['selected']['wins'],row['selected']['games'])
