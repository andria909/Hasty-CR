"""Actual checkpoint inference on development seeds; no quality selection."""
import argparse
import json
from pathlib import Path
import torch
from scripts.attack.evaluate_continuous_pool import load,duel,sha
from scripts.attack.evaluate_full_candidates import test_game,public_duel_metrics
from scripts.attack.verify_full_candidate_evaluation import check_cards
from scripts.attack.evaluation_execution import execution_for,check_row_execution,check_duel_execution
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.runner import resolve_deck


def run(blueprint,output):
    blueprint=Path(blueprint);spec=json.loads(blueprint.read_text());execution=execution_for(spec)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    seeds=[813911,813912,813913]
    heldout={s for test in spec['test_sets']for s in test['seeds']}|set(spec['tournament']['seeds'])
    if heldout.intersection(seeds):raise ValueError('Development seeds overlap independent evaluation')
    entries=[dict(m,file=str(Path(m['checkpoint']).absolute()),sha256=m['checkpoint_sha256'])
             for m in spec['baselines']if m['id']in ('balanced','v25')]
    if len(entries)!=2:raise ValueError('Both input architectures required')
    nets={m['id']:load(m,Path('/'),'cpu')for m in entries}
    rows=[]
    def save(row):
        rows.append(row)
        (output/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(dict(completed=len(rows),total=6,kind=row['kind'])),flush=True)
    for entry in entries:
        for test,seed in zip(spec['test_sets'],seeds):
            row=test_game(nets[entry['id']],entry,test,'defense','air',seed,'cpu',execution=execution)
            check_cards(row);check_row_execution(row,execution)
            save(dict(kind='test',model=entry['id'],row=row))
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    modes={e['id']:e['mode']for e in entries}
    for seat in (1,-1):
        row=duel((nets['balanced'],modes['balanced']),(nets['v25'],modes['v25']),
                 'cpu',seeds[-1],seat,cards,spells,execution=execution,on_complete=public_duel_metrics)
        check_duel_execution(row,execution)
        for metric in row['public_metrics'].values():check_cards(metric)
        save(dict(kind='duel',pair=['balanced','v25'],row=row))
    sources=list(Path('sim').glob('*.py'))+[Path(__file__),Path('scripts/attack/evaluation_execution.py'),
        Path('scripts/attack/evaluate_continuous_pool.py'),Path('scripts/attack/evaluate_full_candidates.py')]
    result=dict(schema='evaluation-execution-technical-smoke-v1',execution=execution,
        blueprint_sha256=sha(blueprint),weights={e['id']:e['sha256']for e in entries},
        technical_games=len(rows),cannon_outside_center=sum(r['row']['cannon_outside_center']for r in rows),
        rows_sha256=sha(output/'rows.json'),sources={str(p):sha(p)for p in sources},
        independent_test_run=False,quality_accepted=False,live_games_started=False)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--blueprint',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.blueprint,a.output)
