"""Per-card usage from accepted Match.play_log events, with unchanged decisions."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics

import torch

from sim.env import DECK_26
from sim.match import Match
from sim.eval_defense_duel import run_match,load_network,summarize,JOINT_MODE


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def card_summary(rows,field='candidate_cards'):
    if not rows:raise ValueError('Completed match rows required')
    elapsed_minutes=sum(r['elapsed_ms']for r in rows)/60000
    result={}
    for card in DECK_26:
        values=[r[field].get(card,0)for r in rows]
        if any(type(n)is not int or n<0 for n in values):raise ValueError('Invalid accepted card count')
        result[card]=dict(total=sum(values),mean_per_match=statistics.mean(values),
            std_population=statistics.pstdev(values),minimum=min(values),maximum=max(values),
            median=statistics.median(values),plays_per_game_minute=sum(values)/elapsed_minutes if elapsed_minutes else None)
    return result


def counted_match(candidate,opponent,device,mode,seed,seat,cards,spells,protocol):
    holder={}
    def factory(**kwargs):
        match=Match(**kwargs,geometry=protocol['geometry']);holder['match']=match;return match
    row=run_match(candidate,opponent,device,mode,'physical_live',seed,seat,cards,spells,
        delay_ms=protocol['action_delay_ms'],max_ms=protocol['max_ms'],match_factory=factory)
    log=holder['match'].play_log
    for side,field in ((seat,'candidate_cards'),(-seat,'opponent_cards')):
        counts=dict(Counter(e['card']for e in log if e['side']==side))
        if sum(counts.values())!=row['plays'][str(side)]:raise ValueError('Accepted command counter differs from card log')
        row[field]=counts
    row['accepted_card_events']=list(log)
    return row


def load_pinned(model,weights,device):
    path=weights/model['file']
    if sha(path)!=model['sha256']:raise ValueError('Policy SHA differs')
    if model.get('refined'):
        from sim.train_attack_ppo import initialize_attack_policy
        blob=torch.load(path,map_location='cpu',weights_only=False)
        if blob['config'].get('method')!='impact_verified_BC_balanced_pairs_public_presence_v3':raise ValueError('Wrong refinement method')
        net,_,_=initialize_attack_policy(blob,device=device,num_scalars=309)
        return net.eval().requires_grad_(False)
    return load_network(path,device,model['mode']).eval().requires_grad_(False)


def execute(args):
    protocol=json.loads(Path(args.protocol).read_text());weights=Path(args.weights)
    if protocol['schema']!='frozen-common-opponent-card-usage-v1':raise ValueError('Wrong usage protocol')
    if not torch.cuda.is_available()or torch.cuda.device_count()<2:raise RuntimeError('Two CUDA devices required')
    torch.set_num_threads(1)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    from sim.defense_assets import load_defense_assets
    from sim.runner import resolve_deck
    cards,spells,_,_=load_defense_assets();cards.update(resolve_deck(cards,DECK_26))
    if spells['fireball'].projectile_speed_mt_per_sec!=12000:raise ValueError('Wrong physics snapshot')
    result=dict(schema='common-opponent-card-usage-result-v1',status='running',protocol_sha256=sha(args.protocol),
        source_sha256=protocol['source_sha256'],models=[m['id']for m in protocol['models']],
        training_performed=False,test_used_for_selection=False,live_games_started=False,parts={})
    print(json.dumps(dict(event='CUDA_READY',gpu=[torch.cuda.get_device_name(i)for i in range(2)],models=len(protocol['models']))),flush=True)
    def worker(index):
        device='cuda:'+str(index);baseline=load_pinned(protocol['opponent'],weights,device)
        for model in protocol['models'][index::2]:
            candidate=load_pinned(model,weights,device);rows=[]
            for seed in protocol['seeds']:
                for seat in protocol['seats']:
                    rows.append(counted_match(candidate,baseline,device,model['mode'],seed,seat,cards,spells,protocol))
            data=dict(schema='model-card-usage-v1',model=model,opponent=protocol['opponent'],
                matches=len(rows),card_counts_mean_card_plays_not_spawned_bodies=True,
                summary=summarize(rows),cards=card_summary(rows),opponent_cards=card_summary(rows,'opponent_cards'),records=rows)
            path=output/(model['id']+'.json');path.write_text(json.dumps(data,indent=2)+'\n')
            print(json.dumps(dict(event='CARD_USAGE_MODEL_COMPLETE',model=model['id'],matches=len(rows))),flush=True)
    try:
        with ThreadPoolExecutor(2)as pool:
            futures=[pool.submit(worker,index)for index in range(2)]
            for future in futures:future.result()
        result.update(status='complete',parts={m['id']+'.json':dict(sha256=sha(output/(m['id']+'.json')))for m in protocol['models']})
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__);raise
    finally:(output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','weights','output'):parser.add_argument('--'+name,required=True)
    execute(parser.parse_args())
