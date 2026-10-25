"""Frozen-policy held-out evaluation on the exact corrected training simulator.

No optimization or candidate selection. Each invocation evaluates one policy;
run both the selected candidate and original parent with the same protocol.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import random
from contextlib import nullcontext
from pathlib import Path

import torch

from sim.fireball_curriculum import FireballEnv
from sim.train_attack_ppo import initialize_attack_policy
from sim.train_defense_ppo import tensor_observations,masked_distribution
from sim.eval_strategy import evaluate as strategy_evaluate
from sim.eval_defense_duel import load_network,run_match,summarize,JOINT_MODE
from sim.match import Match


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_joint(path,device,digest,*,candidate=False):
    if sha(path)!=digest:raise ValueError('Pinned policy SHA differs')
    blob=torch.load(path,map_location='cpu',weights_only=False)
    config=blob.get('config',{})
    if candidate:
        if config.get('method')!='impact_verified_BC_balanced_pairs_public_presence_v3':
            raise ValueError('Wrong balanced candidate method')
    elif config.get('task_mode')!='joint':raise ValueError('Original joint policy required')
    network,_,_=initialize_attack_policy(blob,device=device,num_scalars=309)
    if not all(torch.isfinite(t).all()for t in network.state_dict().values()):raise ValueError('Nonfinite weights')
    return network.eval().requires_grad_(False)


def skills(network,device,seed_start,count,*,king=False):
    kinds=('tower_finish','building_finish','tower_unit')
    if count%6:raise ValueError('Equal task and lane counts required')
    envs={(k,l):FireballEnv(split='test',kind=k,lane=l)for k in kinds for l in ('left','right')}
    rows=[]
    with torch.no_grad():
        for i in range(count):
            kind=kinds[i%3];lane=('left','right')[(i//3)%2];seed=seed_start+i
            env=envs['tower_finish'if king else kind,lane];obs,info=env.reset(seed=seed)
            if king:
                from sim.evaluation_terminal_damage import observe_terminal_damage
                target=env.match.towers[-1]['king']
                for side_lane in ('left','right'):env.match.towers[-1][side_lane].hitpoints=0
                target.hitpoints=random.Random(seed+17000000).randint(1,env._spells['fireball'].damage_to(target))
                env.practice_uids=[target.uid]
                obs,info=env._observe(),env._info()
            context=observe_terminal_damage(env)if king else nullcontext()
            with context as observer:
                for _ in range(30):
                    logits,_=network(*tensor_observations([obs],device))
                    mask=torch.as_tensor(info['action_mask'][None],device=device)
                    action=int(masked_distribution(logits,mask).probs.argmax())
                    obs,_,done,truncated,info=env.step(action)
                    if done or truncated:break
            success=bool(info['metrics']['fireball_success'])
            if king:
                success=any(e['source_kind']=='spell'and e['source_name']=='fireball'and e['source_side']==1
                    and e['target_uid']==target.uid and e['killed']for e in observer.events)
            row=dict(kind='king_finish'if king else kind,lane=lane,seed=seed,success=success,reason=info['terminal_reason'])
            if king:row['terminal_damage_events_v2']=observer.events
            rows.append(row)
    if king:kinds=('king_finish',)
    return dict(episodes=len(rows),successes=sum(r['success']for r in rows),rows=rows,
        by_kind={k:dict(episodes=sum(r['kind']==k for r in rows),
            successes=sum(r['success']for r in rows if r['kind']==k))for k in kinds})


def execute(args):
    protocol_path=Path(args.protocol);protocol=json.loads(protocol_path.read_text())
    if protocol['schema']!='fireball-native-heldout-protocol-v1':raise ValueError('Wrong protocol')
    for name,digest in protocol['training_files'].items():
        if sha(name)!=digest:raise ValueError('Frozen training source differs: '+name)
    if sha(protocol['strategy_protocol'])!=protocol['strategy_protocol_sha256']:raise ValueError('Strategy protocol differs')
    selection=json.loads(Path(args.selection).read_text())
    if not selection.get('candidate_passes_validation_games'):raise ValueError('Audited validation gate required')
    candidate=args.role=='candidate'
    digest=selection['checkpoint_sha256']if candidate else protocol['parent_sha256']
    if selection['source_sha256']!=protocol['training_source_sha256']:raise ValueError('Training source identity differs')
    if not torch.cuda.is_available()or torch.cuda.device_count()<2:raise RuntimeError('Two CUDA devices required')
    torch.set_num_threads(2)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='native-fireball-heldout-result-v1',status='running',role=args.role,
        checkpoint_sha256=digest,protocol_sha256=sha(protocol_path),selection_sha256=sha(args.selection),
        training_source_sha256=protocol['training_source_sha256'],training_performed=False,
        checkpoint_selection_performed=False,quality_accepted=False,live_checkpoint_changed=False,parts={})
    def save(name,data):
        path=output/(name+'.json');path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(event='heldout_part_complete',part=name)),flush=True)
    def exercises():
        net=load_joint(args.policy,'cuda:0',digest,candidate=candidate)
        for group in protocol['fireball_test_strata']:
            save(group['id'],skills(net,'cuda:0',group['seed_start'],group['episodes_per_policy']))
        group=protocol['king_finish_test']
        save('king_finish',skills(net,'cuda:0',group['seed_start'],group['episodes_per_policy'],king=True))
        strategy=json.loads(Path(protocol['strategy_protocol']).read_text())
        save('strategy_test',strategy_evaluate(net,'cuda:0',strategy,'test',terminal_attribution=True))
    def duels():
        from sim.defense_assets import load_defense_assets
        from sim.runner import resolve_deck
        from sim.env import DECK_26
        net=load_joint(args.policy,'cuda:1',digest,candidate=candidate)
        cards,spells,_,_=load_defense_assets();cards.update(resolve_deck(cards,DECK_26))
        groups=[];all_rows=[]
        def factory(**kwargs):return Match(**kwargs,geometry=protocol['environment']['geometry'])
        with (output/'duels_raw.jsonl').open('x')as log:
            for opponent in protocol['opponents']:
                if not candidate and opponent['sha256']==digest:continue
                path=Path(args.opponents)/(opponent['id']+'.pt')
                if sha(path)!=opponent['sha256']:raise ValueError('Opponent SHA differs')
                mode=opponent['mode']
                other=(load_joint(path,'cuda:1',opponent['sha256'])if mode==JOINT_MODE
                       else load_network(path,'cuda:1',mode))
                rows=[]
                for seed in protocol['game_seeds']:
                    for seat in protocol['candidate_seats']:
                        row=run_match(net,other,'cuda:1',JOINT_MODE,mode,seed,seat,cards,spells,
                            delay_ms=protocol['match_delay_ms'],max_ms=protocol['match_max_ms'],match_factory=factory)
                        row.update(baseline_id=opponent['id'],baseline_sha256=opponent['sha256'])
                        rows.append(row);all_rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                groups.append(dict(id=opponent['id'],sha256=opponent['sha256'],summary=summarize(rows),matches=rows))
                print(json.dumps(dict(event='heldout_game_progress',completed=len(all_rows),
                    total=protocol['candidate_games']if candidate else protocol['parent_reference_games_excluding_self'])),flush=True)
        save('duels',dict(groups=groups,summary=summarize(all_rows)))
    print(json.dumps(dict(event='CUDA_READY',role=args.role,gpu=[torch.cuda.get_device_name(i)for i in range(2)],
        checkpoint_sha256=digest)),flush=True)
    try:
        with ThreadPoolExecutor(2)as pool:
            jobs=[pool.submit(exercises),pool.submit(duels)]
            for job in jobs:job.result()
        result.update(status='complete',parts={p.name:dict(sha256=sha(p))for p in output.glob('*.json')})
        result['king_finish_test_complete']=True
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__);raise
    finally:(output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','policy','selection','opponents','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--role',choices=('candidate','parent'),required=True)
    execute(parser.parse_args())
