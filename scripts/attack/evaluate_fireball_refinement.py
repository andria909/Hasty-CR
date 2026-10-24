"""One frozen candidate: new Fireball test, old full test, paired simulator games.

No training, no checkpoint selection. Run on the pinned common evaluation
package, with the three explicitly hashed Fireball extension modules added.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import torch

from scripts.attack.train_fireball_refinement import evaluate,PARENT_SHA
from sim.train_attack_ppo import initialize_attack_policy
from sim.eval_strategy import evaluate as evaluate_strategy
from sim.eval_defense_duel import approved_baselines,load_network,run_match,summarize,JOINT_MODE


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy(path,device,digest,refined=False):
    if sha(path)!=digest:raise ValueError('Policy SHA differs')
    blob=torch.load(path,map_location='cpu',weights_only=False)
    if refined:
        if (blob.get('parent_sha256')!=PARENT_SHA or blob['config'].get('method')!='impact_verified_BC_with_policy_rehearsal'
                or blob['config'].get('test_used_for_selection')is not False):
            raise ValueError('Refined policy contract differs')
    elif blob['config'].get('task_mode')!='joint':raise ValueError('Original joint parent required')
    network,_,_=initialize_attack_policy(blob,device=device,num_scalars=309)
    if not all(torch.isfinite(t).all()for t in network.state_dict().values()):raise ValueError('Nonfinite policy')
    return network.eval().requires_grad_(False)


def execute(args):
    torch.set_num_threads(2)
    if not torch.cuda.is_available()or torch.cuda.device_count()<2:raise RuntimeError('Two CUDA devices required')
    selection=json.loads(Path(args.selection).read_text())
    if selection.get('schema')!='verified-fireball-candidate-v1' or selection.get('parent_sha256')!=PARENT_SHA:
        raise ValueError('Independent candidate audit required')
    candidate_sha=selection['checkpoint_sha256']
    if sha(args.candidate)!=candidate_sha or sha(args.parent)!=PARENT_SHA:raise ValueError('Frozen policy changed')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    protocol_path=Path('data/strategy/evaluation_protocol.json');protocol=json.loads(protocol_path.read_text())
    result=dict(schema='fireball-independent-evaluation-v1',status='running',checkpoint_sha256=candidate_sha,
                parent_sha256=PARENT_SHA,selected_epoch=selection['epoch'],selection_sha256=sha(args.selection),
                protocol_sha256=sha(protocol_path),parts={},quality_accepted=False,live_checkpoint_changed=False)
    def save(name,value):
        path=output/(name+'.json');path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(event='part_completed',part=name)),flush=True)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(event='CUDA_READY',gpu=[torch.cuda.get_device_name(i)for i in range(2)],
                          candidate_sha256=candidate_sha,epoch=selection['epoch'])),flush=True)
    def skills():
        parent=load_policy(args.parent,'cuda:0',PARENT_SHA)
        candidate=load_policy(args.candidate,'cuda:0',candidate_sha,True)
        before=evaluate(parent,'cuda:0','test',384)
        after=evaluate(candidate,'cuda:0','test',384)
        save('fireball_test',dict(schema='paired-fireball-test-v1',candidate_sha256=candidate_sha,
            parent_sha256=PARENT_SHA,checkpoint_selection_performed=False,before=before,after=after))
        raw=evaluate_strategy(candidate,'cuda:0',protocol,'test',terminal_attribution=True)
        raw.update(checkpoint_sha256=candidate_sha,protocol_sha256=sha(protocol_path),
                   input_mode='joint_309_cells_v2')
        save('regression_test',raw)
    def duels():
        from sim.defense_assets import load_defense_assets
        from sim.runner import resolve_deck
        from sim.env import DECK_26
        device='cuda:1'
        candidate=load_policy(args.candidate,device,candidate_sha,True)
        cards,spells,_,_=load_defense_assets();cards.update(resolve_deck(cards,DECK_26))
        baselines=approved_baselines(Path('data/defense/baselines.json'))
        baselines.append(dict(label='parent_joint_economy_soft',path=Path(args.parent),sha256=PARENT_SHA,
                              observation_mode=JOINT_MODE))
        all_rows=[];groups=[]
        with (output/'duels_raw.jsonl').open('x')as log:
            for baseline in baselines:
                mode=baseline['observation_mode']
                net=(load_policy(args.parent,device,PARENT_SHA)if mode==JOINT_MODE
                     else load_network(baseline['path'],device,mode))
                rows=[]
                for seed in range(710001,710011):
                    for side in (1,-1):
                        row=run_match(candidate,net,device,JOINT_MODE,mode,seed,side,cards,spells)
                        row.update(baseline_id=baseline['label'],baseline_sha256=baseline['sha256'])
                        rows.append(row);all_rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                groups.append(dict(id=baseline['label'],sha256=baseline['sha256'],summary=summarize(rows),matches=rows))
                print(json.dumps(dict(event='duel_progress',completed=len(all_rows),total=320)),flush=True)
        save('duels',dict(candidate_sha256=candidate_sha,baselines=groups,matches=all_rows,
                          historical_summary=summarize([r for r in all_rows if r['baseline_id']!='parent_joint_economy_soft']),
                          parent_summary=summarize([r for r in all_rows if r['baseline_id']=='parent_joint_economy_soft'])))
    try:
        with ThreadPoolExecutor(2)as pool:
            futures=[pool.submit(skills),pool.submit(duels)]
            for f in futures:f.result()
        result['parts']={name:dict(sha256=sha(output/(name+'.json')))for name in ('fireball_test','regression_test','duels')}
        result['status']='complete'
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__)
        raise
    finally:(output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','parent','selection','output'):p.add_argument('--'+name,required=True)
    execute(p.parse_args())
