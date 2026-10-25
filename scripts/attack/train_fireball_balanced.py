"""Fresh v3 phase: corrected physics, balanced HP pairs, per-skill and game guards."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from scripts.attack.train_fireball_refinement import demonstration_data,evaluate
from scripts.attack.train_fireball_selectivity import negative_metric
from sim.fireball_decision_pairs import decision_pair
from sim.fireball_balanced import paired_loss,skill_eligibility,deployed_rehearsal
from sim.train_attack_ppo import initialize_attack_policy
from sim.train_defense_ppo import tensor_observations


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_data(count,split):
    offset=870000 if split=='train' else 970000
    pairs=[decision_pair(offset+i,('left','right')[i%2],split)for i in range(count)]
    return dict(positive=[p['positive']for p in pairs],negative=[p['negative']for p in pairs],
        masks=np.array([p['action_mask']for p in pairs]),fireball=np.array([p['fireball_mask']for p in pairs]),
        labels=np.array([p['reference_action']for p in pairs]),contexts=[p['metadata']for p in pairs])


def match_score(report):
    if report['unknown']:return None
    return (report['wins']+.5*report['draws'])/report['matches']


def validation_games(network,opponent,device,opponent_mode,cards,spells,protocol):
    from sim.eval_defense_duel import run_match,summarize,JOINT_MODE
    from sim.match import Match
    factory=partial(Match,geometry=protocol['geometry'])
    rows=[run_match(network,opponent,device,JOINT_MODE,opponent_mode,seed,side,cards,spells,
                   protocol['match_delay_ms'],protocol['match_max_ms'],match_factory=factory)
          for seed in protocol['validation_match_seeds']for side in (1,-1)]
    return dict(matches=rows,summary=summarize(rows))


def train(args):
    protocol=json.loads(Path(args.protocol).read_text())
    if protocol['schema']!='fireball-balanced-protocol-v1'or protocol['resume_old_optimizer']is not False:
        raise ValueError('Explicit new balanced protocol required')
    if sha(args.init)!=protocol['init_sha256']or protocol['teacher_sha256']!=protocol['init_sha256']:
        raise ValueError('Use the preserved original joint PPO as init and frozen teacher')
    torch.set_num_threads(2);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if not torch.cuda.is_available()or torch.cuda.device_count()<2:raise RuntimeError('Two CUDA devices required')
    from sim.defense_assets import load_defense_assets
    cards,spells,_,_=load_defense_assets()
    if spells['fireball'].projectile_speed_mt_per_sec!=protocol['fireball_speed_mt_per_sec']:
        raise ValueError('Fireball native calibration differs')
    device='cuda:0';output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    blob=torch.load(args.init,map_location='cpu',weights_only=False)
    teacher,architecture,_=initialize_attack_policy(blob,device=device,num_scalars=309)
    network,_,_=initialize_attack_policy(blob,device=device,num_scalars=309)
    teacher.eval().requires_grad_(False)
    config=dict(protocol=protocol,protocol_sha256=sha(args.protocol),source_sha256=args.source_sha256,
        method=protocol['method'],architecture=architecture,num_scalars=309,parent_sha256=sha(args.init),
        observation_contract='joint_attack_defense_timer_task_v1',deployment_task=protocol['deployment_task'],
        physics_input_migration=True,resume_old_optimizer=False,test_used_for_selection=False)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(dict(event='CUDA_READY',gpu=[torch.cuda.get_device_name(i)for i in range(2)],
        source_sha256=args.source_sha256,protocol_sha256=sha(args.protocol),method=protocol['method'])),flush=True)
    observations,masks,labels,audit=demonstration_data(protocol['demonstrations'])
    pairs=paired_data(protocol['pairs'],'train');valpairs=paired_data(protocol['validation_negative'],'validation')
    (output/'demonstration_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (output/'pair_contexts.json').write_text(json.dumps(dict(train=pairs['contexts'],validation=valpairs['contexts']),indent=2)+'\n')
    replay,replay_masks,replay_probs,replay_contexts=deployed_rehearsal(teacher,device,protocol['rehearsal_states'],protocol)
    (output/'rehearsal_contexts.json').write_text(json.dumps(replay_contexts,indent=2)+'\n')
    planes,scalars=tensor_observations(observations,device);masks=torch.as_tensor(masks,device=device);labels=torch.as_tensor(labels,device=device)
    pp,ps=tensor_observations(pairs['positive'],device);np_,ns=tensor_observations(pairs['negative'],device)
    pm=torch.as_tensor(pairs['masks'],device=device);pf=torch.as_tensor(pairs['fireball'],device=device);pl=torch.as_tensor(pairs['labels'],device=device)
    vp,vs=tensor_observations(valpairs['negative'],device);vm=torch.as_tensor(valpairs['masks'],device=device);vf=torch.as_tensor(valpairs['fireball'],device=device)
    rp,rs=tensor_observations(replay,device);rm=torch.as_tensor(replay_masks,device=device)
    teacher_negative=negative_metric(teacher,vp,vs,vm,vf)
    initial_positive=evaluate(teacher,device,'validation',protocol['validation_positive'])
    (output/'baseline_skills.json').write_text(json.dumps(dict(positive=initial_positive,negative=teacher_negative),indent=2)+'\n')
    optimizer=torch.optim.Adam(network.parameters(),lr=protocol['learning_rate']);eligible=[];weights=protocol['weights']
    with (output/'metrics.jsonl').open('x')as log:
        for epoch in range(1,protocol['epochs']+1):
            network.train();losses=[]
            for batch in torch.randperm(len(labels),device=device).split(protocol['batch_size']):
                logits,_=network(planes[batch],scalars[batch])
                demonstration=F.cross_entropy(logits.masked_fill(~masks[batch],-1e9),labels[batch])
                idx=torch.randint(len(pairs['labels']),(len(batch),),device=device)
                positive_logits,_=network(pp[idx],ps[idx]);negative_logits,_=network(np_[idx],ns[idx])
                positive,negative,contrast=paired_loss(positive_logits,negative_logits,pm[idx],pf[idx],pl[idx],protocol['pair_log_odds_margin'])
                idx=torch.randint(len(replay),(len(batch),),device=device)
                prior,_=network(rp[idx],rs[idx])
                retention=F.kl_div(F.log_softmax(prior.masked_fill(~rm[idx],-1e9),-1),replay_probs[idx],reduction='batchmean')
                loss=(weights['demonstration_ce']*demonstration+weights['paired_positive_ce']*positive+
                    weights['paired_negative_mass']*negative+weights['paired_log_odds_margin']*contrast+weights['rehearsal_kl']*retention)
                if not torch.isfinite(loss):raise ValueError('Nonfinite balanced loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),.5);optimizer.step();losses.append(float(loss.detach()))
            row=dict(epoch=epoch,percent=100*epoch/protocol['epochs'],loss=sum(losses)/len(losses))
            if epoch%protocol['validation_every']==0:
                network.eval();positive=evaluate(network,device,'validation',protocol['validation_positive'])
                negative=negative_metric(network,vp,vs,vm,vf)
                with torch.no_grad():
                    actual,_=network(rp,rs);agreement=float((actual.masked_fill(~rm,-1e9).argmax(-1)==replay_probs.argmax(-1)).float().mean())
                accepted=skill_eligibility(positive,negative,agreement,teacher_negative,protocol)
                rates=[v['successes']/v['episodes']for v in positive['by_kind'].values()]
                row.update(validation=positive,negative_validation=negative,rehearsal_action_agreement=agreement,
                    eligible=accepted,minimum_skill_rate=min(rates),skill_score=sum(rates)/len(rates)-.25*negative['first_action_fireball_rate'])
                path=output/f'epoch_{epoch}.pt'
                torch.save(dict(state_dict=network.state_dict(),config=config,epoch=epoch,validation=positive,
                    negative_validation=negative,rehearsal_action_agreement=agreement),path)
                if accepted:eligible.append(dict(epoch=epoch,path=path.name,sha256=sha(path),minimum_skill_rate=min(rates),skill_score=row['skill_score']))
            log.write(json.dumps(row)+'\n');log.flush()
            print(json.dumps({k:v for k,v in row.items()if k!='validation'}),flush=True)
    shortlist=sorted(eligible,key=lambda r:(-r['minimum_skill_rate'],-r['skill_score'],r['epoch']))[:protocol['shortlist_size']]
    (output/'shortlist.json').write_text(json.dumps(dict(eligible=eligible,shortlist=shortlist),indent=2)+'\n')
    games=[];selected=None;baseline=None
    if shortlist:
        from sim.eval_defense_duel import load_network,JOINT_MODE
        from sim.env import DECK_26
        from sim.runner import resolve_deck
        cards.update(resolve_deck(cards,DECK_26))
        # Second device holds game validation only; it never sees test seeds.
        teacher_game,_,_=initialize_attack_policy(blob,device='cuda:1',num_scalars=309)
        teacher_game.eval().requires_grad_(False)
        baseline_spec=next(r for r in json.loads(Path('data/defense/baselines.json').read_text())['checkpoints']if r['id']=='v25')
        baseline_path=Path(args.v25)
        if sha(baseline_path)!=baseline_spec['sha256']:raise ValueError('Preserved v25 differs')
        v25=load_network(baseline_path,'cuda:1',baseline_spec['observation_mode'])
        baseline=validation_games(teacher_game,v25,'cuda:1',baseline_spec['observation_mode'],cards,spells,protocol)
        (output/'teacher_validation_games.json').write_text(json.dumps(baseline,indent=2)+'\n')
        baseline_score=match_score(baseline['summary'])
        for candidate in shortlist:
            state=torch.load(output/candidate['path'],map_location='cpu',weights_only=False)
            net,_,_=initialize_attack_policy(state,device='cuda:1',num_scalars=309);net.eval().requires_grad_(False)
            against_parent=validation_games(net,teacher_game,'cuda:1',JOINT_MODE,cards,spells,protocol)
            against_v25=validation_games(net,v25,'cuda:1',baseline_spec['observation_mode'],cards,spells,protocol)
            parent_score=match_score(against_parent['summary']);v25_score=match_score(against_v25['summary'])
            passed=(parent_score is not None and v25_score is not None and baseline_score is not None and
                parent_score>=protocol['minimum_score_vs_teacher']and v25_score>=baseline_score-protocol['score_margin_vs_v25_teacher_reference'])
            row=dict(candidate,parent=against_parent,v25=against_v25,match_guards_passed=passed,
                game_score=(parent_score+v25_score)/2 if parent_score is not None and v25_score is not None else None)
            games.append(row)
            print(json.dumps(dict(event='validation_games_completed',epoch=candidate['epoch'],passed=passed,parent_score=parent_score,v25_score=v25_score)),flush=True)
        passing=[r for r in games if r['match_guards_passed']]
        if passing:
            selected=sorted(passing,key=lambda r:(-r['game_score'],-r['minimum_skill_rate'],r['epoch']))[0]
            shutil.copyfile(output/selected['path'],output/'candidate.pt')
    (output/'validation_games.json').write_text(json.dumps(games,indent=2)+'\n')
    result=dict(status='complete',candidate_available=selected is not None,
        selected_epoch=selected['epoch']if selected else None,candidate_sha256=sha(output/'candidate.pt')if selected else None,
        eligible_skill_epochs=len(eligible),validation_match_candidates=len(shortlist),
        source_sha256=args.source_sha256,protocol_sha256=sha(args.protocol),
        quality_accepted=False,independent_test_required=selected is not None,live_checkpoint_changed=False)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('init','v25','protocol','source-sha256','output'):p.add_argument('--'+name,required=True)
    train(p.parse_args())
