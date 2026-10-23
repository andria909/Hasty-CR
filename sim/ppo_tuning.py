"""Fixed validation-only quality scoring for PPO reward-coefficient trials."""
import argparse
import json
from pathlib import Path


def quality(records,duels,mode):
    if mode not in ('defense','joint'):raise ValueError('Unknown coefficient-study mode')
    expected={(seed,side)for seed in range(970001,970006)for side in (1,-1)}
    pairs=[(r['seed'],r['candidate_side'])for r in duels]
    if len(pairs)!=10 or set(pairs)!=expected:raise ValueError('Invalid paired validation-game schedule')
    if any(r.get('split')!='validation'for r in records):raise ValueError('Coefficient selection forbids train/test results')
    if any(r.get('terminal_reason','').startswith('invalid_script')or r.get('terminal_reason')in('timeout','technical_timeout')or r.get('illegal_actions',0)>0 for r in records):
        return dict(eligible=False,reason='invalid_or_timed_out_validation_episode',score=None)
    if any(r['outcome']not in ('win','loss','draw')for r in duels):return dict(eligible=False,reason='unknown_validation_game_outcome',score=None)
    defense=[r for r in records if r.get('task','defense')=='defense']
    attack=[r for r in records if r.get('task')=='attack']
    def mean(rows,fn):
        if not rows:raise ValueError('Missing required validation stratum')
        return sum(fn(r)for r in rows)/len(rows)
    categories={d:[r for r in defense if r['difficulty']==d]for d in ('simple','medium','complex','impossible')}
    princess=lambda r:all(r['own_towers_alive'][t]for t in ('left','right'))
    components=[mean(categories['simple'],lambda r:r['no_tower_damage']),mean(categories['medium'],lambda r:r['no_tower_damage']),
        mean(categories['complex'],princess),mean(categories['impossible'],lambda r:.5*(princess(r)+r['own_towers_alive']['king']))]
    defense_quality=sum(components)/4
    attack_quality=None;task_quality=defense_quality
    if mode=='joint':
        goals=[mean([r for r in attack if r['goal_kind']==goal],lambda r:r['goal_success'])for goal in ('hog_tower_damage','spell_finish')]
        attack_quality=sum(goals)/2
        task_quality=2*defense_quality*attack_quality/(defense_quality+attack_quality)if defense_quality+attack_quality else 0.
    wins=sum(r['outcome']=='win'for r in duels);win_rate=wins/10
    return dict(eligible=True,score=.5*win_rate+.5*task_quality,validation_win_rate=win_rate,
        defense_quality=defense_quality,attack_quality=attack_quality,task_quality=task_quality,
        king_survival=mean(defense,lambda r:r['own_towers_alive']['king']),
        mean_hp_fraction_loss=mean(defense,lambda r:sum(r['own_tower_fraction_loss'].values())),
        mean_elixir_spent=mean(records,lambda r:r['elixir_spent']),raw_reward_used_for_selection=False)


def evaluate(checkpoint,profile_file,mode,device,contexts_file,expected_steps=102400):
    import torch
    from sim.reward_profiles import load_reward_profile,reward_objects
    from sim.defense_env import DefenseEnv
    from sim.strategy_env import StrategyEnv
    from sim.eval_defense import evaluate_episode
    from sim.eval_defense_duel import load_network,run_match
    from sim.defense_assets import load_defense_assets
    from sim.runner import resolve_deck
    from sim.env import DECK_26
    from sim.defense_protocol import V25_PATH,V25_SHA256
    import hashlib
    data=json.loads(Path(contexts_file).read_text())
    if data.get('split')!='validation':raise ValueError('Validation-only contexts required')
    contexts=data['contexts'][mode]
    if len(contexts)!=128 or len({(r['scenario_id'],r['seed'])for r in contexts})!=128:
        raise ValueError('Exactly128 distinct validation contexts required')
    profile,profile_sha=load_reward_profile(profile_file);defense_rewards,attack_rewards=reward_objects(profile)
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if expected_steps<1 or blob.get('step')!=expected_steps:raise ValueError('Trial did not reach the exact requested step budget')
    if blob['config'].get('reward_profile_sha256')!=profile_sha:raise ValueError('Trial reward profile mismatch')
    joint=mode=='joint';input_mode='joint_enemy_presence_v1'if joint else'physical_live'
    network=load_network(checkpoint,device,input_mode)
    env=(StrategyEnv(split='validation',defense_rewards=defense_rewards,attack_rewards=attack_rewards,deadline_scale=3.,coordinate_mode='physical_live_cells_v2')if joint else
         DefenseEnv(split='validation',rewards=defense_rewards,geometry='legacy_v1',coordinate_mode='physical_live'))
    records=[]
    for context in contexts:
        r=evaluate_episode(env,network,device,context['seed'],context['scenario_id'])
        r.setdefault('task','defense')
        identity=r.get('strategy_scenario_id')if joint else r.get('scenario_id')
        if identity!=context['scenario_id']or r.get('task')!=context['task']:raise ValueError('Validation context mismatch')
        records.append(r)
    baseline_path=Path(V25_PATH)
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest()!=V25_SHA256:raise ValueError('Canonical validation baseline changed')
    baseline=load_network(baseline_path,device,'physical_live')
    cards,spells,_,_=load_defense_assets();cards.update(resolve_deck(cards,DECK_26))
    duels=[run_match(network,baseline,device,input_mode,'physical_live',seed,side,cards,spells,150,400000)
           for seed in range(970001,970006)for side in (1,-1)]
    return dict(schema='ppo-coefficient-validation-result-v1',mode=mode,profile=profile['id'],reward_profile_sha256=profile_sha,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),training_steps=blob['step'],
        records=records,duels=duels,quality=quality(records,duels,mode),test_used_for_selection=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','profile','contexts','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--mode',choices=('defense','joint'),required=True);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--expected-steps',type=int,default=102400)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a new validation output')
    report=evaluate(args.checkpoint,args.profile,args.mode,args.device,args.contexts,args.expected_steps)
    with args.output.open('x')as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(event='TRIAL_VALIDATION',mode=report['mode'],profile=report['profile'],quality=report['quality'])),flush=True)


if __name__=='__main__':main()
