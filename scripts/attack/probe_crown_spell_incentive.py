"""Measure full-bank spell incentives with actual resolved tower contacts."""
import argparse
import json
from pathlib import Path

from scripts.attack.probe_empty_spell_incentive import IdleArena
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.adapter import grid_to_point


def branch(trial,spell,target,weights,cast):
    seed={'fireball':0,'zap':3}[spell]
    keys=('overflow_penalty','fireball_value_reward','reserve_potential_weight','own_hp_penalty',
          'barbarian_train_fraction','lane_reflection_fraction','action_cooldown_ms',
          'cannon_center_only','tower_damage_all','source_radius_reach','target_lock_in_range','hog_sight_clipping')
    profile={key:trial[key]for key in keys};profile.update(shaping_gamma=trial.get('gamma',.999))
    env=IdleArena(seed=seed,action_delay_ms=trial['action_delay_ms'],continuous_profile=profile,
        geometry='native_anchors_candidate_v1',coordinate_mode='physical_live_cells_v2',
        tower_support_weight=trial['tower_support_weight'],**weights)
    env.reset(seed=seed)
    while env.match.players[1].elixir<10000:env.step(0)
    before=env._metrics();king=env.match.towers[-1]['king'];dormant_before=king.target_only_buildings
    if target=='empty':x,y=8,16;uid=None
    else:
        tower=env.match.towers[-1][target];uid=tower.uid
        x,y=min(((x,y)for x in range(18)for y in range(32)),
                key=lambda cell:(grid_to_point(*cell,1).x-tower.pos.x)**2+(grid_to_point(*cell,1).y-tower.pos.y)**2)
    slot=env.match.players[1].hand.index(spell);action=1+slot*576+y*18+x
    if not env._mask()[action]:raise ValueError('Diagnostic spell must be legal')
    rewards=[]
    for i in range(120):
        _,reward,terminated,truncated,_=env.step(action if cast and i==0 else 0)
        if terminated or truncated:raise ValueError('Unexpected diagnostic terminal')
        rewards.append(reward)
    after=env._metrics();events=env.match.battle.spell_cast_log
    hits=[h for e in events for h in e['hits']]
    if (env.pending or env.match.players[1].elixir!=10000 or after['own_fraction_loss'] or
            sum(after['accepted_card_counts'].values())!=int(cast)):
        raise ValueError('One resolved cast and restored full bank required')
    if cast:
        if len(events)!=1 or not events[0]['resolved']:raise ValueError('Unresolved cast')
        if target=='empty'and hits:raise ValueError('Expected empty cast')
        if target!='empty'and (len(hits)!=1 or hits[0]['target_uid']!=uid or not hits[0]['is_tower']or hits[0]['killed']):
            raise ValueError('Expected one nonlethal target tower contact')
    elif events:raise ValueError('WAIT control cast a spell')
    return dict(reward=sum(rewards),discounted_reward=sum(profile['shaping_gamma']**i*r for i,r in enumerate(rewards)),
        elixir_overflow=after['elixir_overflow']-before['elixir_overflow'],
        enemy_fraction_damage=after['enemy_fraction_damage']-before['enemy_fraction_damage'],
        empty_spell_elixir=after['empty_spell_elixir'],empty_spell_penalty=after['empty_spell_penalty'],
        crown_only_elixir=after['nonlethal_crown_only_elixir'],dormant_king_elixir=after['dormant_at_request_king_only_elixir'],
        crown_penalty=after['spell_economy_penalty'],fireball_exchange_value=after['fireball_exchange_value'],
        king_dormant_before=dormant_before,king_dormant_after=king.target_only_buildings,
        final_bank=env.match.players[1].elixir/1000,hits=hits,target_cell=[x,y],
        enemy_hp_reward=env.profile['enemy_hp_reward'],start_ms=before['elapsed_ms'],end_ms=after['elapsed_ms'])


def probe(protocol_path):
    protocol_path=Path(protocol_path);trial=json.loads(protocol_path.read_text())['cases'][0]
    profiles=[('current',.1,0.,0.),('empty05',.5,0.,0.),('king04',.5,0.,.4),('crowns04',.5,.4,.4)]
    rows=[]
    for spell in ('fireball','zap'):
        for target in ('empty','king','left'):
            for name,empty,crown,king in profiles:
                weights=dict(spell_waste_weight=empty,crown_only_weight=crown,dormant_king_weight=king)
                wait,cast=(branch(trial,spell,target,weights,do_cast)for do_cast in(False,True))
                delta=cast['reward']-wait['reward']
                reconstructed=(cast['enemy_hp_reward']*cast['enemy_fraction_damage']+
                    trial['overflow_penalty']*(wait['elixir_overflow']-cast['elixir_overflow'])-
                    cast['empty_spell_penalty']-cast['crown_penalty'])
                if abs(delta-reconstructed)>1e-7:raise ValueError('Unexplained reward component')
                rows.append(dict(spell=spell,target=target,profile=name,weights=weights,wait=wait,cast=cast,
                    cast_minus_wait=delta,discounted_cast_minus_wait=cast['discounted_reward']-wait['discounted_reward'],
                    reconstructed_delta=reconstructed))
    return dict(schema='crown-spell-incentive-probe-v1',protocol_path=str(protocol_path),protocol_sha256=file_sha(protocol_path),
        source_sha256=file_sha(__file__),rows=rows,
        sources={name:file_sha(name)for name in ('sim/continuous_push.py','sim/continuous_spell_waste.py',
            'sim/continuous_spell_economy.py','sim/spell_economy.py','sim/tactical_rewards.py','sim/match.py')},
        scope='Full bank, no enemy deployments, one paid spell, 30 seconds to refill. Diagnostic arithmetic, not learned-policy behavior or win rate.',
        limitations=['No future counterattack cost of activating king measured','Chip damage can be strategically useful',
                    'Same coefficient was previously trained in crown study; this is an interaction diagnostic, not a new candidate',
                    'Finishing casts and unit contacts excluded from this probe'],active_training_changed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=probe(a.protocol)
    with Path(a.output).open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps([{k:r[k]for k in ('spell','target','profile','cast_minus_wait')}for r in result['rows']]))
