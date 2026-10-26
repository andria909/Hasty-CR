"""Short controlled TRAIN-state diagnostic; no enemy actions for15 seconds.

Uses public observations and corrected independent-slot execution. This is
an intervention, not a full match, live game, or held-out quality score.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from scripts.attack.evaluate_continuous_pool import choose,load,sha
from sim.continuous_push import ContinuousPushEnv
from sim.env import DECK_26

STUCK_HAND=('cannon','fireball','skeletons','electro_spirit')


def prepare_state(seed,hand_case):
    if hand_case not in ('shuffled','reported_hand'):raise ValueError('Unknown diagnostic hand')
    env=ContinuousPushEnv(split='train',action_delay_ms=225)
    env.reset(seed=seed,scenario_id='defense::hog/random')
    player=env.match.players[1];player.elixir=10000
    if hand_case=='reported_hand':
        player.hand=list(STUCK_HAND)
        player.queue=[c for c in DECK_26 if c not in STUCK_HAND]
    # Explicitly remove pressure to isolate spending/idle behavior. The model
    # receives only the ordinary public observation, never the hidden deadline.
    env._enemy=lambda:None
    env._previous=env._metrics()
    return env


def run(protocol,weights):
    torch.set_num_threads(1);p=json.loads(Path(protocol).read_text());models=[]
    for entry in p['models']:
        network=load(entry,weights,'cpu');rows=[]
        for hand_case in ('shuffled','reported_hand'):
            for seed in range(101100,101104):
                env=prepare_state(seed,hand_case);initial_hand=list(env.match.players[1].hand);idle_ms=0
                while env.match.elapsed_ms<15000:
                    mask=env._mask();reserved=sum(env.cards[c[1]].cost*1000 for c in env.pending)
                    action=choose(network,env.match,1,entry['mode'],mask,'cpu',reserved)
                    if not action and not env.pending and env.match.players[1].elixir==10000 and mask[1:].any():
                        idle_ms+=env.profile['decision_interval_ms']
                    env.step(action)
                row=env._metrics()
                spells=[dict(spell=e['spell'],cast_ms=e['cast_ms'],resolved=e.get('resolved',False),
                    impact_ms=e.get('impact_ms'),damage=sum(h.get('damage',0)for h in e.get('hits',())))
                    for e in env.match.battle.spell_cast_log if e.get('caster_side')==1]
                rows.append(dict(seed=seed,hand_case=hand_case,initial_hand=initial_hand,
                    first_accepted_action=env.actions[0]if env.actions else None,
                    cap_idle_with_legal_action_ms=idle_ms,elixir_overflow=row['elixir_overflow'],
                    accepted_card_counts=row['accepted_card_counts'],elixir_spent=row['elixir_spent'],
                    accepted_actions=list(env.actions),enemy_fraction_damage=row['enemy_fraction_damage'],
                    spell_outcomes=spells,
                    missed_resolved_spells=sum(e['resolved']and e['damage']==0 for e in spells)))
        groups={}
        for case in ('shuffled','reported_hand'):
            subset=[r for r in rows if r['hand_case']==case];cards=Counter()
            for r in subset:cards.update(r['accepted_card_counts'])
            groups[case]=dict(states=len(subset),no_card_states=sum(r['first_accepted_action']is None for r in subset),
                mean_cap_idle_ms=sum(r['cap_idle_with_legal_action_ms']for r in subset)/len(subset),
                mean_overflow=sum(r['elixir_overflow']for r in subset)/len(subset),
                mean_missed_resolved_spells=sum(r['missed_resolved_spells']for r in subset)/len(subset),
                mean_card_plays={c:n/len(subset)for c,n in cards.items()})
        models.append(dict(model=entry,by_hand=groups,rows=rows))
    return dict(schema='full-elixir-controlled-train-probe-v1',protocol_sha256=sha(protocol),models=models,
        split='train',duration_ms=15000,enemy_suppressed=True,live_games_started=False,
        test_used_for_selection=False,quality_accepted=False,
        limits=['Four seeds per hand; synthetic intervention, not natural full matches',
            'Reported hand fixes hand and queue; four seed repeats are not four independent opening observations',
            'Spell outcomes cover recorded casts only; instant Zap is not covered by spell_cast_log',
            'Legacy controls and saved references only; corrected-slot models still training'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();report=run(a.protocol,a.weights)
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({m['model']['id']:m['by_hand']for m in report['models']}))
