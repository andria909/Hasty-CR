"""Replay paid late-air defenses with optional tower-support instrumentation."""
import argparse
import json
from pathlib import Path
from scripts.attack.audit_air_label_latency import opening
from scripts.attack.sequence_physics import require_current_physics
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.tower_support import TowerSupport
from sim.adapter import grid_to_point
from sim.defense_assets import load_defense_assets
from sim.runner import resolve_deck
from sim.env import DECK_26


def audit(search,output):
    d=json.loads(Path(search).read_text());require_current_physics(d['protocol'])
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    rows=[]
    for context in d['contexts']:
        for mode in ('no_cards','best_paid_plan'):
            m,hp=opening(context['opening'],cards,spells);start=m.elapsed_ms
            tracker=TowerSupport(m.battle).attach()
            commands=list(context['best']['commands']) if mode=='best_paid_plan' else []
            executed=0
            while not m.finished and m.elapsed_ms<start+60000:
                for command in list(commands):
                    if command['placed_ms']>m.elapsed_ms:continue
                    assert m.play_card(1,command['card'],grid_to_point(*command['cell'],1))
                    commands.remove(command);executed+=1
                m.step(50)
            damage=sum(v-m.towers[1][key].hitpoints for key,v in hp.items())
            if mode=='best_paid_plan':
                assert executed==len(context['best']['commands']) and damage==context['best']['total_damage']
            rows.append(dict(context=context['context'],mode=mode,tower_hp_lost=damage,
                cards_played=executed,**tracker.metrics()))
    result=dict(schema='tower-support-diagnostic-v1',source_sha256=file_sha(search),rows=rows,
        paid_plan_outcomes_reproduced=True,reward_enabled=False,quality_accepted=False,
        limitations=['Target-lock based association is not proof of causal damage prevented',
                     'No reward paid for tower-only tanking; effective body value capped per enemy',
                     'Future coefficient must be evaluated against win rate and own tower HP'])
    Path(output).write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--search',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=audit(a.search,a.output);print(json.dumps(dict(episodes=len(r['rows']),paid_plan_outcomes_reproduced=True)))
