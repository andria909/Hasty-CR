"""TRAIN-only feasibility grid; fixed opening hands and legal elixir spending.

No policy training, production action override, or simulator change. Both sides
start with an explicit10-elixir exercise fixture. Threats are actual paid plays.
"""
import argparse
import hashlib
import json
from pathlib import Path

from sim.adapter import grid_to_point
from sim.continuous_push import DECKS
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.match import Match
from sim.runner import resolve_deck


def probe():
    cards,spells,_,_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    def make(seed):return Match(cards=cards,spells=spells,decks=(list(DECK_26),list(DECKS['air'])),seed=seed,geometry='native_anchors_candidate_v1')
    seed=next(s for s in range(1000)if 'musketeer'in make(s).players[1].hand and {'balloon','baby_dragon'}<=set(make(s).players[-1].hand))
    rows=[]
    for supported in (False,True):
        for lane in ('left','right'):
            for x0 in (1,3,6):
                for y in (17,20,23,26):
                    for reaction_ms in (0,1500):
                        m=make(seed);m.players[1].elixir=10000;m.players[-1].elixir=10000
                        x=x0 if lane=='left'else 17-x0
                        enemy_x=3 if lane=='left'else 14
                        before={name:t.hitpoints for name,t in m.towers[1].items()}
                        assert m.play_card(-1,'balloon',grid_to_point(17-enemy_x,16,-1))
                        commands=[(225+reaction_ms,1,'musketeer',x,y)]
                        if supported:commands.append((500,-1,'baby_dragon',17-enemy_x,18))
                        accepted=[]
                        while not m.finished and m.elapsed_ms<60000:
                            for command in list(commands):
                                at,side,card,cx,cy=command
                                if m.elapsed_ms>=at:
                                    assert m.play_card(side,card,grid_to_point(cx,cy,side)),('Illegal probe play',command)
                                    accepted.append(dict(at_ms=m.elapsed_ms,side=side,card=card));commands.remove(command)
                            m.step(50)
                        remaining=sum(e.alive and e.side==-1 and not e.is_tower for e in m.battle.entities.values())
                        damage={name:hp-m.towers[1][name].hitpoints for name,hp in before.items()}
                        rows.append(dict(supported=supported,lane=lane,x=x,y=y,reaction_ms=reaction_ms,
                            own_damage=damage,enemy_units_alive=remaining,success=remaining==0 and not any(damage.values()),
                            elapsed_ms=m.elapsed_ms,accepted_commands=accepted))
    return dict(schema='air-defense-position-feasibility-v1',split='train_fixture',seed=seed,rows=rows,
        scope='Explicit10-elixir drill, one Musketeer play, finite60s observation; not full-game win rate or model quality',
        action_execution_delay_ms=225,live_games_started=False,simulator_changed=False,
        mechanics_not_fully_validated=True,source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest()for name in
            ('sim/match.py','sim/engine.py','sim/entities.py','sim/arena.py','sim/adapter.py','sim/defense_assets.py',
             'sim/gamedata.py','sim/public_gamedata.py','sim/spells.py','sim/projectile_calibration.py','sim/kite_geometry.py')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();r=probe()
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
    for supported in(False,True):
        rows=[x for x in r['rows']if x['supported']==supported]
        print(json.dumps(dict(supported=supported,successes=sum(x['success']for x in rows),cases=len(rows))))
