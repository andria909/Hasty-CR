"""Audit active portable curriculum coverage; never assert current live parity."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.arena import Point
from sim.continuous_push import DECKS
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.gamedata import GAMEDATA_ROOT,load_gamedata
from sim.match import Match,elixir_table
from sim.runner import resolve_deck
from sim.spells import load_spells


def audit():
    cards,spells,public,support=load_defense_assets()
    cards.update(resolve_deck(cards,list(DECK_26)))
    names=sorted(set(DECK_26)|{c for d in DECKS.values()for c in d}|{'hunter','valkyrie','barbarians'})
    missing=sorted(set(names)-set(cards))
    if missing:raise ValueError('Missing curriculum cards: '+str(missing))
    rows={name:dict(cost=cards[name].cost,unit=cards[name].unit is not None,
        spell=name in spells,public_spell_audit=support.get(name)) for name in names}
    deck=['lightning']+list(DECK_26)[:7]
    match=Match(cards=cards,spells=spells,decks=(deck,deck),seed=1,geometry='native_anchors_candidate_v1')
    match.players[1].hand=deck[:4];match.players[1].elixir=10000
    tower=match.towers[-1]['left'];before=tower.hitpoints
    accepted=match.play_card(1,'lightning',Point(tower.pos.x+1500,tower.pos.y+1500))
    for _ in range(100):match.step(50)
    damage=before-tower.hitpoints
    if not accepted or damage!=spells['lightning'].tower_damage_override or match.players[1].elixir_spent!=cards['lightning'].cost:
        raise ValueError('Portable Lightning paid damage differs from loaded spec')
    data=Path('data/royaleapi')
    return dict(schema='portable-curriculum-asset-audit-v1',client_tables_available=GAMEDATA_ROOT.exists(),
        client_tables_path=str(GAMEDATA_ROOT),required_cards=len(names),missing=missing,cards=rows,
        raw_loader_lightning=dict(card='lightning'in load_gamedata(),spell='lightning'in load_spells()),
        lightning_paid_check=dict(accepted=accepted,spent=match.players[1].elixir_spent,tower_damage=damage,
            expected_tower_damage=spells['lightning'].tower_damage_override,spell_spec=asdict(spells['lightning'])),
        collector=dict(in_curriculum='elixir_collector'in names,loaded_income=elixir_table().get('elixir_collector')),
        source_sha256=file_sha(__file__),public_json_sha256={p.name:file_sha(p)for p in sorted(data.glob('*.json'))},
        live_parity_proven=False,quality_accepted=False,
        limitations=['Coverage and one paid spell behavior only, not proof all targeting/abilities are correct',
                    'Historical public tables are not an authoritative current live balance patch',
                    'Legacy tests expecting extracted client forms are not portable curriculum tests'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    result=audit()
    with Path(a.output).open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(required_cards=result['required_cards'],missing=result['missing'],lightning_damage=result['lightning_paid_check']['tower_damage'])))
