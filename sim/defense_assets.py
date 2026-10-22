"""Explicit portable card/spell coverage for the defense curriculum."""
import json
from pathlib import Path

from sim.gamedata import CardSpec,load_buffs,load_gamedata
from sim.public_scenario_spells import load_public_scenario_spells
from sim.public_area_spells import load_public_area_spells
from sim.spells import load_spells

DATA=Path(__file__).resolve().parents[1]/'data/royaleapi/cards.json'


def load_defense_assets():
    cards=dict(load_gamedata(level=11))
    spells=dict(load_spells(level=11))
    public=load_public_scenario_spells(11,available_buffs=load_buffs(),
                                       engine_capabilities={'clone_preserves_shield_presence','graveyard_client_transcription'})
    spells.update(public.spells)
    area=load_public_area_spells(11)
    spells.update(area.spells)
    playable={r['key'].replace('-','_'):r for r in json.loads(DATA.read_text())['cards']}
    for name in spells:
        if name not in cards and name in playable:
            row=playable[name]
            cards[name]=CardSpec(name,int(row['elixir']),str(row['rarity']),None,0,0,0,0)
    return cards,spells,playable,{**public.audit,**area.audit}
