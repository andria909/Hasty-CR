"""Public snapshot does not establish the current fixed Graveyard graph."""
import hashlib
import json
from pathlib import Path

from sim.public_scenario_spells import load_public_scenario_spells

ROOT=Path(__file__).resolve().parents[1]


def test_legacy_random_periodic_row_does_not_satisfy_current_fixed_graph():
    evidence=json.loads((ROOT/'data/validation/graveyard_public_coverage_20261022.json').read_text())
    source=ROOT/evidence['source_snapshot']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==evidence['source_sha256']
    rows=json.loads(source.read_text())
    graveyard=next(r for r in rows if r['name']=='Graveyard')
    assert graveyard['spawn_randomize_sequence'] is True
    legacy_slots=len(range(graveyard['spawn_initial_delay'],graveyard['life_duration'],graveyard['spawn_interval']))
    assert legacy_slots==15 and legacy_slots!=evidence['official_current_count']
    assert evidence['official_pattern']=='fixed'
    assert not evidence['existing_transcription']['independently_verified_here']
    assert len(evidence['missing'])==3


def test_current_graph_guard_remains_closed_despite_existing_hardcoded_schedule():
    from sim.spells import _GRAVEYARD_EVENTS
    assert len(_GRAVEYARD_EVENTS)==12
    catalog=load_public_scenario_spells()
    assert 'graveyard' not in catalog.spells
    assert catalog.audit['graveyard']['missing']==['portable_current_spawn_action_graph']


def test_opt_in_transcription_preserves_exact_existing_delays_positions_and_deploy():
    from sim.arena import Point
    from sim.engine import Battle
    from sim.public_gamedata import load_public_gamedata
    from sim.spells import _GRAVEYARD_EVENTS,cast_spell
    catalog=load_public_scenario_spells(engine_capabilities={'graveyard_client_transcription'})
    spec=catalog.spells['graveyard']
    assert spec.area_spawn_events==_GRAVEYARD_EVENTS
    assert not catalog.audit['graveyard']['independently_verified']
    assert catalog.audit['graveyard']['limitations']
    cards,_=load_public_gamedata()
    battle=Battle()
    battle.unit_lookup=lambda name: cards['skeletons'].unit if name=='Skeleton' else None
    centre=Point(9000,16000)
    cast_spell(battle,spec,centre,1)
    assert len(battle.spell_spawn_events)==12 and not battle.entities
    battle.now_ms=2199;battle._resolve_spell_spawns()
    assert not battle.entities
    for index,(delay,dx,dy) in enumerate(_GRAVEYARD_EVENTS):
        battle.now_ms=delay;battle._resolve_spell_spawns()
        assert len(battle.entities)==index+1
        spawned=max(battle.entities.values(),key=lambda e:e.uid)
        assert spawned.pos==Point(centre.x+dx,centre.y-dy)
        assert spawned.deploy_remaining_ms==500
    assert not battle.spell_spawn_events
