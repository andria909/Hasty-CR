"""Versioned finite push scenarios; family splits prevent lane/setup leakage."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

DIFFICULTIES = ('simple', 'medium', 'complex', 'impossible')
SPLITS = ('train', 'validation', 'test')
SCHEMA = 'defense-v1'


@dataclass(frozen=True)
class ScheduledPlay:
    card: str
    at_ms: int = 0
    position: str = 'bridge'  # enemy bridge, back, corner, or our princess
    near_bridge_after: str | None = None
    max_wait_ms: int = 45000


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    family_id: str
    difficulty: str
    split: str
    lane: str
    plays: tuple[ScheduledPlay, ...]
    elixir_multiplier: int = 1
    enemy_initial_elixir: float = 10
    enemy_bonus_elixir: float = 0
    unavoidable_tower_damage: bool = False
    setup: str = 'bridge'

    def to_dict(self):
        return asdict(self)


MEDIUM = (
    ('giant', 'archers'), ('battle_ram', 'musketeer'), ('hog_rider', 'wizard'),
    ('prince', 'dark_prince'), ('giant', 'graveyard'), ('giant', 'goblin_barrel'),
    ('giant', 'skeleton_barrel'), ('knight', 'musketeer'), ('giant', 'witch'),
    ('ice_golem', 'balloon'), ('goblin_giant', 'sparky'), ('giant', 'bomber'),
    ('giant','archers','bomber'), ('giant','musketeer','skeletons'),
    ('giant','mini_pekka','spear_goblins'), ('giant','knight','archers'),
    ('giant','mega_minion','bats'), ('giant','wizard','zap'),
    ('hog_rider','ice_golem','ice_spirit'), ('hog_rider','musketeer','zap'),
    ('hog_rider','knight','goblins'), ('hog_rider','firecracker','skeletons'),
    ('battle_ram','bandit','zap'), ('battle_ram','musketeer','ice_spirit'),
    ('battle_ram','dark_prince','bats'), ('battle_ram','magic_archer','skeletons'),
    ('prince','dart_goblin','goblins'), ('prince','archers','ice_spirit'),
    ('dark_prince','musketeer','skeletons'), ('dark_prince','bats','goblins'),
    ('knight','goblin_barrel','princess'), ('knight','goblin_barrel','dart_goblin'),
    ('valkyrie','goblin_barrel','spear_goblins'), ('miner','wall_breakers','bats'),
    ('miner','spear_goblins','zap'), ('ice_golem','balloon','bats'),
    ('balloon','ice_spirit','skeletons'), ('royal_hogs','flying_machine','skeletons'),
    ('royal_giant','mega_minion','ice_spirit'), ('goblin_giant','bomber','bats'),
    ('giant','archers','bomber','zap'), ('giant','knight','spear_goblins','bats'),
    ('giant','musketeer','skeletons','ice_spirit'),
    ('hog_rider','ice_golem','ice_spirit','zap'),
    ('hog_rider','musketeer','skeletons','ice_spirit'),
    ('hog_rider','knight','bats','zap'),
    ('battle_ram','bandit','spear_goblins','zap'),
    ('battle_ram','musketeer','skeletons','ice_spirit'),
    ('prince','archers','skeletons','zap'),
    ('dark_prince','dart_goblin','bats','zap'),
    ('knight','goblin_barrel','princess','skeletons'),
    ('valkyrie','goblin_barrel','goblins','ice_spirit'),
    ('miner','wall_breakers','bats','zap'),
    ('ice_golem','balloon','bats','ice_spirit'),
    ('royal_hogs','flying_machine','spear_goblins','zap'),
    ('goblin_giant','bomber','spear_goblins','skeletons'),
)
COMPLEX = (
    ('lumberjack', 'balloon', 'baby_dragon'),
    ('lava_hound', 'balloon', 'mega_minion'),
    ('golem', 'night_witch', 'baby_dragon'),
    ('elixir_golem', 'battle_healer', 'electro_dragon'),
    ('giant', 'witch', 'musketeer'), ('goblin_giant', 'sparky', 'dark_prince'),
    ('royal_giant', 'fisherman', 'hunter'), ('golem', 'witch', 'wizard'),
    ('lumberjack','balloon','freeze'), ('lumberjack','balloon','inferno_dragon'),
    ('lava_hound','balloon','baby_dragon'), ('lava_hound','balloon','inferno_dragon'),
    ('golem','night_witch','electro_dragon'), ('golem','lumberjack','baby_dragon'),
    ('golem','dark_prince','mega_minion'), ('golem','night_witch','rage'),
    ('elixir_golem','battle_healer','baby_dragon'),
    ('elixir_golem','battle_healer','rage'),
    ('goblin_giant','sparky','rage'), ('goblin_giant','sparky','mini_pekka'),
    ('giant','prince','dark_prince'), ('giant','witch','balloon'),
    ('pekka','electro_wizard','battle_ram'), ('pekka','magic_archer','bandit'),
    ('mega_knight','ram_rider','electro_wizard'),
    ('royal_recruits','royal_hogs','flying_machine'),
    ('lumberjack','balloon','baby_dragon','freeze'),
    ('lumberjack','balloon','inferno_dragon','rage'),
    ('lava_hound','balloon','mega_minion','fireball'),
    ('lava_hound','balloon','baby_dragon','zap'),
    ('lava_hound','balloon','inferno_dragon','skeleton_dragons'),
    ('lava_hound','flying_machine','skeleton_dragons','arrows'),
    ('golem','night_witch','baby_dragon','lumberjack'),
    ('golem','night_witch','electro_dragon','rage'),
    ('golem','witch','wizard','mega_minion'),
    ('golem','dark_prince','baby_dragon','lightning'),
    ('elixir_golem','battle_healer','electro_dragon','rage'),
    ('elixir_golem','battle_healer','baby_dragon','electro_dragon'),
    ('goblin_giant','sparky','dark_prince','rage'),
    ('goblin_giant','sparky','mini_pekka','zap'),
    ('giant','prince','dark_prince','musketeer'),
    ('giant','witch','balloon','freeze'),
    ('pekka','electro_wizard','battle_ram','bandit'),
    ('pekka','magic_archer','bandit','zap'),
    ('mega_knight','ram_rider','electro_wizard','fireball'),
    ('royal_giant','fisherman','hunter','lightning'),
    ('royal_recruits','royal_hogs','flying_machine','zappies'),
    ('giant_skeleton','balloon','baby_dragon','freeze'),
)
TARGETED = {'graveyard', 'goblin_barrel', 'goblin_drill', 'miner'}
DIRECT_DAMAGE = {'fireball', 'zap', 'arrows', 'rocket', 'lightning', 'poison',
                 'earthquake', 'giant_snowball', 'tornado', 'void', 'goblin_curse', 'the_log', 'log'}


# Fixed guesses for central defensive buildings; never inspect actual placement.
BUILDING_GUESSES = {
    'center': (8500, 20500), 'near': (8500, 19500),
    'deep': (8500, 21500), 'outer': (7500, 20500), 'inner': (9500, 20500),
}
BUILDING_PRESSURE = (
    ('hog_rider', 'fireball'), ('giant', 'lightning'),
    ('royal_giant', 'lightning'), ('balloon', 'fireball'),
    ('hog_rider', 'ice_spirit', 'fireball'),
    ('hog_rider', 'ice_golem', 'earthquake'),
    ('giant', 'archers', 'fireball'), ('giant', 'witch', 'lightning'),
    ('battle_ram', 'musketeer', 'fireball'),
    ('royal_giant', 'fisherman', 'lightning'),
    ('goblin_giant', 'sparky', 'lightning'),
    ('lumberjack', 'balloon', 'fireball'),
    ('golem', 'night_witch', 'baby_dragon', 'lightning'),
    ('lava_hound', 'balloon', 'mega_minion', 'fireball'),
    ('electro_giant', 'baby_dragon', 'dark_prince', 'lightning'),
    ('hog_rider', 'ice_golem', 'musketeer', 'fireball'),
)


def family_split(family_id: str) -> str:
    number=int(hashlib.sha256((SCHEMA+':'+family_id).encode()).hexdigest()[:8],16)%100
    return 'train' if number < 70 else 'validation' if number < 85 else 'test'


def build_catalog(cards=None):
    """All supported card keys, plus explicitly reported missing required pushes.

    Playable snapshot keys only; NPC bodies and aliases are not extra cards.
    """
    from sim.defense_assets import load_defense_assets
    loaded,spells,playable,audit=load_defense_assets()
    cards=loaded if cards is None else cards
    scenarios=[]
    missing=[]

    def append(family, difficulty, events, *, setup, multiplier=1, bonus=0):
        absent=sorted({e.card for e in events if e.card not in cards})
        unresolved=sorted({e.card for e in events if e.card in cards and
                           cards[e.card].unit is None and
                           not getattr(cards[e.card],'additional_summons',()) and e.card not in spells
                           and not (e.card=='mirror' and getattr(cards[e.card],'mirror_units_only',False))})
        if absent or unresolved:
            missing.append(dict(family=family,setup=setup,missing_cards=absent,
                                missing_spell_implementations=unresolved))
            return
        for lane in ('left','right'):
            sid=f'{family}/{setup}/x{multiplier}/{lane}'
            scenarios.append(Scenario(sid,family,difficulty,family_split(family),lane,
                                      tuple(events),multiplier,10,bonus,
                                      any(e.card in DIRECT_DAMAGE and e.position=='tower' for e in events),setup))

    for name,spec in sorted(cards.items()):
        if name not in playable:
            continue  # NPC bodies and data aliases are not separate playable cards.
        family='solo:'+name
        for setup in ('bridge','back'):
            position='tower' if name in TARGETED or name in DIRECT_DAMAGE else setup
            events=[ScheduledPlay(name,position=position)]
            if name=='mirror':
                events.insert(0,ScheduledPlay('knight',position=setup))
            if name=='elixir_collector':
                # Collector cannot be in the opening hand. Cycle a real card,
                # pay for it, then draw and place the Collector legally.
                events.insert(0,ScheduledPlay('skeletons',position=setup))
            append(family,'simple',events,setup=setup)
    def position(card,default):
        if card in TARGETED:
            return 'tower'
        if card in spells:
            return 'attack_front' if spells[card].only_own_troops or spells[card].area_only_own_troops else 'defenders'
        return default

    for combo in MEDIUM:
        tank=combo[0]
        family='pair:'+':'.join(sorted(combo))
        bridge=[ScheduledPlay(name,at_ms=2000 if name in spells and name not in TARGETED else 0,
                              position=position(name,'bridge' if index==0 else 'bridge_support'))
                for index,name in enumerate(combo)]
        append(family,'medium',bridge,setup='bridge')
        for setup in ('back','corner'):
            leaders=[name for name in combo[1:]if name in cards and cards[name].unit is not None
                     and name not in TARGETED and name not in spells]
            if not leaders:
                events=[ScheduledPlay(tank,position=position(tank,'back'))]
                events += [ScheduledPlay(name,at_ms=12000,position=position(name,'bridge_support'))for name in combo[1:]]
            else:
                support=leaders[0]
                events=[ScheduledPlay(support,position=setup),
                        ScheduledPlay(tank,position=position(tank,'bridge'),near_bridge_after=support)]
                events += [ScheduledPlay(name,position=position(name,'bridge_support'))for name in combo[1:]if name!=support]
            append(family,'medium',events,setup=setup)
    for combo in COMPLEX:
        for multiplier in (1,2,3):
            family='push:'+':'.join(sorted(combo))
            leader=next((name for name in combo if name in cards and cards[name].unit is not None
                         and name not in spells and name not in TARGETED),None)
            events=[ScheduledPlay(name,at_ms=i*1800,position=position(name,'back'),
                                  near_bridge_after=leader if name in spells else None)
                    for i,name in enumerate(combo)]
            append(family,'complex',events,setup='back',multiplier=multiplier)
            # Explicitly unfair stress variant; economics are recorded, not hidden.
            # Reinforce with a spell not already in the push, keeping a legal
            # eight-card deck and avoiding an impossible immediate repeat cycle.
            extra_spell='zap' if 'fireball' in combo else 'fireball'
            extra_troop='battle_ram' if 'hog_rider' in combo else 'hog_rider'
            harder=events+[ScheduledPlay(extra_spell,at_ms=18000,position='defenders'),
                           ScheduledPlay(extra_troop,at_ms=21000,position='bridge')]
            append('stress:'+family,'impossible',harder,setup='reinforced',multiplier=multiplier,bonus=10)
    for combo in BUILDING_PRESSURE:
        spell=combo[-1]
        troops=combo[:-1]
        difficulty='complex' if len(combo)==4 else 'medium'
        for multiplier in (1,2,3):
            for guess in BUILDING_GUESSES:
                events=[ScheduledPlay(name,at_ms=i*1200,
                                      position='back' if difficulty=='complex' else ('bridge' if i==0 else 'bridge_support'))
                        for i,name in enumerate(troops)]
                events.append(ScheduledPlay(spell,position='building_guess:'+guess,
                                            near_bridge_after=troops[0]))
                append('building:'+':'.join(sorted(combo)),difficulty,events,
                       setup='building_guess:'+guess,multiplier=multiplier)
    # Distinct IDs are required even if data has aliases for the same spec.
    unique={}
    for s in scenarios:
        unique.setdefault(s.scenario_id,s)
    return sorted(unique.values(),key=lambda s:s.scenario_id),missing


def select_scenarios(split='train',difficulty=None,scenario_id=None):
    if split not in SPLITS:
        raise ValueError('Unknown scenario split')
    if difficulty is not None and difficulty not in DIFFICULTIES:
        raise ValueError('Unknown difficulty')
    result=[s for s in build_catalog()[0] if s.split==split and
            (difficulty is None or s.difficulty==difficulty)and
            (scenario_id is None or s.scenario_id==scenario_id)]
    if not result:
        raise ValueError('No supported scenarios match split/difficulty/id')
    return result


def catalog_report():
    scenarios,missing=build_catalog()
    from sim.defense_assets import load_defense_assets
    _cards,_spells,playable,audit=load_defense_assets()
    covered={e.card for s in scenarios if s.difficulty=='simple'for e in s.plays}
    absent=sorted(set(playable)-covered)
    return dict(schema=SCHEMA,scenarios=len(scenarios),families=len({s.family_id for s in scenarios}),
                by_split={split:sum(s.split==split for s in scenarios)for split in SPLITS},
                by_difficulty={d:sum(s.difficulty==d for s in scenarios)for d in DIFFICULTIES},
                missing_requested_families=missing,
                supported_playable_cards=len(covered),snapshot_playable_cards=len(playable),
                missing_playable_cards=absent,spell_audit=audit,
                coverage_complete=not missing and not absent,
                note='Coverage of loaded simulator cards is not coverage of every current game card or verified mechanics.')


def scenarios_for_split(split):
    return select_scenarios(split=split)


if __name__=='__main__':
    print(json.dumps(catalog_report(),indent=2))
