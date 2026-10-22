"""Finite defensive pushes, mechanical actions and measured episode outcomes.

Own king death or enemy king death terminates immediately. Once the scripted
attack and delayed hostile events are exhausted, no further cards may be played;
existing bodies/projectiles finish naturally, with a separate timeout boundary.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass
import math
import random

import numpy as np

from sim import arena
from sim.adapter import grid_to_point
from sim.defense_scenarios import Scenario, select_scenarios
from sim.defense_assets import load_defense_assets
from sim.env import (ACTIONS, CARD_ACTIONS, DECK_26, GRID_H, GRID_W,
                     ClashEnv, observe, _placeable)
from sim.gamedata import load_characters, load_gamedata, to_snake_case
from sim.entities import Entity
from sim.match import Match, SINGLE_MS
from sim.reward_events import score_spell_cast
from sim.runner import resolve_deck
from sim.spells import load_spells


@dataclass(frozen=True)
class DefenseRewards:
    own_hp_fraction: float = -20
    enemy_hp_fraction: float = 2
    own_princess_destroyed: float = -25
    own_king_destroyed: float = -100
    enemy_king_destroyed: float = 20
    elixir_spent: float = -.05
    spell_quality: float = .1
    spell_miss: float = -.05
    illegal: float = -.1


class KingDestroyed(Exception):
    """Unwind the exact damage phase before any post-terminal effects."""


class TerminalKing(Entity):
    def _take_hitpoint_damage(self,amount):
        was_alive=self.alive
        dealt=super()._take_hitpoint_damage(amount)
        if was_alive and not self.alive:
            raise KingDestroyed()
        return dealt


class DefenseMatch(Match):
    """Scenario clock does not end at the regular-match crown tiebreak."""
    def _check_end(self):
        own_dead=not self.towers[1]['king'].alive
        enemy_dead=not self.towers[-1]['king'].alive
        if own_dead or enemy_dead:
            self.finished=True
            self.result='draw' if own_dead and enemy_dead else 'top' if own_dead else 'bottom'

    def regen_ms(self):
        return round(SINGLE_MS/getattr(self,'scenario_elixir_multiplier',1))

    def step(self,dt_ms=arena.TICK_MS):
        try:
            super().step(dt_ms)
        except KingDestroyed:
            self._check_end()


@dataclass
class UnitLookup:
    """Pickle-safe replacement of Match's nested closure for exact snapshots."""
    match: Match

    def __call__(self,character):
        table=load_characters(self.match.level)
        exact=table.get(character)
        if exact is not None:
            return exact
        key=to_snake_case(character)
        spec=self.match.cards.get(key)
        if spec is not None and spec.unit is not None:
            return spec.unit
        exact=table.get(key)
        if exact is not None:
            return exact
        for spec in self.match.cards.values():
            if spec.unit is not None and spec.unit.name==key:
                return spec.unit
        self.match.missing_spawns.add(character)
        return None


# Only transient event queues are counted. Histories/caches never prevent end.
PENDING_FIELDS=(
    'in_flight','projectile_area_events','target_poison_controllers','rage_spears',
    'container_drop_events','deploy_barrage_events','control_in_flight','ability_summons',
    'paratroopers','wind_areas','forced_moves','chain_projectiles','ability_area_events',
    'pull_flight','active_pulls','ability_dash_chains',
    'ability_guard_events','ability_deploy_events','ability_lane_switch_events',
    'ability_link_events','areas','spell_impacts','spell_spawn_events','rolling_spells',
    'spell_pulse_events','on_hit_spawn_events','starting_summon_damage_events',
    'level_up_events','ability_taunt_events','taunt_areas','hurl_pending','hurl_events',
    'siege_impacts','split_events','split_damage_events','reroll_events','spin_pending',
    'spin_events','reinforcement_events','parry_events','ability_warp_events',
    'delayed_self_buffs','temporary_form_events')


class DefenseEnv(ClashEnv):
    def __init__(self,seed=0,split='train',difficulty=None,scenario_id=None,
                 action_delay_ms=150,max_ms=240000,initial_elixir=10,
                 rewards=None,scenario=None,public_history_mode='unavailable',
                 geometry=None,coordinate_mode='physical_live'):
        if not (math.isfinite(initial_elixir)and 0<=initial_elixir<=10):
            raise ValueError('Initial elixir must be in [0,10]')
        if not 0<=action_delay_ms<=10000 or not 50<=max_ms<=600000:
            raise ValueError('Invalid finite timing configuration')
        if public_history_mode not in ('unavailable','confirmed'):
            raise ValueError('Unknown public history mode')
        self.geometry=arena.resolve_geometry(geometry)
        if coordinate_mode not in ('legacy_training','physical_live','physical_live_cells_v2'):
            raise ValueError('Unknown defense policy coordinate contract')
        self.coordinate_mode=coordinate_mode
        self.seed=int(seed)
        self.split=split
        self.difficulty=difficulty
        self.scenario_id=scenario_id
        self.scenarios=([scenario] if scenario is not None else
                        select_scenarios(split,difficulty,scenario_id))
        if any(s.split!=split for s in self.scenarios):
            raise ValueError('Scenario does not belong to requested split')
        self._cards,self._spells,self._playable,self._spell_audit=load_defense_assets()
        self._cards.update(resolve_deck(self._cards,DECK_26))
        self.action_delay_ms=int(action_delay_ms)
        self.decide_every_ms=500
        self.max_ms=int(max_ms)
        self.initial_elixir=float(initial_elixir)
        self.rewards=DefenseRewards() if rewards is None else rewards
        self.public_history_mode=public_history_mode
        self.match=None
        self.closed=False

    def reset(self,seed=None,scenario_id=None):
        self.seed=self.seed+1 if seed is None else int(seed)
        candidates=self.scenarios
        if scenario_id is not None:
            candidates=[s for s in candidates if s.scenario_id==scenario_id]
            if not candidates:
                raise ValueError('Scenario ID is outside this split/filter')
        self.scenario=random.Random(self.seed).choice(candidates)
        s=self.scenario
        planned=list(dict.fromkeys(e.card for e in s.plays))
        if any(name not in self._cards for name in planned):
            raise ValueError('Unsupported scenario card')
        if len(planned)>8:
            raise ValueError('Scenario attack uses more than one legal eight-card deck')
        deck=planned+[name for name in DECK_26 if name not in planned]
        deck=deck[:8]
        from sim.match import STARTING_HAND_EXCLUSIONS
        excluded=[card for card in deck if card in STARTING_HAND_EXCLUSIONS]
        eligible=[card for card in deck if card not in STARTING_HAND_EXCLUSIONS]
        # Preserve script order while obeying cards forbidden in opening hand.
        # The first legal opening play draws the first excluded card.
        deck=eligible[:4]+excluded+eligible[4:]
        self.match=DefenseMatch(cards=self._cards,decks=(list(DECK_26),deck),
                                seed=self.seed,spells=self._spells,geometry=self.geometry)
        self.match.battle.unit_lookup=UnitLookup(self.match)
        for side in (1,-1):
            self.match.towers[side]['king'].__class__=TerminalKing
        self.match.scenario_elixir_multiplier=s.elixir_multiplier
        self.match.players[1].elixir=round(self.initial_elixir*1000)
        enemy=self.match.players[-1]
        # Scripted initial enemy hand is explicit, not extra cycle privileges.
        enemy.hand=deck[:4]
        enemy.queue=deck[4:]
        enemy.elixir=round(s.enemy_initial_elixir*1000)
        self._enemy_bonus=round(s.enemy_bonus_elixir*1000)
        self._enemy_bonus_used=0
        self._script_index=0
        self._bridge_ready=set()
        self._script_seen_units=set()
        self._pending=None
        self.phase='defense'
        self.terminal_reason=None
        self._done=False
        self._illegal=0
        self._spell_cursor=0
        self._spell_quality=0.
        self._spell_misses=0
        self._actions=[]
        self._initial_hp={side:{lane:t.hitpoints for lane,t in towers.items()}
                          for side,towers in self.match.towers.items()}
        self._last_metrics=None
        baseline=self._metrics()
        self._process_enemy()
        self._check_boundaries()
        self._terminal_on_reset=self._done
        self._last_metrics=baseline
        return self._observe(),self._info()

    def _target(self,event):
        x=3500 if self.scenario.lane=='left' else 14500
        if event.position in ('defenders','attack_front'):
            side=1 if event.position=='defenders' else -1
            units=[u for u in self.match.battle.entities.values()if u.alive and not u.is_tower
                   and u.side==side and getattr(u,'object_role','')!='pickup_effect_carrier'
                   and (u.pos.x<9000)==(x<9000)]
            if units:
                # Visible-body targeting; never reads the defender's hidden hand.
                chosen=min(units,key=lambda u:abs(u.pos.y-24000))
                return chosen.pos
            return self.match.towers[1][self.scenario.lane].pos if side==1 else arena.Point(x,14000)
        if event.position.startswith('building_guess:'):
            from sim.defense_scenarios import BUILDING_GUESSES
            gx,gy=BUILDING_GUESSES[event.position.split(':',1)[1]]
            return arena.Point(gx if self.scenario.lane=='left' else arena.WIDTH-gx,gy)
        if event.position=='tower':
            return self.match.towers[1][self.scenario.lane].pos
        if event.position=='bridge':
            return arena.Point(x,14000)
        if event.position=='bridge_support':
            return arena.Point(x,12500)
        if event.position=='back':
            return arena.Point(x,1500)
        if event.position=='corner':
            return arena.Point(500 if x<9000 else 17500,1500)
        raise ValueError('Unknown scenario placement')

    def _process_enemy(self):
        if self.phase!='defense' or self.match.finished:
            return
        for unit in self.match.battle.entities.values():
            if unit.alive and unit.side==-1 and not unit.is_tower:
                name=to_snake_case(unit.name)
                self._script_seen_units.add(name)
                if unit.pos.y>=11500:
                    self._bridge_ready.add(name)
        while self._script_index<len(self.scenario.plays):
            event=self.scenario.plays[self._script_index]
            if self.match.elapsed_ms<event.at_ms:
                return
            if event.near_bridge_after:
                name=to_snake_case(self._cards[event.near_bridge_after].unit.name)
                units=[u for u in self.match.battle.entities.values()if u.alive and
                       u.side==-1 and not u.is_tower and to_snake_case(u.name)==name]
                # Arrival is latched: later death must not erase the trigger.
                # If the escort dies on approach, continue the scheduled attack
                # rather than waiting forever for an impossible arrival.
                ready=(name in self._bridge_ready or any(u.pos.y>=11500 for u in units)
                       or (name in self._script_seen_units and not units))
                if not ready:
                    if self.match.elapsed_ms>event.at_ms+event.max_wait_ms:
                        self.terminal_reason='invalid_script_trigger_timeout'
                        self._done=True
                    return
            player=self.match.players[-1]
            cost=self._cards[event.card].cost*1000
            shortfall=max(0,cost-player.elixir)
            if shortfall and self._enemy_bonus:
                grant=min(shortfall,self._enemy_bonus)
                player.elixir+=grant
                self._enemy_bonus-=grant
                self._enemy_bonus_used+=grant
            try:
                accepted=self.match.play_card(-1,event.card,self._target(event))
            except KingDestroyed:
                self.match._check_end()
                self._check_boundaries()
                return
            if not accepted:
                if self.match.elapsed_ms>event.at_ms+event.max_wait_ms:
                    self.terminal_reason='invalid_script_play_timeout'
                    self._done=True
                return
            self._script_index+=1

    def _has_pending_effects(self):
        battle=self.match.battle
        if any(bool(getattr(battle,name,()))for name in PENDING_FIELDS):
            return True
        active_timers=('buff_until_ms','buff_invisible_until_ms','invulnerable_until_ms',
                       'unkillable_until_ms','taunt_until_ms','grounded_until_ms',
                       'cursed_until_ms','deflect_until_ms','forced_move_until_ms',
                       'siege_until_ms','control_cast_until_ms')
        return any(u.alive and any(getattr(u,timer,0)>battle.now_ms for timer in active_timers)
                   for u in battle.entities.values())

    def _has_pending_hostile_spawns(self):
        battle=self.match.battle
        if any(event[3]==-1 for event in battle.spell_spawn_events):
            return True
        for _due,spec,_at,side in battle.spell_impacts:
            if side==-1 and (spec.spawn_character or spec.area_spawn_character or spec.area_spawn_events):
                return True
        if any(event[2]==-1 and event[0].spawn_character for event in battle.rolling_spells):
            return True
        if any(event[2]==-1 and (event[0].area_spawn_character or event[0].area_spawn_events)
               for event in battle.areas):
            return True
        def contains_enemy(value):
            if isinstance(value,Entity):
                return value.side==-1
            if isinstance(value,(tuple,list)):
                return any(contains_enemy(x)for x in value)
            return False
        for field in ('ability_summons','paratroopers','container_drop_events',
                      'deploy_barrage_events','ability_guard_events','ability_deploy_events',
                      'reinforcement_events','split_events','temporary_form_events'):
            if any(contains_enemy(event)for event in getattr(battle,field,())):
                return True
        for event in battle.on_hit_spawn_events:
            parent=battle.entities.get(event[1])
            if parent is not None and parent.side==-1:
                return True
        return False

    def _check_boundaries(self):
        if self._done:
            return
        self.match._check_end()
        if self.match.finished:
            self.terminal_reason=('own_king_destroyed' if not self.match.towers[1]['king'].alive
                                  else 'enemy_king_destroyed')
            self._done=True
            self._pending=None
            return
        if self.match.elapsed_ms>=self.max_ms:
            self.terminal_reason='timeout'
            self._done=True
            self._pending=None
            return
        bodies=[u for u in self.match.battle.entities.values()if u.alive and not u.is_tower
                and getattr(u,'object_role','')!='pickup_effect_carrier']
        if (self.phase=='defense' and self._script_index==len(self.scenario.plays)and
                not any(u.side==-1 for u in bodies)and not self._has_pending_hostile_spawns()):
            self.phase='settle'
            self._pending=None
        if self.phase=='settle' and not bodies and not self._has_pending_effects():
            self.terminal_reason='cleared'
            self._done=True

    def _action_mask(self):
        mask=np.zeros(ACTIONS,dtype=bool);mask[0]=True
        if self._done or self.phase!='defense' or self._pending is not None:
            return mask
        down=tuple(lane for lane in ('left','right')if not self.match.towers[-1][lane].alive)
        p=self.match.players[1]
        for slot,card in enumerate(p.hand):
            if p.elixir<self._cards[card].cost*1000:
                continue
            grid=_placeable(1,down,card in self._spells,geometry=self.geometry)
            mask[1+slot*GRID_W*GRID_H:1+(slot+1)*GRID_W*GRID_H]=grid.reshape(-1)
        return mask

    def _observe(self):
        encoding=('physical_live_cells_v2' if self.coordinate_mode=='physical_live_cells_v2'
                  else 'legacy_training')
        if self.match is None:
            return observe(None,coordinate_mode=encoding)
        view=copy(self.match)
        if self.coordinate_mode=='physical_live':
            # Existing explicit ARM64 compatibility proxy, unchanged for v25.
            view.battle=copy(self.match.battle)
            entities={}
            for uid,entity in self.match.battle.entities.items():
                e=copy(entity);e.pos=arena.Point(entity.pos.x,31000-entity.pos.y);entities[uid]=e
            view.battle.entities=entities
        if self.public_history_mode=='unavailable':
            view.play_log=[]
        return observe(view,coordinate_mode=encoding)

    def _metrics(self):
        towers=self.match.towers
        losses={lane:max(0,self._initial_hp[1][lane]-t.hitpoints)
                for lane,t in towers[1].items()}
        damage={lane:max(0,self._initial_hp[-1][lane]-t.hitpoints)
                for lane,t in towers[-1].items()}
        flags=['simulator_mechanics_not_fully_validated']
        if self.geometry!=arena.LEGACY_GEOMETRY:
            flags.extend('geometry:'+value for value in self.geometry.limitations)
        if self.public_history_mode=='unavailable':
            flags.append('opponent_play_history_unavailable')
        if self.match.missing_spawns:
            flags.append('missing_spawn_profiles')
        if self.match.battle.unmodelled_projectiles:
            flags.append('unmodelled_projectiles')
        for card in {e.card for e in self.scenario.plays}|set(DECK_26):
            unit=self._cards[card].unit
            if unit is not None:
                for limitation in unit.raw.get('_public_profile_limitations',()):
                    flags.append('approximate_unit_mechanics:'+card+':'+limitation)
        for event in self.scenario.plays:
            details=self._spell_audit.get(event.card,{})
            if details.get('limitations'):
                flags.append('approximate_spell_mechanics:'+event.card)
        return dict(scenario_id=self.scenario.scenario_id,family_id=self.scenario.family_id,
                    split=self.split,difficulty=self.scenario.difficulty,lane=self.scenario.lane,
                    seed=self.seed,phase=self.phase,terminal_reason=self.terminal_reason,
                    elapsed_ms=self.match.elapsed_ms,own_tower_hp_loss=losses,
                    own_tower_fraction_loss={lane:losses[lane]/self._initial_hp[1][lane]for lane in losses},
                    own_towers_alive={lane:bool(t.alive)for lane,t in towers[1].items()},
                    enemy_tower_hp_damage=damage,
                    enemy_tower_fraction_damage={lane:damage[lane]/self._initial_hp[-1][lane]for lane in damage},
                    elixir_spent=self.match.players[1].elixir_spent,
                    enemy_elixir_spent=self.match.players[-1].elixir_spent,
                    enemy_bonus_elixir_used=self._enemy_bonus_used/1000,
                    spell_quality=self._spell_quality,spell_misses=self._spell_misses,
                    illegal_actions=self._illegal,no_tower_damage=not any(losses.values()),
                    unavoidable_tower_damage=self.scenario.unavoidable_tower_damage,
                    script_plays_completed=self._script_index,script_plays_total=len(self.scenario.plays),
                    public_history_mode=self.public_history_mode,coverage_flags=flags,
                    geometry=self.geometry.profile_id,geometry_sha256=self.geometry.fingerprint,
                    coordinate_mode=self.coordinate_mode)

    def _reward(self,before,after):
        w=self.rewards
        r=w.own_hp_fraction*(sum(after['own_tower_fraction_loss'].values())-sum(before['own_tower_fraction_loss'].values()))
        r+=w.enemy_hp_fraction*(sum(after['enemy_tower_fraction_damage'].values())-sum(before['enemy_tower_fraction_damage'].values()))
        for lane in ('left','right'):
            if before['own_towers_alive'][lane] and not after['own_towers_alive'][lane]:
                r+=w.own_princess_destroyed
        if before['own_towers_alive']['king'] and not after['own_towers_alive']['king']:
            r+=w.own_king_destroyed
        if after['terminal_reason']=='enemy_king_destroyed' and before['terminal_reason']!='enemy_king_destroyed':
            r+=w.enemy_king_destroyed
        r+=w.elixir_spent*(after['elixir_spent']-before['elixir_spent'])
        r+=w.spell_quality*(after['spell_quality']-before['spell_quality'])
        r+=w.spell_miss*(after['spell_misses']-before['spell_misses'])
        r+=w.illegal*(after['illegal_actions']-before['illegal_actions'])
        return float(r)

    def step(self,action):
        if self.match is None:
            raise RuntimeError('Call reset first')
        before=self._last_metrics
        if self._done:
            if not self._terminal_on_reset:
                raise RuntimeError('Episode is finished; reset before stepping')
            if action!=0:
                self._illegal+=1
            self._terminal_on_reset=False
            after=self._metrics()
            reward=self._reward(before,after)
            self._last_metrics=after
            truncated=self.terminal_reason=='timeout' or self.terminal_reason.startswith('invalid_script')
            return self._observe(),reward,not truncated,bool(truncated),self._info()
        self._check_boundaries()
        if not self._done:
            mask=self._action_mask()
            if not isinstance(action,(int,np.integer))or not 0<=action<ACTIONS or not mask[int(action)]:
                self._illegal+=1
            elif action:
                slot,x,y=self.decode(int(action))
                self._pending=(self.match.elapsed_ms+self.action_delay_ms,
                               self.match.players[1].hand[slot],x,y)
        target=min(self.match.elapsed_ms+self.decide_every_ms,self.max_ms)
        while not self._done and self.match.elapsed_ms<target:
            self._check_boundaries()
            if self._done:
                break
            self._process_enemy()
            if self._done:
                break
            if self._pending is not None and self.phase=='defense' and self.match.elapsed_ms>=self._pending[0]:
                _,card,x,y=self._pending
                try:
                    ok=self.match.play_card(1,card,grid_to_point(x,y,1))
                except KingDestroyed:
                    self._pending=None
                    self.match._check_end()
                    self._check_boundaries()
                    break
                self._actions.append(dict(at_ms=self.match.elapsed_ms,card=card,x=x,y=y,accepted=ok))
                self._illegal+=not ok
                self._pending=None
            self.match.step()
            self._check_boundaries()
        events=self.match.battle.spell_cast_log
        for event in events[self._spell_cursor:]:
            if not event.get('resolved'):
                break
            score=score_spell_cast(event)
            self._spell_quality+=score.quality
            self._spell_misses+=score.missed
            self._spell_cursor+=1
        after=self._metrics()
        reward=self._reward(before,after)
        self._last_metrics=after
        truncated=self._done and (self.terminal_reason=='timeout' or self.terminal_reason.startswith('invalid_script'))
        terminated=self._done and not truncated
        return self._observe(),reward,bool(terminated),bool(truncated),self._info()

    def _info(self):
        return dict(action_mask=self._action_mask(),metrics=self._metrics(),
                    phase=self.phase,terminal_reason=self.terminal_reason)

    def snapshot(self):
        return deepcopy(self.__dict__)

    def restore(self,snapshot):
        previous_geometry=arena.resolve_geometry(snapshot.get('geometry'))
        previous_coordinates=snapshot.get('coordinate_mode','physical_live')
        if previous_geometry!=self.geometry or previous_coordinates!=self.coordinate_mode:
            raise ValueError('Snapshot geometry/coordinate contract differs from this environment')
        self.__dict__=deepcopy(snapshot)
        self.geometry=previous_geometry
        self.coordinate_mode=previous_coordinates
        return self._observe(),self._info()

    def close(self):
        self.closed=True
