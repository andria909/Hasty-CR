"""Independent attack exercises with versioned, optionally hidden hard windows.

Preparation only: reward coefficients/windows are provisional, not acceptance.
At the hard deadline no subsequent projectile impact earns success. King death
has priority over goal success, which has priority over deadline then technical
truncation at the same tick. No 'enemy bodies exhausted' settling rule applies.
"""
from copy import copy, deepcopy
from dataclasses import dataclass
import math
import random

import numpy as np

from sim import arena
from sim.adapter import grid_to_point
from sim.defense_assets import load_defense_assets
from sim.defense_env import DefenseMatch, TerminalKing, UnitLookup, KingDestroyed
from sim.entities import make_unit
from sim.env import ACTIONS, DECK_26, GRID_W, GRID_H, ClashEnv, observe, _placeable
from sim.runner import resolve_deck

OBSERVATION_CONTRACT = 'attack_timer_scalars_v1'


@dataclass(frozen=True)
class AttackRewards:
    success: float = 10.
    hog_damage: float = .005
    elixir: float = -.02
    missed_window: float = -5.
    own_king_death: float = -10.
    enemy_king_death: float = 10.
    illegal: float = -.1


class AttackEnv:
    """Actual Match actions and damage, with an independent finite attack goal."""
    decode = staticmethod(ClashEnv.decode)

    def __init__(self,seed=0,split='train',scenario=None,scenario_id=None,
                 action_delay_ms=150,max_ms=180000,geometry=None,
                 coordinate_mode='physical_live_cells_v2',visible_fraction=.5,rewards=None,
                 initial_hand_mode='scenario'):
        if not math.isfinite(visible_fraction) or not 0<=visible_fraction<=1:
            raise ValueError('Visible fraction must be in [0,1]')
        if not 0<=action_delay_ms<=10000 or not 50<=max_ms<=600000:
            raise ValueError('Invalid finite attack timing')
        if coordinate_mode not in ('legacy_training','physical_live','physical_live_cells_v2'):
            raise ValueError('Unknown attack coordinate contract')
        if initial_hand_mode not in ('scenario','shuffle'):
            raise ValueError('Unknown attack initial hand mode')
        self.initial_hand_mode=initial_hand_mode
        self.seed=int(seed)
        self.split=split
        if scenario is None:
            from sim.attack_scenarios import select_scenarios
            self.scenarios=list(select_scenarios(split=split,scenario_id=scenario_id))
        else:
            self.scenarios=[scenario]
        if not self.scenarios or any(s.split!=split for s in self.scenarios):
            raise ValueError('Attack scenarios must belong to requested split')
        self._geometry_argument=geometry
        self.geometry=arena.resolve_geometry(geometry)
        self.coordinate_mode=coordinate_mode
        self.action_delay_ms=int(action_delay_ms)
        self.decide_every_ms=500
        self.max_ms=int(max_ms)
        self.visible_fraction=visible_fraction
        self.rewards=rewards or AttackRewards()
        self._cards,self._spells,self._playable,self._spell_audit=load_defense_assets()
        self._cards.update(resolve_deck(self._cards,DECK_26))
        self.match=None

    def reset(self,seed=None,scenario_id=None):
        self.seed=self.seed+1 if seed is None else int(seed)
        candidates=[s for s in self.scenarios if scenario_id is None or s.scenario_id==scenario_id]
        if not candidates:raise ValueError('Attack scenario outside split/filter')
        self.scenario=random.Random(self.seed).choice(candidates)
        s=self.scenario
        if s.goal_kind not in ('hog_tower_damage','spell_finish'):
            raise ValueError('Unsupported attack goal')
        low,high=s.deadline_ms_range
        if not (isinstance(low,int) and isinstance(high,int) and 50<=low<=high<=600000):
            raise ValueError('Invalid finite deadline range')
        from sim.attack_scenarios import sample_episode_context
        context_scenario=copy(s)
        if not hasattr(context_scenario,'paired_case_id'):
            context_scenario.paired_case_id=''
        context=sample_episode_context(context_scenario,self.seed,visible_probability=self.visible_fraction)
        self._deadline_ms=getattr(s,'opportunity_start_ms',0)+context['deadline_ms']
        self._timer_visible=context['deadline_visible']
        if self._geometry_argument is None:
            self.geometry=arena.resolve_geometry(getattr(s,'geometry','legacy_v1'))
        self._shuffle_seed=self.seed if self.initial_hand_mode=='shuffle' else context['hand_seed']
        enemy_plays=tuple(s.enemy_plays)
        names=list(dict.fromkeys(e.card for e in enemy_plays))
        if any(name not in self._cards for name in names) or len(names)>4:
            raise ValueError('Script requires at most four supported opening cards')
        deck=(names+[card for card in DECK_26 if card not in names])[:8]
        self.match=DefenseMatch(cards=self._cards,decks=(list(DECK_26),deck),
            spells=self._spells,seed=self._shuffle_seed,geometry=self.geometry)
        self.match.battle.unit_lookup=UnitLookup(self.match)
        self.match.scenario_elixir_multiplier=1
        for side in (1,-1):self.match.towers[side]['king'].__class__=TerminalKing
        elixir=float(s.initial_elixir)
        if not math.isfinite(elixir) or not 0<=elixir<=10:raise ValueError('Invalid attack elixir')
        self.match.players[1].elixir=round(elixir*1000)
        initial_hand=getattr(s,'initial_hand',None)
        if self.initial_hand_mode=='scenario' and initial_hand is not None:
            if len(initial_hand)!=4 or len(set(initial_hand))!=4 or not set(initial_hand)<=set(DECK_26):
                raise ValueError('Invalid attack initial hand')
            player=self.match.players[1]
            player.hand=list(initial_hand)
            player.queue=[card for card in DECK_26 if card not in initial_hand]
        self._initial_hand=tuple(self.match.players[1].hand)
        self._initial_queue=tuple(self.match.players[1].queue)
        self.match.players[-1].elixir=round(getattr(s,'enemy_initial_elixir',10)*1000)
        self.match.players[-1].hand=deck[:4]
        self.match.players[-1].queue=deck[4:]
        for setting in s.tower_hp:
            tower=self.match.towers[setting.side][setting.lane]
            if setting.hp is not None:
                if not isinstance(setting.hp,int) or not 0<=setting.hp<=tower.max_hitpoints:
                    raise ValueError('Invalid tower HP setup')
                tower.hitpoints=setting.hp
        for setting in s.initial_units:
            spec=self._cards.get(setting.card)
            if spec is None or spec.unit is None or setting.side not in (-1,1):
                raise ValueError('Unsupported initial unit')
            unit=make_unit(0,spec.unit,setting.side,arena.Point(setting.x,setting.y))
            unit.deploy_remaining_ms=max(0,unit.deploy_remaining_ms-getattr(setting,'deployed_ms_ago',0))
            self.match.battle.add(unit)
        self._pending=None
        self._done=False
        self.terminal_reason=None
        self._script_index=0
        self._visible_trigger_times={}
        self._illegal=0
        self._actions=[]
        self._redundant_spells=[]
        self._hog_damage=0
        self._damage_cursor=0
        self._goal_uids=self._goal_target_uids()
        self._first_play_ms=None
        self._successful_finish=False
        self._initial_tower_hp={side:{lane:t.hitpoints for lane,t in towers.items()}
                                for side,towers in self.match.towers.items()}
        self._check_boundaries()
        self._terminal_on_reset=self._done
        self._previous_metrics=self._metrics()
        return self._observe(),self._info()

    def _target_lane(self):
        explicit=getattr(self.scenario,'target_lane','auto')
        if explicit in ('left','right','king'):return explicit
        if explicit!='auto':raise ValueError('Unknown goal target lane')
        towers=self.match.towers[-1]
        if not towers['left'].alive or not towers['right'].alive:return 'king'
        left,right=towers['left'].hitpoints,towers['right'].hitpoints
        return self.scenario.lane if left==right else 'left' if left<right else 'right'

    def _goal_target_uids(self):
        towers=self.match.towers[-1]
        if (getattr(self.scenario,'target_lane','auto')=='auto' and
                towers['left'].alive and towers['right'].alive and
                towers['left'].hitpoints==towers['right'].hitpoints):
            return {towers['left'].uid,towers['right'].uid}
        return {towers[self._target_lane()].uid}

    def _capture_damage(self):
        for at_ms,source_uid,target_uid,amount in self.match.battle.damage_log[self._damage_cursor:]:
            source=self.match.battle.entities.get(source_uid)
            if (at_ms>=getattr(self.scenario,'opportunity_start_ms',0) and target_uid in self._goal_uids and source is not None and source.side==1 and
                    source.name=='hog_rider'):
                self._hog_damage+=max(0,amount)
        self._damage_cursor=len(self.match.battle.damage_log)
        for event in self.match.battle.spell_cast_log:
            if event.get('caster_side')!=1 or event.get('side')!=1 or event.get('spell') not in ('fireball','zap'):
                continue
            if any(hit.get('target_uid') in self._goal_uids and hit.get('killed') and
                   hit.get('hit_ms',-1)>=getattr(self.scenario,'opportunity_start_ms',0) and
                   hit.get('damage',0)>0 for hit in event.get('hits',())):
                self._successful_finish=True

    def _check_boundaries(self):
        if not self.match.towers[1]['king'].alive:
            reason='own_king_destroyed'
        elif not self.match.towers[-1]['king'].alive:
            reason='enemy_king_destroyed'
        elif self.scenario.goal_kind=='hog_tower_damage' and self._hog_damage>0:
            reason='goal_success'
        elif self.scenario.goal_kind=='spell_finish' and self._successful_finish:
            reason='goal_success'
        elif self.match.elapsed_ms>=self._deadline_ms:
            reason='goal_window_missed'
        elif self.match.elapsed_ms>=self.max_ms:
            reason='technical_timeout'
        else:return
        self.terminal_reason=reason
        self._done=True
        self._pending=None

    def _observe(self):
        view=copy(self.match)
        if self.coordinate_mode=='physical_live':
            view.battle=copy(self.match.battle)
            view.battle.entities={uid:copy(e)for uid,e in self.match.battle.entities.items()}
            for e in view.battle.entities.values():e.pos=arena.Point(e.pos.x,31000-e.pos.y)
        view.play_log=[]  # no privileged enemy hand/history from scenario
        encoding='physical_live_cells_v2' if self.coordinate_mode=='physical_live_cells_v2' else 'legacy_training'
        result=observe(view,coordinate_mode=encoding)
        remaining=max(0,self._deadline_ms-self.match.elapsed_ms)/600000 if self._timer_visible else 0.
        result['scalars']=np.concatenate((result['scalars'],np.array(
            [remaining,float(self._timer_visible)],dtype=np.float32)))
        return result

    def _action_mask(self):
        mask=np.zeros(ACTIONS,dtype=bool);mask[0]=True
        if self._done or self._pending is not None:return mask
        down=tuple(lane for lane in ('left','right')if not self.match.towers[-1][lane].alive)
        player=self.match.players[1]
        for slot,card in enumerate(player.hand[:4]):
            if player.elixir<self._cards[card].cost*1000:continue
            grid=_placeable(1,down,card in self._spells,geometry=self.geometry)
            mask[1+slot*GRID_W*GRID_H:1+(slot+1)*GRID_W*GRID_H]=grid.reshape(-1)
        return mask

    @staticmethod
    def _spell_covers(spec,point,target):
        if (not target.alive or target.untargetable or not spec.initial_damage or
                not spec.hits_ground or spec.ignore_buildings or spec.only_own_troops):
            return False
        from .spells import spell_overlaps
        return spell_overlaps(spec,point,target)

    def _pending_lethal_evidence(self,card,point):
        """Conservative CURRENT-SIM redundancy evidence, not live guarantees.

        Only actual queued own Fireball/Zap impacts, verified simple damage
        semantics, static target towers without shields/invulnerability, and
        damage arriving within this task's hard window qualify. No mask changes.
        """
        if card not in ('fireball','zap') or self._spell_audit.get(card,{}).get('limitations'):
            return []
        new_spec=self._spells[card]
        evidence=[]
        for lane,target in self.match.towers[-1].items():
            if target.uid not in self._goal_target_uids() or not self._spell_covers(new_spec,point,target):
                continue
            if any(getattr(target,k,0)for k in ('shield_hitpoints','damage_reduction_pct',
                    'invulnerable_until_ms','permanent_invulnerable','on_damage_invulnerable_ms')):
                continue
            queued=[]
            for due,spec,aim,side in self.match.battle.spell_impacts:
                if (side!=1 or spec.name not in ('fireball','zap') or
                        self._spell_audit.get(spec.name,{}).get('limitations') or
                        not self.match.elapsed_ms<=due<=min(self._deadline_ms,self.max_ms) or
                        not self._spell_covers(spec,aim,target)):
                    continue
                # Complex pulse/spawn/wave/limited-target effects are not
                # projected through the simple SpellSpec.damage_to contract.
                if spec.pulse_events or spec.volley_waves or spec.target_limit or spec.waves:
                    continue
                queued.append(dict(spell=spec.name,due_ms=due,damage=spec.damage_to(target)))
            if sum(row['damage']for row in queued)>=target.hitpoints and queued:
                evidence.append(dict(target_uid=target.uid,lane=lane,hp=target.hitpoints,
                    pending_impacts=queued,basis='current_sim_static_tower_and_queued_effects'))
        return evidence

    def _play_card(self,side,card,point):
        try:return self.match.play_card(side,card,point)
        except KingDestroyed:
            self.match._check_end()
            return True

    def _enemy_actions(self):
        events=self.scenario.enemy_plays
        while self._script_index<len(events):
            event=events[self._script_index]
            if self.match.elapsed_ms<event.at_ms:return
            trigger=getattr(event,'trigger_card',None)
            if trigger:
                visible=[e for e in self.match.battle.entities.values()if e.alive and e.side==1 and e.name==trigger]
                if not visible:return
                self._visible_trigger_times.setdefault(self._script_index,self.match.elapsed_ms)
                if self.match.elapsed_ms-self._visible_trigger_times[self._script_index]<getattr(event,'visibility_delay_ms',0):return
            point=arena.Point(event.x,event.y)
            kind=getattr(event,'position_kind','fixed_guess')
            if kind not in ('fixed_guess','reactive_visible','counterspell'):
                raise ValueError('Unsupported enemy targeting policy')
            # Reactive means visible-unit-triggered timing at the declared
            # physical placement, never omniscient current-hand targeting.
            if not self._play_card(-1,event.card,point):
                # Actual cycle/elixir rules remain in force. No free script card.
                if self.match.elapsed_ms-event.at_ms>10000:
                    raise ValueError('Attack enemy script cannot play its card legally')
                return
            self._script_index+=1
            if self.match.finished:return

    def step(self,action):
        if self.match is None:raise RuntimeError('Call reset first')
        before=self._previous_metrics
        if self._done:
            if not self._terminal_on_reset:raise RuntimeError('Attack episode finished')
            self._terminal_on_reset=False
        else:
            mask=self._action_mask()
            if not isinstance(action,(int,np.integer)) or not 0<=action<len(mask) or not mask[action]:
                self._illegal+=1
            elif action:
                slot,x,y=self.decode(int(action))
                if slot>=4:self._illegal+=1
                else:self._pending=(self.match.elapsed_ms+self.action_delay_ms,
                                    self.match.players[1].hand[slot],x,y)
            end=min(self.match.elapsed_ms+self.decide_every_ms,self.max_ms,self._deadline_ms)
            while not self._done and self.match.elapsed_ms<end:
                self._check_boundaries()
                if self._done:break
                self._goal_uids=self._goal_target_uids()
                self._enemy_actions()
                self._capture_damage()
                self._check_boundaries()
                if self._done:break
                if self._pending is not None and self.match.elapsed_ms>=self._pending[0]:
                    _,card,x,y=self._pending
                    point=grid_to_point(x,y,1)
                    telemetry=None
                    if card=='zap':
                        # The generic engine records exact hits when an observer
                        # event exists, but cast_spell creates one only for Fireball/Log.
                        telemetry=dict(spell='zap',caster_side=1,side=1,
                            cast_ms=self.match.elapsed_ms,due_ms=self.match.elapsed_ms+self._spells['zap'].impact_delay_ms,
                            aim=(point.x,point.y),hits=[],resolved=False)
                        self.match.battle.spell_cast_log.append(telemetry)
                    redundant=self._pending_lethal_evidence(card,point)
                    accepted=self._play_card(1,card,point)
                    if accepted and redundant:
                        self._redundant_spells.append(dict(at_ms=self.match.elapsed_ms,card=card,evidence=redundant))
                    if not accepted and telemetry is not None:
                        self.match.battle.spell_cast_log.remove(telemetry)
                    self._actions.append(dict(at_ms=self.match.elapsed_ms,card=card,accepted=accepted))
                    self._illegal+=not accepted
                    goal_cards=('hog_rider',) if self.scenario.goal_kind=='hog_tower_damage' else ('fireball','zap')
                    if accepted and self._first_play_ms is None and card in goal_cards:
                        self._first_play_ms=self.match.elapsed_ms
                    self._pending=None
                # Fractional last tick is allowed; never advance beyond deadline.
                self._capture_damage()
                self._check_boundaries()
                if self._done:break
                self.match.step(min(arena.TICK_MS,end-self.match.elapsed_ms))
                self._capture_damage()
                self._check_boundaries()
        after=self._metrics()
        reward=self.rewards.hog_damage*(after['hog_target_tower_damage']-before['hog_target_tower_damage'])
        reward+=self.rewards.elixir*(after['elixir_spent']-before['elixir_spent'])
        reward+=self.rewards.illegal*(after['illegal_actions']-before['illegal_actions'])
        if after['terminal_reason']!=before['terminal_reason']:
            reward+=self.rewards.success*(after['terminal_reason']=='goal_success')
            reward+=self.rewards.missed_window*(after['terminal_reason']=='goal_window_missed')
            reward+=self.rewards.own_king_death*(after['terminal_reason']=='own_king_destroyed')
            reward+=self.rewards.enemy_king_death*(after['terminal_reason']=='enemy_king_destroyed')
        self._previous_metrics=after
        truncated=self._done and self.terminal_reason=='technical_timeout'
        return self._observe(),float(reward),bool(self._done and not truncated),bool(truncated),self._info()

    def _metrics(self):
        own_events=[event for event in self.match.battle.spell_cast_log if event.get('caster_side')==1]
        resolved=[event for event in own_events if event.get('resolved')]
        wasted=sum(not any(hit.get('damage',0)>0 for hit in event.get('hits',()))for event in resolved)
        flags=['attack_reward_and_window_ranges_provisional','simulator_mechanics_not_fully_validated',
               'redundant_spell_accounting_current_sim_only',
               'deadline_is_hard_cutoff_even_for_inflight_spells','timer_visibility_fraction_requires_validation']
        if self.geometry!=arena.LEGACY_GEOMETRY:flags+=list(self.geometry.limitations)
        flags+=list(getattr(self.scenario,'limitations',()))
        return dict(scenario_id=self.scenario.scenario_id,family_id=self.scenario.family_id,
            split=self.split,difficulty=getattr(self.scenario,'difficulty',None),goal_kind=self.scenario.goal_kind,lane=self.scenario.lane,
            target_lane=self._target_lane(),setup_kind=getattr(self.scenario,'setup_kind',None),
            finisher_spell=getattr(self.scenario,'finisher_spell',None),
            single_cast_sufficient=getattr(self.scenario,'single_cast_sufficient',None),timer_visible=self._timer_visible,deadline_ms=self._deadline_ms,
            initial_hand_mode=self.initial_hand_mode,initial_hand=list(self._initial_hand),
            initial_queue=list(self._initial_queue),
            visible_fraction=self.visible_fraction,shuffle_seed=self._shuffle_seed,seed=self.seed,elapsed_ms=self.match.elapsed_ms,
            terminal_reason=self.terminal_reason,goal_success=self.terminal_reason=='goal_success',
            attack_success=self.terminal_reason in ('goal_success','enemy_king_destroyed'),
            goal_window_missed=self.terminal_reason=='goal_window_missed',
            time_to_first_goal_card_ms=(None if self._first_play_ms is None else
                max(0,self._first_play_ms-getattr(self.scenario,'opportunity_start_ms',0))),
            anticipatory_goal_card_ms=(0 if self._first_play_ms is None else
                max(0,getattr(self.scenario,'opportunity_start_ms',0)-self._first_play_ms)),
            first_goal_card_play_ms=self._first_play_ms,hog_target_tower_damage=self._hog_damage,
            successful_spell_finish=(None if self.terminal_reason=='enemy_king_destroyed' and
                not self._successful_finish else self._successful_finish),
            spell_finish_attribution=('king_terminal_interrupted_hit_ledger' if
                self.terminal_reason=='enemy_king_destroyed' and not self._successful_finish else 'observed_hit_events'),
            hog_damage_ledger_complete=self.terminal_reason!='enemy_king_destroyed',extra_spells=wasted,
            missed_resolved_spells=wasted,redundant_spells=len(self._redundant_spells),
            redundant_spell_evidence=list(self._redundant_spells),
            unresolved_spells=sum(not event.get('resolved')for event in own_events),
            elixir_spent=self.match.players[1].elixir_spent,illegal_actions=int(self._illegal),
            script_plays_completed=self._script_index,geometry=self.geometry.profile_id,
            geometry_sha256=self.geometry.fingerprint,coordinate_mode=self.coordinate_mode,
            observation_contract=OBSERVATION_CONTRACT,coverage_flags=flags)

    def _info(self):
        # Training diagnostics contain privileged deadline, never use info as
        # model features. Action masks carry no deadline hints before terminal.
        return dict(action_mask=self._action_mask(),metrics=self._metrics(),terminal_reason=self.terminal_reason)

    def snapshot(self):return deepcopy(self.__dict__)

    def restore(self,snapshot):
        if (snapshot['geometry']!=self.geometry or snapshot['coordinate_mode']!=self.coordinate_mode or
                snapshot.get('initial_hand_mode','scenario')!=self.initial_hand_mode):
            raise ValueError('Attack snapshot input contract mismatch')
        self.__dict__=deepcopy(snapshot)
        return self._observe(),self._info()
