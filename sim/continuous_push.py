"""Full games against replenishing, legal opponents; no free enemy spawns.

Same public 309 inputs as deployment. Finite regulation/overtime; random
pushes continue until the game ends. Rewards are auxiliary to terminal wins.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

import numpy as np

from sim import arena
from sim.adapter import grid_to_point
from sim.defense_assets import load_defense_assets
from sim.env import ACTIONS, DECK_26, ClashEnv
from sim.eval_defense_duel import mechanical_mask, policy_observation, JOINT_MODE, outcome
from sim.match import Match
from sim.runner import resolve_deck

CONTRACT = 'joint_public_presence_continuous_slots_v2'
DECKS = {
    'hog': ('hog_rider','cannon','musketeer','knight','skeletons','electro_spirit','fireball','zap'),
    'giant': ('giant','musketeer','mini_pekka','bomber','archers','spear_goblins','fireball','arrows'),
    'air': ('balloon','baby_dragon','minions','knight','musketeer','skeletons','fireball','zap'),
    'golem': ('golem','night_witch','baby_dragon','mega_minion','bomber','skeletons','lightning','zap'),
    'swarm': ('goblin_barrel','goblin_gang','skeleton_army','knight','musketeer','inferno_tower','rocket','zap'),
    'bridge': ('battle_ram','bandit','dark_prince','musketeer','minions','knight','fireball','zap'),
}


@dataclass(frozen=True)
class PushCase:
    task: str
    family: str
    style: str
    split: str
    @property
    def scenario_id(self): return f'{self.task}::{self.family}/{self.style}'
    def to_dict(self): return dict(task=self.task,family=self.family,style=self.style,split=self.split)


class ContinuousPushEnv:
    def __init__(self,seed=0,split='train',action_delay_ms=225,max_ms=310000,
                 geometry='native_anchors_candidate_v1',coordinate_mode='physical_live_cells_v2',
                 continuous_profile=None,**unused):
        if split not in ('train','validation','test'): raise ValueError('Unknown split')
        if coordinate_mode!='physical_live_cells_v2': raise ValueError('Continuous requires consistent coordinates')
        if not 0<=action_delay_ms<=10000 or not 180000<=max_ms<=600000: raise ValueError('Invalid finite timing')
        self.seed=seed; self.split=split; self.action_delay_ms=action_delay_ms; self.max_ms=max_ms
        self.geometry=arena.resolve_geometry(geometry); self.coordinate_mode=coordinate_mode
        # The clock/seed split is disjoint, and held-out deck compositions differ
        # by style; these are synthetic opponent tests, not unseen card claims.
        styles={'train':('random','push'),'validation':('heldout_push',),'test':('heldout_random',)}[split]
        self.scenarios=[PushCase(task,family,style,split) for task in ('attack','defense') for family in DECKS for style in styles]
        self.profile=dict(overflow_penalty=.15,missed_fireball_penalty=0.,board_potential_weight=0.,
                          fireball_value_reward=0.,reserve_potential_weight=0.,barbarian_train_fraction=0.,lane_reflection_fraction=0.,shaping_gamma=.999,own_hp_penalty=4.,enemy_hp_reward=3.,
                          win_reward=20.,loss_penalty=20.,action_cooldown_ms=0,decision_interval_ms=250,
                          cannon_center_only=1,tower_damage_all=0,
                          source_radius_reach=1,target_lock_in_range=0,hog_sight_clipping=0)
        if continuous_profile:
            supplied=json.loads(Path(continuous_profile).read_text()) if isinstance(continuous_profile,(str,Path)) else dict(continuous_profile)
            if set(supplied)-set(self.profile): raise ValueError('Unknown continuous coefficient')
            self.profile.update(supplied)
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in self.profile.values()):
            raise ValueError('Invalid continuous coefficient')
        if self.profile['decision_interval_ms']<50:raise ValueError('Decision interval below physics tick')
        if self.profile['shaping_gamma']>1:raise ValueError('Shaping discount above one')
        if self.profile['barbarian_train_fraction']>1:raise ValueError('Invalid Barbarian training fraction')
        if self.profile['lane_reflection_fraction']>1:raise ValueError('Invalid lane reflection fraction')
        if any(self.profile[k]not in (0,1)for k in ('cannon_center_only','tower_damage_all','source_radius_reach','target_lock_in_range','hog_sight_clipping')):
            raise ValueError('Cannon/tower/combat mode flags must be 0 or 1')
        self.cards,self.spells,_,_=load_defense_assets();self.cards.update(resolve_deck(self.cards,list(DECK_26)))
        self._board_values=None
        if self.profile['board_potential_weight']:
            from sim.board_potential import unit_values
            self._board_values=unit_values(self.cards,preferred_cards=DECK_26)
        missing={c for deck in DECKS.values() for c in deck if c not in self.cards}
        if missing: raise ValueError(f'Unsupported opponent cards: {sorted(missing)}')
        self.match=None

    def reset(self,seed=None,scenario_id=None):
        self.seed=self.seed+1 if seed is None else seed
        low={'train':0,'validation':1000000000,'test':2000000000}[self.split]
        self.rng=random.Random(low+self.seed)
        cases=[s for s in self.scenarios if scenario_id is None or s.scenario_id==scenario_id]
        if not cases: raise ValueError('Scenario outside split')
        self.scenario=self.rng.choice(cases)
        self.reflected=bool(self.split=='train'and self.profile['lane_reflection_fraction']and self.rng.random()<self.profile['lane_reflection_fraction'])
        enemy=list(DECKS[self.scenario.family])
        # TRAIN augmentation only; still an ordinary eight-card hand/cycle
        # opponent that pays five elixir. Validation/test composition is fixed.
        if self.split=='train'and self.profile['barbarian_train_fraction']and self.rng.random()<self.profile['barbarian_train_fraction']:
            enemy[3]='barbarians'
        if self.split!='train':
            # Hold out deck compositions while preserving the push archetype.
            enemy[4]='valkyrie' if self.split=='validation' else 'hunter'
        self.match=Match(cards=self.cards,spells=self.spells,decks=(list(DECK_26),enemy),
                         seed=low+self.seed,geometry=self.geometry)
        self.match.battle.normal_range_mode=('edge_to_edge_v1' if self.profile['source_radius_reach'] else 'legacy_target_edge_v1')
        self.match.battle.target_lock_mode=('in_range_v1' if self.profile['target_lock_in_range'] else 'legacy_periodic_v1')
        # Old profiles must retain their measured physics. New training and
        # evaluation explicitly opt into the independently tagged sight fix.
        self.match.battle.hog_sight_mode=('side_clip_v1' if self.profile['hog_sight_clipping'] else 'legacy_radial_v1')
        from sim.defense_env import UnitLookup
        self.match.battle.unit_lookup=UnitLookup(self.match)
        from sim.opponents import ScriptedOpponent
        self.scripted=ScriptedOpponent(self.cards,side=-1,deck=enemy,
            style='beatdown' if self.scenario.family in ('golem','giant','air') else 'cycle',seed=low+self.seed)
        self.scripted.reset()
        self.pending=[];self.slot_ready={};self.next_enemy=0;self.illegal=0
        self.actions=[];self.enemy_actions=[];self.done=False;self.terminal_reason='running'
        self._board_shaping_return=0.
        self._reserve_shaping_return=0.;self._reserve_shaping_discounted_return=0.
        self._board_shaping_discounted_return=0.;self._board_shaping_steps=0;self._initial_board_potential=0.
        self._previous=self._metrics()
        self._initial_board_potential=self._previous['board_potential']
        self._previous['initial_board_potential']=self._initial_board_potential
        return self._observe(),self._info()

    def _mask(self):
        mask=mechanical_mask(self.match,self.cards,self.spells,1)
        reserved=sum(self.cards[e[1]].cost*1000 for e in self.pending)
        available=max(0,self.match.players[1].elixir-reserved)
        busy={e[2]for e in self.pending}
        for slot,card in enumerate(self.match.players[1].hand):
            if (slot in busy or self.match.elapsed_ms<self.slot_ready.get(slot,0)
                    or self.cards[card].cost*1000>available):
                mask[1+slot*576:1+(slot+1)*576]=False
        from sim.cannon_placement import restrict_cannon,CENTRAL_MODE
        mask=restrict_cannon(mask,self.match.players[1].hand,
            CENTRAL_MODE if self.profile['cannon_center_only']else 'unrestricted')
        if self.reflected:
            from sim.lane_reflection import reflect_actions
            mask=reflect_actions(mask)
        return mask

    def _observe(self):
        obs=policy_observation(self.match,1,JOINT_MODE)
        reserved=sum(self.cards[e[1]].cost for e in self.pending)
        obs['scalars']=obs['scalars'].copy()
        obs['scalars'][0]=max(0,obs['scalars'][0]-reserved/10)
        if self.reflected:
            from sim.lane_reflection import reflect_observation
            obs=reflect_observation(obs)
        return obs

    def _enemy(self):
        now=self.match.elapsed_ms
        if now<self.next_enemy:return
        if self.scenario.style!='random' and self.scenario.style!='heldout_random':
            self.next_enemy=now+500
            before=self.match.players[-1].elixir
            result=self.scripted.act(self.match)
            if result:
                card=result[0]
                self.enemy_actions.append(dict(at_ms=now,card=card,elixir_before=before,
                    elixir_after=self.match.players[-1].elixir))
            return
        # No arbitrary extra cards: actual four-card hand/cycle and mana.
        affordable=[c for c in self.match.players[-1].hand if self.match.can_play(-1,c)]
        self.next_enemy=now+self.rng.choice((500,750,1000,1500))
        if self.scenario.task=='attack':self.next_enemy+=500
        if not affordable:return
        mana=self.match.players[-1].elixir
        if self.scenario.style!='random' and mana<7000 and self.rng.random()<.45:return
        card=self.rng.choice(affordable)
        slot=self.match.players[-1].hand.index(card)
        legal=mechanical_mask(self.match,self.cards,self.spells,-1)[1+slot*576:1+(slot+1)*576].reshape(32,18)
        # Units strictly on the enemy's half; spells may target our units.
        if card not in self.spells:
            legal=legal.copy();legal[:16]=False
            if self.scenario.style!='random':
                xs=(2,3,4,13,14,15); preferred=[(y,x) for y,x in np.argwhere(legal) if x in xs and y in (17,18,24,27)]
            else:preferred=[]
        else:preferred=[]
        positions=preferred or list(map(tuple,np.argwhere(legal)))
        if not positions:return
        y,x=self.rng.choice(positions);before=self.match.players[-1].elixir
        if self.match.play_card(-1,card,grid_to_point(int(x),int(y),-1)):
            self.enemy_actions.append(dict(at_ms=now,card=card,elixir_before=before,elixir_after=self.match.players[-1].elixir))

    def step(self,action):
        if self.match is None or self.done:raise RuntimeError('Reset required')
        before=self._previous
        mask=self._mask()
        if not isinstance(action,(int,np.integer)) or not 0<=action<ACTIONS or not mask[int(action)]:self.illegal+=1
        elif action:
            slot,x,y=ClashEnv.decode(int(action))
            if self.reflected:x=17-x
            self.pending.append((self.match.elapsed_ms+self.action_delay_ms,self.match.players[1].hand[slot],slot,x,y))
        target=self.match.elapsed_ms+self.profile['decision_interval_ms']
        while not self.match.finished and self.match.elapsed_ms<min(target,self.max_ms):
            self._enemy()
            due=[e for e in self.pending if self.match.elapsed_ms>=e[0]]
            for command in due:
                _,card,slot,x,y=command
                accepted=self.match.play_card(1,card,grid_to_point(x,y,1))
                self.illegal+=not accepted
                if accepted:self.actions.append(dict(at_ms=self.match.elapsed_ms,card=card,x=x,y=y))
                self.slot_ready[slot]=self.match.elapsed_ms+self.profile['action_cooldown_ms']
                self.pending.remove(command)
            self.match.step(min(50,self.max_ms-self.match.elapsed_ms))
        self.done=self.match.finished or self.match.elapsed_ms>=self.max_ms
        self.terminal_reason='game_'+self.match.result if self.match.finished else 'technical_timeout' if self.done else 'running'
        after=self._metrics();p=self.profile
        reserve_reward=0.
        if p['reserve_potential_weight']:
            from sim.board_potential import shaping
            reserve_reward=p['reserve_potential_weight']*shaping(before['reserve_potential'],after['reserve_potential'],
                p['shaping_gamma'],terminal=self.done)
        self._reserve_shaping_return+=reserve_reward
        self._reserve_shaping_discounted_return+=p['shaping_gamma']**self._board_shaping_steps*reserve_reward
        board_reward=0.
        if p['board_potential_weight']:
            from sim.board_potential import shaping
            board_reward=p['board_potential_weight']*shaping(before['board_potential'],after['board_potential'],
                p['shaping_gamma'],terminal=self.match.finished)
        self._board_shaping_return+=board_reward
        if p['board_potential_weight']:
            self._board_shaping_discounted_return+=p['shaping_gamma']**self._board_shaping_steps*board_reward
        self._board_shaping_steps+=1
        after['board_shaping_return']=self._board_shaping_return
        after['board_shaping_discounted_return']=self._board_shaping_discounted_return
        after['board_shaping_steps']=self._board_shaping_steps
        reward=(p['enemy_hp_reward']*(after['enemy_fraction_damage']-before['enemy_fraction_damage'])
            -p['own_hp_penalty']*(after['own_fraction_loss']-before['own_fraction_loss'])
            -p['overflow_penalty']*(after['elixir_overflow']-before['elixir_overflow'])
            -p['missed_fireball_penalty']*(after['missed_fireball_elixir']-before['missed_fireball_elixir'])
            +p['fireball_value_reward']*(after['fireball_exchange_value']-before['fireball_exchange_value'])
            -.1*(after['illegal_actions']-before['illegal_actions'])+board_reward+reserve_reward)
        if self.done:
            reward+=p['win_reward'] if after['outcome']=='win' else -p['loss_penalty'] if after['outcome']=='loss' else 0
        self._previous=after
        return self._observe(),float(reward),bool(self.match.finished),bool(self.done and not self.match.finished),self._info()

    def _metrics(self):
        from sim.tactical_rewards import fireball_outcomes,reserve_potential
        own={k:t.hp_fraction for k,t in self.match.towers[1].items()}
        enemy={k:t.hp_fraction for k,t in self.match.towers[-1].items()}
        # Charge only instrumented, resolved Fireballs with NO recorded hits.
        # A shield/zero-damage hit is retained; instant Zap lacks this log.
        misses=sum(e.get('caster_side')==1 and e.get('spell')=='fireball'
            and e.get('resolved',False) and not e.get('hits') for e in self.match.battle.spell_cast_log)
        potential=0.
        if self.profile['board_potential_weight']:
            from sim.board_potential import public_board_potential
            if not self.match.finished:potential=public_board_potential(self.match,self._board_values)
        return dict(scenario_id=self.scenario.scenario_id,family_id=self.scenario.family,
            normal_range_mode=self.match.battle.normal_range_mode,
            target_lock_mode=self.match.battle.target_lock_mode,
            hog_sight_mode=self.match.battle.hog_sight_mode,
            cannon_center_only=self.profile['cannon_center_only'],
            action_cooldown_ms=self.profile['action_cooldown_ms'],
            cannon_outside_center=sum(a['card']=='cannon'and not(6<=a['x']<=11 and 18<=a['y']<=23)for a in self.actions),
            episode_seed=self.seed,match_seed=self.match.seed,
            split=self.split,task=self.scenario.task,observation_contract=CONTRACT,
            terminal_reason=self.terminal_reason,elapsed_ms=self.match.elapsed_ms,
            outcome=outcome(self.match.result,1,self.match.finished),
            own_fraction_loss=sum(1-v for v in own.values()),enemy_fraction_damage=sum(1-v for v in enemy.values()),
            own_tower_fraction_loss={k:1-v for k,v in own.items()},
            own_towers_alive={k:t.alive for k,t in self.match.towers[1].items()},
            no_tower_damage=all(v==1 for v in own.values()),
            elixir_overflow=self.match.players[1].natural_elixir_overflow_milli/1000,
            elixir_spent=self.match.players[1].elixir_spent,illegal_actions=self.illegal,
            missed_fireballs=misses,missed_fireball_elixir=misses*self.cards['fireball'].cost,
            **fireball_outcomes(self.match.battle.spell_cast_log),reserve_potential=reserve_potential(self.match),
            reserve_shaping_return=self._reserve_shaping_return,
            reserve_shaping_discounted_return=self._reserve_shaping_discounted_return,
            board_potential=potential,board_shaping_return=self._board_shaping_return,
            initial_board_potential=self._initial_board_potential,
            board_shaping_discounted_return=self._board_shaping_discounted_return,
            board_shaping_steps=self._board_shaping_steps,
            accepted_card_counts=dict(Counter(e['card']for e in self.actions)),
            accepted_card_lane_counts={lane:dict(Counter(e['card']for e in self.actions
                if (e['x']<9)==(lane=='left')))for lane in('left','right')},
            enemy_accepted_card_counts=dict(Counter(e['card']for e in self.enemy_actions)),
            coverage_flags=['simulator_mechanics_not_fully_validated','synthetic_legal_opponent_not_live_rating'])

    def _info(self):return dict(action_mask=self._mask(),metrics=self._metrics(),terminal_reason=self.terminal_reason)
    def snapshot(self):return deepcopy(self.__dict__)
    def restore(self,state):
        if state['profile']!=self.profile:raise ValueError('Continuous profile differs')
        self.__dict__=deepcopy(state)
