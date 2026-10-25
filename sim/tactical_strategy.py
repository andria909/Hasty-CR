"""Public-input tactical curriculum: deploy support, defend, then counterpush.

The former catalog stays available unchanged. New families require useful
Musketeer damage and actual Hog hits, never a bonus for merely playing a card.
"""
from dataclasses import dataclass,replace
import random

import numpy as np

from sim.attack_env import AttackEnv
from sim.attack_scenarios import AttackScenario,UnitSetup,family_split
from sim.strategy_env import StrategyEnv,StrategyScenario

CONTRACT='joint_public_presence_tactical_v1'


@dataclass(frozen=True)
class TacticalScenario(AttackScenario):
    tactical_goal: str='supported_hog'
    required_opening: tuple[str,...]=()


def catalog(geometry):
    result=[]
    groups=(('supported_hog',('knight','valkyrie','mini_pekka','musketeer','hunter','bomber')),
            ('air_defense_counterpush',('minions','baby_dragon','inferno_dragon','electro_dragon','mega_minion','balloon')),
            ('hog_defense_counterpush',('hog_rider','ram_rider','battle_ram','giant','royal_giant','goblin_giant')))
    for goal,opponents in groups:
        for index,opponent in enumerate(opponents):
            family='tactical:'+goal+':'+opponent
            split='train'if index<4 else 'validation'if index==4 else 'test'
            for lane,x in (('left',3500),('right',14500)):
                for mana in (6,8,10):
                    for y in (8500,12500,14500):
                        case=f'{family}/mana{mana}/y{y}/{lane}'
                        # Both lanes and all initial-card permutations of this
                        # matchup share a split. No future spawn labels input.
                        result.append(TacticalScenario(scenario_id=case,family_id=family,
                            split=split,goal_kind='hog_tower_damage',lane=lane,target_lane=lane,
                            initial_elixir=mana,enemy_initial_elixir=10,
                            initial_units=(UnitSetup(opponent,-1,x,y,1500),),
                            deadline_ms_range=(45000,80000),deadline_visibility='hidden',
                            geometry=geometry,setup_kind=goal,tactical_goal=goal,
                            required_opening=('cannon','hog_rider')if goal=='hog_defense_counterpush'else ('musketeer','hog_rider'),
                            paired_case_id=case,
                            limitations=('native_mechanics_not_fully_validated','conditional_opening_hand_curriculum')))
    return result


class TacticalAttackEnv(AttackEnv):
    def reset(self,*args,**kwargs):
        observation,info=super().reset(*args,**kwargs)
        self._tactical_damage=0
        self._support_cursor=0
        self._tactical_initial_enemy={e.uid for e in self.match.battle.entities.values()if e.side==-1 and not e.is_tower}
        self._enemy_unit_uids=set(self._tactical_initial_enemy)
        self._musketeer_uids=set()
        self._tactical_initial_hp=sum(t.hitpoints for t in self.match.towers[1].values())
        required=getattr(self.scenario,'required_opening',())
        if required:
            from sim.env import DECK_26
            # Vary slots, spare cards and cycle while making the immediate
            # response physically available in this introductory curriculum.
            rng=random.Random(self.seed+3700000)
            hand=list(required)+rng.sample([c for c in DECK_26 if c not in required],4-len(required))
            rng.shuffle(hand);queue=[c for c in DECK_26 if c not in hand];rng.shuffle(queue)
            player=self.match.players[1];player.hand=hand;player.queue=queue
            self._initial_hand=tuple(hand);self._initial_queue=tuple(queue)
        return self._observe(),self._info()

    def _capture_damage(self):
        super()._capture_damage()
        if not hasattr(self,'_support_cursor'):return
        self._enemy_unit_uids.update(e.uid for e in self.match.battle.entities.values()if e.side==-1 and not e.is_tower)
        self._musketeer_uids.update(e.uid for e in self.match.battle.entities.values()if e.side==1 and e.name=='musketeer')
        for at,source_id,target_id,amount in self.match.battle.damage_log[self._support_cursor:]:
            if source_id in self._musketeer_uids and target_id in self._enemy_unit_uids:
                self._tactical_damage+=max(0,amount)
        self._support_cursor=len(self.match.battle.damage_log)

    def _goal_satisfied(self):
        if self._hog_damage<=0:return False
        goal=getattr(self.scenario,'tactical_goal',None)
        if goal is None:return True
        if goal in ('supported_hog','air_defense_counterpush'):
            deployed=any(a['accepted']and a['card']=='musketeer'for a in self._actions)
            if not deployed or self._tactical_damage<=0:return False
        if goal.endswith('defense_counterpush'):
            if any(self.match.battle.entities.get(uid) is not None and self.match.battle.entities[uid].alive
                   for uid in self._tactical_initial_enemy):return False
        return True

    def _check_boundaries(self):
        if not getattr(self.scenario,'tactical_goal',None):return super()._check_boundaries()
        if not self.match.towers[1]['king'].alive:reason='own_king_destroyed'
        elif not self.match.towers[-1]['king'].alive:reason='enemy_king_destroyed'
        elif hasattr(self,'_tactical_damage')and self._goal_satisfied():reason='goal_success'
        elif self.match.elapsed_ms>=self._deadline_ms:reason='goal_window_missed'
        elif self.match.elapsed_ms>=self.max_ms:reason='technical_timeout'
        else:return
        self.terminal_reason=reason;self._done=True;self._pending=None

    def step(self,action):
        before_hp=sum(t.hitpoints for t in self.match.towers[1].values())
        before_damage=self._tactical_damage
        obs,reward,done,truncated,info=super().step(action)
        if getattr(self.scenario,'tactical_goal',None):
            lost=max(0,before_hp-sum(t.hitpoints for t in self.match.towers[1].values()))
            reward+=.003*(self._tactical_damage-before_damage)-.01*lost
        return obs,reward,done,truncated,self._info()

    def _metrics(self):
        result=super()._metrics()
        result.update(tactical_goal=getattr(self.scenario,'tactical_goal',None),
            musketeer_unit_damage=getattr(self,'_tactical_damage',0),
            musketeer_accepted_plays=sum(a['accepted']and a['card']=='musketeer'for a in self._actions),
            own_tower_hp_lost=max(0,getattr(self,'_tactical_initial_hp',0)-sum(t.hitpoints for t in self.match.towers[1].values())))
        return result


class TacticalStrategyEnv(StrategyEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        original=self._attack
        self._attack=TacticalAttackEnv(seed=self.seed,split=self.split,
            action_delay_ms=original.action_delay_ms,max_ms=original.max_ms,
            geometry=self.geometry.profile_id,coordinate_mode=self.coordinate_mode,
            initial_hand_mode=self.initial_hand_mode,visible_fraction=original.visible_fraction,
            rewards=self.attack_rewards)
        # Original exercises retain their already declared deadline scale.
        self._attack.scenarios=list(original.scenarios)+[s for s in catalog(self.geometry.profile_id)if s.split==self.split]
        self.scenarios=([StrategyScenario('attack',s)for s in self._attack.scenarios]+
                        [StrategyScenario('defense',s)for s in self._defense.scenarios])

    def _convert(self,observation,info):
        result,info=super()._convert(observation,info)
        from sim.eval_defense_duel import joint_attack_task
        result['scalars']=result['scalars'].copy()
        result['scalars'][-1]=float(joint_attack_task(self.match,1))
        info['metrics']['observation_contract']=CONTRACT
        info['metrics']['task_feature_rule']='public_enemy_presence_v1'
        return result,info
