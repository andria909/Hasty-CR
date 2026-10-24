"""Fireball practice on real simulator impacts, separate from frozen benchmarks.

Family splits are fixed before training. No existing held-out scenario is reused.
The policy keeps its joint309 input. Its task flag follows the public enemy
presence rule used by full-match deployment, not a privileged exercise label.
"""
from types import SimpleNamespace as NS
import random

from sim import arena
from sim.attack_env import AttackEnv
from sim.adapter import grid_to_point

CONTRACT = 'fireball_finish_and_splash_v1'
FAMILIES = {
    'train': ('musketeer', 'wizard', 'witch', 'cannon', 'tesla'),
    'validation': ('archer', 'bomb_tower'),
    'test': ('ice_wizard', 'inferno_tower'),
}


class FireballEnv(AttackEnv):
    def __init__(self, *, split='train', seed=42, kind='tower_finish', lane='left'):
        if split not in FAMILIES or kind not in ('tower_finish', 'building_finish', 'tower_unit'):
            raise ValueError('Unknown Fireball curriculum context')
        if lane not in ('left', 'right'):
            raise ValueError('Unknown Fireball lane')
        self.practice_kind=kind
        self.practice_split=split
        self.practice_lane=lane
        s=NS(scenario_id=CONTRACT, family_id=CONTRACT, split=split,
             goal_kind='spell_finish', lane=lane, target_lane=lane,
             initial_elixir=10, enemy_initial_elixir=10, hand_seed=seed,
             initial_hand=('fireball','zap','hog_rider','knight'),
             tower_hp=(), initial_units=(), enemy_plays=(),
             deadline_ms_range=(8000,12000), deadline_visibility='hidden',
             opportunity_start_ms=0, geometry='native_anchors_candidate_v1')
        super().__init__(scenario=s, split=split, seed=seed, action_delay_ms=150,
                         geometry=s.geometry, coordinate_mode='physical_live_cells_v2')

    def reset(self, seed=None, scenario_id=None):
        seed=self.seed+1 if seed is None else int(seed)
        rng=random.Random(seed)
        # Choose families only from this split, never from the historic test set.
        candidates=[n for n in FAMILIES[self.practice_split] if n in self._cards
                    and self._cards[n].unit is not None]
        buildings=[n for n in candidates if self._cards[n].unit.speed_mt_per_sec==0]
        troops=[n for n in candidates if self._cards[n].unit.speed_mt_per_sec>0]
        from sim.entities import make_unit
        from sim.env import DECK_26
        hand=['fireball']+rng.sample([n for n in DECK_26 if n!='fireball'],3)
        rng.shuffle(hand)
        self.scenarios[0].initial_hand=tuple(hand)
        self.scenarios[0].initial_elixir=rng.randint(4,10)
        self.practice_uids=[]
        super().reset(seed=seed, scenario_id=scenario_id)
        self._previous_elixir=self.match.players[1].elixir_spent
        target=self.match.towers[-1][self.practice_lane]
        self.practice_uids=[]
        if self.practice_kind=='tower_finish':
            target.hitpoints=rng.randint(1,self._spells['fireball'].damage_to(target))
            self.practice_uids=[target.uid]
        else:
            name=rng.choice(buildings if self.practice_kind=='building_finish' else troops)
            # Behind the tower on its own side; movement and flight use Match.
            x=target.pos.x+rng.choice((-500,0,500))
            y=target.pos.y-rng.choice((1000,1500,2000))
            unit=make_unit(0,self._cards[name].unit,-1,arena.Point(x,y))
            unit.deploy_remaining_ms=0
            if self.practice_kind=='building_finish':
                unit.hitpoints=rng.randint(1,self._spells['fireball'].damage_to(unit))
            self.match.battle.add(unit)
            self.practice_uids=[unit.uid]
            if self.practice_kind=='tower_unit':self.practice_uids.append(target.uid)
        self._practice_success=False
        self._previous_metrics=self._metrics()
        return self._observe(),self._info()

    def _check_boundaries(self):
        # During parent reset targets have not yet been assigned.
        if not getattr(self,'practice_uids',None):
            return
        self._practice_success=False
        for event in self.match.battle.spell_cast_log:
            if event.get('caster_side',event.get('side'))!=1 or event.get('spell')!='fireball':
                continue
            hits={h['target_uid']:h for h in event.get('hits',()) if h.get('damage',0)>0}
            if self.practice_kind=='tower_unit':
                success=all(uid in hits for uid in self.practice_uids)
            else:
                success=all(uid in hits and hits[uid].get('killed') for uid in self.practice_uids)
            if success:self._practice_success=True
        if not self.match.towers[1]['king'].alive:reason='own_king_destroyed'
        elif not self.match.towers[-1]['king'].alive:reason='enemy_king_destroyed'
        elif self._practice_success:reason='goal_success'
        elif self.match.elapsed_ms>=self._deadline_ms:reason='goal_window_missed'
        elif self.match.elapsed_ms>=self.max_ms:reason='technical_timeout'
        else:return
        self.terminal_reason=reason;self._done=True;self._pending=None

    def _observe(self):
        import numpy as np
        from sim.eval_defense_duel import joint_attack_task
        obs=super()._observe()
        obs['scalars']=np.concatenate((obs['scalars'],np.array(
            [float(joint_attack_task(self.match,1))],dtype=np.float32)))
        return obs

    def step(self,action):
        obs,_,terminated,truncated,info=super().step(action)
        # Only an observed, task-specific Fireball impact earns the task bonus.
        # Hog/other troops killing a king cannot substitute for this objective.
        cost=self.match.players[1].elixir_spent-self._previous_elixir if hasattr(self,'_previous_elixir') else self.match.players[1].elixir_spent
        self._previous_elixir=self.match.players[1].elixir_spent
        reward=-.02*cost
        if terminated or truncated:reward+=10 if self._practice_success else -5
        info['metrics'].update(fireball_contract=CONTRACT, fireball_task=self.practice_kind,
                               fireball_success=self._practice_success)
        return obs,reward,terminated,truncated,info

    def reference_action(self):
        """Observable geometric candidate; caller must verify actual impacts."""
        if 'fireball' not in self.match.players[1].hand:return 0
        targets=[self.match.battle.entities[u] for u in self.practice_uids]
        spec=self._spells['fireball']
        candidates=[]
        mask=self._action_mask()
        slot=self.match.players[1].hand.index('fireball')
        for y in range(32):
            for x in range(18):
                a=1+slot*576+y*18+x
                if not mask[a]:continue
                point=grid_to_point(x,y,1)
                if all(self._spell_covers(spec,point,t) for t in targets):
                    candidates.append((sum((t.pos.x-point.x)**2+(t.pos.y-point.y)**2 for t in targets),a))
        return min(candidates)[1] if candidates else 0
