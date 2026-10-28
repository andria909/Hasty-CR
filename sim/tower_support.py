"""Optional measurement of crown-tower damage while a defender draws aggro.

No physics, policy inputs or rewards are changed by this module. Attach only
for a diagnostic or a future explicitly enabled training ablation.
"""
import math


class TowerSupport:
    def __init__(self, battle, side=1):
        if side not in (-1,1):raise ValueError('Simulator side required')
        self.battle=battle;self.side=side
        self.total_damage=0.;self.assisted_damage=0.;self.assisted_value=0.
        self.total_paid_value=0.;self.total_credited={}
        self.credited={};self.events=[]

    def record(self, event):
        stamp,source_id,target_id,damage=event
        if not isinstance(damage,(int,float)) or not math.isfinite(damage) or damage<0:
            raise ValueError('Finite nonnegative effective damage required')
        source=self.battle.entities.get(source_id);target=self.battle.entities.get(target_id)
        if not source or not target or not source.is_tower or source.side!=self.side or target.side!=-self.side or target.is_tower:return
        self.total_damage+=damage
        value=max(0,target.elixir_value)/1000
        all_credit=min(max(0.,value-self.total_credited.get(target_id,0.)),value*damage/max(1,target.max_hitpoints))
        self.total_credited[target_id]=self.total_credited.get(target_id,0.)+all_credit
        self.total_paid_value+=all_credit
        defender=self.battle.entities.get(target.target_uid)
        supported=bool(defender and defender.side==self.side and not defender.is_tower and defender.alive)
        credit=0.
        if supported:
            self.assisted_damage+=damage
            value=max(0,target.elixir_value)/1000
            # A healed body cannot yield more than its originally paid value.
            remaining=max(0.,value-self.credited.get(target_id,0.))
            credit=min(remaining,value*damage/max(1,target.max_hitpoints))
            self.credited[target_id]=self.credited.get(target_id,0.)+credit
            self.assisted_value+=credit
        self.events.append(dict(time_ms=stamp,tower=source_id,enemy=target_id,damage=damage,
            defender=defender.uid if supported else None,credited_value=credit,total_credited_value=all_credit))

    def attach(self):
        if isinstance(self.battle.damage_log,ObservedDamageLog):raise ValueError('Already instrumented')
        self.battle.damage_log=ObservedDamageLog(self.battle.damage_log,self.record)
        return self

    def metrics(self):
        return dict(tower_damage=self.total_damage,assisted_tower_damage=self.assisted_damage,
                    assisted_paid_value=self.assisted_value,total_paid_value=self.total_paid_value,events=len(self.events))


class ObservedDamageLog(list):
    def __init__(self, existing, callback):
        super().__init__(existing);self.callback=callback

    def append(self, event):
        super().append(event);self.callback(event)

    def __deepcopy__(self, memo):
        # Snapshotting must not replay old damage through the callback.
        from copy import deepcopy
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        copied.callback = deepcopy(self.callback, memo)
        for event in self:
            list.append(copied, deepcopy(event, memo))
        return copied
