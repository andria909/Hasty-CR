"""Evaluation-only terminal damage attribution; leaves rewards and engine logs intact."""
from contextlib import contextmanager
from threading import Lock
from unittest.mock import patch

from sim.defense_env import KingDestroyed,TerminalKing
from sim.engine import Battle
from sim import spells

_LOCK=Lock()


class TerminalDamageObserver:
    def __init__(self,environment):
        self.environment=environment
        self.events=[]

    def watches(self,battle):
        match=getattr(self.environment,'match',None)
        return match is not None and match.battle is battle

    def capture(self,battle,target,before,source):
        damage=max(0,before-target.hitpoints)
        if damage and before>0 and not target.alive:
            self.events.append(dict(source,hit_ms=battle.now_ms,target_uid=target.uid,
                target_side=target.side,before_hp=before,after_hp=target.hitpoints,damage=damage,killed=True))

    def enrich(self,record):
        result=dict(record)
        result['terminal_damage_events_v2']=list(self.events)
        base=getattr(self.environment,'_active',None) or self.environment
        if record.get('task','attack')!='attack' or not hasattr(base,'_goal_uids'):return result
        start=getattr(base.scenario,'opportunity_start_ms',0)
        relevant=[e for e in self.events if e['target_uid'] in base._goal_uids and e['hit_ms']>=start and e['target_side']==-1]
        own=[e for e in relevant if e['source_side']==1]
        recovered_hog=sum(e['damage'] for e in own if e['source_kind']=='unit' and e['source_name']=='hog_rider')
        result['hog_target_tower_damage_v2']=record.get('hog_target_tower_damage',0)+recovered_hog
        result['hog_damage_ledger_complete_v2']=bool(record.get('hog_damage_ledger_complete') or self.events)
        finish=record.get('successful_spell_finish')
        if any(e['source_kind']=='spell' and e['source_name'] in ('fireball','zap') for e in own):finish=True
        elif finish is None and relevant:finish=False
        result['successful_spell_finish_v2']=finish
        result['terminal_attribution_method_v2']='observed_hp_delta_during_original_exception_unwind'
        return result


@contextmanager
def observe_terminal_damage(environment):
    """One serial episode per context; re-raise the original terminal exception."""
    if not _LOCK.acquire(blocking=False):raise RuntimeError('Terminal attribution contexts cannot overlap')
    observer=TerminalDamageObserver(environment)
    original_spell=spells.apply_spell;original_land=Battle._land
    def spell(battle,spec,at,side):
        targets=[(e,e.hitpoints) for e in battle.entities.values() if isinstance(e,TerminalKing)] if observer.watches(battle) else []
        try:return original_spell(battle,spec,at,side)
        except KingDestroyed:
            for target,before in targets:
                observer.capture(battle,target,before,dict(source_kind='spell',source_name=spec.name,source_side=side))
            raise
    def land(battle,entity,target,*args,**kwargs):
        before=target.hitpoints if observer.watches(battle) and isinstance(target,TerminalKing) else None
        try:return original_land(battle,entity,target,*args,**kwargs)
        except KingDestroyed:
            if before is not None:
                observer.capture(battle,target,before,dict(source_kind='unit',source_uid=entity.uid,source_name=entity.name,source_side=entity.side))
            raise
    try:
        with patch.object(spells,'apply_spell',spell),patch.object(Battle,'_land',land):yield observer
    finally:_LOCK.release()
