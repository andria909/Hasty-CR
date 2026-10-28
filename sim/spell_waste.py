"""Opt-in paid Fireball/Zap outcome accounting without changing combat code."""
from collections import Counter, defaultdict
from sim.tower_support import ObservedDamageLog


class SpellWasteTracker:
    def __init__(self, match):
        self.match=match
        self.receipts=[]

    def attach(self):
        if isinstance(self.match.play_log,ObservedDamageLog):raise ValueError('Play log already observed')
        spec=self.match.spells['zap']
        if spec.impact_delay_ms or spec.travels_from_king or spec.pulse_events:
            raise ValueError('Only current instantaneous base Zap supported')
        self.match.play_log=ObservedDamageLog(self.match.play_log,self.record)
        return self

    def record(self, play):
        name=play['resolved_card']
        if name not in ('zap','fireball'):return
        battle=self.match.battle
        self.receipts.append(dict(spell=name,caster_side=play['side'],cast_ms=battle.now_ms,cost=play['cost']))
        if name=='zap':
            # play_log is appended after payment and before cast_spell. Its
            # existing apply_spell primitive then records actual contacts and
            # marks this event resolved, including shields/zero-body-HP hits.
            from sim.arena import Point
            from sim.spells import reflect_target
            at=Point(play['x'],play['y']);side=play['side']
            bounced=reflect_target(battle,self.match.spells[name],at,side)
            if bounced is not None:at,side=bounced,-side
            battle.spell_cast_log.append(dict(spell=name,caster_side=play['side'],side=side,
                cast_ms=battle.now_ms,due_ms=battle.now_ms,aim=(at.x,at.y),hits=[],resolved=False))

    def metrics(self,side=1):
        receipts=defaultdict(list)
        for receipt in self.receipts:
            receipts[(receipt['spell'],receipt['caster_side'],receipt['cast_ms'])].append(receipt)
        empty=Counter();resolved=Counter();paid=Counter();unresolved=Counter()
        for event in self.match.battle.spell_cast_log:
            name=event['spell']
            if name not in ('fireball','zap') or event.get('caster_side')!=side:continue
            key=(name,side,event['cast_ms'])
            if not receipts[key]:raise ValueError('Spell outcome lacks a matching paid receipt')
            receipt=receipts[key].pop(0)
            if not event.get('resolved'):
                unresolved[name]+=1;continue
            resolved[name]+=1
            # Any actual target contact is conservatively not an empty cast.
            # This does not claim the exchange was good or the stun necessary.
            if not event['hits']:
                empty[name]+=1;paid[name]+=receipt['cost']
        return dict(empty_spell_counts=dict(empty),resolved_spell_counts=dict(resolved),
                    empty_spell_elixir_by_card=dict(paid),empty_spell_elixir=sum(paid.values()),
                    unresolved_spell_counts=dict(unresolved))
