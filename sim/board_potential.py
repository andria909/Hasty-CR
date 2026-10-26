"""Bounded public board estimate for potential-based auxiliary rewards.

This is an approximate value, not an assertion about card exchange rates.
It never reads the enemy hand, queue, bank, targeting state or future events.
"""
import math


def unit_values(cards,preferred_cards=()):
    values={}
    for card in cards.values():
        unit=getattr(card,'unit',None)
        if unit is None or card.cost<=0 or getattr(card,'secondary_unit',None)or getattr(card,'additional_summons',()):continue
        value=card.cost/max(1,card.summon_number)
        values[unit.name]=min(values.get(unit.name,value),value)
    # Our known deck provides replacement prices, not hidden opponent costs.
    # In particular, a single Musketeer must not inherit Three Musketeers'
    # bulk discount and receive a placement penalty just for being deployed.
    for name in preferred_cards:
        card=cards[name];unit=getattr(card,'unit',None)
        if unit is not None and not getattr(card,'secondary_unit',None)and not getattr(card,'additional_summons',()):
            values[unit.name]=card.cost/max(1,card.summon_number)
    return values


def public_board_potential(match,values,side=1,clip=20.):
    if not math.isfinite(clip)or clip<=0:raise ValueError('Finite positive potential bound required')
    bank=match.players[side].elixir/1000
    if not math.isfinite(bank):raise ValueError('Nonfinite own bank')
    result=max(0.,min(10.,bank))
    for entity in match.battle.entities.values():
        if not entity.alive or entity.is_tower:continue
        maximum=max(1,entity.max_hitpoints);hp=entity.hitpoints/maximum
        value=values.get(entity.name,min(1.,maximum/1000))
        if not math.isfinite(hp)or not math.isfinite(value):raise ValueError('Nonfinite public board value')
        result+=(1 if entity.side==side else -1)*value*max(0.,min(1.,hp))
    return max(-clip,min(clip,result))


def shaping(previous,current,gamma,terminal=False):
    if not all(math.isfinite(x)for x in(previous,current,gamma))or not 0<=gamma<=1:
        raise ValueError('Finite potential and valid discount required')
    return gamma*(0. if terminal else current)-previous
