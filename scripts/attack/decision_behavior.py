"""Distinguish a policy's choice to wait from having no legal card action."""
import math

CARDS = ('cannon', 'electro_spirit', 'fireball', 'hog_rider', 'knight',
         'musketeer', 'skeletons', 'zap')


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def economy(records):
    known = [r['decision_economy'] for r in records if 'decision_economy' in r]
    result = dict(known_games=len(known), total_games=len(records), complete=len(known)==len(records))
    if not result['complete']:
        return result
    keys = ('decisions','forced_wait','voluntary_wait','plays','invalid','available_below4')
    totals = dict.fromkeys(keys, 0)
    cards = {}
    for row in known:
        if any(type(row[k]) is not int or row[k] < 0 for k in keys):
            raise ValueError('Nonnegative integer decision counters required')
        if (row['decisions'] != sum(row[k] for k in ('forced_wait','voluntary_wait','plays','invalid'))
                or row['available_below4'] > row['decisions']):
            raise ValueError('Decision counts do not partition observed steps')
        if sum(c['selected'] for c in row['cards'].values()) != row['plays']:
            raise ValueError('Selected card requests do not match play decisions')
        for key in keys:
            totals[key] += row[key]
        for card, counts in row['cards'].items():
            if (any(type(counts[k]) is not int for k in ('in_hand','legal','selected')) or
                    not 0 <= counts['selected'] <= counts['legal'] <= counts['in_hand'] <= row['decisions']):
                raise ValueError('Card opportunity counts are inconsistent')
            target = cards.setdefault(card, dict(in_hand=0, legal=0, selected=0))
            for key in target:
                target[key] += counts[key]
    choice = totals['plays'] + totals['voluntary_wait']
    result.update(totals=totals, valid_choice_decisions=choice,
        forced_wait_fraction=ratio(totals['forced_wait'],totals['decisions']),
        voluntary_wait_given_choice=ratio(totals['voluntary_wait'],choice),
        available_below4_fraction_all_decisions=ratio(totals['available_below4'],totals['decisions']),
        cards={card:dict(counts, selected_given_legal=ratio(counts['selected'],counts['legal']))
               for card,counts in sorted(cards.items())})
    return result


def card_usage(records):
    minutes = sum(r['elapsed_ms'] for r in records)/60000
    if not records or not math.isfinite(minutes) or minutes <= 0:
        raise ValueError('Positive observed match duration required')
    names = sorted(set(CARDS) | {c for r in records for c in r['accepted_card_counts']})
    return {card:dict(per_game=sum(r['accepted_card_counts'].get(card,0) for r in records)/len(records),
                      per_minute=sum(r['accepted_card_counts'].get(card,0) for r in records)/minutes)
            for card in names}


def compare(before, after):
    return dict(before=dict(economy=economy(before),cards=card_usage(before)),
                after=dict(economy=economy(after),cards=card_usage(after)))
