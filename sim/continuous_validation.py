"""Broad full-game validation score, independent of shaping coefficients."""
from collections import defaultdict

FAMILIES=('air','bridge','giant','golem','hog','swarm')


def score_families(records):
    groups=defaultdict(list)
    for r in records:
        if r.get('split')!='validation':raise ValueError('Family selector accepts validation only')
        if r.get('outcome')not in ('win','loss','draw'):raise ValueError('Known terminal outcomes required')
        groups[r['family_id']].append(r)
    if set(groups)!=set(FAMILIES):raise ValueError('Every validation archetype is required')
    rates={name:sum(r['outcome']=='win' or .5*(r['outcome']=='draw')for r in rows)/len(rows)
           for name,rows in groups.items()}
    macro=sum(rates.values())/len(rates);worst=min(rates.values())
    return dict(selection_metric='half_macro_half_worst_archetype_points_v1',
        selection_score=.5*macro+.5*worst,macro_archetype_points=macro,worst_archetype_points=worst,
        by_family={name:dict(games=len(groups[name]),points=rates[name])for name in FAMILIES})
