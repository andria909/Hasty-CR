"""Select a broad warm start and2 full-run profiles using validation only."""
from collections import defaultdict

from sim.continuous_validation import score_families


def validation_score(report):
    records=report['records']
    if len(records)!=72:raise ValueError('Exactly72 balanced validation games required')
    if (any(type(r.get('episode_seed'))is not int or type(r.get('match_seed'))is not int for r in records)or
            len({r['episode_seed']for r in records})!=72 or len({r['match_seed']for r in records})!=72):
        raise ValueError('Every validation game requires a distinct explicit seed')
    result=score_families(records)
    if any(r['games']!=12 for r in result['by_family'].values()):raise ValueError('Unequal family coverage')
    return result


def select_initial(references):
    if not references or 'parent'not in {r['id']for r in references}:raise ValueError('Original parent reference required')
    ranked=[dict(id=r['id'],**validation_score(r['validation']))for r in references]
    ranked.sort(key=lambda r:(-r['selection_score'],r['id']!='parent',r['id']))
    return dict(selected=ranked[0]['id'],ranking=ranked,test_used_for_selection=False)


def select_profiles(runs,seeds=(62,63),floor=.1):
    groups=defaultdict(list)
    for r in runs:
        if not r.get('target_met')or r.get('step')!=102400:raise ValueError('Completed pilot budget required')
        groups[r['trial']['id']].append(r)
    ranking=[]
    for name,rows in groups.items():
        if len(rows)!=len(seeds)or {r['seed']for r in rows}!=set(seeds):raise ValueError('Every independent pilot seed required')
        if any(r['trial']!=rows[0]['trial']for r in rows):raise ValueError('Changed trial profile')
        scores=[validation_score(r['validation'])for r in rows]
        ranking.append(dict(trial=rows[0]['trial'],score=sum(r['selection_score']for r in scores)/len(scores),
            minimum_family_points=min(r['worst_archetype_points']for r in scores),
            eligible=all(r['worst_archetype_points']>=floor for r in scores)))
    if len(ranking)!=4:raise ValueError('All four pilot configurations required')
    ranking.sort(key=lambda r:(-r['score'],r['trial']['entropy_coef'],r['trial']['focus']!='uniform'))
    eligible=[r for r in ranking if r['eligible']]
    return dict(ranking=ranking,selected=eligible[:2]if len(eligible)>=2 else [],
                gate_met=len(eligible)>=2,floor=floor,test_used_for_selection=False)
