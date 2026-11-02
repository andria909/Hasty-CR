"""Fixed409600-step screening, preserving the previous family floor and ranking."""
from collections import defaultdict
from scripts.attack.select_continuous_focus import validation_score


def select_profiles(runs,seeds=(442,443),floor=.1):
    groups=defaultdict(list)
    for row in runs:
        if not row.get('target_met')or row.get('step')!=409600:
            raise ValueError('Completed409600-step pilot required')
        groups[row['trial']['id']].append(row)
    ranking=[]
    for rows in groups.values():
        if len(rows)!=len(seeds)or {r['seed']for r in rows}!=set(seeds):
            raise ValueError('Every independent pilot seed required')
        if any(r['trial']!=rows[0]['trial']for r in rows):raise ValueError('Changed trial profile')
        scores=[validation_score(r['validation'])for r in rows]
        ranking.append(dict(trial=rows[0]['trial'],score=sum(r['selection_score']for r in scores)/len(scores),
            minimum_family_points=min(r['worst_archetype_points']for r in scores),
            eligible=all(r['worst_archetype_points']>=floor for r in scores)))
    if len(ranking)!=4:raise ValueError('All four configurations required')
    ranking.sort(key=lambda r:(-r['score'],r['trial']['entropy_coef'],r['trial']['focus']!='uniform'))
    eligible=[r for r in ranking if r['eligible']]
    return dict(ranking=ranking,selected=eligible[:2]if len(eligible)>=2 else[],
                gate_met=len(eligible)>=2,floor=floor,test_used_for_selection=False)
