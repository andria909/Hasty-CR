from copy import deepcopy

import pytest

from scripts.attack.verify_native_fireball_evaluation import verify_skills


def king_report():
    rows=[dict(kind='king_finish',lane=('left','right')[(i//3)%2],seed=2000000+i,
        success=True,reason='enemy_king_destroyed',terminal_damage_events_v2=[dict(source_kind='spell',
            source_name='fireball',source_side=1,target_side=-1,killed=True,damage=100,before_hp=100,after_hp=0)])for i in range(6)]
    return dict(rows=rows,episodes=6,successes=6,by_kind={'king_finish':dict(episodes=6,successes=6)})


def test_king_success_requires_own_fireball_terminal_damage():
    report=king_report()
    assert verify_skills(report,2000000,6,king=True)['king_finish']['successes']==6
    for field,value in (('source_name','zap'),('source_side',-1),('killed',False),('damage',0)):
        changed=deepcopy(report)
        changed['rows'][0]['terminal_damage_events_v2'][0][field]=value
        with pytest.raises(ValueError,match='observed Fireball'):verify_skills(changed,2000000,6,king=True)


def test_duplicate_seed_cannot_hide_missing_context():
    report=king_report();report['rows'][1]=deepcopy(report['rows'][0])
    with pytest.raises(ValueError,match='schedule'):verify_skills(report,2000000,6,king=True)


def test_success_aggregate_is_recomputed():
    report=king_report();report['by_kind']['king_finish']['successes']=5
    with pytest.raises(ValueError,match='aggregate'):verify_skills(report,2000000,6,king=True)
