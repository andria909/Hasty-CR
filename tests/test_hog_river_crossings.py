from copy import deepcopy

import pytest

from scripts.gamestate.audit_hog_river_crossings import crossings


def rows():
    result=[]
    for i in range(9):
        king=dict(entity_id=1,address='king',name='KingTower',side=0,x=9000,y=3000,hp=3000)
        hog=dict(entity_id=2,address='hog',name='HogRider',side=0,x=7500,y=14000+500*i,hp=1000)
        result.append(dict(status='ok',own_state=dict(status='ok',tick=i*5,world='w'),
                           frame=dict(battle_tick=i*5,entities=[king,hog])))
    return result


def test_native_tick_positions_and_direction_are_preserved():
    source=rows();one=crossings(source)
    assert len(one)==1 and len(one[0]['points'])==9
    assert one[0]['points'][0]==dict(tick=0,x=7500,y=14000,hp=1000,frame=0)
    for row in source:
        row['frame']['entities'][1]['y']=32000-row['frame']['entities'][1]['y']
        row['frame']['entities'][1]['side']=1
    reflected=crossings(source)
    assert len(reflected)==1 and reflected[0]['points'][0]['y']==18000


@pytest.mark.parametrize('reason',['clock','world','gap','identity','missing','conflicting_tick'])
def test_no_crossing_fabricated_across_invalid_middle(reason):
    source=rows()
    if reason=='clock':source[4]['own_state']['tick']+=1
    elif reason=='world':
        for row in source[4:]:row['own_state']['world']='w2'
    elif reason=='gap':
        for row in source[4:]:
            row['own_state']['tick']+=10;row['frame']['battle_tick']+=10
    elif reason=='identity':
        for row in source[4:]:row['frame']['entities'][1]['address']='reused'
    elif reason=='missing':source[4]['frame']['entities'].pop()
    else:
        duplicate=deepcopy(source[4]);duplicate['frame']['entities'][1]['x']+=1
        source.insert(5,duplicate)
    assert crossings(source)==[]
