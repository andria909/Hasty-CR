from types import SimpleNamespace

from scripts.gamestate.audit_early_displacement import contacts
from sim.arena import Point, distance
from sim.engine import Battle
from sim.entities import make_unit
from sim.gamedata import load_gamedata


def test_deploying_body_can_be_displaced_without_becoming_ready():
    cards=load_gamedata(level=11)
    battle=Battle()
    cannon=battle.add(make_unit(0,cards['cannon'].unit,1,Point(9000,20000)))
    skeleton=battle.add(make_unit(0,cards['skeletons'].unit,1,Point(9000,20000)))
    initial=skeleton.pos
    assert not skeleton.active
    battle.step(50)
    assert distance(initial,skeleton.pos)>=100
    assert not skeleton.active and skeleton.deploy_remaining_ms==950
    assert not battle.damage_log
    assert skeleton.target_uid is None
    assert cannon.pos==initial


def test_contact_context_excludes_air_unknown_dead_and_opponent_spawn_identity():
    profiles={name:SimpleNamespace(flying=name=='bat',collision_radius_mt=300)
              for name in ['skeleton','knight','bat']}
    def body(uid,name='Skeleton',side=1,hp=10,x=1000):
        return dict(entity_id=uid,name=name,side=side,hp=hp,x=x,y=1000,
                    play_sequence_candidate=5,deployment_tick_candidate=30)
    source=body(1)
    result=contacts(source,[source,body(2),body(3,side=0),body(4,'Bat'),
        body(5,'Unknown'),body(6,hp=0),body(7,x=1900)],profiles)
    assert [r['entity_id'] for r in result]==[2,3]
    assert result[0]['same_spawn'] and not result[1]['same_spawn']
    assert result[0]['sim_clearance_mt']==-600
