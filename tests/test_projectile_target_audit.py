from scripts.gamestate.audit_projectile_targets import (
    isolated_impact, predict_fresh_target, unique_shooter)
from scripts.gamestate.audit_sim_alignment import unit_profiles
import json


def body(uid, name, side, x, y, hp=500):
    return dict(entity_id=uid, address=hex(uid * 100), name=name,
                side=side, x=x, y=y, hp=hp, max_hp=500)


def test_ambiguous_shooters_are_excluded_instead_of_nearest_guessed():
    shot = body(4000001, 'MusketeerProjectile', 1, 6000, 10000)
    a = body(5000001, 'Musketeer', 1, 6000, 10500)
    b = body(5000002, 'Musketeer', 1, 6200, 10800)
    assert unique_shooter(shot, {a['entity_id']: a}, {'Musketeer'}) == a
    assert unique_shooter(shot, {x['entity_id']: x for x in (a, b)}, {'Musketeer'}) is None


def test_isolated_impact_needs_unique_victim_and_actual_hp_drop():
    shot = body(4000001, 'MusketeerProjectile', 1, 6000, 10000)
    victim = body(5000001, 'Knight', 0, 6100, 10000)
    before = {victim['entity_id']: victim}
    after = {victim['entity_id']: dict(victim, hp=400)}
    assert isolated_impact(shot, before, after) == after[victim['entity_id']]
    assert isolated_impact(shot, before, before) is None
    assert isolated_impact(shot, before, {}) is None  # Missing body is not an attributed kill.
    neighbour = body(5000002, 'Knight', 0, 6200, 10000)
    assert isolated_impact(shot, {**before, neighbour['entity_id']: neighbour}, after) is None


def test_prediction_preserves_native_ids_and_exposes_radius_hypothesis():
    source = body(5000007, 'Musketeer', 1, 6000, 18000)
    # Existing reach is 6 + target radius .5 = 6.5. Source radius adds .5.
    target = body(5000012, 'Knight', 0, 6000, 11200)
    profiles = unit_profiles()
    ordinary, _ = predict_fresh_target([source, target], source['entity_id'], profiles)
    alternative, _ = predict_fresh_target([source, target], source['entity_id'], profiles,
                                         include_source_radius=True)
    assert ordinary is None
    assert alternative == target['entity_id']
    # Audit hypothesis never modifies the source profiles/production engine.
    assert profiles['musketeer'].range_mt == 6000


def test_repeated_projectile_ids_in_new_battle_do_not_replace_previous_lock(tmp_path):
    from scripts.gamestate.audit_projectile_targets import audit
    rows=[]
    for battle_index in range(2):
        king=body(5000000,'KingTower',1,9000,30000)
        king['address']=hex(900000+battle_index)
        source=body(5000007,'Musketeer',1,6000,18000)
        victim=body(5000012,'Knight',0,6000,12000)
        for frame in range(6):
            units=[king,source,dict(victim,hp=500-100*(frame>=2)-100*(frame>=5))]
            if frame%3<2:
                uid=4000001+frame//3
                units.append(body(uid,'MusketeerProjectile',1,6000,17500 if frame%3==0 else 12500))
            # Use the newer host-wrapped format as well as repeated entity IDs.
            rows.append(dict(frame=dict(device_start_ns=(len(rows)+1)*50000000,entities=units)))
    path=tmp_path/'frames.jsonl';path.write_text(''.join(json.dumps(r)+'\n'for r in rows))
    report=audit(path)
    second_shots=[e for e in report['evidence']if e['previous_projectile_id']==4000001]
    assert len(second_shots)==2
    assert len({e['trace_segment']for e in second_shots})==2
    assert all(e.get('prior_inferred_target_id')==5000012 for e in second_shots)
