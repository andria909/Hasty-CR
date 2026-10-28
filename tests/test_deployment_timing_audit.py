import pytest
from scripts.gamestate.audit_deployment_timing import collect


def start(match=1, host=100):
    return dict(event='MATCH_START', match=match, host_ns=host, side=1, tick=0)


def action(match=1, tick=10, card='skeletons'):
    return [dict(event='PLAY', match=match, tick=tick, card=card),
            dict(event='PLAY_ACK', match=match, source_tick=tick, tick=tick+27, card=card)]


def frame(tick, ids=(7, 8, 9), x=1000, host=None, side=1):
    return dict(received_host_ns=host or 100+tick,
        frame=dict(own_state=dict(tick=tick, own_side=1, world='world')),
        units=[dict(entity_id=i, name='Skeleton', side=side, x=x+i, y=1000,
                    deployment_tick_candidate=35, play_sequence_candidate=1) for i in ids])


def test_summon_is_one_placement_and_ack_is_not_readiness():
    result = collect([start(), *action()], [frame(37), frame(54), frame(56, x=1100)])
    assert result['matched_groups'] == 1
    row = result['rows'][0]
    assert row['body_count'] == 3
    assert row['input_to_candidate_ms'] == 1250
    assert row['input_to_ack_ms'] == 1350
    assert row['candidate_to_first_movement_ms'] == 1050


def test_reused_world_entity_and_sequence_are_isolated_by_match():
    result = collect([start(), *action(), start(2, 200), *action(2)],
                     [frame(37), frame(56, x=1100), frame(37, host=237), frame(56, host=256, x=1200)])
    assert [r['match'] for r in result['rows']] == [1, 2]
    assert all(r['body_count'] == 3 for r in result['rows'])


def test_unconfirmed_enemy_ambiguous_and_prestart_not_accepted():
    assert collect([start(), action()[0]], [frame(37)])['rows'] == []
    assert collect([start(), *action()], [frame(37, side=0)])['rows'] == []
    assert collect([start(), *action()], [frame(37, host=99)])['rows'] == []
    assert collect([start(), *action(), *action(tick=11)], [frame(37)])['rows'] == []
    with pytest.raises(ValueError):
        collect([start(), action()[0], *action()], [frame(37)])


def test_stationary_or_duplicate_tick_not_claimed_ready():
    result = collect([start(), *action()], [frame(37), frame(37, x=2000), frame(40)])
    assert result['rows'][0]['first_movement_tick'] is None
