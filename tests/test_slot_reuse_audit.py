import pytest
from scripts.gamestate.audit_slot_reuse import audit_events


def play(match,tick,slot,card):return dict(event='PLAY',match=match,tick=tick,slot=slot,card=card)
def ack(match,tick,source,card):return dict(event='PLAY_ACK',match=match,tick=tick,source_tick=source,card=card)


def test_only_confirmed_same_slot_within_same_match_counts():
    events=[play(1,0,0,'knight'),ack(1,27,0,'knight'),
            play(1,28,1,'fireball'),ack(1,55,28,'fireball'),
            play(1,30,0,'skeletons'),ack(1,57,30,'skeletons'),
            play(1,60,0,'zap'), # not confirmed: exclude
            play(1,110,0,'hog_rider'),ack(1,137,110,'hog_rider'),
            play(2,0,0,'knight'),ack(2,27,0,'knight')]
    r=audit_events(events)
    assert r['confirmed_plays']==5 and r['transition_count']==2
    assert [t['ack_to_next_input_game_ms']for t in r['transitions']]==[150,2650]
    assert r['blocked_by_sim_cooldown']==1


def test_invalid_confirmation_cannot_prove_fast_reuse():
    with pytest.raises(ValueError):audit_events([ack(1,27,0,'knight')])
    with pytest.raises(ValueError):audit_events([play(1,10,0,'knight'),ack(1,9,10,'knight')])
