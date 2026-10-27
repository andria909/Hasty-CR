from sim.match import PlayerState


def test_play_replaces_only_used_slot_and_preserves_cycle():
    p=PlayerState(1,list('abcdefgh'),hand=list('abcd'),queue=list('efgh'))
    assert p.play('b') and p.hand==list('aecd') and p.queue==list('fghb')
    assert p.play('d') and p.hand==list('aecf') and p.queue==list('ghbd')
    assert p.play('a') and p.hand==list('gecf') and p.queue==list('hbda')
    assert p.play('c') and p.hand==list('gehf') and p.queue==list('bdac')
    assert p.play('e') and p.hand==list('gbhf') and p.queue==list('dace')
    assert len(set(p.hand+p.queue))==8


def test_missing_card_does_not_change_hand_or_queue():
    p=PlayerState(1,list('abcdefgh'),hand=list('abcd'),queue=list('efgh'))
    assert not p.play('x') and p.hand==list('abcd') and p.queue==list('efgh')
