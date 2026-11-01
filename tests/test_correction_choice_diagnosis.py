from scripts.attack.diagnose_correction_choices import classify
from sim.env import ClashEnv


def test_wait_is_distinguished_from_missed_placement():
    label=ClashEnv.encode(2,9,22)
    assert classify(0,[label])['category']=='wait'
    assert classify(label,[label])['category']=='accepted_label'


def test_same_position_in_another_slot_is_a_different_card():
    label=ClashEnv.encode(2,9,22)
    result=classify(ClashEnv.encode(1,9,22),[label])
    assert result==dict(category='different_card',distance_tiles=None)


def test_position_distance_uses_nearest_accepted_cell_in_the_same_slot():
    labels=[ClashEnv.encode(2,9,22),ClashEnv.encode(2,9,20)]
    assert classify(ClashEnv.encode(2,9,19),labels)==dict(category='same_card_other_cell',distance_tiles=1)
