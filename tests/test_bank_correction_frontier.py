import pytest
from scripts.attack.filter_bank_correction_frontier import frontier


def context(specs):
    rows=[]
    for a,outcomes,damage in specs:
        for d,outcome,hp in zip((1350,2025),outcomes,damage):
            rows.append(dict(action=a,delay_ms=d,terminal=True,first_command_accepted=True,
                metrics=dict(outcome=outcome),own_damage=hp))
    return dict(successful_actions=[a for a,_,_ in specs],finals=rows)


def test_win_dominates_extra_hp_in_a_loss():
    c=context([(1,('loss','loss'),(1,1)),(2,('loss','win'),(100,100))])
    assert frontier(c)[0]==[2]


def test_unknown_latency_tradeoff_keeps_both_outcome_vectors():
    c=context([(1,('win','loss'),(100,100)),(2,('loss','win'),(1,1))])
    assert frontier(c)[0]==[1,2]


def test_equal_outcomes_keep_hp_tradeoffs_and_equivalent_positions():
    c=context([(1,('loss','loss'),(10,20)),(2,('loss','loss'),(20,10)),
        (3,('loss','loss'),(20,20)),(4,('loss','loss'),(10,20))])
    assert frontier(c)[0]==[1,2,4]


def test_unexecuted_or_missing_branch_never_becomes_a_label():
    c=context([(1,('win','win'),(10,10))]);c['finals'].pop()
    with pytest.raises(ValueError,match='paired'):frontier(c)
