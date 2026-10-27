import pytest
from scripts.attack.verify_sequence_pilot import audit_sampler


def test_uniform_sampler_cannot_pass_weighted_trial():
    good=dict(sequence_wait_fraction=.25,sampling='equal_sequence_mass_bounded_wait_with_replacement')
    audit_sampler(good,dict(sequence_wait_fraction=.25))
    with pytest.raises(ValueError):audit_sampler(dict(good,sampling='uniform_without_replacement'),dict(sequence_wait_fraction=.25))
    with pytest.raises(ValueError):audit_sampler(good,dict(sequence_wait_fraction=.5))
