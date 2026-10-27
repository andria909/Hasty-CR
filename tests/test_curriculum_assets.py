from scripts.attack.audit_curriculum_assets import audit


def test_active_curriculum_has_paid_lightning_and_no_missing_card():
    result=audit()
    assert result['required_cards']==32 and not result['missing']
    assert result['lightning_paid_check']['spent']==6
    assert result['lightning_paid_check']['tower_damage']>0
    assert not result['collector']['in_curriculum']
    assert not result['live_parity_proven']
