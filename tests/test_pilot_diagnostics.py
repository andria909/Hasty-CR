import hashlib
from pathlib import Path
import pytest
from scripts.attack.watch_pilot_diagnostics import selected_weights,render


def test_diagnostic_cannot_use_partial_or_changed_selected_weights(tmp_path):
    models=[]
    for i in range(8):
        p=tmp_path/f'{i}.pt';p.write_bytes(str(i).encode());models.append(dict(id=str(i),checkpoint=str(p),checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    proof=dict(pilot_steps_verified=819200,pilot_models=models)
    assert selected_weights(proof)==models
    Path(models[-1]['checkpoint']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):selected_weights(proof)
    with pytest.raises(ValueError,match='budget'):selected_weights(dict(proof,pilot_steps_verified=409600))


def test_baseline_comparison_rejects_different_fixtures_before_claiming_metrics():
    with pytest.raises(ValueError,match='fixtures changed'):
        render(dict(fixtures=['changed']),dict(),dict(fixtures=['original']))
