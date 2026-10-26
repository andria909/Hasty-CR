import json
import pytest
from scripts.attack.verify_continuous_focus import metric_boundaries


def test_budget_requires_actual_consecutive_cuda_updates_not_terminal_flag(tmp_path):
    p=tmp_path/'metrics.jsonl'
    def row(i):return dict(update=i,step=i*1024,target_environment_steps=3072,percent=100*i/3,actual_cuda='Tesla T4')
    p.write_text('\n'.join(json.dumps(row(i))for i in(1,2,3)))
    assert metric_boundaries(p,3072)==[]
    p.write_text('\n'.join(json.dumps(row(i))for i in(1,3)))
    with pytest.raises(ValueError,match='counters'):metric_boundaries(p,3072)
    p.write_text(json.dumps(row(1)))
    with pytest.raises(ValueError,match='target budget'):metric_boundaries(p,3072)
