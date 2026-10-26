from collections import Counter
import pytest
from sim.continuous_push import ContinuousPushEnv
from sim.continuous_sampling import ContinuousScenarioCycle


def test_hard_focus_preserves_other_families_and_uses_distinct_seeds():
    env=ContinuousPushEnv();cases={s.scenario_id:s for s in env.scenarios}
    sampler=ContinuousScenarioCycle(env.scenarios,focus='air_golem')
    rows=[sampler.next()for _ in range(len(sampler.order))]
    families=Counter(cases[r['scenario_id']].family for r in rows)
    assert families['air']==3*families['hog'] and families['golem']==3*families['giant']
    assert len(families)==6 and len({r['seed']for r in rows})==len(rows)
    assert Counter(cases[r['scenario_id']].task for r in rows)=={'attack':20,'defense':20}


def test_curriculum_resume_exact_and_validation_data_refused():
    env=ContinuousPushEnv();s=ContinuousScenarioCycle(env.scenarios)
    for _ in range(7):s.next()
    state=s.state_dict();expected=[s.next()for _ in range(30)]
    clone=ContinuousScenarioCycle(env.scenarios);clone.load_state_dict(state)
    assert [clone.next()for _ in range(30)]==expected
    with pytest.raises(ValueError):ContinuousScenarioCycle(ContinuousPushEnv(split='validation').scenarios)
    with pytest.raises(ValueError):ContinuousScenarioCycle(env.scenarios,focus='air_golem').load_state_dict(state)
