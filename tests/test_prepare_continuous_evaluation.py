import pytest
from scripts.kaggle.prepare_continuous_evaluation import selected_models


def report():
    return dict(schema='verified-continuous-ppo-training-v1',
        kernel='andreiperelaiko/hastycr-continuous-slots-20261026',
        all_training_budgets_verified=True,
        models=[dict(id=f'full_{name}_s44',steps=4096000)for name in ('balanced','strong')])


def test_control_weights_cannot_be_mislabeled_as_corrected_slots():
    r=report()
    assert len(selected_models(r,'slots'))==2
    with pytest.raises(ValueError,match='cohort'):selected_models(r,'control')


@pytest.mark.parametrize('change',('budget','duplicate','unsafe_id','unverified'))
def test_incomplete_or_ambiguous_selection_rejected(change):
    r=report()
    if change=='budget':r['models'][0]['steps']=102400
    elif change=='duplicate':r['models'][1]=r['models'][0]
    elif change=='unsafe_id':r['models'][0]['id']='../parent'
    else:r['all_training_budgets_verified']=False
    with pytest.raises(ValueError):selected_models(r,'slots')
