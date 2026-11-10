from copy import deepcopy

import pytest

from scripts.attack.summarize_entity_experiment import paired_decoder_report
from tests.test_paired_ppo_behavior import records


def fixture():
    return {c:{s:{p:records()for p in ('selected','final')}for s in (522,523)}
            for c in ('decode64','relative64','sharp64')}


def test_selected_attack_gain_does_not_hide_defense_failure_or_final_collapse():
    data=fixture()
    for seed in (522,523):
        data['relative64'][seed]['selected'][0]['outcome']='win'
        data['relative64'][seed]['selected'][1]['outcome']='win'
        data['relative64'][seed]['selected'][0]['empty_spell_elixir']=4.
    data['decode64'][522]['final'][6]['outcome']='win'
    result=paired_decoder_report(data)
    row=next(x for x in result['rows']if x['case']=='relative64'and x['seed']==522 and x['phase']=='selected')
    assert row['all']['wins_delta']==2
    assert row['task_families']['attack']['air']['after_wins']==2
    assert row['task_families']['defense']['air']['after_wins']==0
    assert row['all']['metrics']['empty_spell_elixir']['delta']==pytest.approx(4/72)
    direction=next(x for x in result['directions']if x['case']=='relative64')
    assert direction['selected_wins_delta_by_seed']==[2,2]
    assert direction['selected_defense_delta_by_seed']==[0,0]
    assert direction['final_wins_delta_by_seed']==[-1,0]
    assert result['selection_changed']is False


@pytest.mark.parametrize('mutation',['seed','context','duplicate','split'])
def test_reporting_rejects_unpaired_evidence(mutation):
    data=fixture()
    if mutation=='seed':data['relative64'][524]=data['relative64'].pop(522)
    if mutation=='context':data['relative64'][522]['selected'][0]['match_seed']+=1
    if mutation=='duplicate':data['relative64'][522]['final'][1]=deepcopy(data['relative64'][522]['final'][0])
    if mutation=='split':data['relative64'][522]['selected'][0]['split']='test'
    with pytest.raises(ValueError):paired_decoder_report(data)
