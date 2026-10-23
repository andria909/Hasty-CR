import json

from scripts.kaggle import watch_selected_ppo_preparation as module


def config(tmp_path):
    runs=[]
    for i in range(4):
        owner='owner'+str(i);folder=tmp_path/owner;folder.mkdir()
        runs.append(dict(owner=owner,collected=str(folder),output=str(folder/'prepared'),launch_config='launch',source_manifest='source',
                         kernel=owner+'/evaluation',evaluation_dataset=owner+'/evaldata',selected_dataset=owner+'/policy'))
    return dict(runs=runs,evaluation_package='evaluation-package')


def test_no_preparation_before_terminal_complete_collection(tmp_path,monkeypatch):
    settings=config(tmp_path)
    for i,run in enumerate(settings['runs']):
        if i==0:continue
        (tmp_path/run['owner']/'receipt.json').write_text(json.dumps(dict(status='COMPLETE'if i==1 else'ERROR',complete=False)))
    monkeypatch.setattr(module,'prepare',lambda *a,**kw:(_ for _ in()).throw(AssertionError('Premature preparation')))
    result=module.observe(settings)
    assert not result['all_prepared']and not result['remote_calls_performed']
    assert result['runs'][0]['state']=='awaiting_collection'
    assert all(r['state']=='training_not_complete'for r in result['runs'][1:])


def test_prepare_once_after_collection_and_preserve_failed_partial_output(tmp_path,monkeypatch):
    settings=config(tmp_path);calls=[]
    for run in settings['runs']:(tmp_path/run['owner']/'receipt.json').write_text(json.dumps(dict(status='COMPLETE',complete=True)))
    monkeypatch.setattr(module,'input_pins',lambda *a:{'fixture':'pins'})
    def prepare(*args,**kwargs):
        calls.append(kwargs['kernel']);output=args[4];output.mkdir()
        if kwargs['kernel'].startswith('owner1/'):raise ValueError('Invalid source')
        (output/'preparation_receipt.json').write_text(json.dumps(dict(kernel=kwargs['kernel'],training_steps=2048000,input_pins={'fixture':'pins'})))
    monkeypatch.setattr(module,'prepare',prepare)
    first=module.observe(settings);assert len(calls)==4
    assert first['runs'][1]['state']=='preparation_failed'
    second=module.observe(settings);assert len(calls)==4
    assert second['runs'][1]['state']=='incomplete_preparation_requires_review'
    assert all(second['runs'][i]['state']=='prepared'for i in (0,2,3))
    monkeypatch.setattr(module,'input_pins',lambda *a:{'fixture':'changed'})
    assert module.observe(settings)['runs'][0]['state']=='incomplete_preparation_requires_review'
