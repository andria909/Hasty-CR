import base64
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

from scripts.kaggle.reuse_frozen_dataset import materialize_reused_inputs,validate_reuse
from scripts.kaggle.local_artifact_transfer import file_sha


def test_materialization_keeps_identical_weights_and_checks_reused_bytes(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'model.pt').write_bytes(b'weights')
    (source/'protocol.json').write_bytes(b'old')
    new=tmp_path/'new.json';new.write_bytes(b'new')
    cfg=dict(pins={'model.pt':file_sha(source/'model.pt'),'protocol.json':file_sha(new)},
             inline_inputs={'protocol.json':base64.b64encode(b'new').decode()})
    out=materialize_reused_inputs(source,tmp_path/'result',cfg)
    assert (out/'model.pt').read_bytes()==b'weights'and(out/'protocol.json').read_bytes()==b'new'
    (source/'model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Reused input'):materialize_reused_inputs(source,tmp_path/'bad',cfg)


def test_materialization_rejects_bad_inline_hash_and_source_symlink(tmp_path):
    source=tmp_path/'source';source.mkdir();weight=source/'model.pt';weight.write_bytes(b'weights')
    cfg=dict(pins={'model.pt':file_sha(weight)},inline_inputs={'model.pt':base64.b64encode(b'wrong').decode()})
    with pytest.raises(ValueError,match='Inline'):materialize_reused_inputs(source,tmp_path/'bad_inline',cfg)
    (source/'link.pt').symlink_to(weight)
    cfg=dict(pins={'link.pt':file_sha(weight)},inline_inputs={})
    with pytest.raises(ValueError,match='Reused'):materialize_reused_inputs(source,tmp_path/'bad_link',cfg)


def test_reused_dataset_submission_never_calls_upload(monkeypatch,tmp_path):
    from scripts.kaggle import collect_final_evaluation as api_module
    from scripts.kaggle.submit_prepared_pilot import submit
    root=tmp_path/'prepared';(root/'job').mkdir(parents=True);(root/'dataset').mkdir()
    (root/'job/train.py').write_text('pass\n')
    (root/'job/kernel-metadata.json').write_text(json.dumps(dict(dataset_sources=['owner/existing'])))
    files={name:b'fixed'for name in ('protocol.json','extensions.tar.gz')}
    for name,data in files.items():(root/'dataset'/name).write_bytes(data)
    cfg=dict(owner='owner',slug='new-unused',source_slug='existing',
             pins={name:file_sha(root/'dataset'/name)for name in files},
             inline_inputs={name:base64.b64encode(data).decode()for name,data in files.items()})
    prep=dict(kernel='owner/new',config=cfg,bootstrap_sha256=file_sha(root/'job/train.py'),reused_dataset='owner/existing')
    (root/'preparation.json').write_text(json.dumps(prep))
    (root/'preflight.json').write_text(json.dumps(dict(exact_resume_16_to_32=True,frozen_sources_verified=True)))
    (root/'transport_preflight.json').write_text(json.dumps(dict(all_input_pins_verified=True,training_run_ast_unchanged=True,
        bootstrap_sha256=prep['bootstrap_sha256'],preparation_sha256=file_sha(root/'preparation.json'))))
    home=tmp_path/'home';(home/'.kaggle').mkdir(parents=True)
    (home/'.kaggle/access_tokens.json').write_text(json.dumps(dict(tokens=['dummy-credential'])))
    monkeypatch.setattr(Path,'home',lambda:home);monkeypatch.setenv('KAGGLE_API_TOKEN','previous')
    calls=[]
    def forbidden(*a,**kw):raise AssertionError('Dataset upload must be skipped')
    api=NS(kernels_list=lambda **kw:[NS(to_dict=lambda:dict(ref='owner/old'))],dataset_create_new=forbidden,
        dataset_status=lambda slug:calls.append(slug)or'ready',
        kernels_push=lambda *a,**kw:NS(to_dict=lambda:dict(ref='/code/owner/new',versionNumber=1)))
    monkeypatch.setattr(api_module,'make_api_client',lambda owner:api)
    monkeypatch.setattr(api_module,'configure_http_timeout',lambda:None)
    assert submit(root,0)['versionNumber']==1
    assert calls==['owner/existing']and not(root/'dataset_submission.json').exists()
    changed=dict(prep,reused_dataset='other/existing')
    with pytest.raises(ValueError):validate_reuse(root,changed)
