import json
import runpy

import pytest

from scripts.kaggle import bootstrap_defense_recovery as bootstrap


def inputs(tmp_path, layout='flat'):
    root = tmp_path/'input'
    source = root/'source' if layout=='flat' else root/'datasets/owner/source'
    weights = root/'weights' if layout=='flat' else root/'datasets/owner/weights'
    source.mkdir(parents=True);weights.mkdir(parents=True)
    (source/'controller.py').write_text('def execute(args):\n    return vars(args)\n')
    (source/'manifest.json').write_text(json.dumps({'archive_sha256':'a'*64}))
    (weights/'recovery_receipt.json').write_text('{}')
    manifest=dict(schema='defense-recovery-controller-v1',not_exact_resume=True,
        controller_sha256=bootstrap.digest(source/'controller.py'),
        source_manifest_sha256=bootstrap.digest(source/'manifest.json'),source_archive_sha256='a'*64)
    (source/'controller_manifest.json').write_text(json.dumps(manifest))
    config=dict(schema='defense-recovery-launch-v1',source_dataset='owner/source',
        recovered_dataset='owner/weights',controller_manifest_sha256=bootstrap.digest(source/'controller_manifest.json'),
        recovery_receipt_sha256=bootstrap.digest(weights/'recovery_receipt.json'))
    return root,source,weights,config


@pytest.mark.parametrize('layout',['flat','namespaced'])
def test_explicit_kaggle_mount_layouts_pass_arguments(tmp_path,layout):
    root,source,weights,config=inputs(tmp_path,layout)
    result=bootstrap.run(config,input_root=root,output=tmp_path/'out')
    assert result['source_input']==source
    assert result['source_extracted_root']==source/'source'
    assert result['recovered_input']==weights
    assert result['recovery_receipt_sha256']==config['recovery_receipt_sha256']


def test_ambiguous_mount_refused(tmp_path):
    root,source,weights,config=inputs(tmp_path)
    (root/'datasets/owner/source').mkdir(parents=True)
    with pytest.raises(ValueError,match='exactly one'):
        bootstrap.run(config,input_root=root)


@pytest.mark.parametrize('changed,expected',[
    ('controller.py','Controller SHA'),('manifest.json','Source manifest SHA'),
    ('controller_manifest.json','Controller manifest SHA'),('receipt','Recovery receipt SHA')])
def test_tampering_refused_before_import(tmp_path,changed,expected):
    root,source,weights,config=inputs(tmp_path)
    path=weights/'recovery_receipt.json' if changed=='receipt' else source/changed
    path.write_text('raise RuntimeError("must not execute")')
    with pytest.raises(ValueError,match=expected):bootstrap.run(config,input_root=root)


def test_prepare_is_private_offline_and_never_replaces_existing_job(tmp_path):
    root,source,weights,config=inputs(tmp_path)
    job=tmp_path/'job'
    metadata=bootstrap.prepare(config,job,'owner/recovery')
    assert metadata['is_private'] and not metadata['enable_internet']
    assert metadata['dataset_sources']==['owner/source','owner/weights']
    # Configuration travels in code_file; no assumption that Kaggle uploads
    # neighboring JSON files with a script kernel.
    assert runpy.run_path(str(job/'bootstrap.py'))['EMBEDDED_CONFIG']==config
    with pytest.raises(FileExistsError):bootstrap.prepare(config,job,'owner/recovery')


@pytest.mark.parametrize('reference',['../source','owner/../../secret','/source','a/b/c','a/'])
def test_nonexplicit_dataset_reference_refused(reference):
    with pytest.raises(ValueError):bootstrap.dataset_parts(reference)
