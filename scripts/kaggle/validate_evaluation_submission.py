"""SDK-only verification of the locally enabled, exact evaluation package."""
import json
from pathlib import Path
from scripts.kaggle.local_artifact_transfer import file_sha


def validate_submission(root, prepared):
    root=Path(root);config=prepared['config']
    ready=json.loads((root/'submission_ready.json').read_text())
    expected=dict(schema='full-evaluation-submission-ready-v1',
        preparation_sha256=file_sha(root/'preparation.json'),
        preflight_sha256=file_sha(root/'preflight.json'),bootstrap_sha256=file_sha(root/'job/train.py'),
        metadata_sha256=file_sha(root/'job/kernel-metadata.json'),
        transport_sha256=file_sha(root/'upload_dataset/evaluation_inputs.b64'))
    if ready!=expected or ready['bootstrap_sha256']!=prepared['bootstrap_sha256'] or ready['transport_sha256']!=config['transport_sha256']:
        raise ValueError('Enabled evaluation package changed')
    report=json.loads((root/'preflight.json').read_text())
    if (report.get('schema')!='full-evaluation-exact-package-preflight-v1'
            or report.get('actual_full_training_reverified') is not True
            or report.get('independent_test_run') is not False
            or report.get('quality_accepted') is not False
            or report.get('source_manifest_sha256')!=config['source_manifest_sha256']
            or report.get('protocol_sha256')!=config['files']['protocol.json']
            or report.get('preparation_sha256')!=ready['preparation_sha256']
            or report.get('bootstrap_sha256')!=ready['bootstrap_sha256']
            or len(report.get('models',[])) not in (5,6,7)
            or report.get('technical_games')!=2*len(report['models'])+1):
        raise ValueError('Actual evaluation preflight required')
    metadata=json.loads((root/'job/kernel-metadata.json').read_text())
    if (metadata['id']!=prepared['kernel'] or metadata['enable_gpu'] is not True
            or metadata['dataset_sources']!=[config['owner']+'/'+config['slug']]
            or metadata['is_private'] is not True or metadata['enable_internet'] is not False):
        raise ValueError('Evaluation submission metadata differs')
    dataset=json.loads((root/'upload_dataset/dataset-metadata.json').read_text())
    if dataset['id']!=config['owner']+'/'+config['slug'] or dataset.get('isPrivate') is not True:
        raise ValueError('Private evaluation dataset target differs')
    return root/'upload_dataset'
