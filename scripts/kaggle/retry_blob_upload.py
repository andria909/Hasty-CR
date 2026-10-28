"""Resume a failed file upload on the same private dataset; never resubmit a kernel.

This narrow recovery applies only to StartBlobUpload HTTP404, before dataset
creation. It uses Kaggle's official web API route for that one SDK operation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def recover(root, token_index):
    from scripts.kaggle.collect_final_evaluation import make_api_client, configure_http_timeout
    from kagglesdk.kaggle_http_client import KaggleHttpClient
    root = Path(root)
    previous = (root/'submit.log').read_text()
    if ('404 Client Error' not in previous or '/StartBlobUpload' not in previous
            or (root/'dataset_create.json').exists()
            or (root/'kernel_push_requested.json').exists()):
        raise ValueError('Not a failed pre-creation blob upload; inspect the exact remote handle')
    prep = json.loads((root/'preparation.json').read_text())
    if token_index not in (0, 1, 2):
        raise ValueError('Only authorized accounts')
    if (root/'blob_route_retry.json').exists():
        raise ValueError('Recovery already attempted; inspect saved state')
    for name, digest in prep['config']['pins'].items():
        if hashlib.sha256((root/'dataset'/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen input changed')
    token = json.loads((Path.home()/'.kaggle/access_tokens.json').read_text())['tokens'][token_index]
    os.environ['KAGGLE_API_TOKEN'] = token if isinstance(token, str) else token['token']
    configure_http_timeout()
    api = make_api_client(prep['config']['owner'])
    original = KaggleHttpClient._get_request_url
    def route(self, service, method):
        if service == 'blobs.BlobApiService' and method == 'StartBlobUpload':
            return 'https://www.kaggle.com/api/v1/'+service+'/'+method
        return original(self, service, method)
    KaggleHttpClient._get_request_url = route
    state = dict(same_dataset=True, kernel_submitted=False,
                 dataset=prep['config']['owner']+'/'+prep['config']['slug'],
                 reason='HTTP404 StartBlobUpload before CreateDataset', requested=True)
    (root/'blob_route_retry.json').write_text(json.dumps(state, indent=2)+'\n')
    try:
        response = api.dataset_create_new(str(root/'dataset'), public=False, quiet=True,
                                         convert_to_csv=False, dir_mode='tar')
        result = response.to_dict()
        (root/'dataset_create.json').write_text(json.dumps(result, indent=2)+'\n')
        if result.get('error') or result.get('status') in ('error', 'failed'):
            raise RuntimeError('Dataset creation was not accepted; inspect receipt')
        print(json.dumps(dict(dataset=state['dataset'], creation_response=result)), flush=True)
    finally:
        KaggleHttpClient._get_request_url = original


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--token-index', type=int, required=True)
    a = p.parse_args(); recover(a.root, a.token_index)
