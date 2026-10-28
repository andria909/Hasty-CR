"""Submit an already frozen, preflight-verified pilot once; tokens stay outside Git."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path



def validate_launch_receipt(receipt,kernel):
    if (not isinstance(receipt,dict) or receipt.get('error') or
            receipt.get('ref')!='/code/'+kernel or receipt.get('versionNumber')!=1):
        raise RuntimeError('Kaggle did not accept the first kernel submission; inspect saved receipt')


def submit(root,token_index,*,evaluation=False):
    from scripts.kaggle.collect_final_evaluation import make_api_client,configure_http_timeout
    root=Path(root);index=int(token_index);prepared=json.loads((root/'preparation.json').read_text());owner=prepared['config']['owner'];kernel=prepared['kernel']
    if index not in (0,1,2):raise ValueError('Only three authorized working accounts allowed')
    if evaluation:
        from scripts.kaggle.validate_evaluation_submission import validate_submission
        upload=validate_submission(root,prepared)
    else:
        preflight=json.loads((root/'preflight.json').read_text())
        if not preflight.get('exact_resume_16_to_32')or not preflight.get('frozen_sources_verified'):
            raise ValueError('Verified frozen preflight required')
        def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if sha(root/'job/train.py')!=prepared['bootstrap_sha256']:raise ValueError('Prepared bootstrap changed')
        for name,digest in prepared['config']['pins'].items():
            if sha(root/'dataset'/name)!=digest:raise ValueError('Prepared input changed: '+name)
        upload=root/prepared.get('upload_directory','dataset')
        if not upload.resolve().is_relative_to(root.resolve()):raise ValueError('Upload folder escapes prepared root')
        for name,digest in prepared.get('upload_pins',{}).items():
            if sha(upload/name)!=digest:raise ValueError('Prepared transport changed')
    r=json.loads((Path.home()/'.kaggle/access_tokens.json').read_text())['tokens'][index]
    os.environ['KAGGLE_API_TOKEN']=(r if isinstance(r,str)else r['token']).strip()
    configure_http_timeout();api=make_api_client(owner)
    try:
     recent=api.kernels_list(mine=True,page_size=1,sort_by='dateRun')
     for row in recent or []:
      ref=row.to_dict().get('ref')
      if ref and not ref.startswith(owner+'/'):raise ValueError('Credential owner differs from prepared owner')
    except Exception as exc:
     (root/'authentication_status.json').write_text(json.dumps(dict(owner=owner,ok=False,http_status=getattr(getattr(exc,'response',None),'status_code',None),error_type=type(exc).__name__),indent=2)+'\n')
     raise
    (root/'authentication_status.json').write_text(json.dumps(dict(owner=owner,ok=True),indent=2)+'\n')
    if not(root/'dataset_submission.json').exists():
     (root/'dataset_submission.json').write_text(json.dumps(dict(requested=True)))
     r=api.dataset_create_new(str(upload),public=False,quiet=True,convert_to_csv=False,dir_mode='tar')
     result=r.to_dict()if hasattr(r,'to_dict')else str(r)
     (root/'dataset_create.json').write_text(json.dumps(result,indent=2)+'\n')
    slug=owner+'/'+prepared['config']['slug']
    for _ in range(40):
     try:status=str(api.dataset_status(slug))
     except Exception as exc:
      if getattr(getattr(exc,'response',None),'status_code',None)!=403:raise
      print('DATASET readiness unconfirmed (HTTP403); same handle retained',flush=True);time.sleep(30);continue
     print('DATASET',status,flush=True)
     if status.lower()=='ready':break
     time.sleep(30)
    else:raise RuntimeError('Dataset not ready; no kernel submitted')
    if(root/'kernel_push_requested.json').exists():raise RuntimeError('Existing push request; inspect exact handle, do not duplicate')
    (root/'kernel_push_requested.json').write_text(json.dumps(dict(kernel=kernel,requested=True)))
    r=api.kernels_push(str(root/'job'),acc='NvidiaTeslaT4');receipt=r.to_dict()if hasattr(r,'to_dict')else str(r)
    (root/'launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    validate_launch_receipt(receipt,kernel)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--token-index',type=int,required=True)
    p.add_argument('--evaluation',action='store_true')
    a=p.parse_args();submit(a.root,a.token_index,evaluation=a.evaluation)
