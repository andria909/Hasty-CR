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


def wait_for_dataset(api,slug):
    for _ in range(40):
        try:status=str(api.dataset_status(slug))
        except Exception as exc:
            code=getattr(getattr(exc,'response',None),'status_code',None)
            transient=(type(exc).__name__ in ('SSLError','ConnectionError','ReadTimeout','Timeout') or
                       code in (403,404,429,500,502,503,504))
            if not transient:raise
            print('DATASET readiness temporarily unconfirmed; same handle retained',flush=True)
            time.sleep(30);continue
        print('DATASET',status,flush=True)
        if status.lower()=='ready':return
        time.sleep(30)
    raise RuntimeError('Dataset not ready; no kernel submitted')


def submit(root,token_index,*,evaluation=False):
    from scripts.kaggle.collect_final_evaluation import make_api_client,configure_http_timeout
    root=Path(root);index=int(token_index);prepared=json.loads((root/'preparation.json').read_text());owner=prepared['config']['owner'];kernel=prepared['kernel']
    if index not in (0,1,2):raise ValueError('Only three authorized working accounts allowed')
    if evaluation:
        from scripts.kaggle.validate_evaluation_submission import validate_submission
        upload=validate_submission(root,prepared)
    else:
        preflight=json.loads((root/'preflight.json').read_text())
        def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if prepared.get('study_variant')=='mixed_bank_initialization_v1':
            if (preflight.get('schema')!='bank-initialization-preflight-v1' or
                    not preflight.get('frozen_sources_verified') or preflight.get('smoke_runs')!=4 or
                    preflight.get('supervised_updates')!=8 or preflight.get('environment_steps')!=0 or
                    preflight.get('preparation_sha256')!=sha(root/'preparation.json') or
                    preflight.get('result_sha256')!=sha(root/'actual_preflight/result/result.json')):
                raise ValueError('Verified BC-only frozen preflight required')
        elif prepared.get('study_variant')in ('bank_correction_pair_v1','bank_fit_depth_v1'):
            p=json.loads((root/'dataset/initialization_protocol.json').read_text())
            schema='bank-fit-depth-preflight-v1'if prepared['study_variant']=='bank_fit_depth_v1'else'bank-correction-pair-preflight-v1'
            if (prepared.get('technical_preflight_only')or p.get('technical_identical')or
                    preflight.get('schema')!=schema or
                    preflight.get('technical_identical')is not False or preflight.get('submission_allowed')is not True or
                    not preflight.get('frozen_sources_verified')or preflight.get('smoke_runs')!=4 or
                    preflight.get('supervised_updates')!=8 or preflight.get('environment_steps')!=0 or
                    preflight.get('bootstrap_sha256')!=prepared['bootstrap_sha256']or
                    preflight.get('preparation_sha256')!=sha(root/'preparation.json')or
                    preflight.get('result_sha256')!=sha(root/'actual_preflight/result/result.json')):
                raise ValueError('Real correction treatment and verified paired frozen preflight required')
            from scripts.attack.verify_bank_correction_pair import verify
            checked=verify(root,technical_smoke=True)
            (root/'preflight_independent_verification.json').write_text(json.dumps(checked,indent=2)+'\n')
        elif prepared.get('study_variant')in ('policy_adaptation_v1','calibrated_spell_cost_v1','group_entropy_v1','air_exposure_v1','spell_interaction_v1','tower_credit_v1'):
            if prepared['study_variant']=='tower_credit_v1':
                from scripts.kaggle.prepare_tower_credit_pilot import validate_protocol
            elif prepared['study_variant']=='spell_interaction_v1':
                from scripts.kaggle.prepare_spell_interaction_pilot import validate_protocol
            elif prepared['study_variant']=='air_exposure_v1':
                from scripts.kaggle.prepare_air_exposure_pilot import validate_protocol
            elif prepared['study_variant']=='group_entropy_v1':
                from scripts.kaggle.prepare_group_entropy_pilot import validate_protocol
            elif prepared['study_variant']=='calibrated_spell_cost_v1':
                from scripts.kaggle.prepare_calibrated_spell_pilot import validate_protocol
            else:
                from scripts.kaggle.prepare_policy_adaptation_pilot import validate_protocol
            p=json.loads((root/'dataset/protocol.json').read_text());validate_protocol(p)
            if (not preflight.get('exact_resume_16_to_32')or not preflight.get('frozen_sources_verified')
                    or preflight.get('preparation_sha256')!=sha(root/'preparation.json')
                    or preflight.get('bootstrap_sha256')!=prepared['bootstrap_sha256']
                    or len(preflight.get('children',[]))!=4 or preflight.get('validation_games')!=48):
                raise ValueError('Actual four-initializer PPO preflight required')
            for trial,child in zip(p['cases'],preflight['children']):
                expected=next(m for m in prepared['config']['models']if m['id']==trial['initial_id'])
                if child.get('explicit_initial_model')!=expected or not child.get('exact_resume_16_to_32'):
                    raise ValueError('Preflight assigned initialization differs')
                if prepared['study_variant']in ('air_exposure_v1','spell_interaction_v1','tower_credit_v1')and (not child.get('air_exposure_verified')or
                        child.get('pilot_steps')!=409600 or child.get('actual_focus')!=trial['focus']or
                        child.get('actual_spell_waste_weight')!=trial['spell_waste_weight']):
                    raise ValueError('Actual400-update pilot preflight required')
                if prepared['study_variant']in ('spell_interaction_v1','tower_credit_v1')and (not child.get('spell_interaction_verified')or
                        child.get('actual_dormant_king_weight')!=trial['dormant_king_weight']or child.get('crown_validation_games')!=12):
                    raise ValueError('Actual dormant-king accounting preflight required')
                if prepared['study_variant']=='tower_credit_v1'and child.get('actual_tower_support_weight')!=trial['tower_support_weight']:
                    raise ValueError('Actual tower coefficient preflight required')
                if prepared['study_variant']=='group_entropy_v1'and child.get('spatial_entropy_verified')!=trial['spatial_entropy_weight']:
                    raise ValueError('Actual entropy preflight required')
        elif not preflight.get('exact_resume_16_to_32')or not preflight.get('frozen_sources_verified'):
            raise ValueError('Verified frozen preflight required')
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
    reused=prepared.get('reused_dataset')
    if reused:
        from scripts.kaggle.reuse_frozen_dataset import validate_reuse
        validate_reuse(root,prepared)
    try:
     recent=api.kernels_list(mine=True,page_size=1,sort_by='dateRun')
     for row in recent or []:
      ref=row.to_dict().get('ref')
      if ref and not ref.startswith(owner+'/'):raise ValueError('Credential owner differs from prepared owner')
    except Exception as exc:
     (root/'authentication_status.json').write_text(json.dumps(dict(owner=owner,ok=False,http_status=getattr(getattr(exc,'response',None),'status_code',None),error_type=type(exc).__name__),indent=2)+'\n')
     raise
    (root/'authentication_status.json').write_text(json.dumps(dict(owner=owner,ok=True),indent=2)+'\n')
    if not reused and not(root/'dataset_submission.json').exists():
     (root/'dataset_submission.json').write_text(json.dumps(dict(requested=True)))
     r=api.dataset_create_new(str(upload),public=False,quiet=True,convert_to_csv=False,dir_mode='tar')
     result=r.to_dict()if hasattr(r,'to_dict')else str(r)
     (root/'dataset_create.json').write_text(json.dumps(result,indent=2)+'\n')
    slug=reused or owner+'/'+prepared['config']['slug']
    wait_for_dataset(api,slug)
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
