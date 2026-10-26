import json
from types import SimpleNamespace as NS
from pathlib import Path

from scripts.kaggle import advance_continuous_focus as advance


def test_no_cloud_call_before_verified_training_and_evaluation(monkeypatch,tmp_path):
    template=tmp_path/'template';template.mkdir();(template/'template_manifest.json').write_text('{"files":{}}')
    clock=[0];monkeypatch.setattr(advance.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(advance.time,'sleep',lambda seconds:clock.__setitem__(0,200000))
    args=NS(state=tmp_path/'state',template=template,output=tmp_path/'job',owner='owner',
            training_report=tmp_path/'missing_training',evaluation_report=tmp_path/'missing_test')
    def forbidden(owner):raise AssertionError('Cloud access before prerequisite proof')
    advance.run(args,api_factory=forbidden)
    assert not args.output.exists()


def test_ambiguous_push_is_observed_but_never_submitted_again(monkeypatch,tmp_path):
    template=tmp_path/'template';template.mkdir();(template/'template_manifest.json').write_text('{"files":{}}')
    state=tmp_path/'state';state.mkdir();kernel='owner/hastycr-continuous-focus-20261026'
    (state/'state.json').write_text(json.dumps(dict(kernel=kernel,dataset_ready=True,push_requested=True,push_receipt_saved=False)))
    output=tmp_path/'job';output.mkdir();(output/'preparation.json').write_text(json.dumps(dict(kernel=kernel)))
    clock=[0];monkeypatch.setattr(advance.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(advance.time,'sleep',lambda seconds:clock.__setitem__(0,200000))
    monkeypatch.setattr(advance,'prerequisites',lambda *args:True)
    calls=[]
    api=NS(kernels_status=lambda handle:(calls.append(handle)or '{"status":"RUNNING"}'),
           kernels_push=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Duplicate GPU push')))
    args=NS(state=state,template=template,output=output,owner='owner',training_report='unused',evaluation_report='unused',status_report=tmp_path/'status.json')
    advance.run(args,api_factory=lambda owner:api)
    assert calls==[kernel]


def test_existing_kernel_found_through_authenticated_ownership_listing():
    pages=[]
    def listing(**kwargs):pages.append(kwargs);return[NS(ref='owner/known')]
    assert advance.kernel_exists(NS(kernels_list=listing),'owner','owner/known')
    assert not advance.kernel_exists(NS(kernels_list=listing),'owner','owner/missing')
    assert all(p['mine']for p in pages)


def test_lost_receipt_can_only_recover_the_exact_prepared_code(tmp_path):
    output=tmp_path/'job';output.mkdir();code=b'print("frozen")\n'
    import hashlib
    (output/'preparation.json').write_text(json.dumps(dict(bootstrap_sha256=hashlib.sha256(code).hexdigest())))
    def pull(kernel,path,**kwargs):Path(path,'remote.py').write_bytes(code)
    assert advance.recover_submission(NS(kernels_pull=pull),'owner/job',output)
    receipt=json.loads((output/'launch_receipt.json').read_text());assert receipt['recovered_from_exact_remote_code']
    def wrong(kernel,path,**kwargs):Path(path,'remote.py').write_text('print("different")')
    assert not advance.recover_submission(NS(kernels_pull=wrong),'owner/job',output)
