import pytest
from scripts.kaggle.submit_prepared_pilot import validate_launch_receipt


def test_dataset_readiness_retries_transport_without_recreating(monkeypatch):
    from requests.exceptions import SSLError
    from types import SimpleNamespace
    from scripts.kaggle import submit_prepared_pilot as uploader
    calls=[]
    def status(slug):
        calls.append(slug)
        if len(calls)==1:raise SSLError('transient EOF')
        return 'ready'
    monkeypatch.setattr(uploader.time,'sleep',lambda seconds:None)
    uploader.wait_for_dataset(SimpleNamespace(dataset_status=status),'owner/same-inputs')
    assert calls==['owner/same-inputs','owner/same-inputs']

@pytest.mark.parametrize('receipt',[
    {'error':'Maximum weekly GPU quota of 30.00 hours reached.'},
    {'ref':'/code/other/job','versionNumber':1},
    {'ref':'/code/owner/job','versionNumber':2},
    {}, 'not a receipt',
])
def test_rejected_or_mismatched_response_is_not_a_launch(receipt):
    with pytest.raises(RuntimeError):validate_launch_receipt(receipt,'owner/job')

def test_matching_first_submission():
    validate_launch_receipt({'ref':'/code/owner/job','versionNumber':1,'kernelId':123},'owner/job')
