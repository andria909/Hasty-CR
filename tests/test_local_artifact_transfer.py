import sys
from types import SimpleNamespace

import pytest
from scripts.kaggle.local_artifact_transfer import download_preserving,copy_verified_cache,file_sha


def test_interrupted_download_retains_partial_and_retries_into_new_file(monkeypatch,tmp_path):
    calls=[];sleeps=[]
    class Failure(Exception):pass
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,size):
            yield b'first'
            if len(calls)==1:raise Failure('connection lost')
            yield b'last'
    def get(*args,**kwargs):calls.append(kwargs);return Response()
    monkeypatch.setitem(sys.modules,'requests',SimpleNamespace(get=get,RequestException=Failure))
    monkeypatch.setattr('time.sleep',lambda delay:sleeps.append(delay))
    output=tmp_path/'weights.pt';download_preserving('private-signed-url',output)
    assert output.read_bytes()==b'firstlast'
    assert (tmp_path/'weights.pt.download_attempt_0').read_bytes()==b'first'
    assert len(calls)==2 and sleeps==[1]
    with pytest.raises(FileExistsError):download_preserving('unused',output)
    assert len(calls)==2


def test_cache_requires_unchanged_bytes_and_rejects_symlink_escape(tmp_path):
    cache=tmp_path/'cache';cache.mkdir();(cache/'weights').write_bytes(b'verified')
    files={'weights':{'sha256':file_sha(cache/'weights')}}
    assert copy_verified_cache(cache,files,'weights',tmp_path/'copied')
    (cache/'weights').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):copy_verified_cache(cache,files,'weights',tmp_path/'bad')
    (cache/'escape').symlink_to(tmp_path/'copied')
    with pytest.raises(ValueError,match='Unsafe'):copy_verified_cache(cache,{'escape':{'sha256':file_sha(tmp_path/'copied')}},'escape',tmp_path/'escaped')
