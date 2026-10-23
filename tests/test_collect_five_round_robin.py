import json
from types import SimpleNamespace as NS

import pytest
import requests

from scripts.kaggle import collect_five_round_robin as collector


def test_complete_job_requires_tournament_and_every_skill_report(monkeypatch,tmp_path):
    data={'round_robin/'+name:b'{}'for name in collector.REQUIRED}
    data['round_robin/project/private.py']=b'excluded'
    monkeypatch.setattr(collector,'list_outputs',lambda api,k:({name:NS(url=name)for name in data},[]))
    monkeypatch.setattr(requests,'get',lambda url,**kw:NS(content=data[url],raise_for_status=lambda:None))
    result=collector.collect(NS(kernels_status=lambda k:'COMPLETE'),'owner/job',tmp_path/'all')
    assert result['complete']and not result['missing_reports']
    assert not (tmp_path/'all/project').exists()
    data.pop('round_robin/v25_common_skill_test.json')
    with pytest.raises(ValueError,match='missing required'):
        collector.collect(NS(kernels_status=lambda k:'COMPLETE'),'owner/job',tmp_path/'partial')
    receipt=json.loads((tmp_path/'partial/receipt.json').read_text())
    assert not receipt['complete']and receipt['missing_reports']==['v25_common_skill_test.json']


def test_timeout_defaults_apply_without_overriding_explicit_download_timeout(monkeypatch):
    from scripts.kaggle.collect_final_evaluation import configure_http_timeout
    monkeypatch.setattr(requests.Session,'send',lambda self,request,**kwargs:kwargs)
    configure_http_timeout();session=requests.Session()
    assert session.send(None)['timeout']==(30,120)
    assert session.send(None,timeout=None)['timeout']==(30,120)
    assert session.send(None,timeout=(5,300))['timeout']==(5,300)
