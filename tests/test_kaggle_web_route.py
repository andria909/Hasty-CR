import sys
from types import SimpleNamespace
from scripts.kaggle.collect_final_evaluation import configure_web_route


def test_web_transport_is_explicit_idempotent_and_preserves_other_endpoints(monkeypatch):
    class Http:
        def __init__(self,endpoint='https://api.kaggle.com',username=None):
            self._endpoint=endpoint;self.username=username
    monkeypatch.setitem(sys.modules,'kagglesdk.kaggle_http_client',SimpleNamespace(KaggleHttpClient=Http))
    configure_web_route();first=Http.__init__;configure_web_route()
    assert Http.__init__ is first
    client=Http(username='our-account')
    assert client._endpoint=='https://www.kaggle.com/api'
    assert client.username=='our-account'
    assert Http('https://test.example')._endpoint=='https://test.example'
