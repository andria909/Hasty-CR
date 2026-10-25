import pytest

from scripts.kaggle.collect_final_evaluation import make_api_client


class Api:
    CONFIG_NAME_USER='username'
    CONFIG_NAME_TOKEN='token'
    def __init__(self):self.config_values={};self.login_calls=0
    def authenticate(self):self.login_calls+=1


def test_explicit_credentials_are_set_for_normal_server_authorized_requests(monkeypatch):
    monkeypatch.setenv('KAGGLE_API_TOKEN','local-test-value')
    client=make_api_client('owner',Api)
    assert client.config_values==dict(username='owner',token='local-test-value')
    assert client.login_calls==0
    assert not hasattr(client,'_authenticated')


def test_explicit_user_without_token_fails_locally(monkeypatch):
    monkeypatch.delenv('KAGGLE_API_TOKEN',raising=False)
    with pytest.raises(ValueError,match='environment|KAGGLE_API_TOKEN'):make_api_client('owner',Api)


def test_default_authentication_workflow_is_preserved():
    assert make_api_client(api_class=Api).login_calls==1
