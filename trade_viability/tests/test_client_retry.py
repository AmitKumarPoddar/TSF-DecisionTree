import pytest
import requests

from datamexico import client as client_mod
from datamexico.client import DataMexicoError, DataMexicoUnreachable, TesseractClient


class FakeResponse:
    def __init__(self, status, body='{"data": []}'):
        self.status_code, self.text = status, body

    def json(self):
        import json
        return json.loads(self.text)


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes, self.calls, self.headers = list(outcomes), 0, {}

    def get(self, url, timeout):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)


def test_retries_transient_errors_then_succeeds():
    session = FakeSession([requests.ConnectTimeout("t"), FakeResponse(503), FakeResponse(200)])
    c = TesseractClient("https://x/tesseract", session=session)
    assert c.get_json("data.jsonrecords") == {"data": []}
    assert session.calls == 3


def test_unreachable_after_retries():
    session = FakeSession([requests.ConnectTimeout("t")] * 3)
    c = TesseractClient("https://x/tesseract", session=session)
    with pytest.raises(DataMexicoUnreachable, match="after 3 attempts"):
        c.get_json("cubes")


def test_client_errors_are_not_retried():
    session = FakeSession([FakeResponse(404, '{"detail": "no level"}')])
    c = TesseractClient("https://x/tesseract", session=session)
    with pytest.raises(DataMexicoError, match="HTTP 404"):
        c.get_json("data.jsonrecords")
    assert session.calls == 1
