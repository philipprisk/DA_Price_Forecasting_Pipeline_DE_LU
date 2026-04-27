from __future__ import annotations

import pytest

from da_price_forecasting.integrations.energy_arena import client as client_module
from da_price_forecasting.integrations.energy_arena.client import EnergyArenaClient


class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.ok = 200 <= status_code < 300

    def json(self):
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload


def test_energy_arena_client_posts_payload_with_api_key_header(monkeypatch) -> None:
    calls = []

    def fake_post(url, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse(201, {"id": "submission-1"})

    monkeypatch.setattr(client_module.requests, "post", fake_post)

    client = EnergyArenaClient(
        api_key="secret",
        api_base_url="https://example.test",
        api_key_header_name="Authorization",
        api_key_prefix="Bearer ",
        timeout_seconds=7,
    )
    response = client.submit_json({"challenge_id": "8", "values": [1.0]})

    assert response == {"id": "submission-1"}
    assert calls == [
        {
            "url": "https://example.test/api/v1/submissions",
            "json": {"challenge_id": "8", "values": [1.0]},
            "headers": {
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
            },
            "timeout": 7,
        }
    ]


def test_energy_arena_client_retries_transient_status(monkeypatch) -> None:
    responses = [
        FakeResponse(503, {"detail": "temporarily unavailable"}),
        FakeResponse(200, {"status": "ok"}),
    ]

    def fake_post(url, json, headers, timeout):
        return responses.pop(0)

    monkeypatch.setattr(client_module.requests, "post", fake_post)

    client = EnergyArenaClient(
        api_key="secret",
        submit_url="https://example.test/submissions",
        retry_delays_seconds=(0,),
    )

    assert client.submit_json({"values": [1.0]}) == {"status": "ok"}
    assert responses == []


def test_energy_arena_client_formats_validation_errors(monkeypatch) -> None:
    def fake_post(url, json, headers, timeout):
        return FakeResponse(
            422,
            {
                "detail": [
                    {"loc": ["body", "values", 0], "msg": "Input should be finite"},
                    {"loc": ["body", "challenge_id"], "msg": "Field required"},
                ]
            },
        )

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    client = EnergyArenaClient(api_key="secret", submit_url="https://example.test/submissions")

    with pytest.raises(RuntimeError, match="values.0: Input should be finite; challenge_id: Field required"):
        client.submit_json({"values": [float("nan")]})
