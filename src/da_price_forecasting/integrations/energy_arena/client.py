from __future__ import annotations

import time
from typing import Any

import requests


class EnergyArenaClient:
    """Minimal JSON client for Energy Arena submissions."""

    def __init__(
        self,
        api_key: str,
        api_base_url: str = "https://api.energy-arena.org",
        submit_url: str | None = None,
        api_key_header_name: str = "X-API-Key",
        api_key_prefix: str = "",
        timeout_seconds: int = 30,
        retry_delays_seconds: tuple[int, ...] = (5, 15, 30),
    ) -> None:
        self.submit_url = submit_url or f"{api_base_url.rstrip('/')}/api/v1/submissions"
        self.api_key = api_key
        self.api_key_header_name = api_key_header_name
        self.api_key_prefix = api_key_prefix
        self.timeout_seconds = timeout_seconds
        self.retry_delays_seconds = retry_delays_seconds

    def submit_json(self, payload: dict) -> dict:
        """Submit a JSON payload and return the parsed response."""
        headers = {
            self.api_key_header_name: f"{self.api_key_prefix}{self.api_key}",
            "Content-Type": "application/json",
        }

        transient_status_codes = {429, 502, 503, 504}
        last_error: Exception | None = None
        last_status: int | None = None
        last_payload: Any = None
        attempts = [0, *self.retry_delays_seconds]
        for delay in attempts:
            if delay:
                time.sleep(delay)
            try:
                return self._post(payload=payload, headers=headers)
            except RuntimeError as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                if status is None or status not in transient_status_codes:
                    raise
                last_status = status
                last_payload = getattr(exc, "payload", None)

        if last_error is not None:
            if last_status is not None:
                raise RuntimeError(
                    f"Energy Arena submission failed with HTTP {last_status}: {last_payload}"
                ) from last_error
            raise last_error

        raise RuntimeError("Energy Arena submission failed without a response.")

    def _post(self, payload: dict, headers: dict[str, str]) -> dict:
        try:
            response = requests.post(
                self.submit_url,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Energy Arena submission failed: {exc}") from exc

        if response.ok:
            try:
                return response.json()
            except ValueError:
                return {
                    "status_code": response.status_code,
                    "body": response.text,
                }

        parsed = self._extract_error_payload(response)
        runtime_error = RuntimeError(
            f"Energy Arena submission failed with HTTP {response.status_code}: {parsed}"
        )
        runtime_error.status_code = response.status_code
        runtime_error.payload = parsed
        raise runtime_error

    @staticmethod
    def _extract_error_payload(response: requests.Response) -> Any:
        try:
            body = response.json()
        except ValueError:
            return response.text.strip()

        if isinstance(body, dict):
            detail = body.get("detail")
            if isinstance(detail, str):
                return detail.strip()
            if isinstance(detail, list):
                messages = []
                for item in detail:
                    if isinstance(item, dict):
                        loc = ".".join(str(part) for part in item.get("loc", []) if str(part) != "body")
                        msg = str(item.get("msg", "")).strip()
                        if loc and msg:
                            messages.append(f"{loc}: {msg}")
                        elif msg:
                            messages.append(msg)
                        else:
                            messages.append(str(item))
                    else:
                        messages.append(str(item))
                return "; ".join(messages).strip()
            if detail is not None:
                return str(detail).strip()
        return body
