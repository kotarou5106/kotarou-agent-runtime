from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from agent_runtime.core.net.http import HttpRequester, RequestBudget


@dataclass(frozen=True)
class NangoConfig:
    base_url: str = ""
    secret_key: str = ""
    github_integration_id: str = ""
    github_connection_id: str = ""

    @classmethod
    def from_env(cls) -> "NangoConfig":
        return cls(
            base_url=os.getenv("NANGO_BASE_URL", "").strip(),
            secret_key=os.getenv("NANGO_SECRET_KEY", "").strip(),
            github_integration_id=os.getenv("NANGO_GITHUB_INTEGRATION_ID", "").strip(),
            github_connection_id=os.getenv("NANGO_GITHUB_CONNECTION_ID", "").strip(),
        )

    @property
    def configured(self) -> bool:
        return all(
            (
                self.base_url,
                self.secret_key,
                self.github_integration_id,
                self.github_connection_id,
            )
        )

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        for env_name, value in (
            ("NANGO_BASE_URL", self.base_url),
            ("NANGO_SECRET_KEY", self.secret_key),
            ("NANGO_GITHUB_INTEGRATION_ID", self.github_integration_id),
            ("NANGO_GITHUB_CONNECTION_ID", self.github_connection_id),
        ):
            if not value:
                missing.append(env_name)
        return missing


class NangoConnector:
    """Small Nango proxy client for GitHub API access.

    GitHub credentials stay inside the Nango connection. This client only uses
    the Nango secret key and connection identifiers.
    """

    def __init__(
        self,
        config: NangoConfig | None = None,
        *,
        requester: HttpRequester | None = None,
    ) -> None:
        self.config = config or NangoConfig.from_env()
        self._requester = requester

    @property
    def configured(self) -> bool:
        return self.config.configured

    def missing_configuration_error(self) -> dict[str, Any]:
        return {
            "error": "nango_not_configured",
            "message": "Nango GitHub connector is not configured.",
            "missing_env": self.config.missing_fields(),
        }

    async def github_get_repo(self, owner: str, repo: str) -> dict[str, Any]:
        return await self.github_request(f"/repos/{owner}/{repo}")

    async def github_list_issues(
        self,
        owner: str,
        repo: str,
        *,
        state: str = "open",
        per_page: int = 30,
        page: int = 1,
    ) -> dict[str, Any]:
        return await self.github_request(
            f"/repos/{owner}/{repo}/issues",
            params={
                "state": state,
                "per_page": max(1, min(int(per_page), 100)),
                "page": max(1, int(page)),
            },
        )

    async def github_list_pull_requests(
        self,
        owner: str,
        repo: str,
        *,
        state: str = "open",
        per_page: int = 30,
        page: int = 1,
    ) -> dict[str, Any]:
        return await self.github_request(
            f"/repos/{owner}/{repo}/pulls",
            params={
                "state": state,
                "per_page": max(1, min(int(per_page), 100)),
                "page": max(1, int(page)),
            },
        )

    async def github_request(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.configured:
            return self.missing_configuration_error()
        if self._requester is None:
            from agent_runtime.core.net.http import get_default_http_requester

            self._requester = get_default_http_requester("external_default")

        url = self._proxy_url(path)
        response = await self._requester.get(
            url,
            params=params,
            headers=self._headers(),
            timeout_s=30.0,
            budget=RequestBudget(total_timeout_s=45.0),
        )
        text = response.text
        try:
            payload: Any = response.json()
        except Exception:
            try:
                payload = json.loads(text)
            except Exception:
                payload = {"text": text}
        if response.status_code >= 400:
            return {
                "error": "nango_proxy_error",
                "status_code": response.status_code,
                "url": url,
                "payload": payload,
            }
        return {
            "ok": True,
            "status_code": response.status_code,
            "data": payload,
        }

    def _proxy_url(self, github_path: str) -> str:
        base = self.config.base_url.rstrip("/")
        path = github_path if github_path.startswith("/") else f"/{github_path}"
        return f"{base}/proxy{path}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.secret_key}",
            "Provider-Config-Key": self.config.github_integration_id,
            "Connection-Id": self.config.github_connection_id,
            "Accept": "application/json",
        }
