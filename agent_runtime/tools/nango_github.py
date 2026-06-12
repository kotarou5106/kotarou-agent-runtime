from __future__ import annotations

import json
from typing import Any

from agent_runtime.tools.base import Tool
from connectors.nango import NangoConnector


_REPO_PARAMS = {
    "type": "object",
    "properties": {
        "owner": {"type": "string", "description": "GitHub owner or organization"},
        "repo": {"type": "string", "description": "GitHub repository name"},
    },
    "required": ["owner", "repo"],
}

_LIST_PARAMS = {
    "type": "object",
    "properties": {
        "owner": {"type": "string", "description": "GitHub owner or organization"},
        "repo": {"type": "string", "description": "GitHub repository name"},
        "state": {
            "type": "string",
            "enum": ["open", "closed", "all"],
            "description": "Item state. Default: open",
        },
        "per_page": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "description": "Page size, 1-100. Default: 30",
        },
        "page": {
            "type": "integer",
            "minimum": 1,
            "description": "Page number. Default: 1",
        },
    },
    "required": ["owner", "repo"],
}


class NangoGitHubGetRepoTool(Tool):
    name = "nango_github_get_repo"
    description = "Get GitHub repository information through the configured Nango GitHub connection."
    parameters = _REPO_PARAMS

    def __init__(self, connector: NangoConnector) -> None:
        self._connector = connector

    async def execute(self, owner: str, repo: str, **_: Any) -> str:
        result = await self._connector.github_get_repo(owner, repo)
        return json.dumps(result, ensure_ascii=False)


class NangoGitHubListIssuesTool(Tool):
    name = "nango_github_list_issues"
    description = "List GitHub repository issues through the configured Nango GitHub connection."
    parameters = _LIST_PARAMS

    def __init__(self, connector: NangoConnector) -> None:
        self._connector = connector

    async def execute(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        per_page: int = 30,
        page: int = 1,
        **_: Any,
    ) -> str:
        result = await self._connector.github_list_issues(
            owner,
            repo,
            state=state,
            per_page=per_page,
            page=page,
        )
        return json.dumps(result, ensure_ascii=False)


class NangoGitHubListPullRequestsTool(Tool):
    name = "nango_github_list_pull_requests"
    description = "List GitHub repository pull requests through the configured Nango GitHub connection."
    parameters = _LIST_PARAMS

    def __init__(self, connector: NangoConnector) -> None:
        self._connector = connector

    async def execute(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        per_page: int = 30,
        page: int = 1,
        **_: Any,
    ) -> str:
        result = await self._connector.github_list_pull_requests(
            owner,
            repo,
            state=state,
            per_page=per_page,
            page=page,
        )
        return json.dumps(result, ensure_ascii=False)
