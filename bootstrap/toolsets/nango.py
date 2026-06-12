from __future__ import annotations

from agent_runtime.tools.nango_github import (
    NangoGitHubGetRepoTool,
    NangoGitHubListIssuesTool,
    NangoGitHubListPullRequestsTool,
)
from bootstrap.toolsets.protocol import ToolsetDeps, ToolsetProvider, build_registration_result
from connectors.nango import NangoConnector


class NangoToolsetProvider(ToolsetProvider):
    def register(self, registry, deps: ToolsetDeps):
        before = set(registry.get_registered_names())
        requester = (
            deps.http_resources.external_default
            if deps.http_resources is not None
            else None
        )
        connector = NangoConnector(requester=requester)
        registry.register(
            NangoGitHubGetRepoTool(connector),
            always_on=False,
            risk="external-side-effect",
            search_hint="github repo repository nango saas connector",
        )
        registry.register(
            NangoGitHubListIssuesTool(connector),
            always_on=False,
            risk="external-side-effect",
            search_hint="github issues nango saas connector",
        )
        registry.register(
            NangoGitHubListPullRequestsTool(connector),
            always_on=False,
            risk="external-side-effect",
            search_hint="github pull requests prs nango saas connector",
        )
        return build_registration_result(
            registry=registry,
            source_name="nango",
            before=before,
            extras={"nango_connector": connector},
        )
