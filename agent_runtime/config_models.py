from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from proactive_system.config import ProactiveConfig


@dataclass
class TelegramChannelConfig:
    token: str
    allow_from: list[str] = field(default_factory=list)
    channel_name: str = "telegram"
    live_edit: bool = False
    stream_response: bool = False
    show_thinking: bool = False


@dataclass
class QQGroupConfig:
    group_id: str
    allow_from: list[str] = field(default_factory=list)
    require_at: bool = True


@dataclass
class QQChannelConfig:
    bot_uin: str
    allow_from: list[str] = field(default_factory=list)
    groups: list[QQGroupConfig] = field(default_factory=list)
    websocket_open_timeout_seconds: float = 5.0


@dataclass
class QQBotGroupConfig:
    group_openid: str
    allow_from: list[str] = field(default_factory=list)
    require_at: bool = True
    allow_proactive: bool = False


@dataclass
class QQBotChannelConfig:
    app_id: str
    client_secret: str
    allow_from: list[str] = field(default_factory=list)
    groups: list[QQBotGroupConfig] = field(default_factory=list)


@dataclass
class ChannelsConfig:
    telegram: TelegramChannelConfig | None = None
    qq: QQChannelConfig | None = None
    qqbot: QQBotChannelConfig | None = None
    socket: str = "/tmp/kotarou.sock"


@dataclass
class MemoryEmbeddingConfig:
    model: str = "text-embedding-v3"
    api_key: str = ""
    base_url: str = ""


@dataclass
class MemoryConfig:
    enabled: bool = False
    engine: str = ""
    embedding: MemoryEmbeddingConfig = field(default_factory=MemoryEmbeddingConfig)


@dataclass
class KnowledgeConfig:
    enabled: bool = False
    db_path: str = "knowledge.db"
    top_k: int = 6
    score_threshold: float = 0.20
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 120
    inject_max_chars: int = 2400


@dataclass
class FitbitIntegrationConfig:
    enabled: bool = False


@dataclass
class PeerAgentConfig:
    name: str
    base_url: str
    launcher: list[str]          # 拉起命令，如 ["uv", "run", "python", "-m", "app.a2a_server"]
    cwd: str | None = None       # 子进程工作目录，None 表示继承父进程
    description: str = ""        # 工具描述，用于 LLM 路由；服务器在线时会被 AgentCard 覆盖
    health_path: str = "/health"
    startup_timeout_s: int = 30
    shutdown_timeout_s: int = 10


@dataclass
class WiringConfig:
    context: str = "default"
    memory: str = "default"
    toolsets: list[str] = field(
        default_factory=lambda: [
            "meta_common",
            "spawn",
            "schedule",
            "mcp",
        ]
    )


@dataclass
class OrchestrationConfig:
    backend: str = "native"
    interrupt_high_risk_tools: bool = True
    checkpoint_enabled: bool = True


@dataclass
class Config:
    provider: str
    model: str
    api_key: str
    system_prompt: str
    max_tokens: int = 8192
    max_iterations: int = 10
    memory_window: int = 40
    base_url: str | None = None
    extra_body: dict = field(default_factory=dict)
    channels: ChannelsConfig = field(default_factory=ChannelsConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    memory_optimizer_enabled: bool = True
    memory_optimizer_interval_seconds: int = 64800
    light_model: str = ""
    light_api_key: str = ""
    light_base_url: str = ""
    agent_model: str = ""
    agent_api_key: str = ""
    agent_base_url: str = ""
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    fitbit: FitbitIntegrationConfig = field(default_factory=FitbitIntegrationConfig)
    multimodal: bool = True
    vl_model: str = ""
    vl_api_key: str = ""
    vl_base_url: str = ""
    tool_search_enabled: bool = False
    spawn_enabled: bool = True
    dev_mode: bool = False
    peer_agents: list[PeerAgentConfig] = field(default_factory=list)
    wiring: WiringConfig = field(default_factory=WiringConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)

    @classmethod
    def load(cls, path: str | Path = "config.toml") -> Config:
        from importlib import import_module

        return import_module("agent_runtime.config").load_config(path)


__all__ = [
    "ChannelsConfig",
    "Config",
    "FitbitIntegrationConfig",
    "KnowledgeConfig",
    "MemoryConfig",
    "MemoryEmbeddingConfig",
    "OrchestrationConfig",
    "PeerAgentConfig",
    "QQChannelConfig",
    "QQBotChannelConfig",
    "QQBotGroupConfig",
    "QQGroupConfig",
    "TelegramChannelConfig",
    "WiringConfig",
]
