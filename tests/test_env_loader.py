from __future__ import annotations

import os
from pathlib import Path

from agent_runtime.config import Config
from agent_runtime.env_loader import load_dotenv_for_config


def _minimal_config(path: Path) -> None:
    path.write_text(
        """
[llm]
provider = "deepseek"

[llm.main]
model = "deepseek-chat"
api_key = "${DEEPSEEK_API_KEY}"
base_url = "https://api.deepseek.com/v1"

[proactive]
profile = "daily"
""",
        encoding="utf-8",
    )


def test_dotenv_loads_from_config_directory(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _minimal_config(config_path)
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    result = load_dotenv_for_config(config_path, log=False)

    assert result.dotenv_path == tmp_path / ".env"
    assert result.dotenv_loaded is True
    assert os.environ["DEEPSEEK_API_KEY"] == "from-dotenv"


def test_config_resolves_env_after_dotenv_load(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _minimal_config(config_path)
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=resolved-key\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    config = Config.load(config_path)

    assert config.api_key == "resolved-key"


def test_dotenv_loads_when_cwd_is_not_config_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    config_path = project / "config.toml"
    _minimal_config(config_path)
    (project / ".env").write_text("DEEPSEEK_API_KEY=from-config-dir\n", encoding="utf-8")
    monkeypatch.chdir(other)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    config = Config.load(config_path)

    assert config.api_key == "from-config-dir"


def test_system_env_is_not_overridden_by_dotenv(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _minimal_config(config_path)
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-system")

    config = Config.load(config_path)

    assert config.api_key == "from-system"


def test_proxy_env_loads_before_config_resolution(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _minimal_config(config_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "DEEPSEEK_API_KEY=resolved-key",
                "HTTP_PROXY=http://127.0.0.1:15547",
                "HTTPS_PROXY=http://127.0.0.1:15547",
                "ALL_PROXY=socks5://127.0.0.1:15547",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for key in ("DEEPSEEK_API_KEY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(key, raising=False)

    config = Config.load(config_path)

    assert config.api_key == "resolved-key"
    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:15547"
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:15547"
    assert os.environ["ALL_PROXY"] == "socks5://127.0.0.1:15547"


def test_system_proxy_env_is_not_overridden_by_dotenv(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _minimal_config(config_path)
    (tmp_path / ".env").write_text(
        "DEEPSEEK_API_KEY=from-dotenv\nHTTP_PROXY=http://127.0.0.1:15547\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://system-proxy:8080")

    config = Config.load(config_path)

    assert config.api_key == "from-dotenv"
    assert os.environ["HTTP_PROXY"] == "http://system-proxy:8080"
