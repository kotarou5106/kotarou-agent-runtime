from __future__ import annotations

from agent_runtime.network_proxy import parse_proxy_url, select_http_proxy, select_telethon_proxy


def test_select_http_proxy_prefers_https(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:15547")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:15548")
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:15549")

    proxy = select_http_proxy()

    assert proxy is not None
    assert proxy.source == "HTTPS_PROXY"
    assert proxy.url == "http://127.0.0.1:15548"


def test_parse_proxy_url_for_telethon_socks5() -> None:
    assert parse_proxy_url("socks5://127.0.0.1:15547") == (
        "socks5",
        "127.0.0.1",
        15547,
    )


def test_select_telethon_proxy_prefers_all_proxy(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:15547")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:15548")
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:15549")

    proxy, source = select_telethon_proxy()

    assert source == "ALL_PROXY"
    assert proxy == ("socks5", "127.0.0.1", 15549)


def test_select_telethon_proxy_ignores_invalid_proxy(monkeypatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "not-a-url")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:15547")
    monkeypatch.delenv("HTTP_PROXY", raising=False)

    proxy, source = select_telethon_proxy()

    assert source == "HTTPS_PROXY"
    assert proxy == ("http", "127.0.0.1", 15547)
