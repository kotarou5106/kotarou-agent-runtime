"""
IPC server channel.

Uses a Unix domain socket on POSIX systems and loopback TCP on Windows so the
local CLI can talk to the running agent process.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from agent_runtime.config import _normalize_cli_socket_endpoint
from agent_runtime.events.events import InboundMessage, OutboundMessage
from agent_runtime.events.queue import MessageBus

if TYPE_CHECKING:
    from proactive_system.loop import ProactiveLoop

logger = logging.getLogger(__name__)

CHANNEL = "cli"


def _parse_tcp_endpoint(endpoint: str) -> tuple[str, int] | None:
    if endpoint.count(":") != 1:
        return None
    host, port = endpoint.rsplit(":", 1)
    if not host:
        return None
    try:
        return host, int(port)
    except ValueError:
        return None


def _normalize_endpoint(endpoint: str) -> str:
    return _normalize_cli_socket_endpoint(endpoint)


def _short_unix_socket_path(path: str) -> str:
    if len(os.fsencode(path)) <= 100:
        return path
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    fallback = str(Path("/private/tmp") / f"kotarou-{digest}.sock")
    logger.warning("IPC socket path too long, using fallback path: %s", fallback)
    return fallback


class IPCServerChannel:
    def __init__(
        self,
        bus: MessageBus,
        socket_path: str,
        proactive_loop: "ProactiveLoop | None" = None,
    ) -> None:
        self._bus = bus
        self._socket_path = _normalize_endpoint(socket_path)
        self._proactive_loop = proactive_loop
        self._writers: dict[str, asyncio.StreamWriter] = {}
        self._server: asyncio.AbstractServer | None = None
        bus.subscribe_outbound(CHANNEL, self._on_response)

    async def start(self) -> None:
        tcp_endpoint = _parse_tcp_endpoint(self._socket_path)
        if tcp_endpoint is not None:
            host, port = tcp_endpoint
            self._server = await asyncio.start_server(
                self._handle_connection,
                host=host,
                port=port,
            )
            logger.info("IPC server listening on tcp://%s:%s", host, port)
            return

        if not hasattr(asyncio, "start_unix_server"):
            raise RuntimeError("Unix sockets are unavailable on this platform; use a host:port endpoint instead.")
        self._socket_path = _short_unix_socket_path(self._socket_path)
        Path(self._socket_path).unlink(missing_ok=True)
        try:
            self._server = await asyncio.start_unix_server(
                self._handle_connection,
                path=self._socket_path,
            )
            os.chmod(self._socket_path, 0o600)
            logger.info("IPC server listening on %s", self._socket_path)
        except OSError as exc:
            logger.warning(
                "IPC unix socket unavailable at %s: %s; falling back to tcp://127.0.0.1:0",
                self._socket_path,
                exc,
            )
            Path(self._socket_path).unlink(missing_ok=True)
            try:
                self._server = await asyncio.start_server(
                    self._handle_connection,
                    host="127.0.0.1",
                    port=0,
                )
                sock = self._server.sockets[0] if self._server.sockets else None
                port = sock.getsockname()[1] if sock is not None else 0
                self._socket_path = f"127.0.0.1:{port}"
                logger.info("IPC server listening on tcp://%s", self._socket_path)
            except OSError as tcp_exc:
                self._server = None
                logger.warning("IPC server disabled; tcp fallback also unavailable: %s", tcp_exc)

    async def stop(self) -> None:
        if not self._server:
            return
        self._server.close()
        await self._server.wait_closed()
        if _parse_tcp_endpoint(self._socket_path) is None:
            Path(self._socket_path).unlink(missing_ok=True)

    def set_proactive_loop(self, proactive_loop: "ProactiveLoop") -> None:
        self._proactive_loop = proactive_loop
        logger.info("[cli] ProactiveLoop attached")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername") or "local"
        chat_id = f"cli-{id(writer)}"
        self._writers[chat_id] = writer
        logger.info("[cli] client connected session=%s peer=%s", chat_id, peer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("[cli] received non-JSON payload")
                    continue

                if data.get("type") == "command":
                    await self._handle_command(data, chat_id, writer)
                    continue

                content = str(data.get("content", "")).strip()
                if not content:
                    continue
                preview = content[:60] + "..." if len(content) > 60 else content
                logger.info("[cli] received session=%s content=%r", chat_id, preview)
                await self._bus.publish_inbound(
                    InboundMessage(
                        channel=CHANNEL,
                        sender="cli-user",
                        chat_id=chat_id,
                        content=content,
                    )
                )
        finally:
            self._writers.pop(chat_id, None)
            writer.close()
            await writer.wait_closed()
            logger.info("[cli] client disconnected session=%s", chat_id)

    async def _handle_command(
        self,
        data: dict,
        chat_id: str,
        writer: asyncio.StreamWriter,
    ) -> None:
        cmd = data.get("command", "")
        logger.info("[cli] received command cmd=%r session=%s", cmd, chat_id)
        await self._write_command_result(
            writer,
            ok=False,
            message=f"unknown command: {cmd!r}",
        )

    @staticmethod
    async def _write_command_result(
        writer: asyncio.StreamWriter,
        *,
        ok: bool,
        message: str,
    ) -> None:
        payload = (
            json.dumps(
                {"type": "command_result", "ok": ok, "message": message},
                ensure_ascii=False,
            )
            + "\n"
        )
        writer.write(payload.encode("utf-8"))
        await writer.drain()

    async def _on_response(self, msg: OutboundMessage) -> None:
        writer = self._writers.get(msg.chat_id)
        if writer and not writer.is_closing():
            payload = (
                json.dumps(
                    {
                        "type": "assistant",
                        "content": msg.content,
                        "metadata": msg.metadata or {},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            writer.write(payload.encode("utf-8"))
            await writer.drain()
