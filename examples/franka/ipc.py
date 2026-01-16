"""Lightweight IPC helpers (length-prefixed msgpack over TCP)."""

from __future__ import annotations

import socket
import struct
from typing import Any

import msgpack

_HEADER = struct.Struct("!I")  # uint32 length prefix


def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = num_bytes
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while reading message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_msg(sock: socket.socket, payload: dict[str, Any]) -> None:
    data = msgpack.packb(payload, use_bin_type=True)
    sock.sendall(_HEADER.pack(len(data)))
    sock.sendall(data)


def recv_msg(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exact(sock, _HEADER.size)
    (length,) = _HEADER.unpack(header)
    data = _recv_exact(sock, length)
    return msgpack.unpackb(data, raw=False)
