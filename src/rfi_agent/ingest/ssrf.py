from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

BLOCKED_HOSTS = {"localhost", "metadata.google.internal"}


def host_on_allowlist(host: str, allowlist: str) -> bool:
    items = [item.strip().lower().lstrip(".") for item in (allowlist or "").split(",") if item.strip()]
    if not items:
        return True
    host = host.lower()
    return any(host == item or host.endswith("." + item) for item in items)


class UnsafeURL(ValueError):
    pass


def assert_public_https(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise UnsafeURL("Only https URLs are allowed.")
    host = parsed.hostname
    if not host:
        raise UnsafeURL("URL host is missing.")
    if host.lower() in BLOCKED_HOSTS or host.endswith(".localhost"):
        raise UnsafeURL("Host is not allowed.")
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeURL(f"Could not resolve host: {host}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or             ip.is_unspecified
        ):
            raise UnsafeURL("URL resolves to a private or reserved address.")
    from rfi_agent.config import get_settings

    if not host_on_allowlist(host, get_settings().fetch_allowlist):
        raise UnsafeURL(f"Host {host} is not on FETCH_ALLOWLIST.")
    return url
