"""LAN discovery helpers so the desktop app and /health report reachable URLs."""
import ipaddress
import socket


def _preference(address: str) -> tuple[int, str]:
    """Prefer normal home/office Wi-Fi ranges over virtual adapters."""
    if address.startswith("192.168."):
        return (0, address)
    if address.startswith("10."):
        return (1, address)
    return (2, address)


def lan_urls(port: int) -> list[str]:
    addresses: set[str] = set()

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        addresses.add(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        probe.close()

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(info[4][0])
    except OSError:
        pass

    usable: list[str] = []
    for address in addresses:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        if parsed.is_loopback or parsed.is_link_local or not parsed.is_private:
            continue
        usable.append(address)

    return [f"http://{address}:{port}" for address in sorted(usable, key=_preference)]
