"""Print the URLs a phone on the same Wi-Fi can use to reach this computer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.network import lan_urls  # noqa: E402


def main() -> int:
    urls = lan_urls(8000)
    if not urls:
        print("  No LAN address found. Connect this computer to Wi-Fi or Ethernet.")
        return 0
    for url in urls:
        print(f"  Backend from phone: {url}")
        print(f"  Web app from phone: {url.replace(':8000', ':3000')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
