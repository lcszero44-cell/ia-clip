from __future__ import annotations

import logging

from common import ensure_directories
from queue_service import run_worker


def main() -> int:
    ensure_directories()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return run_worker(idle_seconds=10)


if __name__ == "__main__":
    raise SystemExit(main())
