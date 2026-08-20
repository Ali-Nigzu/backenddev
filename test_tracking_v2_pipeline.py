"""Run a manually selected local pipeline visualization."""

from __future__ import annotations

import os
from pathlib import Path

from visualize import Visualize

SERVICE_ACCOUNT_PATH = Path(__file__).resolve().parent / "TestAdminSA.json"
DEVICE_ID = 1
TIMEFRAME = {
    "start": "2026-08-20T19:00:00.000Z",
    "end": "2026-08-20T19:05:00.000Z",
}


def main() -> None:
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(SERVICE_ACCOUNT_PATH))
    Visualize(DEVICE_ID, TIMEFRAME)


if __name__ == "__main__":
    main()
