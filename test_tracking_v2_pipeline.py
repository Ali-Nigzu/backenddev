"""Run the real production pipeline for one configured device."""

from __future__ import annotations

from analyse import Analyse
from analyse.visualise import Visualise

DEVICE_ID = 1


def main() -> None:
    success = Analyse(DEVICE_ID)
    print(success)


if __name__ == "__main__":
    main()

    # OPTIONAL LOCAL VISUALISATION
    # Comment out main() above when using this.
    #
    # Visualise(
    #     1,
    #     {
    #         "start": "2026-08-20T19:00:00.000Z",
    #         "end": "2026-08-20T19:05:00.000Z",
    #     },
    # )
