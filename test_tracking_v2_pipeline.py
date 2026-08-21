"""Run the real production pipeline for one configured device."""

from __future__ import annotations

import json

from analyze import Analyze
from visualise import Visualise

DEVICE_ID = 1


def main() -> None:
    output_batch = Analyze(DEVICE_ID)
    print(json.dumps(output_batch, sort_keys=True))


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
