"""Run the real production pipeline for one configured device."""

from __future__ import annotations

import json
import os
from pathlib import Path

from assemble import Assemble
from demographics import Demographic
from detect import Detect
from events import Event
from initialise import initialise
from load.load import load
from send import Send
from track import Track
from update import update
from visualize import Visualize

SERVICE_ACCOUNT_PATH = Path(__file__).resolve().parent / "TestAdminSA.json"
DEVICE_ID = 1


def main() -> None:
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(SERVICE_ACCOUNT_PATH))
    with SERVICE_ACCOUNT_PATH.open(encoding="utf-8") as file:
        service_account_info = json.load(file)

    context = initialise(DEVICE_ID)
    frame_batch = load(
        context["gcs_source_uri"],
        context["timeframe"],
        service_account_info,
    )
    detection_batch = Detect()(frame_batch)
    track_batch = Track()(detection_batch)
    event_batch = Event()(track_batch, context["analysis_config"])
    demographics_batch = Demographic()(event_batch, frame_batch)
    output_batch = Assemble()(
        event_batch,
        demographics_batch,
        context["timeframe"]["start"],
        context["device_id"],
    )
    Send()(output_batch, context["bigquery_destination"])
    update(context["device_id"], context["timeframe"]["end"])
    print(json.dumps(output_batch, sort_keys=True))


if __name__ == "__main__":
    main()

    # OPTIONAL LOCAL VISUALIZATION
    # Comment out main() above when using this.
    #
    # Visualize(
    #     1,
    #     {
    #         "start": "2026-08-20T19:00:00.000Z",
    #         "end": "2026-08-20T19:05:00.000Z",
    #     },
    # )
