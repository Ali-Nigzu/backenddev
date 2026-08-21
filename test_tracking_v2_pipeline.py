"""Run the real production pipeline for one configured device."""

from __future__ import annotations

import json

from assemble import Assemble
from demographics import Demographic
from detect import Detect
from events import Event
from initialise import initialise
from load.load import load
from send import Send
from track import Track
from update import update
from visualise import Visualise

DEVICE_ID = 1


def main() -> None:
    context = initialise(DEVICE_ID)
    frame_batch = load(
        context["gcs_source_uri"],
        context["timeframe"],
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
