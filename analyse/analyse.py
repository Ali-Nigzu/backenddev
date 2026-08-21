"""Production pipeline orchestration for one device."""

from __future__ import annotations

from .assemble import Assemble
from .demographics import Demographic
from .detect import Detect
from .events import Event
from .initialise import initialise
from .load import load
from .send import Send
from .track import Track
from .update import update


def Analyse(device_id: int) -> bool:
    """Run the complete production pipeline for one device."""
    try:
        context = initialise(device_id)
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
        return True
    except Exception as exc:
        print(f"Analyse failed for device {device_id}: {exc}")
        return False
