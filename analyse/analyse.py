from .assemble import Assemble
from .demographics import Demographic
from .detect import Detect
from .events import Event
from .initialise import initialise
from .load import load, load_package, load_selected
from .send import Send
from .track import Track
from .update import update

def Analyse(device_id: int) -> bool:

    try:
        context = initialise(device_id)
        source = load(context)
        if not source["packages"]:
            return True

        detect = Detect()
        detections = []
        for package in source["packages"]:
            frame_batch = load_package(source, package)
            detections.extend(detect(frame_batch)["detections"])
            del frame_batch

        detection_batch = {"detections": detections}
        track_batch = Track()(detection_batch)
        event_batch = Event()(track_batch, context["line_config"])
        if not event_batch["events"]:
            update(context["device_id"], source["consumed_until"])
            return True

        selected_frames = load_selected(
            source,
            {event["best_crop"]["frame_id"] for event in event_batch["events"]},
        )
        demographics_batch = Demographic()(event_batch, selected_frames)
        output_batch = Assemble()(
            event_batch,
            demographics_batch,
            source["source_origin"],
            context["organisation_id"],
            context["site_id"],
            context["device_id"],
        )
        del detection_batch, track_batch, event_batch, demographics_batch, selected_frames
        Send()(output_batch)
        update(context["device_id"], source["consumed_until"])
        return True
    except Exception as exc:
        print(f"Analyse failed for device {device_id}: {exc}")
        return False
