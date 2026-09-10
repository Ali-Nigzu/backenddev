from datetime import datetime, timezone
from io import BytesIO
from pathlib import PurePosixPath
import tarfile

import cv2
import numpy as np
from google.cloud import storage


BUCKET_NAME = "camos-prod-0"
SOURCE_TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S.%fZ"


def _source_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime(SOURCE_TIMESTAMP_FORMAT)[:-4] + "Z"


def _parse_source_timestamp(value: str) -> datetime:
    return datetime.strptime(value, SOURCE_TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)


def _window_start(value: datetime, interval_minutes: int) -> datetime:
    interval_seconds = interval_minutes * 60
    seconds = int(value.timestamp())
    return datetime.fromtimestamp(
        seconds - seconds % interval_seconds,
        tz=timezone.utc,
    )


def _package_times(object_name: str):
    start, end = PurePosixPath(object_name).name.removesuffix(".tar").split("__")
    return _parse_source_timestamp(start), _parse_source_timestamp(end)


def _capture_time(member_name: str) -> datetime:
    value = PurePosixPath(member_name).name.removesuffix(".jpg").split("__", 1)[0]
    return _parse_source_timestamp(value)


def _decode(image_bytes: bytes):
    bgr_image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def load(context: dict) -> dict:
    prefix = (
        f'{context["organisation_id"]}/{context["site_id"]}/'
        f'{context["device_id"]}/'
    )
    analyzed_until = context["analyzed_until"]
    source_start = analyzed_until or context["created_at"]
    listing_start = analyzed_until or _window_start(
        source_start,
        context["frame_package_interval_minutes"],
    )
    client = storage.Client()
    packages = []
    for blob in client.list_blobs(
        BUCKET_NAME,
        prefix=prefix,
        start_offset=f"{prefix}{_source_timestamp(listing_start)}",
    ):
        window_start, window_end = _package_times(blob.name)
        if analyzed_until is not None or window_end > source_start:
            packages.append(
                {
                    "blob": blob,
                    "object_name": blob.name,
                    "start": window_start,
                    "end": window_end,
                }
            )
    packages.sort(key=lambda package: (package["start"], package["object_name"]))
    return {
        "packages": packages,
        "source_origin": packages[0]["start"] if packages else None,
        "consumed_until": packages[-1]["end"] if packages else None,
        "frames": {},
    }


def load_package(source: dict, package: dict) -> dict:
    frames = []
    with tarfile.open(
        fileobj=BytesIO(package["blob"].download_as_bytes()),
        mode="r:",
    ) as archive:
        for member in archive:
            image_bytes = archive.extractfile(member).read()
            frame_id = f'{package["object_name"]}/{member.name}'
            source["frames"][frame_id] = (package, member.name)
            frames.append(
                {
                    "frame_id": frame_id,
                    "timestamp": float(
                        (_capture_time(member.name) - source["source_origin"]).total_seconds()
                    ),
                    "image": _decode(image_bytes),
                }
            )
    return {"frames": frames}


def load_selected(source: dict, frame_ids) -> dict:
    selected_by_package = {}
    for frame_id in frame_ids:
        package, member_name = source["frames"][frame_id]
        selected_by_package.setdefault(package["object_name"], (package, set()))[1].add(
            member_name
        )

    frames = []
    for package, selected_members in selected_by_package.values():
        with tarfile.open(
            fileobj=BytesIO(package["blob"].download_as_bytes()),
            mode="r:",
        ) as archive:
            for member in archive:
                if member.name in selected_members:
                    frame_id = f'{package["object_name"]}/{member.name}'
                    frames.append(
                        {
                            "frame_id": frame_id,
                            "timestamp": float(
                                (
                                    _capture_time(member.name)
                                    - source["source_origin"]
                                ).total_seconds()
                            ),
                            "image": _decode(archive.extractfile(member).read()),
                        }
                    )
    return {"frames": frames}
