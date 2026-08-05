"""Load timestamp-named JPG frames from Google Cloud Storage."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath
from urllib.parse import urlparse

import cv2
import numpy as np
from google.cloud import storage
from google.oauth2 import service_account

FRAME_FILENAME_FORMAT = "%Y-%m-%dT%H-%M-%S.%fZ.jpg"
TIMEFRAME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def load(
    source_uri: str,
    timeframe: dict,
    service_account_info: dict,
) -> dict:
    """Return a FrameBatch for timestamp-named JPG frames in a GCS prefix and timeframe."""
    bucket_name, object_prefix = _parse_gcs_uri(source_uri)

    timeframe_start = _parse_utc_timestamp(timeframe["start"])
    timeframe_end = _parse_utc_timestamp(timeframe["end"])

    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    client = storage.Client(
        credentials=credentials,
        project=service_account_info.get("project_id"),
    )

    selected_blobs = []
    for blob in client.list_blobs(bucket_name, prefix=object_prefix):
        capture_time = _parse_frame_capture_time(blob.name)
        if capture_time is None:
            continue
        if timeframe_start <= capture_time < timeframe_end:
            selected_blobs.append((capture_time, blob))

    selected_blobs.sort(key=lambda item: (item[0], item[1].name))

    frames = []
    for capture_time, blob in selected_blobs:
        image_bytes = blob.download_as_bytes()
        image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr_image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
        if bgr_image is None:
            raise ValueError(f"Could not decode selected JPG frame: {blob.name}")
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        frames.append(
            {
                "frame_id": blob.name,
                "timestamp": float((capture_time - timeframe_start).total_seconds()),
                "image": rgb_image,
            }
        )

    return {"frames": frames}


def _parse_gcs_uri(source_uri: str) -> tuple[str, str]:
    parsed_uri = urlparse(source_uri)
    if parsed_uri.scheme != "gs" or not parsed_uri.netloc:
        raise ValueError(f"Expected canonical gs:// GCS URI: {source_uri}")
    return parsed_uri.netloc, parsed_uri.path.lstrip("/")


def _parse_utc_timestamp(value: str) -> datetime:
    return datetime.strptime(value, TIMEFRAME_FORMAT).replace(tzinfo=timezone.utc)


def _parse_frame_capture_time(object_name: str) -> datetime | None:
    basename = PurePosixPath(object_name).name
    try:
        return datetime.strptime(basename, FRAME_FILENAME_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
