import os
from pathlib import Path

from .cloud_sql import cloud_sql_connection

SA_PATH = Path(__file__).resolve().parent / "SA.json"
_DEVICE_CONTEXT_QUERY = """
SELECT
    devices.id,
    devices.site_id,
    sites.organisation_id,
    devices.created_at,
    devices.analyzed_until,
    devices.frame_package_interval_minutes,
    devices.line_ax,
    devices.line_ay,
    devices.line_bx,
    devices.line_by
FROM public.devices AS devices
JOIN public.sites AS sites ON sites.id = devices.site_id
WHERE devices.id = %s
"""

def initialise(device_id: int) -> dict:

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(SA_PATH)

    try:
        with cloud_sql_connection() as connection:
            cursor = connection.cursor()
            try:
                cursor.execute(_DEVICE_CONTEXT_QUERY, (device_id,))
                row = cursor.fetchone()
            finally:
                cursor.close()
    except Exception as exc:
        raise RuntimeError(f"Unable to read processing context for device {device_id}") from exc

    if row is None:
        raise ValueError(f"Device not found: {device_id}")

    (
        resolved_device_id,
        site_id,
        organisation_id,
        created_at,
        analyzed_until,
        frame_package_interval_minutes,
        line_ax,
        line_ay,
        line_bx,
        line_by,
    ) = row
    return {
        "device_id": resolved_device_id,
        "site_id": site_id,
        "organisation_id": organisation_id,
        "created_at": created_at,
        "analyzed_until": analyzed_until,
        "frame_package_interval_minutes": frame_package_interval_minutes,
        "line_config": {
            "point_a": {"x": line_ax, "y": line_ay},
            "point_b": {"x": line_bx, "y": line_by},
        },
    }
