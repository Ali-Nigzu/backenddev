"""Advance a device analysis watermark after a successful send."""

from __future__ import annotations

from cloud_sql import cloud_sql_connection

_UPDATE_ANALYZED_UNTIL = """
UPDATE devices
SET analyzed_until = %s
WHERE id = %s
"""


def update(device_id: int, analyzed_until: str) -> None:
    """Set a device's analyzed-until watermark to the caller-supplied timestamp."""
    try:
        with cloud_sql_connection() as connection:
            cursor = connection.cursor()
            try:
                cursor.execute(_UPDATE_ANALYZED_UNTIL, (analyzed_until, device_id))
                if cursor.rowcount != 1:
                    raise ValueError(f"Device not found: {device_id}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                cursor.close()
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Unable to update analyzed_until for device {device_id}"
        ) from exc
