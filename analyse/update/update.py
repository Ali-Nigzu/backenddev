from ..initialise.cloud_sql import cloud_sql_connection

_UPDATE_ANALYZED_UNTIL = """
UPDATE devices
SET analyzed_until = %s
WHERE id = %s
"""

def update(device_id: int, analyzed_until: str) -> None:

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
