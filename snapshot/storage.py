"""Cloud SQL context and atomic optimistic Snapshot persistence."""

import json
from contextlib import contextmanager
from pathlib import Path

from .site_engine import stamp


INSTANCE = "camosbase:europe-west2:camos-prod-postgres"
DATABASE = "camos_prod"


class ConcurrentSnapshotUpdate(RuntimeError):
    pass


def credentials():
    from google.oauth2 import service_account

    path = Path(__file__).with_name("sa.json")
    return service_account.Credentials.from_service_account_file(str(path))


@contextmanager
def connection(connector, snapshot_credentials):
    email = snapshot_credentials.service_account_email
    value = connector.connect(
        INSTANCE,
        "pg8000",
        user=email.removesuffix(".gserviceaccount.com"),
        db=DATABASE,
        enable_iam_auth=True,
    )
    try:
        yield value
    finally:
        value.close()


def _json(value):
    if isinstance(value, str):
        value = json.loads(value)
    return value


def load_context(connection, organisation_id):
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT id,enabled FROM public.organisations WHERE id=%s",
            (organisation_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Snapshot organisation not found: {organisation_id}")
        organisation = {
            "id": int(row[0]),
            "enabled": bool(row[1]),
        }

        cursor.execute(
            "SELECT id,name,organisation_id,max_capacity,enabled,created_at "
            "FROM public.sites WHERE organisation_id=%s ORDER BY id",
            (organisation_id,),
        )
        sites = [
            {
                "id": int(row[0]),
                "name": row[1],
                "organisation_id": int(row[2]),
                "max_capacity": int(row[3]),
                "enabled": bool(row[4]),
                "created_at": stamp(row[5]),
            }
            for row in cursor.fetchall()
        ]
        site_ids = [site["id"] for site in sites]
        devices_by_site = {site_id: [] for site_id in site_ids}
        if site_ids:
            cursor.execute(
                "SELECT id,name,site_id,enabled,analyzed_until,created_at "
                "FROM public.devices WHERE site_id = ANY(%s) ORDER BY site_id,id",
                (site_ids,),
            )
            for row in cursor.fetchall():
                devices_by_site[int(row[2])].append(
                    {
                        "id": int(row[0]),
                        "name": row[1],
                        "site_id": int(row[2]),
                        "enabled": bool(row[3]),
                        "analyzed_until": (
                            None if row[4] is None else stamp(row[4])
                        ),
                        "created_at": stamp(row[5]),
                    }
                )
            cursor.execute(
                "SELECT site_id,ts,state,updated_at FROM public.site_snapshots "
                "WHERE site_id = ANY(%s)",
                (site_ids,),
            )
            site_rows = {
                int(row[0]): {
                    "site_id": int(row[0]),
                    "ts": stamp(row[1]),
                    "state": _json(row[2]),
                    "version": row[3],
                }
                for row in cursor.fetchall()
            }
        else:
            site_rows = {}
        missing = sorted(set(site_ids) - set(site_rows))
        if missing:
            raise ValueError(f"Missing provisioned site_snapshots rows: {missing}")

        cursor.execute(
            "SELECT organisation_id,ts,state,updated_at "
            "FROM public.organisation_snapshots WHERE organisation_id=%s",
            (organisation_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Missing provisioned organisation_snapshots row: {organisation_id}")
        org_row = {
            "organisation_id": int(row[0]),
            "ts": stamp(row[1]),
            "state": _json(row[2]),
            "version": row[3],
        }
        context = {
            "organisation": organisation,
            "sites": sites,
            "devices_by_site": devices_by_site,
            "site_snapshots": site_rows,
            "organisation_snapshot": org_row,
        }
        context["fingerprint"] = membership_fingerprint(context)
        return context
    finally:
        cursor.close()


def membership_fingerprint(context):
    return (
        (context["organisation"]["id"], context["organisation"]["enabled"]),
        tuple(
            (
                site["id"],
                site["name"],
                site["organisation_id"],
                site["enabled"],
                site["created_at"],
                site["max_capacity"],
            )
            for site in context["sites"]
        ),
        tuple(
            (
                site_id,
                device["id"],
                device["name"],
                device["enabled"],
                device["created_at"],
            )
            for site_id in sorted(context["devices_by_site"])
            for device in context["devices_by_site"][site_id]
        ),
    )


def load_fingerprint(connection, organisation_id):
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT id,enabled FROM public.organisations WHERE id=%s",
            (organisation_id,),
        )
        organisation = cursor.fetchone()
        if organisation is None:
            raise ConcurrentSnapshotUpdate(
                "Organisation membership/configuration changed"
            )
        cursor.execute(
            "SELECT id,name,organisation_id,enabled,created_at,max_capacity "
            "FROM public.sites WHERE organisation_id=%s ORDER BY id",
            (organisation_id,),
        )
        sites = [
            {
                "id": int(row[0]),
                "name": row[1],
                "organisation_id": int(row[2]),
                "enabled": bool(row[3]),
                "created_at": stamp(row[4]),
                "max_capacity": int(row[5]),
            }
            for row in cursor.fetchall()
        ]
        site_ids = [site["id"] for site in sites]
        devices_by_site = {site_id: [] for site_id in site_ids}
        if site_ids:
            cursor.execute(
                "SELECT id,name,site_id,enabled,created_at FROM public.devices "
                "WHERE site_id = ANY(%s) ORDER BY site_id,id",
                (site_ids,),
            )
            for row in cursor.fetchall():
                devices_by_site[int(row[2])].append(
                    {
                        "id": int(row[0]),
                        "name": row[1],
                        "enabled": bool(row[3]),
                        "created_at": stamp(row[4]),
                    }
                )
        return membership_fingerprint(
            {
                "organisation": {
                    "id": int(organisation[0]),
                    "enabled": bool(organisation[1]),
                },
                "sites": sites,
                "devices_by_site": devices_by_site,
            }
        )
    finally:
        cursor.close()


def persist(connection, context, site_candidates, org_candidate):
    fresh = load_fingerprint(connection, context["organisation"]["id"])
    if fresh != context["fingerprint"]:
        raise ConcurrentSnapshotUpdate("Organisation membership/configuration changed")
    cursor = connection.cursor()
    try:
        for site_id, candidate in sorted(site_candidates.items()):
            cursor.execute(
                "UPDATE public.site_snapshots "
                "SET ts=%s,payload=%s::jsonb,state=%s::jsonb,updated_at=CURRENT_TIMESTAMP "
                "WHERE site_id=%s AND updated_at=%s RETURNING updated_at",
                (
                    candidate[0],
                    json.dumps(candidate[1]),
                    json.dumps(candidate[2]),
                    site_id,
                    context["site_snapshots"][site_id]["version"],
                ),
            )
            if cursor.fetchone() is None:
                raise ConcurrentSnapshotUpdate(f"Concurrent site snapshot update: {site_id}")
        if org_candidate is not None:
            cursor.execute(
                "UPDATE public.organisation_snapshots "
                "SET ts=%s,payload=%s::jsonb,state=%s::jsonb,updated_at=CURRENT_TIMESTAMP "
                "WHERE organisation_id=%s AND updated_at=%s RETURNING updated_at",
                (
                    org_candidate[0],
                    json.dumps(org_candidate[1]),
                    json.dumps(org_candidate[2]),
                    context["organisation"]["id"],
                    context["organisation_snapshot"]["version"],
                ),
            )
            if cursor.fetchone() is None:
                raise ConcurrentSnapshotUpdate("Concurrent organisation snapshot update")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
