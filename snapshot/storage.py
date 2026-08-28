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
    path = Path(__file__).resolve().parent.parent / "sa.json"
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot service account file not found: {path}")
    return service_account.Credentials.from_service_account_file(str(path))


@contextmanager
def connection(resources):
    email = resources.credentials.service_account_email
    if not email.endswith(".gserviceaccount.com"):
        raise RuntimeError("Snapshot credentials must be a service account")
    value = resources.connector.connect(INSTANCE, "pg8000", user=email.removesuffix(".gserviceaccount.com"),
                                        db=DATABASE, enable_iam_auth=True)
    try:
        yield value
    finally:
        value.close()


def _json(value, field):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"Snapshot {field} must be a JSON object")
    return value


def _status(value, entity, entity_id):
    if value not in ("enabled", "disabled"):
        raise ValueError(f"{entity} {entity_id} has invalid status: {value!r}")
    return value


def load_context(connection, organisation_id, destination_parser):
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT id, name, status, updated_at FROM public.organisations WHERE id=%s", (organisation_id,))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Snapshot organisation not found: {organisation_id}")
        organisation = {"id": int(row[0]), "name": row[1],
                        "status": _status(row[2], "Organisation", row[0]), "updated_at": row[3]}
        cursor.execute("SELECT id,name,organisation_id,bigquery_destination,max_capacity,status,created_at,updated_at FROM public.sites WHERE organisation_id=%s ORDER BY id", (organisation_id,))
        sites = []
        for row in cursor.fetchall():
            if row[4] is None or row[4] <= 0:
                raise ValueError(f"Site {row[0]} max_capacity must be positive")
            sites.append({"id": int(row[0]), "name": row[1], "organisation_id": int(row[2]),
                          "bigquery_destination": row[3], "destination": destination_parser(row[3]),
                          "max_capacity": int(row[4]), "status": _status(row[5], "Site", row[0]),
                          "created_at": stamp(row[6]), "updated_at": stamp(row[7])})
        site_ids = [site["id"] for site in sites]
        devices_by_site = {site_id: [] for site_id in site_ids}
        if site_ids:
            cursor.execute("SELECT id,name,site_id,status,analysis_config,analyzed_until,created_at,updated_at FROM public.devices WHERE site_id = ANY(%s) ORDER BY site_id,id", (site_ids,))
            for row in cursor.fetchall():
                config = json.loads(row[4]) if isinstance(row[4], str) else row[4]
                devices_by_site[int(row[2])].append({"id": int(row[0]), "name": row[1], "site_id": int(row[2]),
                    "status": _status(row[3], "Device", row[0]), "analysis_config": config, "analyzed_until": None if row[5] is None else stamp(row[5]),
                    "created_at": stamp(row[6]), "updated_at": stamp(row[7])})
            cursor.execute("SELECT site_id,ts,payload,state,updated_at FROM public.site_snapshots WHERE site_id = ANY(%s)", (site_ids,))
            site_rows = {int(row[0]): {"site_id": int(row[0]), "ts": stamp(row[1]), "payload": _json(row[2], "payload"),
                "state": _json(row[3], "state"), "version": row[4]} for row in cursor.fetchall()}
        else:
            site_rows = {}
        missing = sorted(set(site_ids) - set(site_rows))
        if missing:
            raise ValueError(f"Missing provisioned site_snapshots rows: {missing}")
        cursor.execute("SELECT organisation_id,ts,payload,state,updated_at FROM public.organisation_snapshots WHERE organisation_id=%s", (organisation_id,))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Missing provisioned organisation_snapshots row: {organisation_id}")
        org_row = {"organisation_id": int(row[0]), "ts": stamp(row[1]), "payload": _json(row[2], "payload"),
                   "state": _json(row[3], "state"), "version": row[4]}
        return {"organisation": organisation, "sites": sites, "devices_by_site": devices_by_site,
                "site_snapshots": site_rows, "organisation_snapshot": org_row}
    finally:
        cursor.close()


def membership_fingerprint(context):
    return (
        (context["organisation"]["id"], context["organisation"]["status"]),
        tuple((site["id"], site["organisation_id"], site["status"], site["destination"], site["created_at"], site["max_capacity"])
              for site in context["sites"]),
        tuple((site_id, device["id"], device["status"], device["created_at"], repr(device["analysis_config"]))
              for site_id in sorted(context["devices_by_site"]) for device in context["devices_by_site"][site_id]),
    )


def persist(connection, context, site_candidates, org_candidate, destination_parser):
    # Re-read configuration, excluding watermarks so later Analyse progress is allowed.
    fresh = load_context(connection, context["organisation"]["id"], destination_parser)
    if membership_fingerprint(fresh) != membership_fingerprint(context):
        raise ConcurrentSnapshotUpdate("Organisation membership/configuration changed")
    cursor = connection.cursor()
    try:
        for site_id, candidate in sorted(site_candidates.items()):
            cursor.execute("UPDATE public.site_snapshots SET ts=%s,payload=%s::jsonb,state=%s::jsonb,updated_at=CURRENT_TIMESTAMP WHERE site_id=%s AND updated_at=%s RETURNING updated_at",
                           (candidate.ts, json.dumps(candidate.payload), json.dumps(candidate.state), site_id,
                            context["site_snapshots"][site_id]["version"]))
            if cursor.fetchone() is None:
                raise ConcurrentSnapshotUpdate(f"Concurrent site snapshot update: {site_id}")
        if org_candidate is not None:
            cursor.execute("UPDATE public.organisation_snapshots SET ts=%s,payload=%s::jsonb,state=%s::jsonb,updated_at=CURRENT_TIMESTAMP WHERE organisation_id=%s AND updated_at=%s RETURNING updated_at",
                           (org_candidate.ts, json.dumps(org_candidate.payload), json.dumps(org_candidate.state),
                            context["organisation"]["id"], context["organisation_snapshot"]["version"]))
            if cursor.fetchone() is None:
                raise ConcurrentSnapshotUpdate("Concurrent organisation snapshot update")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
