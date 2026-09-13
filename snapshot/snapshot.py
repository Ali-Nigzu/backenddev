"""Organisation-scoped Snapshot orchestration."""

from datetime import datetime, timezone

from . import organisation_engine as organisation
from . import site_engine as site_engine
from .source import SourceRange, fetch_events
from .storage import ConcurrentSnapshotUpdate, connection, credentials, load_context, persist

MAX_RETRIES = 2


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0)


def _site_plan(context, snapshot_now):
    classifications, previous, horizons = {}, {}, {}
    for site in context["sites"]:
        site_id = site["id"]
        devices = context["devices_by_site"][site_id]
        horizon = site_engine.site_horizons(site, devices, snapshot_now)
        horizons[site_id] = None if horizon is None else {"stable_until": horizon[1]}
        classifications[site_id], previous[site_id] = site_engine.classify_site(
            site, devices, context["site_snapshots"][site_id], snapshot_now)
    return classifications, previous, horizons


def _org_classification(context, horizons, snapshot_now):
    row = context["organisation_snapshot"]
    state = row["state"]
    if state == {}:
        return "REBUILD", None
    latest = site_engine.parse_ts(row["ts"])
    current_membership = organisation.membership(context["sites"])
    if state["membership"].get("site_ids") != current_membership["site_ids"]:
        return "REBUILD", state
    current_devices = organisation.device_watermarks(
        context["devices_by_site"], snapshot_now
    )
    if set(current_devices) != set(state["device_watermarks"]):
        return "REBUILD", state
    current_metadata = organisation.metadata(
        context["sites"], context["organisation"]["enabled"]
    )
    metadata_only = current_metadata != state["metadata"]
    old_site_enabled = state["metadata"].get("site_enabled", {})
    new_site_enabled = current_metadata["site_enabled"]
    retired = any(old_site_enabled.get(key) and not value
                  for key, value in new_site_enabled.items())
    activated = any(not old_site_enabled.get(key) and value
                    for key, value in new_site_enabled.items())
    for key, value in current_devices.items():
        old = state["device_watermarks"][key]
        if any(old.get(field) != value.get(field) for field in ("site_id", "created_at")):
            return "REBUILD", state
        retired = retired or old.get("enabled") and not value.get("enabled")
        activated = activated or not old.get("enabled") and value.get("enabled")
        if site_engine.parse_ts(value["consumed_until"]) < site_engine.parse_ts(old["consumed_until"]):
            return "REBUILD", state
    if activated:
        if any(not state["device_watermarks"][key].get("enabled")
               and value.get("enabled")
               and site_engine.parse_ts(state["device_watermarks"][key]["consumed_until"])
                   < site_engine.parse_ts(state["stable_until"])
               for key, value in current_devices.items()):
            return "REBUILD", state
        stable_candidate = organisation.organisation_horizons(horizons, snapshot_now)
        if stable_candidate < site_engine.parse_ts(state["stable_until"]):
            return "REBUILD", state
        return "SOURCE_ACTIVATION", state
    source_changed = any(state["device_watermarks"][key].get("consumed_until") != value.get("consumed_until")
                         for key, value in current_devices.items())
    if retired:
        return "SOURCE_RETIREMENT", state
    if source_changed:
        prior_stable = site_engine.parse_ts(state["stable_until"])
        if any(site_engine.parse_ts(state["device_watermarks"][key]["consumed_until"]) < prior_stable
               and site_engine.parse_ts(value["consumed_until"]) >
                   site_engine.parse_ts(state["device_watermarks"][key]["consumed_until"])
               for key, value in current_devices.items()):
            return "REBUILD", state
        return "INCREMENTAL", state
    stable_candidate = organisation.organisation_horizons(horizons, snapshot_now)
    if stable_candidate > site_engine.parse_ts(state["stable_until"]):
        return "PROMOTE_ONLY", state
    if current_devices != state["device_watermarks"] or metadata_only:
        return "METADATA_ONLY", state
    if snapshot_now != site_engine.parse_ts(row["ts"]):
        return "TIME_ONLY", state
    return "NO_OP", state


def _ranges(context, classifications, previous, org_classification, snapshot_now):
    ranges = []
    for site in context["sites"]:
        site_id = site["id"]
        classification = classifications[site_id]
        if org_classification == "REBUILD" or classification == "REBUILD":
            start = site_engine.parse_ts(site["created_at"])
        elif classification in ("INCREMENTAL", "SOURCE_RETIREMENT", "SOURCE_ACTIVATION"):
            start = None  # resolved per device from its previously consumed horizon
        else:
            continue
        for device in context["devices_by_site"][site_id]:
            end = site_engine.device_horizon(device, snapshot_now)
            if start is None and org_classification != "REBUILD":
                old = previous[site_id]["device_watermarks"][str(device["id"])]
                device_start = max(
                    site_engine.parse_ts(old["consumed_until"]),
                    site_engine.parse_ts(device["created_at"]),
                )
                active = site["enabled"] and device["enabled"]
                terminal_delta = not active and device_start < end
                if not active and not terminal_delta:
                    continue
            else:
                device_start = max(start, site_engine.parse_ts(device["created_at"]))
            if device_start < end:
                ranges.append(SourceRange(
                    context["organisation"]["id"], site_id, device["id"],
                    device_start, end,
                ))
    return ranges


def _run_attempt(sql_connection, organisation_id, bq_client, snapshot_now):
    context = load_context(sql_connection, organisation_id)
    # End the read transaction before potentially long BigQuery/reducer work.  The
    # write transaction starts later with explicit configuration revalidation.
    sql_connection.commit()
    classifications, previous, horizons = _site_plan(context, snapshot_now)
    # Raises the required domain error before any BigQuery work.
    organisation.organisation_horizons(horizons, snapshot_now)
    org_classification, org_previous = _org_classification(context, horizons, snapshot_now)
    ranges = _ranges(
        context, classifications, previous, org_classification, snapshot_now
    )
    events = fetch_events(bq_client(), ranges) if ranges else []
    events_by_site = {site["id"]: [] for site in context["sites"]}
    for event in events:
        events_by_site[event["site_id"]].append(event)
    site_candidates = {}
    for site in context["sites"]:
        site_id = site["id"]
        classification = classifications[site_id]
        if classification == "NO_OP":
            continue
        site_candidates[site_id] = site_engine.compute_site(site, context["devices_by_site"][site_id],
            context["site_snapshots"][site_id], classification, previous[site_id],
            events_by_site[site_id], snapshot_now)
    org_candidate = None
    if org_classification != "NO_OP":
        org_candidate = organisation.compute(
            context["sites"], context["devices_by_site"], horizons, org_previous,
            context["organisation_snapshot"]["ts"], events, org_classification,
            snapshot_now, context["organisation"]["enabled"])
    if not site_candidates and org_candidate is None:
        return True
    persist(sql_connection, context, site_candidates, org_candidate)
    return True


def Snapshot(organisation_id):
    """Advance all required site snapshots and one organisation snapshot atomically."""
    from google.cloud.sql.connector import Connector

    snapshot_now = _now()
    snapshot_credentials = credentials()
    connector = Connector(credentials=snapshot_credentials)
    client = None

    def bq_client():
        nonlocal client
        if client is None:
            from google.cloud import bigquery
            client = bigquery.Client(credentials=snapshot_credentials,
                                     project=snapshot_credentials.project_id)
        return client

    try:
        with connection(connector, snapshot_credentials) as sql_connection:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    return _run_attempt(
                        sql_connection, organisation_id, bq_client, snapshot_now
                    )
                except ConcurrentSnapshotUpdate:
                    sql_connection.rollback()
                    if attempt == MAX_RETRIES:
                        raise RuntimeError(f"Snapshot update repeatedly conflicted for organisation_id={organisation_id}") from None
    finally:
        connector.close()
