"""Organisation-scoped Snapshot orchestration."""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

from . import organisation_engine as organisation
from . import site_engine as site_engine
from .models import AttemptStats, SnapshotCandidate, SourceRange
from .source import fetch_events
from .storage import ConcurrentSnapshotUpdate, connection, credentials, load_context, persist

MAX_RETRIES = 2
_DESTINATION = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+$")
_LINK = re.compile(r"(?:^|!)1s(?P<project>[A-Za-z0-9_-]+)!2s(?P<dataset>[A-Za-z0-9_]+)!3s(?P<table>[A-Za-z0-9_]+)(?:!|$)")


def destination(value):
    if _DESTINATION.fullmatch(value):
        return value
    workspace = parse_qs(urlparse(value).query).get("ws", [""])[0]
    match = _LINK.search(workspace)
    if not match:
        raise ValueError("Invalid BigQuery destination")
    return ".".join(match.group("project", "dataset", "table"))


@dataclass
class Resources:
    credentials: object
    connector: object
    bq_client: object


def _resources():
    from google.cloud import bigquery
    from google.cloud.sql.connector import Connector
    creds = credentials()
    return Resources(creds, Connector(credentials=creds), bigquery.Client(credentials=creds, project=creds.project_id))


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0)


def _site_plan(context, snapshot_now):
    classifications, previous, horizons = {}, {}, {}
    for site in context["sites"]:
        site_id = site["id"]
        devices = context["devices_by_site"][site_id]
        horizon = site_engine.site_horizons(site, devices)
        horizons[site_id] = None if horizon is None else {"stable_until": horizon[1]}
        classifications[site_id], previous[site_id] = site_engine.classify_site(
            site, devices, context["site_snapshots"][site_id], snapshot_now)
    return classifications, previous, horizons


def _org_classification(context, horizons, snapshot_now):
    row = context["organisation_snapshot"]
    state = organisation.validate_state(row["state"], row["ts"], context["sites"], context["devices_by_site"])
    if state is None:
        return "REBUILD", None
    current_membership = organisation.membership(context["sites"], [key for key, value in horizons.items() if value])
    if state["membership"] != current_membership:
        return "REBUILD", state
    current_devices = organisation.device_watermarks(context["sites"], context["devices_by_site"], snapshot_now)
    if set(current_devices) != set(state["device_watermarks"]):
        return "REBUILD", state
    metadata_only = organisation.metadata(context["sites"], context["devices_by_site"]) != state["metadata"]
    for key, value in current_devices.items():
        old = state["device_watermarks"][key]
        if any(old.get(field) != value.get(field) for field in ("site_id", "status", "created_at", "destination", "analysis_config")):
            return "REBUILD", state
        if old.get("analyzed_until") and (not value.get("analyzed_until") or site_engine.parse_ts(value["analyzed_until"]) < site_engine.parse_ts(old["analyzed_until"])):
            return "REBUILD", state
    source_changed = any(state["device_watermarks"][key].get("source_horizon") != value.get("source_horizon")
                         for key, value in current_devices.items())
    if source_changed:
        return "INCREMENTAL", state
    effective_stable = organisation.organisation_horizons(horizons, snapshot_now)
    if effective_stable > site_engine.parse_ts(state["stable_until"]):
        return "PROMOTE_ONLY", state
    if any(site_engine.parse_ts(event["timestamp"]) < snapshot_now
           for event in state["pending_events"]):
        return "PROMOTE_ONLY", state
    if current_devices != state["device_watermarks"] or metadata_only:
        return "METADATA_ONLY", state
    if snapshot_now != site_engine.parse_ts(row["ts"]):
        return "TIME_ONLY", state
    return "NO_OP", state


def _ranges(context, classifications, previous, org_classification, org_previous, horizons):
    ranges = []
    for site in context["sites"]:
        site_id = site["id"]
        if horizons[site_id] is None:
            continue
        classification = classifications[site_id]
        if org_classification == "REBUILD" or classification == "REBUILD":
            start = site_engine.parse_ts(site["created_at"])
        elif classification == "INCREMENTAL":
            start = None  # resolved per device from its previously consumed horizon
        else:
            continue
        for device in context["devices_by_site"][site_id]:
            if device["status"] != "enabled":
                continue
            end = site_engine.device_horizon(device)
            if classification == "INCREMENTAL" and org_classification != "REBUILD":
                old = previous[site_id]["device_watermarks"][str(device["id"])]
                old_horizon = site_engine.parse_ts(old.get("source_horizon", old["analyzed_until"] or old["created_at"]))
                device_start = max(old_horizon, site_engine.parse_ts(device["created_at"]))
            else:
                device_start = max(start, site_engine.parse_ts(device["created_at"]))
            if device_start < end:
                ranges.append(SourceRange(site["destination"], site_id, device["id"], device_start, end))
    return ranges


def _run_attempt(resources, sql_connection, organisation_id, stats):
    snapshot_now = _now()
    context = load_context(sql_connection, organisation_id, destination)
    context["snapshot_now"] = snapshot_now
    # End the read transaction before potentially long BigQuery/reducer work.  The
    # write transaction starts later with explicit configuration revalidation.
    sql_connection.commit()
    classifications, previous, horizons = _site_plan(context, snapshot_now)
    # Raises the required domain error before any BigQuery work.
    organisation.organisation_horizons(horizons)
    org_classification, org_previous = _org_classification(context, horizons, snapshot_now)
    for value in classifications.values():
        stats.classifications[value] = stats.classifications.get(value, 0) + 1
    ranges = _ranges(context, classifications, previous, org_classification, org_previous, horizons)
    events = fetch_events(resources.bq_client, ranges, stats) if ranges else []
    events_by_site = {site["id"]: [] for site in context["sites"]}
    for event in events:
        events_by_site[event["site_id"]].append(event)
    site_candidates = {}
    for site in context["sites"]:
        site_id = site["id"]
        classification = classifications[site_id]
        if classification == "NO_OP":
            continue
        result = site_engine.compute_site(site, context["devices_by_site"][site_id],
            context["site_snapshots"][site_id], classification, previous[site_id],
            events_by_site[site_id], snapshot_now)
        site_candidates[site_id] = SnapshotCandidate(*result)
    stats.changed_sites = len(site_candidates)
    org_candidate = None
    if org_classification != "NO_OP":
        result = organisation.compute(context["sites"], context["devices_by_site"], horizons,
                                      org_previous, context["organisation_snapshot"]["ts"], events,
                                      org_classification, snapshot_now)
        org_candidate = SnapshotCandidate(*result)
    if not site_candidates and org_candidate is None:
        return True
    persist(sql_connection, context, site_candidates, org_candidate, destination)
    return True


def Snapshot(organisation_id):
    """Advance all required site snapshots and one organisation snapshot atomically."""
    resources = _resources()
    try:
        with connection(resources) as sql_connection:
            for attempt in range(MAX_RETRIES + 1):
                stats = AttemptStats()
                try:
                    return _run_attempt(resources, sql_connection, organisation_id, stats)
                except ConcurrentSnapshotUpdate:
                    sql_connection.rollback()
                    if attempt == MAX_RETRIES:
                        raise RuntimeError(f"Snapshot update repeatedly conflicted for organisation_id={organisation_id}") from None
    finally:
        resources.connector.close()
    raise AssertionError("unreachable")
