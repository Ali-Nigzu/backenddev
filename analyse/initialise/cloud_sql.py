"""Shared Cloud SQL connection for pipeline database stages."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import google.auth
from google.cloud.sql.connector import Connector

_CLOUD_SQL_INSTANCE = "camosbase:europe-west2:camos-prod-postgres"
_CLOUD_SQL_DATABASE = "camos_prod"
_SERVICE_ACCOUNT_DOMAIN = ".gserviceaccount.com"


def _iam_database_user() -> str:
    credentials, _project_id = google.auth.default()
    service_account_email = getattr(
        credentials, "service_account_email", None
    ) or getattr(credentials, "signer_email", None)
    if not service_account_email or not service_account_email.endswith(
        _SERVICE_ACCOUNT_DOMAIN
    ):
        raise RuntimeError(
            "Application Default Credentials must identify a service account"
        )
    return service_account_email.removesuffix(_SERVICE_ACCOUNT_DOMAIN)


@contextmanager
def cloud_sql_connection() -> Iterator:
    """Yield a pg8000 connection authenticated with the active service account."""
    with Connector() as connector:
        connection = connector.connect(
            _CLOUD_SQL_INSTANCE,
            "pg8000",
            user=_iam_database_user(),
            db=_CLOUD_SQL_DATABASE,
            enable_iam_auth=True,
        )
        try:
            yield connection
        finally:
            connection.close()
