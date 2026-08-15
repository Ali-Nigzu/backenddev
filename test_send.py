"""Unit tests for BigQuery delivery without real cloud writes."""

import copy
import unittest
from unittest.mock import patch

from send import Send
from send.send import _parse_bigquery_link

DESTINATION = (
    "https://console.cloud.google.com/bigquery"
    "?ws=!1m5!1m4!4m3!1scamosbase"
    "!2sOrg_Test01"
    "!3sSite_Test01_Logs"
)


def row():
    return {
        "device_id": 0,
        "event_id": 123456789012345678,
        "event": 1,
        "timestamp": "2026-08-15T11:34:56.333Z",
        "sex": 1,
        "age_bucket": 3,
    }


class SendTests(unittest.TestCase):
    def test_valid_url_extracts_complete_destination(self):
        self.assertEqual(
            _parse_bigquery_link(DESTINATION),
            ("camosbase", "Org_Test01", "Site_Test01_Logs"),
        )

    def test_invalid_urls_are_rejected(self):
        invalid = (
            "http://console.cloud.google.com/bigquery?ws=!1sa!2sb!3sc",
            "https://example.com/bigquery?ws=!1sa!2sb!3sc",
            "https://console.cloud.google.com/storage?ws=!1sa!2sb!3sc",
            "https://console.cloud.google.com/bigquery",
            "https://console.cloud.google.com/bigquery?ws=!1sa!2sb",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                _parse_bigquery_link(value)

    @patch("send.send.bigquery.Client")
    def test_empty_batch_is_noop_before_client_or_destination(self, client_class):
        self.assertIsNone(Send()({"rows": []}, "not a URL"))
        client_class.assert_not_called()

    def test_malformed_output_batch_is_rejected(self):
        for value in (None, {}, {"rows": {}}, {"rows": [], "extra": True}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Send()(value, DESTINATION)

    def test_missing_extra_and_old_fields_are_rejected(self):
        cases = []
        missing = row()
        del missing["sex"]
        cases.append(missing)
        extra = row()
        extra["org_id"] = 1
        cases.append(extra)
        old = row()
        old["ts"] = old.pop("timestamp")
        cases.append(old)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                Send()({"rows": [value]}, DESTINATION)

    def test_wrong_types_and_malformed_timestamps_are_rejected(self):
        cases = []
        for field in ("device_id", "event_id", "event", "sex", "age_bucket"):
            value = row()
            value[field] = True
            cases.append(value)
        for timestamp in (
            "2026-08-15T11:34:56.333+00:00",
            "2026-08-15T11:34:56Z",
            "not-a-timestamp",
        ):
            value = row()
            value["timestamp"] = timestamp
            cases.append(value)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                Send()({"rows": [value]}, DESTINATION)

    @patch("send.send.bigquery.Client")
    def test_successful_append_passes_original_rows_and_options(self, client_class):
        client = client_class.return_value
        client.insert_rows_json.return_value = []
        rows = [row()]
        original = copy.deepcopy(rows)

        self.assertIsNone(Send()({"rows": rows}, DESTINATION))

        client_class.assert_called_once_with(project="camosbase")
        client.insert_rows_json.assert_called_once_with(
            "camosbase.Org_Test01.Site_Test01_Logs",
            rows,
            row_ids=[str(rows[0]["event_id"])],
            skip_invalid_rows=False,
            ignore_unknown_values=False,
        )
        self.assertEqual(rows, original)
        self.assertFalse(client.create_table.called)
        self.assertFalse(client.update_table.called)
        self.assertFalse(client.delete_table.called)

    @patch("send.send.bigquery.Client")
    def test_all_returned_row_errors_are_raised(self, client_class):
        errors = [
            {"index": 0, "errors": [{"reason": "invalid"}]},
            {"index": 1, "errors": [{"reason": "stopped"}]},
        ]
        client_class.return_value.insert_rows_json.return_value = errors
        with self.assertRaisesRegex(RuntimeError, "invalid.*stopped"):
            Send()({"rows": [row(), {**row(), "event_id": 2}]}, DESTINATION)

    @patch("send.send.bigquery.Client")
    def test_client_api_authentication_and_not_found_errors_propagate(
        self, client_class
    ):
        for error in (
            RuntimeError("API failure"),
            PermissionError("authentication failure"),
            FileNotFoundError("table not found"),
        ):
            with self.subTest(error=error):
                client_class.reset_mock()
                client_class.return_value.insert_rows_json.side_effect = error
                with self.assertRaises(type(error)):
                    Send()({"rows": [row()]}, DESTINATION)


if __name__ == "__main__":
    unittest.main()
