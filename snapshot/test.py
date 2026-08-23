import copy
import json
from pathlib import Path

from snapshot import Snapshot


ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / "local"


def read(name):
    return json.loads((LOCAL / name).read_text(encoding="utf-8"))


def write(name, value):
    (LOCAL / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    baseline = {name: read(name) for name in ("devices.json", "events.json", "snapshot.json")}
    try:
        assert Snapshot(1)
        row = read("snapshot.json")
        machine = row["state"]["current_machine"]
        assert row["ts"] == "2026-08-23T15:00:00Z"
        assert row["state"]["stable_until"] == row["ts"]
        assert machine["occupancy"] == 0 and machine["entry_fifo"] == []
        assert row["state"]["recent_events"] == []
        assert machine["all_time"]["entrances"] == [15]
        assert machine["all_time"]["exits"] == [15]
        assert machine["all_time"]["age_counts"] == [3, 3, 3, 2, 2, 2]
        assert machine["all_time"]["sex_counts"] == [8, 7]
        assert row["payload"]["traffic_devices"] == [{"device_id": 1, "name": "Front Door"}]
        first = copy.deepcopy(row)
        assert Snapshot(1) and read("snapshot.json") == first
        devices = read("devices.json")
        devices[0]["analyzed_until"] = "2026-08-23T16:15:00Z"
        write("devices.json", devices)
        assert Snapshot(1)
        zero = read("snapshot.json")
        assert zero["ts"] == "2026-08-23T16:15:00Z"
        assert zero["state"]["current_machine"]["occupancy"] == 0
        devices.append({"id": 2, "name": "Rear Door", "site_id": 1, "gcs_source_uri": "gs://example/rear-door/", "status": "enabled", "analysis_interval_minutes": 15, "analysis_config": {}, "analyzed_until": "2026-08-23T14:30:00Z", "created_at": "2026-08-23T14:30:00Z", "updated_at": "2026-08-23T14:30:00Z"})
        write("devices.json", devices)
        assert Snapshot(1)
        assert len(read("snapshot.json")["payload"]["traffic_devices"]) == 2
        print("snapshot MVP tests passed")
    finally:
        for name, value in baseline.items():
            write(name, value)


if __name__ == "__main__":
    main()

