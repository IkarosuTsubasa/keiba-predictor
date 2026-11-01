from __future__ import annotations

from datetime import date, timedelta
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_simulator_calibration import prepare, candidates


class SimulatorCalibrationDataTest(unittest.TestCase):
    def test_labels_never_enter_inputs_and_latest_dates_remain_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); snapshot = root / "source"; audit_dir = snapshot / "agent_backtest_audit"
            audit_dir.mkdir(parents=True)
            first_id = "202601010101"
            for i in range(12):
                rid = str(int(first_id) + i); day = (date(2026, 7, 1) + timedelta(days=i)).isoformat()
                horses = []
                for no in range(1, 6):
                    runs = [{"race_id": "202501010101", "date": "2025-01-01", "distance": 1000, "surface": "ダート", "finish": no, "field_size": 5, "final_3f": 40., "odds": 1., "popularity": 1}, {"race_id": rid, "date": day, "finish": 1}, {"date": "2099-01-01"}, {"date": None}]
                    if i:
                        runs.append({"race_id": first_id, "date": "2026-07-01", "distance": 1000, "surface": "ダート", "finish": no, "field_size": 5, "final_3f": 38. + no / 10})
                    horses.append({"horse_no": no, "horse_id": str(no), "horse_name": str(no), "odds": 1., "popularity": 1, "recent_runs": runs})
                payload = {"race_id": rid, "audit_kind": "full", "inputs": {"race_data": {"race_info": {"race_id": rid, "race_date": day, "course": "川崎", "surface": "ダート", "distance": 1000, "source": "local"}, "horses": horses}}, "outcomes": {"result": {"finish_order": [{"horse_no": no, "finish": no, "time": f"1:0{no}.0", "odds": 1} for no in range(1, 6)]}}}
                (audit_dir / f"{rid}.json.gz").write_bytes(gzip.compress(json.dumps(payload).encode()))
            output = root / "output"; prepare(snapshot, output)
            cases = json.loads((output / "cases.json").read_text(encoding="utf-8"))
            first = cases[0]
            self.assertEqual(first["labels"][0]["final_3f"], 38.1)
            self.assertEqual(first["race"]["horses"][0]["recent_runs"][0]["final_3f"], 40.)
            for c in cases:
                text = json.dumps(c["race"])
                for field in ('"odds"', '"popularity"', '"labels"', '"running_style"'):
                    self.assertNotIn(field, text)
                for h in c["race"]["horses"]:
                    self.assertTrue(all(r["date"] < c["race"]["race_info"]["race_date"] for r in h["recent_runs"]))
            dev = [c for c in cases if c["split"] == "development"]; hold = [c for c in cases if c["split"] == "holdout"]
            self.assertLess(max(c["race"]["race_info"]["race_date"] for c in dev), min(c["race"]["race_info"]["race_date"] for c in hold))
            self.assertEqual(len({c["fold"] for c in dev}), 5)
            self.assertTrue(all(c["fold"] is None for c in hold))
            with self.assertRaises(ValueError): prepare(snapshot, output)
            with self.assertRaises(ValueError): prepare(snapshot, snapshot / "derived")

    def test_final_version_wins_over_root_and_early(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); rid = "202601010101"
            for relative, stage, value in ((f"{rid}.json.gz", "final", "root"), (f"_versions/{rid}/early/new.json.gz", "early", "early"), (f"_versions/{rid}/final/old.json.gz", "final", "final")):
                p = root / relative; p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(gzip.compress(json.dumps({"race_id": rid, "prediction_stage": stage, "value": value}).encode()))
            rows = list(candidates(root))
            self.assertEqual(len(rows), 1); self.assertEqual(rows[0][2]["value"], "final")


if __name__ == "__main__": unittest.main()
