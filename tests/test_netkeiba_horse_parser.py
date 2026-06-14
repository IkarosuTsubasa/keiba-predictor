from __future__ import annotations

import unittest
from pathlib import Path

from keiba_llm_agent.parsers.netkeiba_horse_parser import parse_horse_recent_runs


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "netkeiba_horse_sample.html"


class NetkeibaHorseParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.html = FIXTURE_PATH.read_text(encoding="utf-8")

    def test_parse_horse_recent_runs_returns_list(self) -> None:
        recent_runs = parse_horse_recent_runs(self.html, limit=5)
        self.assertGreater(len(recent_runs), 0)
        self.assertEqual(recent_runs[0].date, "2026-05-17")
        self.assertEqual(recent_runs[0].course, "東京")
        self.assertEqual(recent_runs[0].surface, "芝")
        self.assertEqual(recent_runs[0].distance, 1600)
        self.assertEqual(recent_runs[0].track_condition, "良")
        self.assertEqual(recent_runs[0].finish, 17)
        self.assertEqual(recent_runs[0].field_size, 18)
        self.assertEqual(recent_runs[0].jockey, "騎手A")
        self.assertEqual(recent_runs[0].odds, 127.3)
        self.assertEqual(recent_runs[0].popularity, 17)
        self.assertEqual(recent_runs[0].passing_order, "10-10-9-8")
        self.assertEqual(recent_runs[0].corner_positions, [10, 10, 9, 8])
        self.assertEqual(recent_runs[0].final_3f, 33.9)
        self.assertEqual(recent_runs[0].margin, "1.2")
        self.assertIsNotNone(recent_runs[0].date)
        self.assertIsNotNone(recent_runs[0].course)
        self.assertIsNotNone(recent_runs[0].surface)
        self.assertIsNotNone(recent_runs[0].distance)
        self.assertIsNotNone(recent_runs[0].finish)
        self.assertIsNotNone(recent_runs[0].field_size)

    def test_limit_parameter_is_respected(self) -> None:
        recent_runs = parse_horse_recent_runs(self.html, limit=3)
        self.assertEqual(len(recent_runs), 3)

    def test_default_limit_returns_full_result_table(self) -> None:
        recent_runs = parse_horse_recent_runs(self.html)
        limited_runs = parse_horse_recent_runs(self.html, limit=5)
        self.assertGreater(len(recent_runs), len(limited_runs))

    def test_returns_empty_list_when_table_missing(self) -> None:
        recent_runs = parse_horse_recent_runs("<html><body>no table</body></html>", limit=5)
        self.assertEqual(recent_runs, [])


if __name__ == "__main__":
    unittest.main()
