from __future__ import annotations

import json
import unittest
from pathlib import Path

from keiba_llm_agent.parsers.netkeiba_result_parser import parse_netkeiba_result_html


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "netkeiba_result_sample.html"


class NetkeibaResultParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.html = FIXTURE_PATH.read_text(encoding="utf-8")

    def test_parse_result_html_extracts_race_id_and_top3(self) -> None:
        result_data = parse_netkeiba_result_html(self.html)
        self.assertEqual(result_data.race_id, "202605020811")
        self.assertEqual(result_data.result.first, 12)
        self.assertEqual(result_data.result.second, 8)
        self.assertEqual(result_data.result.third, 7)

    def test_payouts_missing_does_not_fail(self) -> None:
        html = self.html.replace("<table class=\"PayTable\">", "<table class=\"OtherTable\">")
        result_data = parse_netkeiba_result_html(html)
        self.assertIsInstance(result_data.payouts, list)

    def test_result_data_can_serialize_to_json(self) -> None:
        result_data = parse_netkeiba_result_html(self.html)
        payload = json.dumps(result_data.model_dump(by_alias=True), ensure_ascii=False)
        self.assertIn("\"1st\": 12", payload)


if __name__ == "__main__":
    unittest.main()
