import unittest
from pathlib import Path

from keiba_llm_agent.parsers.netkeiba_result_parser import parse_netkeiba_result_html


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "netkeiba_result_detailed_sample.html"


class ResultParserFinishOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.html = FIXTURE_PATH.read_text(encoding="utf-8")

    def test_parse_full_finish_order(self) -> None:
        result_data = parse_netkeiba_result_html(self.html)
        self.assertEqual(len(result_data.finish_order), 4)
        first = result_data.finish_order[0]
        self.assertEqual(first.finish, 1)
        self.assertEqual(first.horse_no, 4)
        self.assertEqual(first.horse_name, "ロブチェン")
        self.assertEqual(first.jockey, "松山")
        self.assertEqual(first.popularity, 1)
        self.assertEqual(first.odds, 4.0)

    def test_result_top3_is_derived_from_finish_order(self) -> None:
        result_data = parse_netkeiba_result_html(self.html)
        self.assertEqual(result_data.result.first, 4)
        self.assertEqual(result_data.result.second, 15)
        self.assertEqual(result_data.result.third, 9)


if __name__ == "__main__":
    unittest.main()
