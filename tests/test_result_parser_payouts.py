import unittest
from pathlib import Path

from keiba_llm_agent.parsers.netkeiba_result_parser import parse_netkeiba_result_html


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "netkeiba_result_detailed_sample.html"


class ResultParserPayoutsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.html = FIXTURE_PATH.read_text(encoding="utf-8")

    def test_parse_single_win_payout(self) -> None:
        result_data = parse_netkeiba_result_html(self.html)
        win = next(item for item in result_data.payouts if item.bet_type == "単勝")
        self.assertEqual(win.combination, "4")
        self.assertEqual(win.payout, 400)
        self.assertEqual(win.popularity, 1)

    def test_parse_multiple_place_payouts(self) -> None:
        result_data = parse_netkeiba_result_html(self.html)
        places = [item for item in result_data.payouts if item.bet_type == "複勝"]
        self.assertEqual([item.combination for item in places], ["4", "15", "9"])
        self.assertEqual([item.payout for item in places], [170, 250, 500])

    def test_parse_multiple_wide_payouts(self) -> None:
        result_data = parse_netkeiba_result_html(self.html)
        wides = [item for item in result_data.payouts if item.bet_type == "ワイド"]
        self.assertEqual([item.combination for item in wides], ["4-15", "4-9", "9-15"])
        self.assertEqual([item.payout for item in wides], [580, 1320, 2310])

    def test_parse_supported_types_and_integer_payouts(self) -> None:
        result_data = parse_netkeiba_result_html(self.html)
        bet_types = {item.bet_type for item in result_data.payouts}
        self.assertTrue({"単勝", "複勝", "ワイド", "馬連", "馬単", "三連複", "三連単"} <= bet_types)
        self.assertTrue(all(isinstance(item.payout, int) for item in result_data.payouts))


if __name__ == "__main__":
    unittest.main()
