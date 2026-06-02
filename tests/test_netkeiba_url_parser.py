from __future__ import annotations

import unittest

from keiba_llm_agent.parsers.netkeiba_url_parser import extract_race_id


class NetkeibaUrlParserTests(unittest.TestCase):
    def test_extracts_race_id_from_shutuba_url(self) -> None:
        url = "https://race.netkeiba.com/race/shutuba.html?race_id=202605180511"
        self.assertEqual(extract_race_id(url), "202605180511")

    def test_extracts_race_id_with_extra_query_params(self) -> None:
        url = "https://race.netkeiba.com/race/shutuba.html?race_id=202605180511&rf=race_list"
        self.assertEqual(extract_race_id(url), "202605180511")

    def test_extracts_race_id_from_result_url(self) -> None:
        url = "https://race.netkeiba.com/race/result.html?race_id=202605180511"
        self.assertEqual(extract_race_id(url), "202605180511")

    def test_extracts_race_id_from_http_url(self) -> None:
        url = "http://race.netkeiba.com/race/shutuba.html?race_id=202605180511"
        self.assertEqual(extract_race_id(url), "202605180511")

    def test_raises_when_race_id_missing(self) -> None:
        url = "https://race.netkeiba.com/race/shutuba.html?rf=race_list"
        with self.assertRaisesRegex(ValueError, "race_id not found in netkeiba URL"):
            extract_race_id(url)

    def test_raises_when_domain_not_supported(self) -> None:
        url = "https://example.com/race/shutuba.html?race_id=202605180511"
        with self.assertRaisesRegex(ValueError, "not a supported netkeiba URL"):
            extract_race_id(url)

    def test_race_id_keeps_string_format(self) -> None:
        url = "https://race.netkeiba.com/race/shutuba.html?race_id=012345678901"
        race_id = extract_race_id(url)
        self.assertEqual(race_id, "012345678901")
        self.assertIsInstance(race_id, str)


if __name__ == "__main__":
    unittest.main()
