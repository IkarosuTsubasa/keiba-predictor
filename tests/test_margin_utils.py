from __future__ import annotations

import unittest

from keiba_llm_agent.utils.margin import parse_margin_value


class MarginParsingTests(unittest.TestCase):
    def test_symbolic_and_numeric_margins(self) -> None:
        cases = {
            "ハナ": 0.05,
            "アタマ": 0.1,
            "クビ": 0.2,
            "大差": 4.0,
            "0.8": 0.8,
            " １／２ ": 0.5,
            "1/2": 0.5,
            "1 1/4": 1.25,
            "1.1/4": 1.25,
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertAlmostEqual(parse_margin_value(raw) or 0.0, expected)

    def test_invalid_margin_returns_none(self) -> None:
        self.assertIsNone(parse_margin_value(""))
        self.assertIsNone(parse_margin_value(None))
        self.assertIsNone(parse_margin_value("計測不能"))


if __name__ == "__main__":
    unittest.main()
