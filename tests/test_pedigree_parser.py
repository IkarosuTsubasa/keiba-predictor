from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent import main as main_module
from keiba_llm_agent.fetchers.netkeiba_horse_fetcher import DB_NETKEIBA_ENCODING
from keiba_llm_agent.pedigree import pedigree_analyzer
from keiba_llm_agent.pedigree.pedigree_analyzer import fetch_pedigree_html
from keiba_llm_agent.pedigree.pedigree_parser import parse_pedigree_info


ROOT_DIR = Path(__file__).resolve().parents[1]
HORSE_FIXTURE_PATH = ROOT_DIR / "tests" / "fixtures" / "netkeiba_horse_sample.html"
PEDIGREE_FIXTURE_PATH = ROOT_DIR / "tests" / "fixtures" / "netkeiba_pedigree_sample.html"


class FakeResponse:
    def __init__(self, text: str) -> None:
        self._text = text
        self.apparent_encoding = "windows-1251"
        self.encoding = None

    @property
    def text(self) -> str:
        return self._text

    def raise_for_status(self) -> None:
        return None


class PedigreeParserTests(unittest.TestCase):
    def test_parse_sire_dam_damsire_from_fixture(self) -> None:
        html = PEDIGREE_FIXTURE_PATH.read_text(encoding="utf-8")
        pedigree = parse_pedigree_info(html, horse_id="2021104073", horse_name="サンプルホース")
        self.assertEqual(pedigree.sire, "ハーツクライ")
        self.assertEqual(pedigree.dam, "サンプル母")
        self.assertEqual(pedigree.damsire, "キングカメハメハ")

    def test_missing_pedigree_does_not_fail(self) -> None:
        pedigree = parse_pedigree_info("<html><body><div>no pedigree</div></body></html>", horse_id="x")
        self.assertIsNone(pedigree.sire)
        self.assertIsNone(pedigree.dam)
        self.assertIsNone(pedigree.damsire)

    def test_parse_pedigree_falls_back_from_horse_page_to_pedigree_page(self) -> None:
        horse_html = """
        <html><body>
          <div class="db_head_regist fc">
            <ul class="db_detail_menu">
              <li><a href="https://db.netkeiba.com/horse/ped/2021104073/">血統</a></li>
            </ul>
          </div>
        </body></html>
        """
        pedigree_html = PEDIGREE_FIXTURE_PATH.read_text(encoding="utf-8")
        initial = parse_pedigree_info(horse_html, horse_id="2021104073", horse_name="サンプルホース")
        self.assertIsNone(initial.sire)
        with patch("keiba_llm_agent.main.fetch_pedigree_html", return_value=pedigree_html):
            result = main_module.run_parse_pedigree("2021104073", horse_name="サンプルホース")
        self.assertEqual(result["sire"], "ハーツクライ")
        self.assertEqual(result["dam"], "サンプル母")
        self.assertEqual(result["damsire"], "キングカメハメハ")

    def test_fetch_pedigree_html_forces_db_euc_jp_encoding(self) -> None:
        response = FakeResponse(PEDIGREE_FIXTURE_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            pedigree_analyzer, "PEDIGREE_HTML_CACHE_DIR", Path(temp_dir)
        ), patch("keiba_llm_agent.pedigree.pedigree_analyzer.requests.get", return_value=response):
            html = fetch_pedigree_html("2021104073", force_refresh=True)

        self.assertIn("ハーツクライ", html)
        self.assertEqual(response.encoding, DB_NETKEIBA_ENCODING)


if __name__ == "__main__":
    unittest.main()
