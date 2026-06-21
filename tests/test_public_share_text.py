import sys
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from web_pages.public_llm import build_public_share_text
import web_public


def _to_int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class PublicShareTextTests(unittest.TestCase):
    def test_share_header_includes_race_name(self) -> None:
        text = build_public_share_text(
            {
                "location": "東京",
                "race_id": "202606210511",
                "race_name": "テストステークス",
            },
            "gemini",
            {"1": "◎", "2": "○", "3": "▲"},
            [],
            max_chars=140,
            share_detail_label="詳細はこちら",
            share_url=lambda _row: "https://example.test/race/202606210511",
            share_hashtag="#いかいもAI競馬 #競馬",
            to_int_or_none=_to_int_or_none,
        )

        self.assertEqual(text.splitlines()[0], "#東京競馬 11R テストステークス")

    def test_injected_share_button_uses_race_subtitle(self) -> None:
        runtime = web_public._public_share_runtime_html()

        self.assertIn('const PUBLIC_API_BASE_PATH = "/keiba/api/public";', runtime)
        self.assertIn("const buildDetailedShareText = (race, fallbackTitle = \"\", fallbackName = \"\") => {", runtime)
        self.assertIn("const fetchRaceDetail = async (raceCard) => {", runtime)
        self.assertIn("const text = await buildShareTextForRace({", runtime)
        self.assertIn('lines.push("", SHARE_HASHTAG);', runtime)
        self.assertIn("await navigator.share({ text });", runtime)
        self.assertNotIn("await drawShareImage(payload)", runtime)


if __name__ == "__main__":
    unittest.main()
