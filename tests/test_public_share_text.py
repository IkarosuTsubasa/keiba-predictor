import sys
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from web_pages.public_llm import build_public_share_text


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


if __name__ == "__main__":
    unittest.main()
