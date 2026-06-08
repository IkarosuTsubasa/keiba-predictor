import sys
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from web_data import agent_predictions


class AgentPredictionsScopeTests(unittest.TestCase):
    def test_scope_matches_local_from_race_info_scope_key(self) -> None:
        payload = {
            "race_id": "202644031102",
            "race_info": {
                "scope_key": "local",
                "source": "local",
                "course": "",
                "surface": "ダート",
            },
        }

        self.assertTrue(agent_predictions._scope_matches(payload, "local"))
        self.assertFalse(agent_predictions._scope_matches(payload, "central_dirt"))


if __name__ == "__main__":
    unittest.main()
