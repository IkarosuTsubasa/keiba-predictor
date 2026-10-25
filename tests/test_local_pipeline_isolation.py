import base64
import builtins
from contextlib import contextmanager
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import web_app
from web_data import agent_predictions
from keiba_llm_agent import main
from keiba_llm_agent.backtest.audit_snapshot import read_backtest_audit_payload, attach_result_to_backtest_audit
from keiba_llm_agent.fetchers import netkeiba_fetcher, netkeiba_horse_fetcher
from keiba_llm_agent.parsers.netkeiba_race_parser import parse_netkeiba_shutuba_html
from keiba_llm_agent.scoring import recent_run_scorer
from tests.test_netkeiba_fetcher import FakeResponse

RACE_ID = "202644102601"
URL = "https://nar.netkeiba.com/race/shutuba.html?race_id=" + RACE_ID
OPTIONAL_MODULES = ("keiba_llm_agent.scoring.workout_shadow", "keiba_llm_agent.fetchers.netkeiba_workout_fetcher",
                    "pipeline.workout_shadow_metrics", "workout_shadow_metrics")


def local_html():
    rows = ''.join(f'<tr class="HorseList"><td class="Waku">{n}</td><td class="Umaban">{n}</td>'
                   f'<td class="HorseInfo"><a href="https://db.netkeiba.com/horse/202310000{n}/">テスト馬{n}</a></td>'
                   '<td class="Jockey">テスト騎手</td><td class="Weight">56</td></tr>' for n in range(1, 7))
    return ('<meta charset="utf-8"><div class="RaceName">地方テスト</div>'
            '<div class="RaceData01">2026年10月26日 大井 1R ダート1400m / 天候:晴 / 馬場:良</div>'
            '<table class="Shutuba_Table">' + rows + '</table>')


@contextmanager
def blocked_optional_modules():
    original = builtins.__import__
    attempted = []
    def guarded(name, *args, **kwargs):
        if name in OPTIONAL_MODULES:
            attempted.append(name)
            raise ImportError("optional module intentionally unavailable")
        return original(name, *args, **kwargs)
    with patch("builtins.__import__", side_effect=guarded):
        yield attempted


class LocalPipelineIsolationTests(unittest.TestCase):
    def test_local_full_flow_survives_missing_workout_modules_and_keeps_ranking(self):
        horse_html = (ROOT / "tests/fixtures/netkeiba_horse_sample.html").read_text(encoding="utf-8")
        requested = []
        def fetch(url, **kwargs):
            requested.append(url)
            if url == URL:
                return FakeResponse(local_html())
            if url.startswith("https://db.netkeiba.com/horse/result/"):
                return FakeResponse(horse_html)
            raise AssertionError("Unexpected network request: " + url)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lessons = root / "lessons.json"
            lessons.write_text("[]", encoding="utf-8")
            base = root / "render"
            with (patch.object(netkeiba_fetcher, "HTML_CACHE_DIR", root / "html"),
                  patch.object(netkeiba_fetcher, "RACE_DATA_DIR", root / "race"),
                  patch.object(netkeiba_horse_fetcher, "HORSE_HTML_CACHE_DIR", root / "horse"),
                  patch.object(main, "DEFAULT_PREDICTIONS_DIR", root / "predictions"),
                  patch.object(main, "DEFAULT_BACKTEST_AUDIT_DIR", root / "audit"),
                  patch.object(web_app, "BASE_DIR", base),
                  patch.object(netkeiba_fetcher.requests, "get", side_effect=fetch),
                  patch.object(recent_run_scorer, "build_pedigree_analyses_for_race", return_value=[]),
                  patch.object(main, "capture_public_workout", side_effect=AssertionError("local cannot capture workouts")) as capture,
                  blocked_optional_modules() as attempted):
                result = main.run_predict_race(URL, force_refresh=True, skip_social=True,
                                               llm_provider="mock", lessons_path=lessons)
                prediction_path = Path(result["prediction_path"])
                prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
                audit_path = Path(result["backtest_audit_path"])
                audit = read_backtest_audit_payload(audit_path)
                self.assertEqual(result["scope_key"], "local")
                self.assertEqual(result["scoring_profile"], "local_accuracy_default")
                self.assertEqual(prediction["scope_reranker_result"]["scope"], "local")
                self.assertTrue(prediction["scope_reranker_result"]["applied"])
                self.assertEqual(len(prediction["marks"]), 5)
                self.assertTrue(all(h["recent_runs"] for h in audit["inputs"]["race_data"]["horses"]))
                self.assertNotIn("shadow_predictions", audit)
                self.assertNotIn("supplementary_observations", audit["inputs"])
                baseline, _ = main.run_analysis(result["race_data_path"], output_path=root / "baseline.json",
                                                lessons_path=lessons, llm_provider="mock", scoring_profile="local_accuracy_default")
                self.assertEqual(baseline.model_dump()["horse_scores"], prediction["horse_scores"])
                self.assertEqual(baseline.model_dump()["marks"], prediction["marks"])
                callback = {"result": {
                    "content_base64": base64.b64encode(prediction_path.read_bytes()).decode(),
                    "audit_content_base64": base64.b64encode(audit_path.read_bytes()).decode()}}
                stored = []
                for stage in ("early", "final"):
                    task = {"race_id": RACE_ID, "task_id": "local-test-" + stage,
                            "bundle_meta": {"prediction_stage": stage, "final_prediction_expected": True}}
                    stored.append(web_app._apply_remote_agent_prediction_result(task, callback))
                podium = {"race_id": RACE_ID, "result": {"1st": 1, "2nd": 2, "3rd": 3}}
                agent_predictions.attach_result_to_agent_backtest_audit(base, RACE_ID, podium)
                attach_result_to_backtest_audit(audit_path, podium)
                for saved in stored:
                    restored = read_backtest_audit_payload(saved["backtest_audit_path"])
                    self.assertEqual(restored["outcomes"]["result"], podium)
                    self.assertNotIn("public_workout_comparison", restored["outcomes"])
                    public = json.loads(Path(saved["prediction_path"]).read_text(encoding="utf-8"))
                    self.assertEqual(public["marks"], prediction["marks"])
                capture.assert_not_called()
                self.assertEqual(attempted, [])
        self.assertEqual(requested.count(URL), 1)
        self.assertEqual(len(requested), 7)

    def test_local_url_analysis_and_incomplete_scope_never_capture(self):
        race = parse_netkeiba_shutuba_html(local_html(), RACE_ID)
        with patch.object(main, "capture_public_workout", side_effect=AssertionError) as capture:
            for source, scope in (("local", "local"), (None, None), ("central", "central_turf")):
                race.race_info.source = source
                race.race_info.scope_key = scope
                self.assertIsNone(main._capture_prediction_supplements(race))
            capture.assert_not_called()
        race.race_info.source = race.race_info.scope_key = "local"
        with tempfile.TemporaryDirectory() as temp, patch.object(main, "fetch_and_parse_netkeiba_race", return_value=race), \
                patch.object(main, "save_race_data", return_value=Path(temp) / "race.json"), \
                patch.object(main, "run_analysis", return_value=(None, Path(temp) / "pred.json")) as analyze, \
                blocked_optional_modules() as attempted:
            main.run_analyze_url(URL)
            self.assertIsNone(analyze.call_args.kwargs["supplementary_observations"])
            self.assertEqual(attempted, [])

    def test_optional_import_failure_cannot_break_central_audit_or_settlement(self):
        from tests.test_workout_shadow import fixture
        race, prediction, observation = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(main, "DEFAULT_BACKTEST_AUDIT_DIR", root / "audit"), blocked_optional_modules():
                path = main._write_prediction_backtest_audit(race_data=race, prediction=prediction,
                        supplementary_observations={"public_workout": observation})
                audit = read_backtest_audit_payload(path)
                self.assertEqual(audit["shadow_predictions"]["public_workout"]["status"], "failed")
                result = {"race_id": race.race_info.race_id, "result": {"1st": 1, "2nd": 2, "3rd": 3}}
                attach_result_to_backtest_audit(path, result)
                saved = agent_predictions.write_agent_backtest_audit_payload(root / "render", race.race_info.race_id, audit)
                agent_predictions.attach_result_to_agent_backtest_audit(root / "render", race.race_info.race_id, result)
            for entry in (path, saved):
                payload = read_backtest_audit_payload(entry)
                self.assertEqual(payload["outcomes"]["result"], result)
                self.assertEqual(payload["outcomes"]["public_workout_comparison"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
